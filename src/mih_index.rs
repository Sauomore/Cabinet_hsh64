//! HSH-64 多索引哈希（MIH）粗排索引
//!
//! 把 52-bit sim 切成 B 段等长子串，每段单独建立倒排索引。
//! 52 可被 4 整除，因此默认 segment_count=4，每段 13 bit。

use crate::{
    embedding::{EmbeddingModel, FileCachedEmbedding},
    encoder::Encoder,
    error::EncodeError,
    hsh64::hamming_distance64,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// 搜索结果
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub word: String,
    pub score: f32,
    pub hamming: u32,
}

/// MIH 语义索引
pub struct MihSemanticIndex {
    encoder: Encoder,
    embedding: Arc<dyn EmbeddingModel>,
    segment_indices: Vec<HashMap<u32, Vec<String>>>,
    segment_count: usize,
    segment_bits: usize,
    reranker: Option<Arc<dyn EmbeddingModel>>,
    strict_vocab: bool,
}

impl MihSemanticIndex {
    /// 从已有编码器和 embedding 缓存文件构建 MIH 索引
    pub fn build(
        encoder: Encoder,
        embedding_path: impl AsRef<Path>,
        segment_count: usize,
        strict_vocab: bool,
    ) -> Result<Self, EncodeError> {
        let embedding: Arc<dyn EmbeddingModel> =
            match FileCachedEmbedding::new(embedding_path.as_ref(), encoder.embed_dim()) {
                Ok(model) => Arc::new(model),
                Err(e) => {
                    return Err(EncodeError::Config(format!(
                        "无法加载 embedding 缓存 '{}': {}",
                        embedding_path.as_ref().display(),
                        e
                    )));
                }
            };

        Self::build_with_embedding(encoder, embedding, segment_count, strict_vocab)
    }

    /// 非对称距离评分：query 用连续投影幅度，document 用量化后的 bit
    ///
    /// score = Σ_j u_q[j] * (2*b_d[j] - 1)
    pub fn asymmetric_score(query_proj: &[f32], sim_code: u64) -> f32 {
        let n = query_proj.len().min(52);
        let mut score = 0.0f32;
        for i in 0..n {
            let bit_val = if (sim_code >> i) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            };
            score += query_proj[i] * bit_val;
        }
        score
    }

    /// 用已加载的 embedding 模型构建 MIH 索引
    pub fn build_with_embedding(
        encoder: Encoder,
        embedding: Arc<dyn EmbeddingModel>,
        segment_count: usize,
        strict_vocab: bool,
    ) -> Result<Self, EncodeError> {
        if segment_count == 0 || segment_count > 52 {
            return Err(EncodeError::Config(format!(
                "segment_count 必须在 [1, 52] 范围内: {}",
                segment_count
            )));
        }
        if 52 % segment_count != 0 {
            return Err(EncodeError::Config(format!(
                "segment_count 必须能整除 52-bit sim: {} 不能整除 52",
                segment_count
            )));
        }

        let vocab = embedding.vocab();
        if vocab.is_empty() {
            return Err(EncodeError::Config("embedding 缓存为空".to_string()));
        }

        let segment_bits = 52 / segment_count;
        let mut segment_indices: Vec<HashMap<u32, Vec<String>>> =
            (0..segment_count).map(|_| HashMap::new()).collect();

        for word in vocab {
            let code = match encoder.encode_word_with_pos(&word, "n") {
                c => c,
            };
            let sim = code.sim();
            for seg in 0..segment_count {
                let value = segment_value(sim, seg, segment_bits);
                segment_indices[seg]
                    .entry(value)
                    .or_default()
                    .push(word.clone());
            }
        }

        Ok(Self {
            encoder,
            embedding,
            segment_indices,
            segment_count,
            segment_bits,
            reranker: None,
            strict_vocab,
        })
    }

    /// 用独立精排模型构建 MIH 索引
    pub fn build_with_reranker(
        encoder: Encoder,
        embedding: Arc<dyn EmbeddingModel>,
        reranker: Arc<dyn EmbeddingModel>,
        segment_count: usize,
        strict_vocab: bool,
    ) -> Result<Self, EncodeError> {
        let mut index =
            Self::build_with_embedding(encoder, embedding, segment_count, strict_vocab)?;
        index.reranker = Some(reranker);
        Ok(index)
    }

    /// 查询与单个词最相似的 top_k 个词
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        radius: u32,
        coarse_factor: usize,
    ) -> Result<Vec<SearchResult>, EncodeError> {
        if self.strict_vocab {
            let vocab: HashSet<String> = self.embedding.vocab().into_iter().collect();
            if !vocab.contains(query) {
                return Err(EncodeError::Config(format!(
                    "查询词 '{}' 不在词表中",
                    query
                )));
            }
        }

        let query_code = self.encoder.encode_word_with_pos(query, "n");
        let query_sim = query_code.sim();
        let _query_vec = self.get_query_vec(query);

        let candidates = self.collect_candidates(query_sim, radius);
        let mut scored: Vec<(String, u32)> = candidates
            .into_iter()
            .map(|word| {
                let code = self.encoder.encode_word_with_pos(&word, "n");
                let dist = code.sim_hamming_distance(&query_code);
                (word, dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.cmp(&b.1));

        let coarse_limit = if coarse_factor == usize::MAX {
            scored.len()
        } else {
            (coarse_factor * top_k).min(scored.len())
        };
        let coarse = &scored[..coarse_limit];

        let mut results: Vec<SearchResult> = if let Some(reranker) = &self.reranker {
            let query_rerank = reranker.embed(query);
            let mut ranked: Vec<SearchResult> = coarse
                .iter()
                .map(|(word, hamming)| {
                    let v = reranker.embed(word);
                    let score = cosine_similarity(&query_rerank, &v);
                    SearchResult {
                        word: word.clone(),
                        score,
                        hamming: *hamming,
                    }
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked
        } else {
            coarse
                .iter()
                .map(|(word, hamming)| SearchResult {
                    word: word.clone(),
                    score: -(*hamming as f32),
                    hamming: *hamming,
                })
                .collect()
        };

        results.truncate(top_k);
        Ok(results)
    }

    /// 自适应 radius 搜索：逐步扩大 Hamming 半径，直到候选池增长饱和。
    ///
    /// 策略：半径从 0 开始以步长 2 扩展。当同时满足
    ///   1) 候选数 >= max(top_k, coarse_factor * top_k)
    ///   2) 本轮新增候选占比 < growth_threshold（默认 10%）
    /// 时停止，避免在邻居尚未完全召回时过早截断。
    pub fn search_adaptive(
        &self,
        query: &str,
        top_k: usize,
        coarse_factor: usize,
        max_radius: u32,
    ) -> Result<(u32, Vec<SearchResult>), EncodeError> {
        self.search_adaptive_with_threshold(query, top_k, coarse_factor, max_radius, 0.1)
    }

    /// 带自定义增长阈值的可自适应 radius 搜索。
    pub fn search_adaptive_with_threshold(
        &self,
        query: &str,
        top_k: usize,
        coarse_factor: usize,
        max_radius: u32,
        growth_threshold: f32,
    ) -> Result<(u32, Vec<SearchResult>), EncodeError> {
        if self.strict_vocab {
            let vocab: HashSet<String> = self.embedding.vocab().into_iter().collect();
            if !vocab.contains(query) {
                return Err(EncodeError::Config(format!(
                    "查询词 '{}' 不在词表中",
                    query
                )));
            }
        }

        let query_code = self.encoder.encode_word_with_pos(query, "n");
        let query_sim = query_code.sim();
        let target = (coarse_factor * top_k).max(top_k);
        let mut prev_count: usize = 0;

        for radius in (0..=max_radius).step_by(2) {
            let candidates = self.collect_candidates(query_sim, radius);
            let count = candidates.len();
            let growth = if prev_count == 0 {
                f32::INFINITY
            } else {
                (count.saturating_sub(prev_count)) as f32 / prev_count as f32
            };

            if count >= target && (growth < growth_threshold || radius == max_radius) {
                // 自适应阶段只负责选 radius；精排阶段使用全部候选，避免 coarse_factor 截断真邻居。
                return Ok((radius, self.search(query, top_k, radius, usize::MAX)?));
            }
            prev_count = count;
        }
        Ok((
            max_radius,
            self.search(query, top_k, max_radius, usize::MAX)?,
        ))
    }

    /// 非对称距离搜索
    ///
    /// 与 `search` 不同：候选排序使用 query 的连续投影幅度与 document bit 的内积，
    /// 而不是 Hamming 距离。这能减小 sign 量化的信息损失。
    pub fn search_asymmetric(
        &self,
        query: &str,
        top_k: usize,
        radius: u32,
        coarse_factor: usize,
    ) -> Result<Vec<SearchResult>, EncodeError> {
        if self.strict_vocab {
            let vocab: HashSet<String> = self.embedding.vocab().into_iter().collect();
            if !vocab.contains(query) {
                return Err(EncodeError::Config(format!(
                    "查询词 '{}' 不在词表中",
                    query
                )));
            }
        }

        let query_code = self.encoder.encode_word_with_pos(query, "n");
        let query_sim = query_code.sim();
        let query_proj = self.encoder.project_word(query, query_code.feat());

        let candidates = self.collect_candidates(query_sim, radius);
        let coarse_limit = if coarse_factor == usize::MAX {
            candidates.len()
        } else {
            (coarse_factor * top_k).min(candidates.len())
        };

        let mut scored: Vec<(String, u32, f32)> = candidates
            .into_iter()
            .map(|word| {
                let code = self.encoder.encode_word_with_pos(&word, "n");
                let hamming = code.sim_hamming_distance(&query_code);
                let asym_score = Self::asymmetric_score(&query_proj, code.sim());
                (word, hamming, asym_score)
            })
            .collect();

        // 按非对称分数降序排列
        scored.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let coarse = &scored[..coarse_limit];

        let mut results: Vec<SearchResult> = if let Some(reranker) = &self.reranker {
            let query_rerank = reranker.embed(query);
            let mut ranked: Vec<SearchResult> = coarse
                .iter()
                .map(|(word, hamming, _)| {
                    let v = reranker.embed(word);
                    let score = cosine_similarity(&query_rerank, &v);
                    SearchResult {
                        word: word.clone(),
                        score,
                        hamming: *hamming,
                    }
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked
        } else {
            coarse
                .iter()
                .map(|(word, hamming, asym_score)| SearchResult {
                    word: word.clone(),
                    score: *asym_score,
                    hamming: *hamming,
                })
                .collect()
        };

        results.truncate(top_k);
        Ok(results)
    }

    /// 自适应非对称距离搜索
    pub fn search_adaptive_asymmetric(
        &self,
        query: &str,
        top_k: usize,
        coarse_factor: usize,
        max_radius: u32,
    ) -> Result<(u32, Vec<SearchResult>), EncodeError> {
        if self.strict_vocab {
            let vocab: HashSet<String> = self.embedding.vocab().into_iter().collect();
            if !vocab.contains(query) {
                return Err(EncodeError::Config(format!(
                    "查询词 '{}' 不在词表中",
                    query
                )));
            }
        }

        let query_code = self.encoder.encode_word_with_pos(query, "n");
        let query_sim = query_code.sim();
        let target = (coarse_factor * top_k).max(top_k);
        let mut prev_count: usize = 0;
        let growth_threshold = 0.1f32;

        for radius in (0..=max_radius).step_by(2) {
            let candidates = self.collect_candidates(query_sim, radius);
            let count = candidates.len();
            let growth = if prev_count == 0 {
                f32::INFINITY
            } else {
                (count.saturating_sub(prev_count)) as f32 / prev_count as f32
            };

            if count >= target && (growth < growth_threshold || radius == max_radius) {
                // 自适应阶段只负责选 radius；精排阶段使用全部候选。
                return Ok((
                    radius,
                    self.search_asymmetric(query, top_k, radius, usize::MAX)?,
                ));
            }
            prev_count = count;
        }
        Ok((
            max_radius,
            self.search_asymmetric(query, top_k, max_radius, usize::MAX)?,
        ))
    }

    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// 收集 query_sim 在 Hamming 半径 radius 内的所有候选词。
    ///
    /// 使用标准 MIH 思想：把 52-bit sim 切成 `segment_count` 段，每段
    /// `segment_bits` 位。对每一段枚举该段上 Hamming 距离不超过
    /// `min(radius, segment_bits)` 的所有段值，收集对应倒排桶中的词；
    /// 最后用完整 sim Hamming 距离 `<= radius` 做精确过滤。
    ///
    /// 该实现保证不遗漏任何完整 Hamming 距离 <= radius 的候选。
    pub fn collect_candidates(&self, query_sim: u64, radius: u32) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        // 每段最多允许的 Hamming 距离：完整距离 <= radius 的候选，
        // 必然在每一段上的距离也不超过 radius（同时不超过段长）。
        let segment_radius = (radius as usize).min(self.segment_bits);
        let max_seg_value = 1u32 << self.segment_bits;

        for seg in 0..self.segment_count {
            let query_seg = segment_value(query_sim, seg, self.segment_bits);

            for value in 0..max_seg_value {
                if hamming_distance64(value as u64, query_seg as u64) > segment_radius as u32 {
                    continue;
                }
                if let Some(words) = self.segment_indices[seg].get(&value) {
                    for word in words {
                        if seen.insert(word.clone()) {
                            result.push(word.clone());
                        }
                    }
                }
            }
        }

        // 用完整 sim Hamming 距离精确过滤
        result
            .into_iter()
            .filter(|word| {
                let code = self.encoder.encode_word_with_pos(word, "n");
                hamming_distance64(code.sim(), query_sim) <= radius
            })
            .collect()
    }

    fn get_query_vec(&self, query: &str) -> Vec<f32> {
        self.embedding.embed(query)
    }
}

/// 计算 sim 码第 seg 段的值
fn segment_value(sim: u64, seg: usize, segment_bits: usize) -> u32 {
    let shift = seg * segment_bits;
    let mask = (1u64 << segment_bits) - 1;
    ((sim >> shift) & mask) as u32
}

/// 余弦相似度
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedding;

    fn build_test_index() -> MihSemanticIndex {
        let encoder = Encoder::new();
        let words: Vec<String> = (0..100).map(|i| format!("词{}", i)).collect();
        let embedding = Arc::new(MockEmbedding::with_vocab(
            encoder.embed_dim(),
            words.clone(),
        ));
        MihSemanticIndex::build_with_embedding(encoder, embedding, 4, false).unwrap()
    }

    #[test]
    fn test_mih_build() {
        let index = build_test_index();
        assert_eq!(index.segment_count, 4);
        assert_eq!(index.segment_bits, 13);
    }

    #[test]
    fn test_mih_search_returns_results() {
        let index = build_test_index();
        let results = index.search("词0", 5, 0, 10).unwrap();
        assert!(!results.is_empty());
    }
}
