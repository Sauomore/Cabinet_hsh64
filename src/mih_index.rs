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
        let mut index = Self::build_with_embedding(encoder, embedding, segment_count, strict_vocab)?;
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
            ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            ranked
        } else {
            coarse
                .iter()
                .map(|(word, hamming)| SearchResult {
                    word: word.clone(),
                    score: -( *hamming as f32),
                    hamming: *hamming,
                })
                .collect()
        };

        results.truncate(top_k);
        Ok(results)
    }

    /// 自适应 radius 搜索：从 0 开始扩展直到候选足够
    pub fn search_adaptive(
        &self,
        query: &str,
        top_k: usize,
        coarse_factor: usize,
        max_radius: u32,
    ) -> Result<Vec<SearchResult>, EncodeError> {
        for radius in (0..=max_radius).step_by(2) {
            let results = self.search(query, top_k, radius, coarse_factor)?;
            if results.len() >= top_k {
                return Ok(results);
            }
        }
        self.search(query, top_k, max_radius, coarse_factor)
    }

    fn collect_candidates(&self, query_sim: u64, radius: u32) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        // 每段允许的最大 Hamming 距离
        let segment_radius = (radius as usize) / self.segment_count;

        for seg in 0..self.segment_count {
            let query_seg = segment_value(query_sim, seg, self.segment_bits);
            let start = if query_seg >= segment_radius as u32 {
                query_seg - segment_radius as u32
            } else {
                0
            };
            let end = (query_seg + segment_radius as u32 + 1)
                .min(1 << self.segment_bits);

            for value in start..end {
                if let Some(words) = self.segment_indices[seg].get(&value) {
                    for word in words {
                        if seen.insert(word.clone()) {
                            result.push(word.clone());
                        }
                    }
                }
            }
        }

        // 再按完整 sim Hamming 距离过滤
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
        let embedding = Arc::new(MockEmbedding::with_vocab(encoder.embed_dim(), words.clone()));
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
