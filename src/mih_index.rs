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
        // ⚠️ segment_count = 1 是【退化】情形，明确拒绝而不是静默给错结果。
        //
        //    原因：段值宽度 = 52 bit，但索引键是 u32，会被截断成低 32 位；
        //    且「枚举段内汉明球」在 52 位上有 2^52 个值，不可能枚举。
        //    历史上允许 seg=1，结果是对所有 radius 都返回 0 候选 ——
        //    看起来像「索引坏了」，实际是这个配置在数学上就不成立。
        //    多索引哈希（MIH）本身要求 m >= 2 段。
        if segment_count == 1 {
            return Err(EncodeError::Config(
                "segment_count = 1 不支持：段值会占满 52 位，既超出 u32 键宽、\
                 也无法枚举段内汉明球。多索引哈希要求至少 2 段（建议 4 或 13）。"
                    .to_string(),
            ));
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
            // 词性从 encoder 的 pos_tags 表解析（双语词表时中文用 jieba 标签、
            // 英文用 PTB 标签）。历史上这里硬编码 "n"，导致 feat 恒为 0x0、
            // 第二种语言的词全部落入 FALLBACK。
            let code = encoder.encode_word_auto(&word);
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

        let query_code = self.encoder.encode_word_auto(query);
        let query_sim = query_code.sim();
        let _query_vec = self.get_query_vec(query);

        let candidates = self.collect_candidates(query_sim, radius);
        let mut scored: Vec<(String, u32)> = candidates
            .into_iter()
            .map(|word| {
                let code = self.encoder.encode_word_auto(&word);
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

        let query_code = self.encoder.encode_word_auto(query);
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

        let query_code = self.encoder.encode_word_auto(query);
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
                let code = self.encoder.encode_word_auto(&word);
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

        let query_code = self.encoder.encode_word_auto(query);
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

        for seg in 0..self.segment_count {
            let query_seg = segment_value(query_sim, seg, self.segment_bits);
            let dict = &self.segment_indices[seg];

            // 两种枚举策略，取更省的那一种：
            //   · 球体积小  -> 枚举汉明球内的取值（C(bits, <=d) 个）
            //   · 球体积大  -> 直接遍历字典键（段字典通常远小于 2^bits）
            //
            // 旧实现无条件遍历 0..2^segment_bits：
            //   seg=4 (13 bit) 时 8192 次，尚可；
            //   seg=2 (26 bit) 时 6700 万次 —— QPS 掉到个位数。
            // 以 6109 词为例，字典每段只有几千个键，遍历字典快好几个数量级。
            let ball = binomial_sum_le(self.segment_bits, segment_radius);
            let use_ball = ball <= (dict.len() as u64).saturating_mul(2);

            if use_ball {
                for value in enumerate_ball(query_seg, self.segment_bits, segment_radius) {
                    if let Some(words) = dict.get(&value) {
                        for word in words {
                            if seen.insert(word.clone()) {
                                result.push(word.clone());
                            }
                        }
                    }
                }
            } else {
                for (&value, words) in dict.iter() {
                    if popcount_u32(value ^ query_seg) <= segment_radius as u32 {
                        for word in words {
                            if seen.insert(word.clone()) {
                                result.push(word.clone());
                            }
                        }
                    }
                }
            }
        }

        // 用完整 sim Hamming 距离精确过滤
        result
            .into_iter()
            .filter(|word| {
                let code = self.encoder.encode_word_auto(word);
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

/// 32 位 popcount（段值宽度 <= 26，故够了）
fn popcount_u32(x: u32) -> u32 {
    x.count_ones()
}

/// 二项式和 C(bits, 0..=d)：汉明球的体积。
///
/// 用来说明「枚举球」和「遍历字典」哪个更省。
fn binomial_sum_le(bits: usize, d: usize) -> u64 {
    if d >= bits {
        return 1u64 << bits.min(52); // 全空间
    }
    // 递推算 C(n, k)
    let mut total: u64 = 1; // k = 0
    let mut c: u64 = 1;
    for k in 1..=d {
        c = c.saturating_mul((bits - k + 1) as u64) / k as u64;
        total = total.saturating_add(c);
    }
    total
}

/// 枚举与 center 的 Hamming 距离 <= d 的所有取值（码宽 bits）。
///
/// 直接生成组合 C(bits, k)，而不是遍历 0..2^bits。
/// bits 最多 26（因为 segment_count >= 2），所以 1u32 << b 不会溢出。
///
/// 返回的向量含 center 本身（距离 0），长度应为 sum_{k=0..=d} C(bits, k)。
fn enumerate_ball(center: u32, bits: usize, d: usize) -> Vec<u32> {
    let d = d.min(bits);
    let mut out = Vec::with_capacity(binomial_sum_le(bits, d) as usize);
    out.push(center);

    for k in 1..=d {
        if k > bits {
            break;
        }
        let mut comb: Vec<usize> = (0..k).collect();
        'emit: loop {
            // 翻转 comb 指定的 k 位
            let mut v = center;
            for &b in &comb {
                v ^= 1u32 << b;
            }
            out.push(v);

            // 组合推进（字典序）
            let mut i = k;
            loop {
                if i == 0 {
                    break 'emit; // 本 k 的组合已全部枚举
                }
                i -= 1;
                if comb[i] < bits - (k - i) {
                    comb[i] += 1;
                    for j in (i + 1)..k {
                        comb[j] = comb[j - 1] + 1;
                    }
                    continue 'emit;
                }
            }
        }
    }
    out
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

    /// 球枚举必须与暴力枚举的结果【完全一致】。
    ///
    /// 这是性能优化（2^bits -> C(bits,<=d)）的正确性保障：
    /// 优化只改枚举方式，不能改候选集合。
    #[test]
    fn test_enumerate_ball_matches_bruteforce() {
        for bits in [4usize, 8, 13] {
            for d in [0usize, 1, 2, 3] {
                if d > bits {
                    continue;
                }
                let space = 1u32 << bits;
                for &center in &[0u32, 1, space - 1, space / 3] {
                    let ball = enumerate_ball(center, bits, d);
                    let mut brute: Vec<u32> = (0..space)
                        .filter(|&v| popcount_u32(v ^ center) <= d as u32)
                        .collect();
                    let mut got = ball.clone();
                    got.sort_unstable();
                    got.dedup();
                    brute.sort_unstable();
                    assert_eq!(
                        got, brute,
                        "bits={} d={} center={}：球枚举与暴力不一致",
                        bits, d, center
                    );
                    // 长度应等于二项式和
                    assert_eq!(
                        ball.len() as u64,
                        binomial_sum_le(bits, d),
                        "bits={} d={}：球体积与二项式和不符",
                        bits, d
                    );
                }
            }
        }
    }

    #[test]
    fn test_binomial_sum_le() {
        assert_eq!(binomial_sum_le(13, 0), 1);
        assert_eq!(binomial_sum_le(13, 1), 1 + 13);
        assert_eq!(binomial_sum_le(13, 2), 1 + 13 + 78);
        assert_eq!(binomial_sum_le(4, 4), 1 << 4);
    }

    /// segment_count = 1 必须被明确拒绝，而不是静默返回 0 候选。
    #[test]
    fn test_reject_single_segment() {
        let encoder = Encoder::new();
        let words: Vec<String> = (0..10).map(|i| format!("w{}", i)).collect();
        let embedding = Arc::new(MockEmbedding::with_vocab(encoder.embed_dim(), words));
        let r = MihSemanticIndex::build_with_embedding(encoder, embedding, 1, false);
        assert!(r.is_err(), "segment_count=1 应当报错");
    }
}
