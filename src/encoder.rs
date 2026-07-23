//! HSH-64 编码器

use crate::{
    deep_hash::DeepHashProjection,
    embedding::{EmbeddingModel, MockEmbedding},
    error::EncodeError,
    hsh64::HSHCode64,
    pca::PcaProjection,
    perfect_hash::compute_abs,
    pos_map::{pos_to_feat, FeatureCode},
};
use std::path::PathBuf;
use std::sync::Arc;

/// 编码器配置
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub pos_threshold: u32,
    pub embed_dim: usize,
    pub embedding_cache_path: Option<PathBuf>,
    pub pca_path: Option<PathBuf>,
    pub deep_hash_path: Option<PathBuf>,
    pub sim_override_path: Option<PathBuf>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            pos_threshold: 50,
            embed_dim: 768,
            embedding_cache_path: None,
            pca_path: None,
            deep_hash_path: None,
            sim_override_path: None,
        }
    }
}

/// 投影方式：PCA 线性投影 或 Deep Hash v3 非线性投影
#[derive(Clone, Debug)]
enum Projection {
    Pca(PcaProjection),
    DeepHash(DeepHashProjection),
}

impl Projection {
    /// 投影到连续低维空间
    fn project(&self, vector: &[f32]) -> Vec<f32> {
        match self {
            Projection::Pca(pca) => pca.project(vector),
            Projection::DeepHash(dh) => dh.project(vector),
        }
    }

    /// 投影并 sign 量化为 52-bit sim 码
    fn project_to_sim(&self, vector: &[f32]) -> u64 {
        match self {
            Projection::Pca(pca) => pca.project_to_sim(vector),
            Projection::DeepHash(dh) => dh.quantize(vector),
        }
    }
}

/// HSH-64 编码器
#[derive(Clone)]
pub struct Encoder {
    #[allow(dead_code)]
    config: EncoderConfig,
    embedding: Arc<dyn EmbeddingModel>,
    projection: Projection,
    common_words: std::collections::HashSet<String>,
    /// 每个 (feat, sim_low8) 对应的完美哈希种子
    seed_table: std::collections::HashMap<(u8, u8), u8>,
    /// 后处理优化后的 sim 码覆盖表：word -> sim
    sim_override: std::collections::HashMap<String, u64>,
}

impl Encoder {
    /// 使用默认配置创建编码器
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default()).unwrap()
    }

    /// 使用配置创建编码器
    pub fn with_config(config: EncoderConfig) -> Result<Self, EncodeError> {
        let embedding: Arc<dyn EmbeddingModel> = if let Some(ref path) = config.embedding_cache_path
        {
            match crate::embedding::FileCachedEmbedding::new(path, config.embed_dim) {
                Ok(model) => Arc::new(model),
                Err(e) => {
                    return Err(EncodeError::Config(format!(
                        "无法加载 embedding 缓存: {}",
                        e
                    )))
                }
            }
        } else {
            Arc::new(MockEmbedding::new(config.embed_dim))
        };

        let projection = if let Some(ref path) = config.deep_hash_path {
            let bytes = std::fs::read(path).map_err(|e| EncodeError::Config(e.to_string()))?;
            let dh = DeepHashProjection::from_bytes(&bytes)
                .map_err(|e| EncodeError::Config(format!("无法解析 Deep Hash 模型: {}", e)))?;
            if dh.dim != embedding.dim() {
                return Err(EncodeError::Config(format!(
                    "Deep Hash dim (={}) 与 embedding dim (={}) 不一致",
                    dh.dim,
                    embedding.dim()
                )));
            }
            if dh.n_bits != 52 {
                return Err(EncodeError::Config(format!(
                    "HSH-64 Deep Hash 需要 52 个输出位，得到 {}",
                    dh.n_bits
                )));
            }
            Projection::DeepHash(dh)
        } else if let Some(ref path) = config.pca_path {
            let bytes = std::fs::read(path).map_err(|e| EncodeError::Config(e.to_string()))?;
            let pca = PcaProjection::from_bytes(&bytes)?;
            if pca.dim != embedding.dim() {
                return Err(EncodeError::Config(format!(
                    "PCA dim (={}) 与 embedding dim (={}) 不一致",
                    pca.dim,
                    embedding.dim()
                )));
            }
            Projection::Pca(pca)
        } else {
            Projection::Pca(PcaProjection::mock(embedding.dim()))
        };

        let sim_override = if let Some(ref path) = config.sim_override_path {
            let bytes = std::fs::read(path).map_err(|e| EncodeError::Config(e.to_string()))?;
            parse_sim_override(&bytes)?
        } else {
            std::collections::HashMap::new()
        };

        Ok(Self {
            config,
            embedding,
            projection,
            common_words: std::collections::HashSet::new(),
            seed_table: std::collections::HashMap::new(),
            sim_override,
        })
    }

    /// 编码单个词
    pub fn encode_word(&self, word: &str, feat: u8) -> HSHCode64 {
        let sim = if let Some(&sim) = self.sim_override.get(word) {
            sim
        } else {
            let vector = self.embedding.embed(word);
            self.projection.project_to_sim(&vector)
        };
        let abs = compute_abs(word, self.seed_for(feat, sim));
        HSHCode64::new(feat, sim, abs)
    }

    /// 编码单个词（带词性）
    pub fn encode_word_with_pos(&self, word: &str, pos: &str) -> HSHCode64 {
        let feat = if self.is_common_word(word) {
            FeatureCode::COMMON.as_u8()
        } else {
            pos_to_feat(pos)
                .map(|f| f.as_u8())
                .unwrap_or(FeatureCode::FALLBACK.as_u8())
        };
        self.encode_word(word, feat)
    }

    /// 批量编码词表并构建种子表
    pub fn build_seed_table(&mut self, words: &[(String, u8)]) {
        use crate::perfect_hash::search_seed;

        let mut buckets: std::collections::HashMap<(u8, u8), Vec<String>> =
            std::collections::HashMap::new();
        for (word, feat) in words {
            let sim = self.sim_override.get(word).copied().unwrap_or_else(|| {
                let vector = self.embedding.embed(word);
                self.projection.project_to_sim(&vector)
            });
            let key = (*feat, (sim & 0xFF) as u8);
            buckets.entry(key).or_default().push(word.clone());
        }

        self.seed_table.clear();
        for ((feat, sim_low), bucket_words) in buckets {
            let (seed, _, _) = search_seed(&bucket_words);
            self.seed_table.insert((feat, sim_low), seed);
        }
    }

    fn seed_for(&self, feat: u8, sim: u64) -> u8 {
        let key = (feat, (sim & 0xFF) as u8);
        *self.seed_table.get(&key).unwrap_or(&0)
    }

    /// 添加常用词
    pub fn add_common_word(&mut self, word: &str) {
        self.common_words.insert(word.to_string());
    }

    /// 判断是否为常用词
    pub fn is_common_word(&self, word: &str) -> bool {
        self.common_words.contains(word)
    }

    /// 获取 embedding 维度
    pub fn embed_dim(&self) -> usize {
        self.embedding.dim()
    }

    /// 获取连续投影（PCA 或 Deep Hash）
    pub fn project_word(&self, word: &str, _feat: u8) -> Vec<f32> {
        // 若存在后处理优化的 sim 码，无法恢复连续投影幅度，回退到原始投影
        let vector = self.embedding.embed(word);
        self.projection.project(&vector)
    }
}

const OVERRIDE_MAGIC: u32 = 0xCAB1_0D01;
const OVERRIDE_VERSION: u32 = 1;

/// 解析 sim 码覆盖缓存
fn parse_sim_override(bytes: &[u8]) -> Result<std::collections::HashMap<String, u64>, EncodeError> {
    use std::convert::TryInto;

    if bytes.len() < 16 {
        return Err(EncodeError::Config("sim 覆盖缓存过短".to_string()));
    }
    let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    if magic != OVERRIDE_MAGIC {
        return Err(EncodeError::Config(format!(
            "sim 覆盖缓存 magic 不匹配: 0x{:08X}",
            magic
        )));
    }
    let version = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    if version != OVERRIDE_VERSION {
        return Err(EncodeError::Config(format!(
            "sim 覆盖缓存 version 不支持: {}",
            version
        )));
    }
    let n_bits = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if n_bits != 52 {
        return Err(EncodeError::Config(format!(
            "sim 覆盖缓存 n_bits 错误: {}",
            n_bits
        )));
    }
    let count = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;

    let mut map = std::collections::HashMap::with_capacity(count);
    let mut offset = 16usize;
    for _ in 0..count {
        if offset + 2 > bytes.len() {
            return Err(EncodeError::Config(
                "sim 覆盖缓存截断（word_len）".to_string(),
            ));
        }
        let len = u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + len > bytes.len() {
            return Err(EncodeError::Config("sim 覆盖缓存截断（word）".to_string()));
        }
        let word = String::from_utf8(bytes[offset..offset + len].to_vec())
            .map_err(|e| EncodeError::Config(format!("sim 覆盖缓存非法 utf-8: {}", e)))?;
        offset += len;
        if offset + 8 > bytes.len() {
            return Err(EncodeError::Config(
                "sim 覆盖缓存截断（sim_code）".to_string(),
            ));
        }
        let sim = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        map.insert(word, sim);
    }
    Ok(map)
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_basic() {
        let encoder = Encoder::new();
        let code = encoder.encode_word_with_pos("测试", "n");
        assert!(code.feat() <= 0x0F);
        assert!(code.sim() <= HSHCode64::MAX_SIM);
    }

    #[test]
    fn test_common_word_promotion() {
        let mut encoder = Encoder::new();
        encoder.add_common_word("测试");
        let code = encoder.encode_word_with_pos("测试", "n");
        assert_eq!(code.feat(), FeatureCode::COMMON.as_u8());
    }
}
