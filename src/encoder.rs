//! HSH-64 编码器

use crate::{
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
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            pos_threshold: 50,
            embed_dim: 768,
            embedding_cache_path: None,
            pca_path: None,
        }
    }
}

/// HSH-64 编码器
pub struct Encoder {
    #[allow(dead_code)]
    config: EncoderConfig,
    embedding: Arc<dyn EmbeddingModel>,
    pca: PcaProjection,
    common_words: std::collections::HashSet<String>,
    /// 每个 (feat, sim_low8) 对应的完美哈希种子
    seed_table: std::collections::HashMap<(u8, u8), u8>,
}

impl Encoder {
    /// 使用默认配置创建编码器
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default()).unwrap()
    }

    /// 使用配置创建编码器
    pub fn with_config(config: EncoderConfig) -> Result<Self, EncodeError> {
        let embedding: Arc<dyn EmbeddingModel> = if let Some(ref path) = config.embedding_cache_path {
            match crate::embedding::FileCachedEmbedding::new(path, config.embed_dim) {
                Ok(model) => Arc::new(model),
                Err(e) => return Err(EncodeError::Config(format!("无法加载 embedding 缓存: {}", e))),
            }
        } else {
            Arc::new(MockEmbedding::new(config.embed_dim))
        };

        let pca = if let Some(ref path) = config.pca_path {
            let bytes = std::fs::read(path).map_err(|e| EncodeError::Config(e.to_string()))?;
            PcaProjection::from_bytes(&bytes)?
        } else {
            PcaProjection::mock(embedding.dim())
        };

        if pca.dim != embedding.dim() {
            return Err(EncodeError::Config(format!(
                "PCA dim (={}) 与 embedding dim (={}) 不一致",
                pca.dim, embedding.dim()
            )));
        }

        Ok(Self {
            config,
            embedding,
            pca,
            common_words: std::collections::HashSet::new(),
            seed_table: std::collections::HashMap::new(),
        })
    }

    /// 编码单个词
    pub fn encode_word(&self, word: &str, feat: u8) -> HSHCode64 {
        let vector = self.embedding.embed(word);
        let sim = self.pca.project_to_sim(&vector);
        let abs = compute_abs(word, self.seed_for(feat, sim));
        HSHCode64::new(feat, sim, abs)
    }

    /// 编码单个词（带词性）
    pub fn encode_word_with_pos(&self, word: &str, pos: &str) -> HSHCode64 {
        let feat = if self.is_common_word(word) {
            FeatureCode::COMMON.as_u8()
        } else {
            pos_to_feat(pos).map(|f| f.as_u8()).unwrap_or(FeatureCode::FALLBACK.as_u8())
        };
        self.encode_word(word, feat)
    }

    /// 批量编码词表并构建种子表
    pub fn build_seed_table(&mut self, words: &[(String, u8)]) {
        use crate::perfect_hash::search_seed;

        let mut buckets: std::collections::HashMap<(u8, u8), Vec<String>> = std::collections::HashMap::new();
        for (word, feat) in words {
            let vector = self.embedding.embed(word);
            let sim = self.pca.project_to_sim(&vector);
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

    /// 获取 PCA 投影
    pub fn project_word(&self, word: &str, _feat: u8) -> Vec<f32> {
        let vector = self.embedding.embed(word);
        self.pca.project(&vector)
    }
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
