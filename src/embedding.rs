//! Embedding 模型抽象层

use std::collections::HashMap;
use std::path::Path;

/// Embedding 模型接口
pub trait EmbeddingModel: Send + Sync {
    /// 对单个词返回稠密向量
    fn embed(&self, word: &str) -> Vec<f32>;

    /// 向量维度
    fn dim(&self) -> usize;

    /// 批量编码
    fn embed_batch(&self, words: &[&str]) -> Vec<Vec<f32>> {
        words.iter().map(|w| self.embed(w)).collect()
    }

    /// 是否基于真实预训练模型
    fn is_real(&self) -> bool {
        false
    }

    /// 返回缓存中已知的所有词汇
    fn vocab(&self) -> Vec<String> {
        Vec::new()
    }
}

/// MVP 默认：字符级三角函数 mock 向量
pub struct MockEmbedding {
    dim: usize,
    vocab: Vec<String>,
}

impl MockEmbedding {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vocab: Vec::new(),
        }
    }

    pub fn with_vocab(dim: usize, vocab: Vec<String>) -> Self {
        Self { dim, vocab }
    }
}

impl EmbeddingModel for MockEmbedding {
    fn embed(&self, word: &str) -> Vec<f32> {
        let chars: Vec<u32> = word.chars().map(|c| c as u32).collect();
        let mut vec = vec![0.0f32; self.dim];
        for (i, &ch) in chars.iter().enumerate() {
            let weight = 1.0 / ((i + 1) as f32).sqrt();
            for d in 0..self.dim {
                let theta = (d + 1) as f32 * 0.031415926535;
                let phi = ch as f32 * 0.01745329252;
                let psi = (i + 1) as f32 * 0.5;
                vec[d] += weight
                    * ((theta * ch as f32 + phi + psi).sin() * 0.5
                        + (theta * (ch as f32 + 1.0) + phi + psi).cos() * 0.5);
            }
        }
        normalize(&mut vec);
        vec
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn vocab(&self) -> Vec<String> {
        self.vocab.clone()
    }
}

/// 从预计算文件加载的词向量缓存
pub struct FileCachedEmbedding {
    dim: usize,
    cache: HashMap<String, Vec<f32>>,
    fallback: MockEmbedding,
}

impl FileCachedEmbedding {
    pub fn new<P: AsRef<Path>>(path: P, fallback_dim: usize) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let (dim, cache) = Self::parse(&bytes)?;
        Ok(Self {
            dim,
            cache,
            fallback: MockEmbedding::new(fallback_dim),
        })
    }

    pub fn empty(fallback_dim: usize) -> Self {
        Self {
            dim: fallback_dim,
            cache: HashMap::new(),
            fallback: MockEmbedding::new(fallback_dim),
        }
    }

    fn parse(bytes: &[u8]) -> anyhow::Result<(usize, HashMap<String, Vec<f32>>)> {
        use anyhow::{bail, Context};
        use std::convert::TryInto;

        if bytes.len() < 16 {
            bail!("embedding cache file too short");
        }

        let magic = u32::from_be_bytes(bytes[0..4].try_into()?);
        if magic != 0xCAB1_EBED {
            bail!("invalid embedding cache magic: 0x{:08X}", magic);
        }

        let version = u32::from_be_bytes(bytes[4..8].try_into()?);
        if version != 1 {
            bail!("unsupported embedding cache version: {}", version);
        }

        let dim = u32::from_be_bytes(bytes[8..12].try_into()?) as usize;
        let vocab_size = u32::from_be_bytes(bytes[12..16].try_into()?) as usize;

        let mut cache = HashMap::with_capacity(vocab_size);
        let mut offset = 16usize;

        for _ in 0..vocab_size {
            if offset + 2 > bytes.len() {
                bail!("truncated word length");
            }
            let len = u16::from_be_bytes(bytes[offset..offset + 2].try_into()?) as usize;
            offset += 2;

            if offset + len > bytes.len() {
                bail!("truncated word bytes");
            }
            let word = String::from_utf8(bytes[offset..offset + len].to_vec())
                .context("invalid utf-8 word")?;
            offset += len;

            if offset + dim * 4 > bytes.len() {
                bail!("truncated vector for word '{}'", word);
            }
            let mut vec = vec![0.0f32; dim];
            for j in 0..dim {
                vec[j] = f32::from_be_bytes(bytes[offset..offset + 4].try_into()?);
                offset += 4;
            }
            cache.insert(word, vec);
        }

        Ok((dim, cache))
    }
}

impl EmbeddingModel for FileCachedEmbedding {
    fn embed(&self, word: &str) -> Vec<f32> {
        if let Some(v) = self.cache.get(word) {
            return v.clone();
        }
        self.fallback.embed(word)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, words: &[&str]) -> Vec<Vec<f32>> {
        words.iter().map(|w| self.embed(w)).collect()
    }

    fn is_real(&self) -> bool {
        !self.cache.is_empty()
    }

    fn vocab(&self) -> Vec<String> {
        let mut words: Vec<String> = self.cache.keys().cloned().collect();
        words.sort();
        words
    }
}

/// 向量单位化
pub fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for v in vec {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_dim() {
        let model = MockEmbedding::new(128);
        let v = model.embed("测试");
        assert_eq!(v.len(), 128);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_mock_embedding_stable() {
        let model = MockEmbedding::new(64);
        let v1 = model.embed("稳定");
        let v2 = model.embed("稳定");
        assert_eq!(v1, v2);
    }
}
