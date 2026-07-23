//! PCA 投影实现（HSH-64 版本，输出 52 维）

use crate::error::EncodeError;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

pub const PCA_MAGIC: u32 = 0xCAB1_3CA2;
pub const PCA_VERSION: u32 = 1;
pub const PCA_N_COMPONENTS: usize = 52;

/// PCA 投影参数
#[derive(Clone, Debug, PartialEq)]
pub struct PcaProjection {
    pub dim: usize,
    pub n_components: usize,
    pub mean: Vec<f32>,
    pub components: Vec<Vec<f32>>,
    pub explained_variance_ratio: Vec<f32>,
}

impl PcaProjection {
    /// 从字节解析 PCA 缓存
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EncodeError> {
        if bytes.len() < 16 {
            return Err(EncodeError::Config("PCA 缓存过短".to_string()));
        }

        let mut cursor = Cursor::new(bytes);
        let magic = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| EncodeError::Config(e.to_string()))?;
        if magic != PCA_MAGIC {
            return Err(EncodeError::Config(format!(
                "PCA magic 不匹配: 0x{:08X}",
                magic
            )));
        }

        let version = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| EncodeError::Config(e.to_string()))?;
        if version != PCA_VERSION {
            return Err(EncodeError::Config(format!(
                "PCA version 不支持: {}",
                version
            )));
        }

        let dim = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| EncodeError::Config(e.to_string()))? as usize;
        let n_components = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| EncodeError::Config(e.to_string()))? as usize;

        if n_components != PCA_N_COMPONENTS {
            return Err(EncodeError::Config(format!(
                "HSH-64 PCA 需要 {} 个主成分，得到 {}",
                PCA_N_COMPONENTS, n_components
            )));
        }

        let mut mean = vec![0.0f32; dim];
        for i in 0..dim {
            mean[i] = cursor
                .read_f32::<BigEndian>()
                .map_err(|e| EncodeError::Config(e.to_string()))?;
        }

        let mut components = vec![vec![0.0f32; dim]; n_components];
        for i in 0..n_components {
            for j in 0..dim {
                components[i][j] = cursor
                    .read_f32::<BigEndian>()
                    .map_err(|e| EncodeError::Config(e.to_string()))?;
            }
        }

        let mut explained_variance_ratio = vec![0.0f32; n_components];
        for i in 0..n_components {
            explained_variance_ratio[i] = cursor
                .read_f32::<BigEndian>()
                .map_err(|e| EncodeError::Config(e.to_string()))?;
        }

        Ok(Self {
            dim,
            n_components,
            mean,
            components,
            explained_variance_ratio,
        })
    }

    /// 序列化为字节
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<BigEndian>(PCA_MAGIC).unwrap();
        buf.write_u32::<BigEndian>(PCA_VERSION).unwrap();
        buf.write_u32::<BigEndian>(self.dim as u32).unwrap();
        buf.write_u32::<BigEndian>(self.n_components as u32)
            .unwrap();

        for &v in &self.mean {
            buf.write_f32::<BigEndian>(v).unwrap();
        }
        for row in &self.components {
            for &v in row {
                buf.write_f32::<BigEndian>(v).unwrap();
            }
        }
        for &v in &self.explained_variance_ratio {
            buf.write_f32::<BigEndian>(v).unwrap();
        }
        buf
    }

    /// 投影单个向量到 PCA 空间
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "输入维度不匹配");
        let mut result = vec![0.0f32; self.n_components];
        for i in 0..self.n_components {
            let mut acc = 0.0f32;
            for j in 0..self.dim {
                acc += (vector[j] - self.mean[j]) * self.components[i][j];
            }
            result[i] = acc;
        }
        result
    }

    /// 投影并 sign 量化为 52-bit sim 码
    pub fn project_to_sim(&self, vector: &[f32]) -> u64 {
        let proj = self.project(vector);
        let mut sim: u64 = 0;
        for (i, &v) in proj.iter().enumerate() {
            if v >= 0.0 {
                sim |= 1 << i;
            }
        }
        sim
    }

    /// 生成 mock PCA（单位阵前 dim 行截断到 52 维）
    pub fn mock(dim: usize) -> Self {
        let n_components = PCA_N_COMPONENTS.min(dim);
        let mut components = vec![vec![0.0f32; dim]; n_components];
        for i in 0..n_components {
            components[i][i] = 1.0;
        }
        Self {
            dim,
            n_components,
            mean: vec![0.0f32; dim],
            components,
            explained_variance_ratio: vec![1.0 / n_components as f32; n_components],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_mock_project() {
        let pca = PcaProjection::mock(64);
        let vec: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let proj = pca.project(&vec);
        assert_eq!(proj.len(), 52);
    }

    #[test]
    fn test_pca_serde_roundtrip() {
        let pca = PcaProjection::mock(64);
        let bytes = pca.to_bytes();
        let decoded = PcaProjection::from_bytes(&bytes).unwrap();
        assert_eq!(pca, decoded);
    }
}
