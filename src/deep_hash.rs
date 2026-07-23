//! Deep Hash v3：MLP 投影 + 纯 Rust 推理（HSH-64 版本）
//!
//! 对应 Python 脚本 `scripts/train_deep_hash_v3_64.py` 导出的二进制格式，
//! 用一个多层 MLP（dim -> hidden x depth -> n_bits）替代线性 PCA。
//!
//! HSH-64 中 n_bits 固定为 52。

use crate::embedding::EmbeddingModel;
use crate::hsh64::HSHCode64;
use std::sync::Arc;

pub const DEEP_HASH_MAGIC: u32 = 0xCAB1_DE3D;
pub const DEEP_HASH_VERSION_V1: u32 = 1;
pub const DEEP_HASH_VERSION_V2: u32 = 2;

/// Deep Hash v3 单个隐藏层
#[derive(Debug, Clone)]
pub struct DeepHashHiddenLayer {
    /// W: hidden_dim × input_dim（行优先，每行对应一个隐藏神经元）
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    pub bn_gamma: Vec<f32>,
    pub bn_beta: Vec<f32>,
    pub bn_running_mean: Vec<f32>,
    pub bn_running_var: Vec<f32>,
}

/// Deep Hash v3 投影模型
///
/// 网络结构：x -> [Linear -> BatchNorm -> ReLU] x depth -> Linear -> u
/// 其中 u 为 n_bits 维连续值，再用 sign_quantize 得到 52-bit sim。
#[derive(Debug, Clone)]
pub struct DeepHashProjection {
    pub dim: usize,
    pub hidden_dim: usize,
    pub depth: usize,
    pub n_bits: usize,
    pub mean: Vec<f32>,
    pub hidden_layers: Vec<DeepHashHiddenLayer>,
    /// W_out: n_bits × hidden_dim（行优先）
    pub w_out: Vec<Vec<f32>>,
    pub b_out: Vec<f32>,
}

impl DeepHashProjection {
    /// 从二进制字节解析 Deep Hash v3 模型
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        use std::convert::TryInto;

        if bytes.len() < 20 {
            return Err("DeepHash 缓存过短（<20 字节）".to_string());
        }
        let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        if magic != DEEP_HASH_MAGIC {
            return Err(format!("DeepHash magic 不匹配: 0x{:08X}", magic));
        }
        let version = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        if version != DEEP_HASH_VERSION_V1 && version != DEEP_HASH_VERSION_V2 {
            return Err(format!("DeepHash version 不支持: {}", version));
        }

        let dim = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let hidden_dim = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;

        let (depth, n_bits, header_size): (usize, usize, usize);
        if version == DEEP_HASH_VERSION_V1 {
            depth = 1;
            n_bits = u32::from_be_bytes(bytes[16..20].try_into().unwrap()) as usize;
            header_size = 20;
        } else {
            if bytes.len() < 24 {
                return Err("DeepHash v2 缓存过短（<24 字节）".to_string());
            }
            depth = u32::from_be_bytes(bytes[16..20].try_into().unwrap()) as usize;
            n_bits = u32::from_be_bytes(bytes[20..24].try_into().unwrap()) as usize;
            header_size = 24;
        }

        if depth == 0 {
            return Err("DeepHash depth 不能为 0".to_string());
        }

        let mut offset = header_size;

        // 计算期望总长度
        let mut expect_total = header_size + dim * 4;
        for layer in 0..depth {
            let input_dim = if layer == 0 { dim } else { hidden_dim };
            expect_total += hidden_dim * input_dim * 4; // W
            expect_total += hidden_dim * 4 * 5; // b + 4 BN params
        }
        expect_total += n_bits * hidden_dim * 4; // W_out
        expect_total += n_bits * 4; // b_out

        if bytes.len() < expect_total {
            return Err(format!(
                "DeepHash 缓存截断：期望至少 {} 字节，实际 {}",
                expect_total,
                bytes.len()
            ));
        }

        let read_f32 = |bytes: &[u8], offset: &mut usize| -> f32 {
            let v = f32::from_be_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            v
        };

        let mut mean = vec![0.0f32; dim];
        for i in 0..dim {
            mean[i] = read_f32(bytes, &mut offset);
        }

        let mut hidden_layers = Vec::with_capacity(depth);
        for layer in 0..depth {
            let input_dim = if layer == 0 { dim } else { hidden_dim };

            let mut w = vec![vec![0.0f32; input_dim]; hidden_dim];
            for i in 0..hidden_dim {
                for j in 0..input_dim {
                    w[i][j] = read_f32(bytes, &mut offset);
                }
            }

            let mut b = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                b[i] = read_f32(bytes, &mut offset);
            }

            let mut bn_gamma = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                bn_gamma[i] = read_f32(bytes, &mut offset);
            }

            let mut bn_beta = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                bn_beta[i] = read_f32(bytes, &mut offset);
            }

            let mut bn_running_mean = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                bn_running_mean[i] = read_f32(bytes, &mut offset);
            }

            let mut bn_running_var = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                bn_running_var[i] = read_f32(bytes, &mut offset);
            }

            hidden_layers.push(DeepHashHiddenLayer {
                w,
                b,
                bn_gamma,
                bn_beta,
                bn_running_mean,
                bn_running_var,
            });
        }

        let mut w_out = vec![vec![0.0f32; hidden_dim]; n_bits];
        for i in 0..n_bits {
            for j in 0..hidden_dim {
                w_out[i][j] = read_f32(bytes, &mut offset);
            }
        }

        let mut b_out = vec![0.0f32; n_bits];
        for i in 0..n_bits {
            b_out[i] = read_f32(bytes, &mut offset);
        }

        Ok(Self {
            dim,
            hidden_dim,
            depth,
            n_bits,
            mean,
            hidden_layers,
            w_out,
            b_out,
        })
    }

    /// 连续投影：x -> [Linear -> BatchNorm -> ReLU] x depth -> Linear -> u
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "输入维度与模型不匹配");
        const EPS: f32 = 1e-5;

        // x = vector - mean
        let mut h = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            h[i] = vector[i] - self.mean[i];
        }

        for layer in 0..self.depth {
            let layer_ref = &self.hidden_layers[layer];
            let input_dim = h.len();
            let mut next_h = vec![0.0f32; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = layer_ref.b[i];
                for j in 0..input_dim {
                    sum += layer_ref.w[i][j] * h[j];
                }
                // BatchNorm eval
                let bn = (sum - layer_ref.bn_running_mean[i])
                    / (layer_ref.bn_running_var[i].sqrt() + EPS);
                let y = layer_ref.bn_gamma[i] * bn + layer_ref.bn_beta[i];
                next_h[i] = y.max(0.0);
            }
            h = next_h;
        }

        // u = W_out·h + b_out
        let mut u = vec![0.0f32; self.n_bits];
        for i in 0..self.n_bits {
            let mut sum = self.b_out[i];
            for j in 0..self.hidden_dim {
                sum += self.w_out[i][j] * h[j];
            }
            u[i] = sum;
        }

        u
    }

    /// 投影并量化为 52-bit sim 码
    pub fn quantize(&self, vector: &[f32]) -> u64 {
        let proj = self.project(vector);
        let mut sim: u64 = 0;
        for (i, &v) in proj.iter().enumerate() {
            if v >= 0.0 {
                sim |= 1 << i;
            }
        }
        sim
    }
}

/// 基于 Deep Hash v3 投影的 HSH-64 编码器
///
/// 注意：通用 [`Encoder`] 已支持 Deep Hash 投影；本结构保留给需要显式
/// 使用 DeepHashProjection 的场景。
pub struct DeepHashEncoder64 {
    projection: DeepHashProjection,
    embedding: Arc<dyn EmbeddingModel>,
    common_words: std::collections::HashSet<String>,
    seed_table: std::collections::HashMap<(u8, u8), u8>,
}

impl DeepHashEncoder64 {
    /// 创建 DeepHash 编码器
    pub fn new(projection: DeepHashProjection, embedding: Arc<dyn EmbeddingModel>) -> Self {
        Self {
            projection,
            embedding,
            common_words: std::collections::HashSet::new(),
            seed_table: std::collections::HashMap::new(),
        }
    }

    /// 获取 embedding 维度
    pub fn embed_dim(&self) -> usize {
        self.embedding.dim()
    }

    /// 计算单个词的连续投影（保留幅度信息）
    pub fn project_word(&self, word: &str, _feat: u8) -> Vec<f32> {
        let vec = self.embedding.embed(word);
        self.projection.project(&vec)
    }

    /// 编码单个词为 HSH-64
    pub fn encode_word(&self, word: &str, feat: u8) -> HSHCode64 {
        let vec = self.embedding.embed(word);
        let sim = self.projection.quantize(&vec);
        let abs = crate::perfect_hash::compute_abs(word, self.seed_for(feat, sim));
        HSHCode64::new(feat, sim, abs)
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

    /// 移除常用词
    pub fn remove_common_word(&mut self, word: &str) {
        self.common_words.remove(word);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedding;
    use std::sync::Arc;

    /// 构造一个最小可用的 DeepHash v1 二进制：
    /// 输入 dim=4，hidden=3，n_bits=52
    fn build_minimal_deep_hash_v1_bytes() -> Vec<u8> {
        let dim = 4usize;
        let hidden_dim = 3usize;
        let n_bits = 52usize;

        let mut buf = Vec::new();
        buf.extend_from_slice(&DEEP_HASH_MAGIC.to_be_bytes());
        buf.extend_from_slice(&DEEP_HASH_VERSION_V1.to_be_bytes());
        buf.extend_from_slice(&(dim as u32).to_be_bytes());
        buf.extend_from_slice(&(hidden_dim as u32).to_be_bytes());
        buf.extend_from_slice(&(n_bits as u32).to_be_bytes());

        // mean
        for _ in 0..dim {
            buf.extend_from_slice(&0.0f32.to_be_bytes());
        }
        // W1
        for _ in 0..hidden_dim {
            for _ in 0..dim {
                buf.extend_from_slice(&0.1f32.to_be_bytes());
            }
        }
        // b1
        for _ in 0..hidden_dim {
            buf.extend_from_slice(&0.0f32.to_be_bytes());
        }
        // bn gamma, beta, running_mean, running_var
        for _ in 0..hidden_dim {
            buf.extend_from_slice(&1.0f32.to_be_bytes()); // gamma
        }
        for _ in 0..hidden_dim {
            buf.extend_from_slice(&0.0f32.to_be_bytes()); // beta
        }
        for _ in 0..hidden_dim {
            buf.extend_from_slice(&0.0f32.to_be_bytes()); // running_mean
        }
        for _ in 0..hidden_dim {
            buf.extend_from_slice(&1.0f32.to_be_bytes()); // running_var
        }
        // W2
        for _ in 0..n_bits {
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&0.1f32.to_be_bytes());
            }
        }
        // b2
        for _ in 0..n_bits {
            buf.extend_from_slice(&0.0f32.to_be_bytes());
        }

        buf
    }

    /// 构造一个最小可用的 DeepHash v2 二进制：
    /// 输入 dim=4，hidden=3，depth=2，n_bits=52
    fn build_minimal_deep_hash_v2_bytes() -> Vec<u8> {
        let dim = 4usize;
        let hidden_dim = 3usize;
        let depth = 2usize;
        let n_bits = 52usize;

        let mut buf = Vec::new();
        buf.extend_from_slice(&DEEP_HASH_MAGIC.to_be_bytes());
        buf.extend_from_slice(&DEEP_HASH_VERSION_V2.to_be_bytes());
        buf.extend_from_slice(&(dim as u32).to_be_bytes());
        buf.extend_from_slice(&(hidden_dim as u32).to_be_bytes());
        buf.extend_from_slice(&(depth as u32).to_be_bytes());
        buf.extend_from_slice(&(n_bits as u32).to_be_bytes());

        // mean
        for _ in 0..dim {
            buf.extend_from_slice(&0.0f32.to_be_bytes());
        }

        for layer in 0..depth {
            let input_dim = if layer == 0 { dim } else { hidden_dim };
            // W
            for _ in 0..hidden_dim {
                for _ in 0..input_dim {
                    buf.extend_from_slice(&0.1f32.to_be_bytes());
                }
            }
            // b
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&0.0f32.to_be_bytes());
            }
            // bn gamma, beta, running_mean, running_var
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&1.0f32.to_be_bytes());
            }
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&0.0f32.to_be_bytes());
            }
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&0.0f32.to_be_bytes());
            }
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&1.0f32.to_be_bytes());
            }
        }

        // W_out
        for _ in 0..n_bits {
            for _ in 0..hidden_dim {
                buf.extend_from_slice(&0.1f32.to_be_bytes());
            }
        }
        // b_out
        for _ in 0..n_bits {
            buf.extend_from_slice(&0.0f32.to_be_bytes());
        }

        buf
    }

    #[test]
    fn test_deep_hash_projection_from_bytes_v1() {
        let bytes = build_minimal_deep_hash_v1_bytes();
        let proj = DeepHashProjection::from_bytes(&bytes).unwrap();
        assert_eq!(proj.dim, 4);
        assert_eq!(proj.hidden_dim, 3);
        assert_eq!(proj.depth, 1);
        assert_eq!(proj.n_bits, 52);
    }

    #[test]
    fn test_deep_hash_projection_from_bytes_v2() {
        let bytes = build_minimal_deep_hash_v2_bytes();
        let proj = DeepHashProjection::from_bytes(&bytes).unwrap();
        assert_eq!(proj.dim, 4);
        assert_eq!(proj.hidden_dim, 3);
        assert_eq!(proj.depth, 2);
        assert_eq!(proj.n_bits, 52);
    }

    #[test]
    fn test_deep_hash_projection_inference_v1() {
        let bytes = build_minimal_deep_hash_v1_bytes();
        let proj = DeepHashProjection::from_bytes(&bytes).unwrap();
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        let u = proj.project(&v);
        assert_eq!(u.len(), 52);

        let sim1 = proj.quantize(&v);
        let sim2 = proj.quantize(&v);
        assert_eq!(sim1, sim2);
        assert!(sim1 <= HSHCode64::MAX_SIM);
    }

    #[test]
    fn test_deep_hash_projection_inference_v2() {
        let bytes = build_minimal_deep_hash_v2_bytes();
        let proj = DeepHashProjection::from_bytes(&bytes).unwrap();
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        let u = proj.project(&v);
        assert_eq!(u.len(), 52);

        let sim1 = proj.quantize(&v);
        let sim2 = proj.quantize(&v);
        assert_eq!(sim1, sim2);
        assert!(sim1 <= HSHCode64::MAX_SIM);
    }

    #[test]
    fn test_deep_hash_encoder64() {
        let bytes = build_minimal_deep_hash_v1_bytes();
        let proj = DeepHashProjection::from_bytes(&bytes).unwrap();
        let embedding: Arc<dyn EmbeddingModel> = Arc::new(MockEmbedding::new(4));
        let encoder = DeepHashEncoder64::new(proj, embedding);

        let code1 = encoder.encode_word("测试", 0x01);
        let code2 = encoder.encode_word("测试", 0x01);
        assert_eq!(code1, code2);
        assert!(code1.feat() <= HSHCode64::MAX_FEAT);
        assert!(code1.sim() <= HSHCode64::MAX_SIM);
        assert!(code1.abs() <= HSHCode64::MAX_ABS);
    }
}
