//! HSH-64：64-bit 语义哈希编码方案
//!
//! 结构：[feat: 4 bit] + [sim: 52 bit] + [abs: 8 bit]
//!
//! 与 HSH-32（feat4 + sim20 + abs8）相比，HSH-64 将 sim 码从 20 bit 扩展到 52 bit，
//! 大幅提升语义区分能力，同时保持 64-bit 硬件友好（一个 u64 存储，单次 popcnt 比较）。

pub mod embedding;
pub mod encoder;
pub mod error;
pub mod hsh64;
pub mod mih_index;
pub mod pca;
pub mod perfect_hash;
pub mod pos_map;

pub use encoder::{Encoder, EncoderConfig};
pub use hsh64::{HSHCode64, hamming_distance64};
pub use mih_index::{MihSemanticIndex, SearchResult};
pub use pca::{PcaProjection, PCA_MAGIC, PCA_VERSION, PCA_N_COMPONENTS};
