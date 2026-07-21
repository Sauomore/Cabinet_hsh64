//! HSH-64 核心编码类型
//!
//! 位布局（从高到低）：
//!   [63:60] feat  (4 bit)
//!   [59: 8] sim   (52 bit)
//!   [ 7: 0] abs   (8 bit)

/// HSH-64 编码，feat4 + sim52 + abs8
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HSHCode64 {
    raw: u64,
}

impl HSHCode64 {
    /// feat 位宽
    pub const FEAT_BITS: u32 = 4;
    /// sim 位宽
    pub const SIM_BITS: u32 = 52;
    /// abs 位宽
    pub const ABS_BITS: u32 = 8;

    /// feat 最大值（含）
    pub const MAX_FEAT: u8 = (1 << Self::FEAT_BITS) - 1;
    /// sim 最大值（含）
    pub const MAX_SIM: u64 = (1 << Self::SIM_BITS) - 1;
    /// abs 最大值（含）
    pub const MAX_ABS: u8 = ((1u16 << Self::ABS_BITS) - 1) as u8;

    /// sim 字段起始位
    pub const SIM_SHIFT: u32 = Self::ABS_BITS;
    /// feat 字段起始位
    pub const FEAT_SHIFT: u32 = Self::SIM_SHIFT + Self::SIM_BITS;

    /// 创建 HSH-64 码，参数越界时 panic
    pub fn new(feat: u8, sim: u64, abs: u8) -> Self {
        assert!(feat <= Self::MAX_FEAT, "feat 越界: {} > {}", feat, Self::MAX_FEAT);
        assert!(sim <= Self::MAX_SIM, "sim 越界: {} > {}", sim, Self::MAX_SIM);
        assert!(abs <= Self::MAX_ABS, "abs 越界: {} > {}", abs, Self::MAX_ABS);

        let raw = ((feat as u64) << Self::FEAT_SHIFT)
            | ((sim & Self::MAX_SIM) << Self::SIM_SHIFT)
            | (abs as u64);

        Self { raw }
    }

    /// 从原始 u64 构造，不检查字段范围
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// 返回原始 u64
    pub const fn raw(&self) -> u64 {
        self.raw
    }

    /// 提取 feat 字段
    pub fn feat(&self) -> u8 {
        ((self.raw >> Self::FEAT_SHIFT) & (Self::MAX_FEAT as u64)) as u8
    }

    /// 提取 sim 字段
    pub fn sim(&self) -> u64 {
        (self.raw >> Self::SIM_SHIFT) & Self::MAX_SIM
    }

    /// 提取 abs 字段
    pub fn abs(&self) -> u8 {
        (self.raw & (Self::MAX_ABS as u64)) as u8
    }

    /// 与另一个 HSH-64 码的完整 Hamming 距离
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        hamming_distance64(self.raw, other.raw)
    }

    /// 与另一个 HSH-64 码的 sim 字段 Hamming 距离
    pub fn sim_hamming_distance(&self, other: &Self) -> u32 {
        let mask = Self::MAX_SIM << Self::SIM_SHIFT;
        hamming_distance64(self.raw & mask, other.raw & mask)
    }
}

/// 计算两个 u64 的 Hamming 距离（popcount）
pub fn hamming_distance64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsh64_new_and_extract() {
        let code = HSHCode64::new(0x0A, 0x123456789ABCD, 0x7F);
        assert_eq!(code.feat(), 0x0A);
        assert_eq!(code.sim(), 0x123456789ABCD);
        assert_eq!(code.abs(), 0x7F);
    }

    #[test]
    fn test_hsh64_raw_roundtrip() {
        let code = HSHCode64::new(0x0F, HSHCode64::MAX_SIM, 0xFF);
        assert_eq!(HSHCode64::from_raw(code.raw()), code);
    }

    #[test]
    fn test_hamming_distance64() {
        let a = HSHCode64::new(0, 0, 0);
        let b = HSHCode64::new(0, 1, 0);
        let c = HSHCode64::new(0, 3, 0);
        assert_eq!(a.hamming_distance(&a), 0);
        assert_eq!(a.hamming_distance(&b), 1);
        assert_eq!(a.hamming_distance(&c), 2);
    }

    #[test]
    fn test_sim_hamming_distance_ignores_abs_and_feat() {
        let a = HSHCode64::new(0x01, 0x100, 0xFF);
        let b = HSHCode64::new(0x0F, 0x101, 0x00);
        // sim 差 1 bit，abs/feat 不同但应被忽略
        assert_eq!(a.sim_hamming_distance(&b), 1);
    }
}
