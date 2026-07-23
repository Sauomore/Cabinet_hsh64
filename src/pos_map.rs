//! 词性到特征码（feat）的映射表
//!
//! 基于 jieba 词性标注，映射到 4-bit 特征码（0x0-0xF）

/// 特征码常量定义
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureCode(pub u8);

impl FeatureCode {
    pub const NOUN: FeatureCode = FeatureCode(0x0);
    pub const VERB: FeatureCode = FeatureCode(0x1);
    pub const ADJ: FeatureCode = FeatureCode(0x2);
    pub const ADV: FeatureCode = FeatureCode(0x3);
    pub const PRONOUN: FeatureCode = FeatureCode(0x4);
    pub const PREP: FeatureCode = FeatureCode(0x5);
    pub const CONJ: FeatureCode = FeatureCode(0x6);
    pub const AUX: FeatureCode = FeatureCode(0x7);
    pub const NUM: FeatureCode = FeatureCode(0x8);
    pub const MEASURE: FeatureCode = FeatureCode(0x9);
    pub const TIME: FeatureCode = FeatureCode(0xA);
    pub const LOC: FeatureCode = FeatureCode(0xB);
    pub const PUNCT: FeatureCode = FeatureCode(0xC);
    pub const STRING: FeatureCode = FeatureCode(0xD);
    pub const COMMON: FeatureCode = FeatureCode(0xE);
    pub const FALLBACK: FeatureCode = FeatureCode(0xF);

    pub fn as_u8(&self) -> u8 {
        self.0
    }
}

/// 词性 → 特征码映射表
pub fn pos_to_feat(pos: &str) -> Option<FeatureCode> {
    match pos {
        "n" | "nr" | "nr1" | "nr2" | "nrj" | "nrf" | "ns" | "nsf" | "nt" | "nz" | "nl" | "ng" => {
            Some(FeatureCode::NOUN)
        }
        "v" | "vd" | "vn" | "vf" | "vx" | "vi" | "vl" | "vg" => Some(FeatureCode::VERB),
        "a" | "ad" | "an" | "ag" | "al" => Some(FeatureCode::ADJ),
        "d" | "df" | "dg" => Some(FeatureCode::ADV),
        "r" | "rr" | "rz" | "rzt" | "rzs" | "rzv" | "ry" | "ryt" | "rys" | "ryv" | "rg" | "ryy" => {
            Some(FeatureCode::PRONOUN)
        }
        "p" | "pba" | "pbei" => Some(FeatureCode::PREP),
        "c" | "cc" => Some(FeatureCode::CONJ),
        "u" | "ud" | "ug" | "uj" | "ul" | "uv" | "uz" | "y" | "z" => Some(FeatureCode::AUX),
        "m" | "mq" => Some(FeatureCode::NUM),
        "q" | "qv" | "qt" => Some(FeatureCode::MEASURE),
        "t" | "tg" => Some(FeatureCode::TIME),
        "f" | "fg" | "s" => Some(FeatureCode::LOC),
        "w" | "wkz" | "wky" | "wyz" | "wyy" | "wj" | "ww" | "wt" | "wd" | "wf" | "wn" | "wm"
        | "ws" | "wp" | "wb" | "wh" => Some(FeatureCode::PUNCT),
        "x" | "xx" | "xu" | "xi" | "wjb" | "nx" => Some(FeatureCode::STRING),
        _ => Some(FeatureCode::FALLBACK),
    }
}

/// 获取特征码的中文名称
pub fn feat_name(feat: FeatureCode) -> &'static str {
    match feat.0 {
        0x0 => "名词",
        0x1 => "动词",
        0x2 => "形容词",
        0x3 => "副词",
        0x4 => "代词",
        0x5 => "介词",
        0x6 => "连词",
        0x7 => "助词",
        0x8 => "数词",
        0x9 => "量词",
        0xA => "时间词",
        0xB => "方位词",
        0xC => "标点",
        0xD => "字符串",
        0xE => "常用词",
        0xF => "兜底",
        _ => "未知",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_mapping() {
        assert_eq!(pos_to_feat("n"), Some(FeatureCode::NOUN));
        assert_eq!(pos_to_feat("v"), Some(FeatureCode::VERB));
        assert_eq!(pos_to_feat("a"), Some(FeatureCode::ADJ));
        assert_eq!(pos_to_feat("unknown"), Some(FeatureCode::FALLBACK));
    }
}
