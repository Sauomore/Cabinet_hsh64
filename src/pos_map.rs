//! 词性到特征码（feat）的映射表 —— 中英双语
//!
//! 4-bit 特征码（0x0-0xF）共 16 个槽位，中文（jieba 标签）与英文
//! （Penn Treebank 标签）映射到**同一组槽位**，这样才能把两种语言的词
//! 放进同一个码本（混合索引）。
//!
//! # 槽位分配原则
//!
//! 1. **能合并的合并**：中英共有的大类（名词/动词/形容词/副词/介词/连词/
//!    数词/标点）直接共用槽位。
//! 2. **合并不了的保留**：中文特有的「量词」独占 0x9 —— 英文没有对应类别，
//!    而中文量词是真实语义类别（碗/根/双），与数词混在一起会污染语义。
//! 3. **需要的新增**：英文的限定词（the/a/this）与情态动词（can/will/must）
//!    在中文里没有直接对应，但它们是高频功能词，必须有自己的槽位。
//!
//! # 已知局限（实测，非猜测）
//!
//! 词表里的词是**孤立单词**，没有句子上下文。jieba 与 nltk 的标注器都是
//! 按句子训练的，对孤立词会显著退化：
//!
//! - 中文：3109 词产出 **29 种**标签，「同意」→ 副词、「丘陵」→ 人名
//! - 英文：40 词中约 **36% 标错**，「tree」→ 形容词、「beautiful」→ 介词
//!
//! 因此 `feat` 在这套流程里**天然是模糊的**。下游不应假设 feat 精确，
//! 也不应给 feat 过多权重；它更像一个粗粒度的形态线索。

/// 特征码常量定义
///
/// 槽位是 16 个，中英共用。注释标出每种语言对应的标签来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureCode(pub u8);

impl FeatureCode {
    /// 名词 —— 中: n/nr/ns/nt/nz/ng/nl | 英: NN/NNS/NNP/NNPS/FW
    pub const NOUN: FeatureCode = FeatureCode(0x0);
    /// 动词 —— 中: v/vd/vn/vg | 英: VB/VBD/VBG/VBN/VBP/VBZ
    pub const VERB: FeatureCode = FeatureCode(0x1);
    /// 形容词/区别词 —— 中: a/ad/an/ag/b | 英: JJ/JJR/JJS
    pub const ADJ: FeatureCode = FeatureCode(0x2);
    /// 副词 —— 中: d/df/dg | 英: RB/RBR/RBS/WRB
    pub const ADV: FeatureCode = FeatureCode(0x3);
    /// 代词/限定词 —— 中: r/rr/rz/rg | 英: PRP/PRP$/WP/WP$/DT/PDT/WDT/EX
    ///
    /// 中英在此合并：中文的代词与英文的「代词+限定词」共用。理由是两者
    /// 都是封闭类功能词，语义角色相近，而 4-bit 空间有限。
    pub const PRON: FeatureCode = FeatureCode(0x4);
    /// 介词/方位词/小品词 —— 中: p/f/fg/s | 英: IN/TO/RP
    pub const PREP: FeatureCode = FeatureCode(0x5);
    /// 连词 —— 中: c/cc | 英: CC
    pub const CONJ: FeatureCode = FeatureCode(0x6);
    /// 助词/语气词/情态 —— 中: u/ud/ug/uj/ul/uv/uz/y/z/zg | 英: MD
    ///
    /// 中英在此合并：中文「的/了/着」与英文情态动词都是语法功能标记。
    pub const AUX: FeatureCode = FeatureCode(0x7);
    /// 数词 —— 中: m/mq | 英: CD/LS
    pub const NUM: FeatureCode = FeatureCode(0x8);
    /// 量词 —— 中: q/qv/qt | 英: （无对应）
    ///
    /// **中文特有，独占槽位**。英文没有量词这一词类（"a cup of" 里的 cup
    /// 是名词）。与数词分开是刻意的：量词承载「以什么单位计量」的语义，
    /// 与「数量是多少」不同。
    pub const MEASURE: FeatureCode = FeatureCode(0x9);
    /// 时间词 —— 中: t/tg | 英: （并入 ADV）
    ///
    /// 中文有时间词这一独立类（今天/上午/七月）。英文的 today/yesterday
    /// 在 PTB 标签集里就是 RB（副词），故不单独设槽。
    pub const TIME: FeatureCode = FeatureCode(0xA);
    /// 字符串/专名/外来语 —— 中: x/xx/xu/xi/wjb/nx/eng/j
    pub const STRING: FeatureCode = FeatureCode(0xB);
    /// 标点 —— 中: w 系列 | 英: . , : `` '' -LRB- -RRB- 等
    pub const PUNCT: FeatureCode = FeatureCode(0xC);
    /// 其他/习语/感叹 —— 中: i（成语）| 英: UH/POS
    pub const OTHER: FeatureCode = FeatureCode(0xD);
    /// 常用词 —— 由 `Encoder` 的 common_words 表在运行时判定，优先级最高
    pub const COMMON: FeatureCode = FeatureCode(0xE);
    /// 兜底 —— 未识别的标签
    pub const FALLBACK: FeatureCode = FeatureCode(0xF);

    pub fn as_u8(&self) -> u8 {
        self.0
    }

    /// 由 4-bit 值构造，越界返回 FALLBACK
    pub fn from_u8(v: u8) -> FeatureCode {
        if v <= 0xF {
            FeatureCode(v)
        } else {
            FeatureCode::FALLBACK
        }
    }
}

/// 标注语言
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosLang {
    /// 中文（jieba 词性标签）
    Zh,
    /// 英文（Penn Treebank 标签）
    En,
}

impl PosLang {
    /// 按词表内容自动判断：含 CJK 字符即为中文
    pub fn detect(word: &str) -> PosLang {
        if word.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)) {
            PosLang::Zh
        } else {
            PosLang::En
        }
    }
}

/// 中文词性 → 特征码（jieba 标签集）
///
/// 标签列表来自对 `tests/data/vocab_3109.txt` 的**实际标注结果**（29 种标签），
/// 不是照抄文档。
pub fn pos_to_feat_zh(pos: &str) -> FeatureCode {
    match pos {
        // 名词类
        "n" | "nr" | "nr1" | "nr2" | "nrj" | "nrf" | "nrt" | "ns" | "nsf" | "nt"
        | "nz" | "nl" | "ng" => FeatureCode::NOUN,
        // 动词类
        "v" | "vd" | "vn" | "vf" | "vx" | "vi" | "vl" | "vg" => FeatureCode::VERB,
        // 形容词 + 区别词（b）
        "a" | "ad" | "an" | "ag" | "al" | "b" => FeatureCode::ADJ,
        // 副词
        "d" | "df" | "dg" => FeatureCode::ADV,
        // 代词
        "r" | "rr" | "rz" | "rzt" | "rzs" | "rzv" | "ry" | "ryt" | "rys" | "ryv"
        | "rg" | "ryy" => FeatureCode::PRON,
        // 介词 / 方位词 / 处所词
        "p" | "pba" | "pbei" | "f" | "fg" | "s" => FeatureCode::PREP,
        // 连词
        "c" | "cc" => FeatureCode::CONJ,
        // 助词/语气词/状态词
        "u" | "ud" | "ug" | "uj" | "ul" | "uv" | "uz" | "y" | "z" | "zg" => {
            FeatureCode::AUX
        }
        // 数词（mq 是数量词，归数词）
        "m" | "mq" => FeatureCode::NUM,
        // 量词 —— 中文特有，独占
        "q" | "qv" | "qt" => FeatureCode::MEASURE,
        // 时间词
        "t" | "tg" => FeatureCode::TIME,
        // 标点
        "w" | "wkz" | "wky" | "wyz" | "wyy" | "wj" | "ww" | "wt" | "wd" | "wf"
        | "wn" | "wm" | "ws" | "wp" | "wb" | "wh" => FeatureCode::PUNCT,
        // 字符串/专名/外来语/简称
        "x" | "xx" | "xu" | "xi" | "wjb" | "nx" | "eng" | "j" => FeatureCode::STRING,
        // 成语/习语
        "i" => FeatureCode::OTHER,
        _ => FeatureCode::FALLBACK,
    }
}

/// 英文词性 → 特征码（Penn Treebank 标签集）
///
/// 实指标注产出的 19 种标签 + PTB 完整标签集。
pub fn pos_to_feat_en(pos: &str) -> FeatureCode {
    match pos {
        // 名词（含外来词）
        "NN" | "NNS" | "NNP" | "NNPS" | "FW" => FeatureCode::NOUN,
        // 动词
        "VB" | "VBD" | "VBG" | "VBN" | "VBP" | "VBZ" => FeatureCode::VERB,
        // 形容词
        "JJ" | "JJR" | "JJS" => FeatureCode::ADJ,
        // 副词（含疑问副词 WRB）
        "RB" | "RBR" | "RBS" | "WRB" => FeatureCode::ADV,
        // 代词 + 限定词 + 存在句 there
        "PRP" | "PRP$" | "WP" | "WP$" | "DT" | "PDT" | "WDT" | "EX" => {
            FeatureCode::PRON
        }
        // 介词 + 不定式 to + 小品词
        "IN" | "TO" | "RP" => FeatureCode::PREP,
        // 连词
        "CC" => FeatureCode::CONJ,
        // 情态动词
        "MD" => FeatureCode::AUX,
        // 数词
        "CD" | "LS" => FeatureCode::NUM,
        // 标点（PTB 里标点自成标签）
        "." | "," | ":" | "``" | "''" | "-LRB-" | "-RRB-" | "-NONE-" | "$"
        | "#" | "SYM" => FeatureCode::PUNCT,
        // 感叹词 / 所有格标记
        "UH" | "POS" => FeatureCode::OTHER,
        _ => FeatureCode::FALLBACK,
    }
}

/// 按语言分派
pub fn pos_to_feat_lang(pos: &str, lang: PosLang) -> FeatureCode {
    match lang {
        PosLang::Zh => pos_to_feat_zh(pos),
        PosLang::En => pos_to_feat_en(pos),
    }
}

/// 兼容旧接口：默认按中文处理
///
/// 保留它是为了不破坏既有调用点与测试。新代码请用 `pos_to_feat_lang`。
pub fn pos_to_feat(pos: &str) -> Option<FeatureCode> {
    Some(pos_to_feat_zh(pos))
}

/// 获取特征码的名称（中英双语）
pub fn feat_name(feat: FeatureCode) -> &'static str {
    match feat.0 {
        0x0 => "名词/noun",
        0x1 => "动词/verb",
        0x2 => "形容词/adjective",
        0x3 => "副词/adverb",
        0x4 => "代词·限定词/pronoun·determiner",
        0x5 => "介词·方位/preposition·locative",
        0x6 => "连词/conjunction",
        0x7 => "助词·情态/particle·modal",
        0x8 => "数词/numeral",
        0x9 => "量词/measure word",
        0xA => "时间词/time word",
        0xB => "字符串·专名/string·proper noun",
        0xC => "标点/punctuation",
        0xD => "其他·习语/other·idiom",
        0xE => "常用词/common word",
        0xF => "兜底/fallback",
        _ => "未知/unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_mapping_zh() {
        assert_eq!(pos_to_feat_zh("n"), FeatureCode::NOUN);
        assert_eq!(pos_to_feat_zh("nr"), FeatureCode::NOUN);
        assert_eq!(pos_to_feat_zh("v"), FeatureCode::VERB);
        assert_eq!(pos_to_feat_zh("a"), FeatureCode::ADJ);
        assert_eq!(pos_to_feat_zh("b"), FeatureCode::ADJ);
        assert_eq!(pos_to_feat_zh("q"), FeatureCode::MEASURE);
        assert_eq!(pos_to_feat_zh("t"), FeatureCode::TIME);
        assert_eq!(pos_to_feat_zh("f"), FeatureCode::PREP);
        assert_eq!(pos_to_feat_zh("i"), FeatureCode::OTHER);
        assert_eq!(pos_to_feat_zh("unknown"), FeatureCode::FALLBACK);
    }

    #[test]
    fn test_pos_mapping_en() {
        assert_eq!(pos_to_feat_en("NN"), FeatureCode::NOUN);
        assert_eq!(pos_to_feat_en("NNPS"), FeatureCode::NOUN);
        assert_eq!(pos_to_feat_en("VBZ"), FeatureCode::VERB);
        assert_eq!(pos_to_feat_en("JJS"), FeatureCode::ADJ);
        assert_eq!(pos_to_feat_en("RB"), FeatureCode::ADV);
        assert_eq!(pos_to_feat_en("DT"), FeatureCode::PRON);
        assert_eq!(pos_to_feat_en("MD"), FeatureCode::AUX);
        assert_eq!(pos_to_feat_en("CD"), FeatureCode::NUM);
        assert_eq!(pos_to_feat_en("."), FeatureCode::PUNCT);
        assert_eq!(pos_to_feat_en("TO"), FeatureCode::PREP);
        assert_eq!(pos_to_feat_en("unknown"), FeatureCode::FALLBACK);
    }

    #[test]
    fn test_lang_dispatch() {
        // 同一个字符串 "q"：中文里是量词，英文里未定义
        assert_eq!(pos_to_feat_lang("q", PosLang::Zh), FeatureCode::MEASURE);
        assert_eq!(pos_to_feat_lang("q", PosLang::En), FeatureCode::FALLBACK);
        // "NN" 中文里未定义，英文里是名词
        assert_eq!(pos_to_feat_lang("NN", PosLang::Zh), FeatureCode::FALLBACK);
        assert_eq!(pos_to_feat_lang("NN", PosLang::En), FeatureCode::NOUN);
    }

    #[test]
    fn test_lang_detect() {
        assert_eq!(PosLang::detect("苹果"), PosLang::Zh);
        assert_eq!(PosLang::detect("apple"), PosLang::En);
        assert_eq!(PosLang::detect("T恤"), PosLang::Zh);
        assert_eq!(PosLang::detect("plumber"), PosLang::En);
    }

    #[test]
    fn test_all_slots_reachable() {
        // 0x0-0xD 每个槽位都应有标签能映射到
        // （0xE COMMON 由运行时词表决定，0xF FALLBACK 是默认分支）
        let zh: Vec<u8> = [
            "n", "v", "a", "d", "r", "p", "c", "u", "m", "q", "t", "x", "w", "i",
        ]
        .iter()
        .map(|t| pos_to_feat_zh(t).as_u8())
        .collect();
        let en: Vec<u8> = [
            "NN", "VB", "JJ", "RB", "PRP", "IN", "CC", "MD", "CD", ".", "UH",
        ]
        .iter()
        .map(|t| pos_to_feat_en(t).as_u8())
        .collect();
        let mut all: Vec<u8> = zh.into_iter().chain(en).collect();
        all.sort_unstable();
        all.dedup();
        for slot in 0x0..=0xD {
            assert!(all.contains(&slot), "槽位 0x{:X} 无任何标签映射到", slot);
        }
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(FeatureCode::from_u8(0x5).as_u8(), 0x5);
        assert_eq!(FeatureCode::from_u8(0xF).as_u8(), 0xF);
        assert_eq!(FeatureCode::from_u8(0xFF), FeatureCode::FALLBACK);
    }
}
