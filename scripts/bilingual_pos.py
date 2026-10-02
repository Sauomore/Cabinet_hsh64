# -*- coding: utf-8 -*-
"""
中英双语词性标注 —— 与 Rust 侧 `src/pos_map.rs` 严格对应。

⚠️ 两张表必须保持一致。改了一边就要改另一边，否则「Python 生成码本、
   Rust 查询码本」会给出不同结果。本文件的表与 pos_map.rs 逐步对齐，
   并由 `tests/` 下的对拍脚本校验（见 scripts/check_pos_parity.py）。

设计要点（与 Rust 侧相同）：
    16 个 feat 槽位，中英共用。中文用 jieba 标签，英文用 Penn Treebank 标签。
    - 能合并的合并（名词/动词/形容词/…）
    - 合并不了的保留（中文「量词」独占 0x9）
    - 需要的新增（英文限定词、情态动词）

已知局限：
    词表里是【孤立单词】，没有句子上下文。两个标注器都会显著退化
    （中文 29 种标签、英文约 36% 标错）。所以 feat 天然模糊，
    下游不应假设它精确。
"""

from __future__ import annotations

import os
from enum import IntEnum
from pathlib import Path

# ---- nltk 数据路径（本机 hosts 屏蔽了 nltk 官网，数据是手工下载的）----
_NLTK_DATA = Path(r"F:\nltk_data")
if _NLTK_DATA.exists():
    os.environ.setdefault("NLTK_DATA", str(_NLTK_DATA))


class Feat(IntEnum):
    """16 个 feat 槽位（与 pos_map.rs 的 FeatureCode 对应）"""

    NOUN = 0x0      # 名词          中 n/nr/ns/nt/nz/ng/nl   英 NN/NNS/NNP/NNPS/FW
    VERB = 0x1      # 动词          中 v/vd/vn/vg             英 VB/VBD/VBG/VBN/VBP/VBZ
    ADJ = 0x2       # 形容词/区别词  中 a/ad/an/ag/b           英 JJ/JJR/JJS
    ADV = 0x3       # 副词          中 d/df/dg                英 RB/RBR/RBS/WRB
    PRON = 0x4      # 代词/限定词    中 r/rr/rz/rg             英 PRP/PRP$/WP/WP$/DT/PDT/WDT/EX
    PREP = 0x5      # 介词/方位      中 p/f/fg/s               英 IN/TO/RP
    CONJ = 0x6      # 连词          中 c/cc                   英 CC
    AUX = 0x7       # 助词/情态      中 u/ud/ug/uj/ul/uv/uz/y/z/zg  英 MD
    NUM = 0x8       # 数词          中 m/mq                   英 CD/LS
    MEASURE = 0x9   # 量词          中 q/qv/qt                英 （无）
    TIME = 0xA      # 时间词        中 t/tg                   英 （并入 ADV）
    STRING = 0xB    # 字符串/专名    中 x/xx/xu/xi/wjb/nx/eng/j
    PUNCT = 0xC     # 标点          中 w 系列                 英 . , : `` '' 等
    OTHER = 0xD     # 其他/习语      中 i                     英 UH/POS
    COMMON = 0xE    # 常用词（运行时判定）
    FALLBACK = 0xF  # 兜底


# ---------------------------------------------------------------- 中文表
_ZH: dict[str, Feat] = {}
for _tags, _f in [
    (("n", "nr", "nr1", "nr2", "nrj", "nrf", "nrt", "ns", "nsf", "nt",
      "nz", "nl", "ng"), Feat.NOUN),
    (("v", "vd", "vn", "vf", "vx", "vi", "vl", "vg"), Feat.VERB),
    (("a", "ad", "an", "ag", "al", "b"), Feat.ADJ),
    (("d", "df", "dg"), Feat.ADV),
    (("r", "rr", "rz", "rzt", "rzs", "rzv", "ry", "ryt", "rys", "ryv",
      "rg", "ryy"), Feat.PRON),
    (("p", "pba", "pbei", "f", "fg", "s"), Feat.PREP),
    (("c", "cc"), Feat.CONJ),
    (("u", "ud", "ug", "uj", "ul", "uv", "uz", "y", "z", "zg"), Feat.AUX),
    (("m", "mq"), Feat.NUM),
    (("q", "qv", "qt"), Feat.MEASURE),
    (("t", "tg"), Feat.TIME),
    (("w", "wkz", "wky", "wyz", "wyy", "wj", "ww", "wt", "wd", "wf",
      "wn", "wm", "ws", "wp", "wb", "wh"), Feat.PUNCT),
    (("x", "xx", "xu", "xi", "wjb", "nx", "eng", "j"), Feat.STRING),
    (("i",), Feat.OTHER),
]:
    for _t in _tags:
        _ZH[_t] = _f

# ---------------------------------------------------------------- 英文表
_EN: dict[str, Feat] = {}
for _tags, _f in [
    (("NN", "NNS", "NNP", "NNPS", "FW"), Feat.NOUN),
    (("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"), Feat.VERB),
    (("JJ", "JJR", "JJS"), Feat.ADJ),
    (("RB", "RBR", "RBS", "WRB"), Feat.ADV),
    (("PRP", "PRP$", "WP", "WP$", "DT", "PDT", "WDT", "EX"), Feat.PRON),
    (("IN", "TO", "RP"), Feat.PREP),
    (("CC",), Feat.CONJ),
    (("MD",), Feat.AUX),
    (("CD", "LS"), Feat.NUM),
    ((".", ",", ":", "``", "''", "-LRB-", "-RRB-", "-NONE-", "$", "#",
      "SYM"), Feat.PUNCT),
    (("UH", "POS"), Feat.OTHER),
]:
    for _t in _tags:
        _EN[_t] = _f


def pos_to_feat_zh(pos: str) -> Feat:
    return _ZH.get(pos, Feat.FALLBACK)


def pos_to_feat_en(pos: str) -> Feat:
    return _EN.get(pos, Feat.FALLBACK)


def detect_lang(word: str) -> str:
    """含 CJK 字符即为中文。"""
    for c in word:
        if "\u4e00" <= c <= "\u9fff":
            return "zh"
    return "en"


# ---------------------------------------------------------------- 标注器
_jieba_ready = False
_nltk_ready = False


def _ensure_jieba():
    global _jieba_ready
    if not _jieba_ready:
        import jieba.posseg as pseg  # noqa: F401
        import jieba
        jieba.setLogLevel(60)          # 关掉构建前缀词典的日志
        _jieba_ready = True


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        if _NLTK_DATA.exists():
            nltk.data.path.insert(0, str(_NLTK_DATA))
        _nltk_ready = True


def tag_zh(words: list[str]) -> list[str]:
    """中文词性标注。返回与输入等长的标签列表。"""
    _ensure_jieba()
    import jieba.posseg as pseg
    out = []
    for w in words:
        pairs = list(pseg.cut(w))
        out.append(pairs[0].flag if pairs else "")
    return out


def tag_en(words: list[str]) -> list[str]:
    """英文词性标注。返回与输入等长的标签列表。"""
    _ensure_nltk()
    import nltk
    tagged = nltk.pos_tag(words)
    return [t for _, t in tagged]


def tag_mixed(words: list[str]) -> list[tuple[str, str, Feat]]:
    """
    混合词表标注：按词自动分派语言。

    返回 [(词, 标签, feat), ...]，顺序与输入一致。

    ⚠️ 英文为了效率会批量标注（nltk 的 pos_tag 接受列表），
       但中文必须逐词（jieba 对每个词单独切分）。
       混合词表因此会被分成两批分别处理，最后按原顺序拼回。
    """
    zh_idx = [i for i, w in enumerate(words) if detect_lang(w) == "zh"]
    en_idx = [i for i, w in enumerate(words) if detect_lang(w) == "en"]

    result: list[tuple[str, str, Feat] | None] = [None] * len(words)

    if zh_idx:
        zh_tags = tag_zh([words[i] for i in zh_idx])
        for i, t in zip(zh_idx, zh_tags):
            result[i] = (words[i], t, pos_to_feat_zh(t))
    if en_idx:
        en_tags = tag_en([words[i] for i in en_idx])
        for i, t in zip(en_idx, en_tags):
            result[i] = (words[i], t, pos_to_feat_en(t))

    return [r for r in result if r is not None]


def feat_name(f: Feat) -> str:
    return {
        Feat.NOUN: "名词/noun",
        Feat.VERB: "动词/verb",
        Feat.ADJ: "形容词/adjective",
        Feat.ADV: "副词/adverb",
        Feat.PRON: "代词·限定词/pronoun·determiner",
        Feat.PREP: "介词·方位/preposition·locative",
        Feat.CONJ: "连词/conjunction",
        Feat.AUX: "助词·情态/particle·modal",
        Feat.NUM: "数词/numeral",
        Feat.MEASURE: "量词/measure word",
        Feat.TIME: "时间词/time word",
        Feat.STRING: "字符串·专名/string·proper noun",
        Feat.PUNCT: "标点/punctuation",
        Feat.OTHER: "其他·习语/other·idiom",
        Feat.COMMON: "常用词/common word",
        Feat.FALLBACK: "兜底/fallback",
    }.get(f, "未知/unknown")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    demo = ["苹果", "计算机", "碗", "今天", "漂亮",
            "apple", "computer", "the", "can", "beautiful"]
    print("=" * 62)
    print("双语词性标注演示")
    print("=" * 62)
    print(f"  {'词':<12}{'语言':<6}{'标签':<8}{'feat':<6}说明")
    print("  " + "-" * 58)
    for w, t, f in tag_mixed(demo):
        print(f"  {w:<12}{detect_lang(w):<6}{t:<8}0x{f.value:X}   {feat_name(f)}")
