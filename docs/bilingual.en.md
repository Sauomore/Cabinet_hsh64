# HSH-64 Chinese–English Extension

This document records the design, implementation and measured results of extending HSH-64
from **Chinese-only** to a **mixed Chinese–English index**.

---

## 1. Why bilingual

The original implementation supported Chinese only:

- Vocabulary `tests/data/vocab_3109.txt`, 3,109 words, of which only 4 contain Latin letters
- POS tagging went through jieba (a Chinese segmenter's tag set)
- Embeddings used `BAAI/bge-small-zh-v1.5` (a Chinese-only model)

Making it bilingual is not just "swap in a bigger model" — the encoding has **language-dependent
assumptions**, and each has to be handled individually.

---

## 2. The language-dependent part of the encoding

HSH-64's bit layout:

```text
[63:60] feat (4 bit)   ← POS category, 16 slots      ★ language-dependent
[59: 8] sim  (52 bit)  ← semantic code, from Deep Hash  ✅ language-independent
[ 7: 0] abs  (8 bit)   ← within-bucket unique ID       ✅ language-independent
```

**Only `feat` is language-dependent.** `sim` is determined entirely by the embedding, and `abs`
is just a string hash.

**So the bilingual work is concentrated in the `feat` mapping table** — which is a fortunate
property of the design: the 4-bit `feat` was always meant to be a "hard-coded coarse category",
not a carrier of fine-grained semantics.

---

## 3. Allocation of `feat`'s 16 slots

Chinese and English each have their own POS tag set:

- Chinese: jieba tags (`n` / `nr` / `ns` / `v` / `a` / `q` / `t` …)
- English: Penn Treebank tags (`NN` / `VB` / `JJ` / `DT` / `MD` …)

Both map onto **one shared set of 16 slots**, so that words from the two languages can enter the
same codebook.

### Allocation principles

1. **Merge what can be merged** — categories both languages share (noun/verb/adjective/adverb/preposition/conjunction/numeral/punctuation) are shared directly
2. **Keep what cannot merge** — the Chinese-only "measure word" gets slot 0x9 exclusively
3. **Add what is needed** — English determiners and modal verbs have no direct Chinese counterpart, but they are high-frequency function words

### Slot table

| Slot | Meaning | Chinese source (jieba) | English source (PTB) |
|---|---|---|---|
| 0x0 | noun | `n` `nr` `ns` `nt` `nz` `ng` `nl` … | `NN` `NNS` `NNP` `NNPS` `FW` |
| 0x1 | verb | `v` `vd` `vn` `vg` … | `VB` `VBD` `VBG` `VBN` `VBP` `VBZ` |
| 0x2 | adjective / distinguishing word | `a` `ad` `an` `ag` `b` | `JJ` `JJR` `JJS` |
| 0x3 | adverb | `d` `df` `dg` | `RB` `RBR` `RBS` `WRB` |
| 0x4 | **pronoun / determiner** | `r` `rr` `rz` `rg` … | `PRP` `PRP$` `WP` `WP$` `DT` `PDT` `WDT` `EX` |
| 0x5 | preposition / locative / particle | `p` `f` `fg` `s` | `IN` `TO` `RP` |
| 0x6 | conjunction | `c` `cc` | `CC` |
| 0x7 | **particle / modal** | `u` `ud` `ug` `uj` … `y` `z` `zg` | `MD` |
| 0x8 | numeral | `m` `mq` | `CD` `LS` |
| 0x9 | **measure word** | `q` `qv` `qt` | (no counterpart) |
| 0xA | **time word** | `t` `tg` | (English folds into ADV) |
| 0xB | string / proper name | `x` `xx` `xu` `xi` `wjb` `nx` `eng` `j` | (English folds into NOUN/FW) |
| 0xC | punctuation | the 16 `w`-series tags | `.` `,` `:` ``` `` ``` `''` … |
| 0xD | other / idiom | `i` (idiom) | `UH` `POS` |
| 0xE | common word | decided at runtime by `add_common_word` | same |
| 0xF | fallback | unrecognised tag | same |

### The cost of each merge (all of them have one, recorded here)

| Slot | What was merged | Cost |
|---|---|---|
| 0x4 | Chinese pronouns + English pronouns and determiners | English `the`/`a` share a code with `he`/`it`; Chinese has no such distinction, so no loss |
| 0x5 | Chinese prepositions/locatives/place words + English prepositions/infinitive to/particles | Chinese "上/下" share a code with "在/从" |
| 0x7 | Chinese particles/mood words + English modal verbs | Different grammatical functions, but both are "grammatical markers" |
| 0x9 | Not merged; Chinese measure words exclusively | That slot is empty for English (no loss) |
| 0xA | Not merged; Chinese time words exclusively | English `today`/`yesterday` are already `RB` in PTB and land in 0x3 |
| 0xB | Chinese "string/proper name" kept as is | English has no corresponding class |

---

## 4. ⚠️ Measured limitation: `feat` is in practice quite blurry

**This section reports measurements, not conjecture.**

### Limitation 1: POS tagging of isolated words is unreliable

The vocabulary stores **isolated words** with no sentence context. Both jieba's and nltk's taggers
are trained on sentences and degrade noticeably on single words:

**Chinese** (3,109 words, producing **29 distinct** tags):

```text
同意    → d  (adverb)       should be a verb
丘陵    → nr (person name)  should be a noun
乌龟    → nr (person name)  should be a noun
伏特加  → nr (person name)  should be a noun
```

**English** (40-word probe, roughly **36% mis-tagged**):

```text
tree      → JJ    should be NN
flower    → JJR   should be NN
teacher   → RB    should be NN
beautiful → IN    should be JJ (single-call) / VB (batch call)
cold      → VBP   should be JJ
```

**`beautiful` yields different results on different call paths (`IN` vs `VB`)** — tagging of
isolated words is not merely wrong, it is **unstable**.

### Limitation 2: the `feat` distribution is extremely skewed

Actual distribution over the mixed Chinese–English vocabulary (6,109 words):

```text
0x0 noun          2,925  (47.9%)
0x2 adjective     2,344  (38.4%)
0x1 verb            408  ( 6.7%)
0x3 adverb          203  ( 3.3%)
0x5 preposition      89  ( 1.5%)
0x4 pronoun          44  ( 0.7%)
other 10 slots       96  ( 1.6%)
─────────────────────────────
unused: 0xC (punctuation), 0xE (common word)
```

**Two slots absorb 86% of the vocabulary; the effective entropy is under 1.5 bit.**

### Conclusion: `feat` is only a hard-coded coarse cue, it does not carry semantic separation

This is **acceptable by design** — semantic separation is carried by the 52-bit `sim`
(measured share of unique `sim` codes: 99%). What `feat` does is:

1. Place different word classes into different `(feat, sim_low8)` buckets, **reducing `abs` collisions**
2. Provide an interpretable retrieval path ("why did this word come to mind")
3. Support filtering by word class

**The more weight `feat` is given in retrieval scoring, the more noise it injects.**

---

## 5. Implementation

### 5.1 Rust side (`src/pos_map.rs`)

```rust
pub enum PosLang { Zh, En }

pub fn pos_to_feat_zh(pos: &str) -> FeatureCode   // jieba tag set
pub fn pos_to_feat_en(pos: &str) -> FeatureCode   // PTB tag set
pub fn pos_to_feat_lang(pos: &str, lang: PosLang) -> FeatureCode

impl PosLang {
    /// Contains a CJK character means Chinese
    pub fn detect(word: &str) -> PosLang
}
```

### 5.2 Loading tags from a file (`src/encoder.rs`)

Before bilingual support, every call site **hard-coded** `encode_word_with_pos(word, "n")` —
that is, `feat` was always `0x0`. With a bilingual vocabulary that breaks: passing `"n"` for an
English word makes it be treated as a Chinese tag and fall into `FALLBACK`.

```rust
pub struct EncoderConfig {
    ...
    /// word → POS tag (optional). When empty, falls back to the built-in "n", preserving old behaviour
    pub pos_tags: HashMap<String, String>,
}

impl EncoderConfig {
    /// Load from TSV (word<TAB>lang<TAB>tag<TAB>feat)
    pub fn load_pos_tags(&mut self, path: &Path) -> Result<usize, EncodeError>
}

impl Encoder {
    /// Recommended bilingual entry point: looks up the tag table and detects the language
    pub fn encode_word_auto(&self, word: &str) -> HSHCode64
}
```

**Compatibility**: when `pos_tags` is not supplied, behaviour is identical to before the change
(falling back to `"n"`), so every old call site can be switched to `encode_word_auto` directly.

### 5.3 Python side (`scripts/bilingual_pos.py`)

Two mappings that correspond strictly to the Rust tables — jieba for Chinese, nltk for English.
For the mixed vocabulary, the language is detected per word and each side tagged in a batch.

### 5.4 Parity test (`tests/feat_parity.rs`)

**The two tables exist in two places and must agree** — otherwise "Python builds the codebook,
Rust looks it up" gives different answers, and that class of bug looks fine from either side
alone.

```text
scripts/bilingual_pos.py
        │  scripts/46_export_feat_table.py
        ▼
tests/data/feat_table.json
        │  cargo test --test feat_parity
        ▼
src/pos_map.rs  asserted entry by entry
```

Coverage: 92 Chinese tags, 46 English tags, reachability of all 16 slots, no cross-language
conflict.

**The test has been mutation-verified** — deliberately deleting the `FW` mapping on the Rust
side makes it fail immediately:

```text
英文表有 1 处不一致（Python vs Rust）:
  FW          Python=0x0  Rust=0xF
```

---

## 6. Vocabulary construction

### 6.1 Data sources

| Side | Source | Size |
|---|---|---|
| Chinese | reuses `tests/data/vocab_3109.txt` | 3,109 words |
| English | [ECDICT](https://github.com/skywind3000/ECDICT) (English–Chinese dictionary, MIT) | filtered from 770,611 entries |

**The Chinese side reuses the original vocabulary** so that older experiments stay comparable.

### 6.2 English word-selection rules

```text
· Frequency: bnc or frq (COCA) rank <= 6000, taking the better of the two
· Shape: pure ASCII lowercase letters only (no hyphens, spaces, accents)
· Length: 2 ~ 14
```

Result: **3,000 words**, rank range **1 ~ 2744**.

```text
top 15 by frequency: the be and of in to have it he for that not you with on
15 that made it in:  complicated olympic alongside dirt bullet lend rice
                     considerably tight bible replacement chart proof practise solar
```

### 6.3 Final vocabulary

```text
data/mixed_vocab.txt        6,109 words
  Chinese      3,095  (50.7%)
  English      3,002  (49.1%)
  mixed           12  ( 0.2%)   T恤、U盘、大U盘…
```

**The Chinese–English ratio is close to 1:1**, which makes cross-lingual retrieval evaluation
convenient.

---

## 7. Cross-lingual alignment validation (preliminary experiment)

A mixed index rests on one premise: **the embedding must place "苹果" and "apple" near each
other.** That cannot be assumed, so it was validated first.

### 7.1 Model choice

| Model | Dimension | Note |
|---|---|---|
| `BAAI/bge-large-zh-v1.5` | 1024 | Chinese-only (control) |
| **`BAAI/bge-m3`** | **1024** | **multilingual (adopted)** |

### 7.2 Results (40 Chinese–English translation pairs + 6 negative pairs)

| Metric | bge-m3 | bge-large-zh |
|---|---|---|
| Positive mean | **+0.8054** | +0.7205 |
| Negative mean | +0.4910 | +0.4244 |
| **AUC (ranking quality)** | **1.0000** | 0.9573 |
| Smallest positive | +0.5446 | +0.4036 |
| Largest negative | +0.5393 | +0.5094 |
| Distribution overlap | **none** ✅ | exists ❌ |

**All 40 positives for bge-m3 rank above all 6 negatives, with no overlap**
(smallest positive +0.5446 > largest negative +0.5393).

Typical results:

```text
学校 school   +0.9136      苹果 apple     +0.9046
狗   dog      +0.9061      汽车 car       +0.8890
学生 student  +0.9061      计算机 computer +0.8642

weaker (proper nouns):
北京 Beijing  +0.7015      上海 Shanghai  +0.6654
```

> ⚠️ **Sample-size caveat**: 40 positives / 6 negatives is a small sample; AUC = 1.0 does not
> extrapolate. The conclusion is limited to "the premise for cross-lingual alignment holds, so
> it is worth continuing".

---

## 8. End-to-end pipeline

```bash
# 1. Build the mixed vocabulary (download ECDICT first)
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000

# 2. Bilingual POS tagging
python scripts/49_tag_vocab.py

# 3. bge-m3 encoding
python scripts/generate_embeddings.py \
    --vocab data/mixed_vocab.txt \
    --local-model /path/to/bge-m3 \
    -o data/mixed_embedding.cache

# 4. PCA down to 52 dimensions
python scripts/train_pca_64.py \
    --embedding-cache data/mixed_embedding.cache \
    -o data/mixed_pca_52.bin

# 5. Deep Hash training
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache data/mixed_embedding.cache \
    -o data/mixed_deep_hash_52.bin \
    --n-bits 52 --epochs 500

# 6. Rust-side evaluation (--pos-tags is what makes it bilingual)
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding data/mixed_embedding.cache \
    --pca data/mixed_pca_52.bin \
    --deep-hash data/mixed_deep_hash_52.bin \
    --pos-tags data/mixed_tags.tsv \
    --top-k 10 --queries 100
```

---

## 9. Current state and known limitations

### 9.1 Working end to end

| Stage | Result |
|---|---|
| Mixed vocabulary | 6,109 words (Chinese–English ≈ 1:1) |
| Bilingual tagging | 28 tags for Chinese / 31 tags for English |
| Share of unique codes | **6058 / 6109 = 99.2%** |
| Deep Hash convergence | info loss 1.5758 → 0.0080 |
| POS wiring | ✅ 14 hard-coded call sites replaced |

### 9.2 Not yet done: third-stage post-processing

```text
current (bare DeepHash)              Recall@10 = 0.2750
recorded (DeepHash + post-processing) Recall@10 = 0.728
```

The difference is that **the bilingual model has not yet had third-stage post-processing**
(greedy bit flipping). The recorded `0.728` is the output of `sim_override_*.bin`, not the bare
Deep Hash.

Post-processing is necessary for this scheme, not an optional refinement — measured on the
natural Chinese vocabulary, the bare code at 0.246 is even below the zero-learning PCA-52
baseline at 0.376, and only reaches 0.608 after post-processing
(see [scale_evaluation.en.md](scale_evaluation.en.md) §3.2). So the `0.2750` above does not
represent the final capability of bilingual retrieval.

### 9.3 Known limitations

**① `feat` has limited discriminative power**

See §4. The 16 slots of `feat` come from hard-coded POS categories, so words with very different
meanings under the same tag land in the same slot (for example "同意" is tagged as an adverb and
"丘陵" as a person name). Its role is to coarsely partition the code space, not to classify
semantics precisely.

**② The English POS mapping has clear errors**

About 36% of a 40-word probe is judged incorrectly (`tree` → JJ, `beautiful` → IN or VB depending
on the call path). The reason is that ECDICT provides word frequency but not POS, so the English
tags come from rule-based inference rather than a statistical tagger.

**③ The perfect-hash seed table for `abs` has never taken effect**

`Encoder::build_seed_table` has no call site anywhere in the codebase, so `seed_for` always
returns 0:

```rust
*self.seed_table.get(&key).unwrap_or(&0)
```

The consequence is that `abs` loses its uniqueness guarantee and may collide silently. `abs` does
not participate in MIH retrieval (which uses `sim` segmentation plus `feat` filtering), so recall
is unaffected, but the field currently lacks the uniqueness the design intends.

**④ Bilingual evaluation is not finished**

The existing number (bare code 0.2750) lacks post-processing and cannot be compared directly
against the Chinese-only results.
