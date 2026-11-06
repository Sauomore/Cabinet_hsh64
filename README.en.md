# HSH-64

[![DOI](https://zenodo.org/badge/1307857867.svg)](https://doi.org/10.5281/zenodo.21721838)
[![GitHub release](https://img.shields.io/github/v/release/Sauomore/Cabinet_hsh64?include_prereleases&style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases)
[![Version](https://img.shields.io/badge/version-2.0-blue?style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases/tag/v2.0)

[中文](README.md) ｜ **English**

> Give every word a **64-bit integer** as its "semantic fingerprint".
> Words that mean similar things differ in only a few bits, and comparing two words costs
> one XOR plus one popcount.
> Pure CPU — a single core compares tens of millions of pairs per second.

**Current version v2.0** — adds natural-vocabulary models (3k / 6k / 10k / 16k) with a scale
evaluation, and fixes two index defects.
See the [release page](https://github.com/Sauomore/Cabinet_hsh64/releases/tag/v2.0).

---

## Supporting this project

If you make use of my project, **please consider giving it a star** — it genuinely matters to
me. Thank you very much.

If you hit a bug while using it, or have an improvement in mind, please
[open an issue](https://github.com/Sauomore/Cabinet_hsh64/issues).
Even if it is just a sentence in the docs that did not make sense — that usually means the docs
really do need fixing.

---

## The problem it solves

Suppose you want to build a "find similar words" feature. The most direct approach is:

1. Use a strong model (bge-large, say) to turn each word into a 1024-dimensional float vector
2. Similarity between two words = cosine similarity of the two vectors
3. Nearest neighbours = compute against every vector in the vocabulary, keep the top few

That method is **accurate, but expensive**: 1024 float32 values is 4 KB, so a million words is
4 GB; each comparison costs 1024 multiply-adds.

What HSH-64 does is: **train a small network that compresses that 1024-dimensional vector into a
single 64-bit integer**, with the requirement that "two words that were close before are still
only a few bits apart afterwards".

So:

| | Store one word | One comparison | A million words |
|---|---|---|---|
| Raw vector (1024-dim float32) | 4 KB | 1024 multiply-adds | 4 GB |
| HSH-64 code | **8 bytes** | **1 XOR + 1 popcount** | **8 MB** |

The price is some loss of accuracy (how much is in [Measured results](#measured-results)). What
you get back is **500x less storage, two to three orders of magnitude faster comparison, and no
GPU**.

### An analogy: postal codes

Real postal codes are layered: the first digits are the province, the middle ones the city, the
last ones the district. To tell whether two addresses are in the same district, you compare
codes — no need to read the street number.

The 64 bits of HSH-64 are layered the same way:

```text
 63    60 59                          8 7         0
┌────────┬───────────────────────────┬───────────┐
│  feat  │           sim             │    abs    │
│  4 bit │          52 bit           │   8 bit   │
└────────┴───────────────────────────┴───────────┘
 POS class       semantic print        in-bucket ID
"which province" "where exactly"    "which entry in the cell"
```

- **`feat` (4 bits)**: coarse category, like "province". Noun / verb / adjective … 16 slots in
  total. Its job is to separate words of different parts of speech so they do not share a bucket.
- **`sim` (52 bits)**: the semantic fingerprint, like "the detailed address within a province".
  **Retrieval quality is almost entirely determined by it.** 52 bits can express
  2⁵² ≈ 4.5×10¹⁵ distinct values.
- **`abs` (8 bits)**: the in-bucket index. `sim` has 52 bits but there are only 4,096 buckets, so
  a single `(feat, low 8 bits of sim)` bucket can hold several words; `abs` tells them apart
  (at most 256 per bucket).

Comparing how alike two codes are is this single line of Rust:

```rust
let distance = (a.sim() ^ b.sim()).count_ones();   // 0..=52, smaller means more alike
```

`count_ones()` compiles to one `POPCNT` instruction. That is why it runs so fast on a plain CPU.

---

## Why 64 bits

It is not an arbitrary number — three constraints pin it to this position:

**① It has to fit in one register.** 64 bits = one `u64` = one CPU register. A single core can
XOR and popcount several codes at once, with no SIMD and no unpacking.

**② It has to be large enough to separate enough words.** The code structure fixes its
"guaranteed unique" capacity:

```text
feat(16) × low 8 bits of sim(256) × abs(256) = 1,048,576 distinguishable codes
```

Measured capacity boundary (see
[docs/scale_evaluation.en.md](docs/scale_evaluation.en.md) §4.3):

| Vocabulary size | Largest bucket | Headroom to the 256 limit |
|---|---|---|
| 6,109 | 24 | 232 |
| 20,000 | 79 | 177 |
| 50,000 | 196 | 60 |
| 100,000 | 393 | **overflow** |

**The safe ceiling is around 50,000 words.** What binds is not the 52-bit `sim` but `abs` having
only 8 bits, together with the skew of `feat` (nouns take 47.9%).

**③ It has to be small enough that the index fits in memory.** A million codes take only 8 MB —
small enough to sit near the CPU caches, so random access does not hit main memory everywhere.

---

## How it is trained

Three stages, all offline, producing a 1.17 MB MLP.

```text
Stage 1  continuous pre-training   the net learns to preserve the teacher's neighbourhood structure
          │                        student input 512-dim → 512 hidden → 52 continuous values
          ▼
Stage 2  STE discrete refinement   sign-quantise those values into 52 bits,
          │                        using a straight-through estimator (STE) to pass gradients through
          ▼
Stage 3  recall-oriented post-proc greedy bit flipping directly against Recall@10:
                                   "if I flip this bit, how many more true neighbours do I recover?"
                                   the result is exported as sim_override.bin
```

> **Stage three is not an optional refinement.** Measured: skipping it leaves the bare code at
> 0.4988; adding it gives 0.7426. On natural vocabularies it is more extreme — the bare code at
> 0.2463 is **below even the zero-learning PCA baseline at 0.3757**, and post-processing lifts it
> to 0.6082.
>
> In other words, **"HSH-64 works" strictly means "HSH-64 plus greedy bit flipping works".**
> The derivation is in [docs/explained.en.md §8.3](docs/explained.en.md).

Retrieval is two-stage:

```text
query word ──→ small net ──→ 64-bit code ──→ MIH coarse pass (tens of candidates)
                                                    │
                                                    ▼
                                      bge-large rerank (optional) ──→ final top-10
```

The coarse pass uses only the codes, so it is fast; the rerank brings in the large model, so it is
accurate. The rerank can be switched off, which degrades gracefully to pure hash retrieval.

---

## Measured results

### On the paper's benchmark vocabulary (3,109 words, Recall@10)

"Ground truth" is defined by bge-large cosine similarity — that is, whichever word the teacher
model considers most similar is the answer.

| Method | Recall@10 | What it needs online |
|---|---|---|
| PCA-52 (zero-learning baseline) | 0.5318 | one PCA matrix |
| Deep Hash h512 (**no** post-processing) | 0.4265 | 1.17 MB MLP |
| **Deep Hash h512 + post-processing** | **0.7404** | 1.17 MB MLP |
| 4-model ensemble + post-processing | **0.7724** | ≈4 MB |
| **Deep Hash + MIH + rerank** | **0.8641** | 1.17 MB MLP + bge-large |
| Ensemble + MIH + rerank | **0.8970** | ≈4 MB + bge-large |

Independently reproduced in this repository (student input uses bge-large's PCA-512 in place of
the paper's bge-small):

```text
bare code 0.4988   + post-processing 0.7426   two-stage 0.8890
```

Post-processing and two-stage match the paper; the bare code is higher because the student input
is stronger.

### ⚠️ The scope of that table

A later evaluation found that **65% of the 3,109-word vocabulary above is synthetic.** The reason
is that no suitable corpus was available at the time — no longer a concern.

The construction was "10 modifiers × roughly 200 base words": `大U盘`, `大一月`, `大三月`, `红U盘`,
`快丝瓜` … leaving only about 1,114 genuinely natural base words.

Neighbourhoods in such a vocabulary are **2.18x sharper than natural text** (normalised top-10
boundary gap 0.0196 vs 0.0090), so the numbers above reflect capability on a
"structurally simpler task".

Re-run on the **natural Chinese vocabularies** found later (words drawn from a frequency list,
only 10.2% overlap with the vocabulary above):

| Vocabulary | Size | Bare code | + post-processing | vs PCA-52 baseline |
|---|---|---|---|---|
| Synthetic (table above) | 3,109 | 0.499 | **0.7426** | +0.131 |
| Natural | 3,000 | 0.246 | **0.6082** | **+0.233 (1.62x)** |
| Natural | 6,000 | 0.245 | 0.5990 | — |
| Natural | 10,000 | 0.247 | 0.5936 | — |
| Natural | 16,000 | 0.252 | **0.5965** | **+0.284 (1.91x)** |

**Two things worth noticing:**

1. **Scale barely affects recall.** A 5.3x larger vocabulary moves 0.6082 to 0.5965 — a drop of
   0.012. And this is not "the vocabulary was too small to be difficult": at 16,000 words,
   **62.3% of every query's true neighbours are words that do not exist in the 3,000-word
   vocabulary**, and 100% of queries have at least one new neighbour. The task genuinely got
   harder, and the codes held up.

2. **The advantage over the baseline grows with scale.** PCA-52 falls from 0.376 to 0.313
   (−0.063) while HSH-64 goes from 0.608 to 0.597 (−0.012). **Learned codes are more
   scale-stable than unlearned ones.**

Full data, mechanism analysis and index parameter selection:
**[docs/scale_evaluation.en.md](docs/scale_evaluation.en.md)**.

---

## What it is not good at

The usual trap in a project like this is writing "what it can do" as "what it can do anything
with". Below are the boundaries measured here:

**① It only does in-vocabulary retrieval, not sentence understanding.**
The input is a word and the output is the ten most similar words in the vocabulary. It does not
understand sentences, answer questions, or generate text. If you want to use it for document
deduplication or RAG, you have to wrap it in your own sentence splitting and joining.

**② The vocabulary is fixed.**
The model is only responsible for words it saw during training. New words (slang, technical
terms) require retraining. Measured, growing the vocabulary from 3,000 to 16,000 is fine (recall
does not fall), but somewhere around 50,000 words it hits the `abs` capacity ceiling —
**this will be addressed in a future version**.

**③ Recall depends heavily on post-processing, and post-processing does not scale.**
Stage three is an O(N²) greedy search. 3,000 words takes 27 s; 16,000 words takes 12.8 minutes
and saturates 8 GB of VRAM. Beyond 16k the implementation has to change (mini-batch with
hard-negative mining) — adding data alone will not do it.

**④ Those 4 `feat` bits are weaker than they look.**
They come from POS tagging, but the vocabulary holds **isolated words with no context**, and both
taggers degrade noticeably: 3,109 Chinese words produce 29 distinct tags ("同意" gets tagged as an
adverb), and about 36% of English words are mis-tagged (`tree` tagged as an adjective).
As a result the `feat` distribution is extremely skewed (nouns 47.9% + adjectives 38.4% = 86%),
with effective entropy under 1.5 bit.

This is acceptable by design — **semantic separation is carried by the 52-bit `sim`** (measured
share of unique codes: 99.9%), and `feat` only does coarse bucketing. But do not give it much
weight.

**⑤ Chinese–English support currently has bare-code results only.**
The bilingual pipeline (6,109 words, Chinese–English ≈ 1:1) runs end to end, but **has no
post-processing yet**; Recall@10 is 0.2750 and cannot be compared directly with any number above.
Following the pattern in ④②, adding post-processing should raise it substantially — but that
needs to be measured, not assumed.

---

## Quick start

### Requirements

- Rust 1.72+
- Python 3.8+ (only for the training scripts)
- **No GPU** (inference is pure CPU; the training scripts run on CPU too, just slowly)

```bash
cargo build --release
cargo test        # 34 tests, including the Python↔Rust mapping parity check
```

### Option 1: use pretrained models (recommended, 5 minutes)

Download a bundle from [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases), unpack it,
then:

```bash
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding  student_pca512.cache \
    --deep-hash  deep_hash_52.bin \
    --sim-override sim_override_52.bin \
    --reranker   teacher_bge_large.cache \
    --top-k 10 --queries 3000
```

**Natural-vocabulary models shipped in v2.0** (recommended — these correspond to the "Natural"
rows above):

| Bundle | Size | Vocabulary | Recall@10 after post-processing |
|---|---|---|---|
| `hsh64_pretrained_natural_3000.zip` | 19 MB | 3,000 natural Chinese words | 0.6082 |
| `hsh64_pretrained_natural_6000.zip` | 36 MB | 6,000 | 0.5990 |
| `hsh64_pretrained_natural_10000.zip` | 58 MB | 10,000 | 0.5936 |
| `hsh64_pretrained_natural_16000.zip` | 91 MB | 16,000 | 0.5965 |

Each bundle contains: the vocabulary, the student embedding cache (512-dim, **the model's actual
input**), the teacher embedding cache (1024-dim, **the source of evaluation ground truth**), the
DeepHash model, the post-processing override, the PCA basis, `MANIFEST.json` (sha256) and a README
with the reproduction command.

**Paper-benchmark-vocabulary models shipped in v1** (these correspond to the synthetic rows above):

| Bundle | Size | Contents |
|---|---|---|
| `hsh64_pretrained_3109_light.zip` | 18 MB | vocab + embedding caches + PCA-52 + Deep Hash h256 + post-processing |
| `hsh64_pretrained_3109_best.zip` | 18 MB | as above, Deep Hash h512 s42 (best single model) |
| `hsh64_pretrained_3109_full.zip` | 22 MB | multiple models and post-processing sets (reproduce the ensemble) |

> **Do not mix the two conventions**: `--reranker` serves as both the rerank model and the source
> of ground truth. With `--reranker` you get the teacher-truth convention (0.6082); without it the
> truth comes from the student embeddings, which is **a different metric** (0.5947 measured on
> 3,000 words). The number in each bundle's `eval_result.json` is the teacher-truth one.

> The repository itself carries no `.bin` / `.cache` (see [.gitignore](.gitignore)) — get them from
> Releases.

### Option 2: train from scratch

```bash
# 1. Student embeddings (bge-small, 512-dim)
python scripts/generate_embeddings.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/embedding.cache

# 2. Teacher embeddings (bge-large, 1024-dim)
python scripts/generate_reranker_embeddings_64.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/reranker_embedding.cache

# 3. PCA-52 baseline
python scripts/train_pca_64.py \
    --embedding-cache tests/data/embedding.cache \
    -o tests/data/pca_52.bin

# 4. Deep Hash (stages 1+2)
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --hidden-dim 256 --seed 2025 \
    -o tests/data/deep_hash_v3_64_3109_h256_s2025.bin

# 5. Recall-oriented post-processing (stage 3) — the step that takes 0.4988 to 0.7426
python scripts/post_optimize_codes_recall_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --model-type deep_hash \
    --model-path tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --pos-k 10 --k 10 --neg-weight 1.0 --max-iters 10 \
    -o tests/data/sim_override_3109_h256_s2025_recall_mi100.bin

# 6. Evaluate
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding.cache \
    --pca tests/data/pca_52.bin \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109
```

> For exact flags, defer to `python scripts/<script> --help`.

### Custom vocabulary size

To change the vocabulary size, use the v2.0 pipeline (one pass: build vocabulary → embed → PCA →
train → post-process → evaluate):

```bash
python scripts/53_build_natural_vocab.py --sizes 3000 6000 10000 16000
python scripts/54_run_scale_curve.py --sizes 6000 10000
python scripts/62_natural_baselines.py          # compare against random codes / PCA-52
```

---

## Index: how to set the radius

This one bit us, so it gets its own section.

MIH (multi-index hashing) prunes using "segments + radius". **Too small a radius retrieves not a
single positive sample** — positive pairs on a 52-bit code sit roughly 9-10 bits apart, while the
default parameters once swept only up to 8.

Measured (3,109 words, pure Hamming):

| Radius | Mean candidates | Recall@10 |
|---|---|---|
| 8 | 4.3 | 0.330 |
| 18 | 10.0 | 0.605 |
| **22** | **10.0** | **0.605** |

At radius 22, MIH reaches 95% of the brute-force scan. **So `--radii` has to reach at least 22**,
or use `--adaptive` and let it expand until the candidate pool saturates.

The segment count `seg` affects speed but not recall (everything is ultimately filtered by the full
52-bit distance): `seg=2` 1304 QPS, `seg=4` 834 QPS, `seg=13` 382 QPS.

> `seg=1` is unsupported — the segment value would occupy all 52 bits, exceeding the index key
> width and making the in-segment Hamming ball impossible to enumerate. Passing 1 produces an
> explicit configuration error rather than silently returning empty results.
> Details in [docs/scale_evaluation.en.md](docs/scale_evaluation.en.md) §4.

---

## Chinese–English support

The 4 bits of `feat` are **the only language-dependent part**. Chinese (jieba tags) and English
(Penn Treebank tags) map onto one shared set of 16 slots, so words from both languages can enter
the same codebook.

```text
0x0 noun        n/nr/ns/…        NN/NNS/NNP/…        0x8  numeral   m/mq        CD/LS
0x1 verb        v/vd/vn/…        VB/VBD/VBG/…        0x9  measure   q/qv/qt     (Chinese only)
0x2 adjective   a/ad/an/ag/b     JJ/JJR/JJS          0xA  time word t/tg        (Chinese only)
0x3 adverb      d/df/dg          RB/RBR/RBS/WRB      0xB  string    x/xx/…/eng
0x4 pron·det    r/rr/rz/…        PRP/PRP$/DT/…       0xC  punct     w series    . , : `` ''
0x5 prep·loc    p/f/fg/s         IN/TO/RP            0xD  other     i           UH/POS
0x6 conjunction c/cc             CC                  0xE  common word (decided at runtime)
0x7 particle·md u/ud/…/z/zg      MD                  0xF  fallback
```

**Vocabulary**: 6,109 words = 3,095 Chinese + 3,002 English + 12 mixed (`T恤`, `U盘`)

```bash
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000   # select English by frequency from ECDICT
python scripts/49_tag_vocab.py                                 # bilingual POS tagging
```

Cross-lingual alignment was validated with **bge-m3**: 40 Chinese–English translation pairs plus 6
negative pairs, AUC = 1.0000, smallest positive +0.5446 > largest negative +0.5393, no
distribution overlap.

> ⚠️ But **the sample is only 40 positives / 6 negatives, so AUC = 1.0 does not extrapolate**.
> The conclusion is limited to "the premise for cross-lingual alignment holds".
> And the bilingual model **has no post-processing yet** (bare code 0.2750).
> Current state and known limitations: [docs/bilingual.en.md](docs/bilingual.en.md).

---

## API

### Rust

```rust
use hsh64::{Encoder, EncoderConfig, HSHCode64, hamming_distance64};

// The most basic thing: distance
let dist = hamming_distance64(a.sim(), b.sim());   // one XOR + POPCNT

// Encoder
let mut config = EncoderConfig {
    embed_dim: 512,
    embedding_cache_path: Some("student_pca512.cache".into()),
    pca_path: Some("pca_52.bin".into()),
    deep_hash_path: Some("deep_hash_52.bin".into()),
    sim_override_path: Some("sim_override_52.bin".into()),
    ..Default::default()
};
// Bilingual use needs the tag table; without it every word is treated as "n" (old behaviour)
config.load_pos_tags("data/mixed_tags.tsv".as_ref())?;

let encoder = Encoder::with_config(config)?;

let code = encoder.encode_word_auto("北京");                        // auto tag lookup + language detection
let code = encoder.encode_word_with_pos_lang("apple", "NN", PosLang::En);  // explicit
```

```rust
use hsh64::MihSemanticIndex;

let index = MihSemanticIndex::build_with_embedding(encoder, embedding, 13, false)?;

let (radius, results) = index.search_adaptive(query, top_k, coarse_factor, max_radius)?;
let (radius, results) = index.search_adaptive_asymmetric(query, top_k, coarse_factor, max_radius)?;
```

### Python scripts

| Script | Purpose |
|---|---|
| `generate_embeddings.py` | generate the student embedding cache |
| `generate_reranker_embeddings_64.py` | generate the teacher / rerank embedding cache |
| `train_pca_64.py` | train the PCA-52 baseline |
| `train_deep_hash_v3_64.py` | train Deep Hash (stages 1+2) |
| `post_optimize_codes_64.py` | basic greedy bit flipping |
| `post_optimize_codes_recall_64.py` | recall-oriented post-processing (stage 3) |
| `bilingual_pos.py` | bilingual POS tagging (strictly mirrors `pos_map.rs`) |
| `48_build_vocab.py` / `49_tag_vocab.py` | build and tag the mixed vocabulary |
| `53_build_natural_vocab.py` | build nested natural Chinese vocabularies |
| `52_eval_pipeline.py` | one command for train + post-process + evaluate + export |
| `54_run_scale_curve.py` | run the scale curve in batch |
| `62_natural_baselines.py` | compare against random codes / PCA-52 |
| `58_package_release.py` | package release assets |

---

## Repository layout

```text
hsh64/
├── src/                        Rust core library
│   ├── hsh64.rs                HSHCode64 / Hamming distance
│   ├── encoder.rs              Encoder / EncoderConfig (including tag-table loading)
│   ├── deep_hash.rs            DeepHashEncoder64 / DeepHashProjection
│   ├── pca.rs                  PcaProjection
│   ├── mih_index.rs            MihSemanticIndex adaptive search
│   ├── embedding.rs            Embedding trait / FileCachedEmbedding
│   ├── perfect_hash.rs         in-cluster perfect hashing
│   ├── pos_map.rs              POS mapping (bilingual, 16 shared slots)
│   └── error.rs                error types
├── examples/
│   ├── benchmark_pure_hsh64.rs     single-model benchmark (brute force / MIH / asymmetric / rerank)
│   └── benchmark_ensemble_hsh64.rs ensemble benchmark
├── scripts/                    Python training and evaluation
├── tests/
│   ├── end_to_end.rs
│   ├── feat_parity.rs          Python↔Rust POS mapping parity
│   └── data/
├── data/                       working data (generated, mostly not committed)
├── docs/
│   ├── README.en.md            documentation index
│   ├── explained.en.md         design document (from first principles)
│   ├── scale_evaluation.en.md  every measured result and how to reproduce it
│   ├── bilingual.en.md         Chinese–English extension
│   └── *.md                    Chinese versions of each document
```

---

## Status

| Component | Status |
|---|---|
| Code structure `feat(4) + sim(52) + abs(8)` | ✅ done |
| Three-stage training pipeline | ✅ done |
| Bilingual `feat` mapping | ✅ done (bilingual evaluation still needs post-processing) |
| Mixed vocabulary, 6,109 words | ✅ done |
| Natural Chinese vocabularies 3k / 6k / 10k / 16k | ✅ done |
| Scale evaluation (3k→16k recall curve + baselines) | ✅ done |
| Python↔Rust parity test | ✅ done (mutation-verified) |
| Unit + integration + parity tests | ✅ all 34 pass |
| `abs` perfect-hash seed table | ⚠️ inert (`build_seed_table` has no call site, so `abs` loses its uniqueness guarantee) |
| Post-processing for the bilingual model | ⚠️ not done (bare code 0.2750) |

> The inert seed table **does not affect recall** — `abs` does not participate in MIH retrieval
> (which uses `sim` segmentation plus `feat` filtering) — but the field currently lacks the
> uniqueness the design intends. See [docs/bilingual.en.md](docs/bilingual.en.md) §9.3.

---

## Documentation

| Document | Contents |
|---|---|
| **[docs/](docs/README.en.md)** | **documentation index** — organised by what you want to do |
| **[docs/explained.en.md](docs/explained.en.md)** | design document from first principles: vocabulary and embeddings, what Recall@K means, codec, DeepHash, three-stage training, asymmetric distance, MIH, two-stage retrieval |
| **[docs/scale_evaluation.en.md](docs/scale_evaluation.en.md)** | every measured result: reproduction of the paper's benchmark, the natural-vocabulary 3k→16k curve, comparison against a zero-learning baseline, index radii, the `abs` capacity ceiling, reproduction commands |
| **[docs/bilingual.en.md](docs/bilingual.en.md)** | Chinese–English support: `feat` slot allocation, vocabulary construction, cross-lingual alignment validation, known limitations |

Every document has a Chinese version without the `.en` suffix — see
[docs/README.md](docs/README.md).

The paper is not distributed in this repository. The paper PDFs are inside the source
archive on [Zenodo](https://doi.org/10.5281/zenodo.21721838) — the v1 and v2 bundles
contain `paper/main.pdf` and `paper/main_chinese.pdf`. Use that DOI for citation.

---

## Citation

```bibtex
@software{hsh64,
  author  = {Sauomore},
  title   = {HSH-64: 64-bit Learnable Semantic Hashing},
  doi     = {10.5281/zenodo.21721838},
  url     = {https://github.com/Sauomore/Cabinet_hsh64}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
