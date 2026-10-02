# HSH-64

[![DOI](https://zenodo.org/badge/1307857867.svg)](https://doi.org/10.5281/zenodo.21721838)
[![GitHub release](https://img.shields.io/github/v/release/Sauomore/Cabinet_hsh64?include_prereleases&style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases)

[中文](README.md) ｜ **English**

> A 64-bit learnable semantic hashing scheme for lightweight **Chinese–English**
> vocabulary-level retrieval.
> One `u64` per item, one `popcnt` per comparison, runs on CPU only.

---

## What this is

HSH-64 extends the semantic hash code from 20 bits in HSH-32 to **52 bits**,
keeping the hardware-friendly shape while substantially increasing how much
semantic structure the discrete space can carry.

```text
[ feat: 4 bit ] + [ sim: 52 bit ] + [ abs: 8 bit ] = 64 bit
```

| Field | Bits | Meaning |
|---|---|---|
| `feat` | 4 | part-of-speech class, 16 slots, usable as a hard filter |
| `sim` | 52 | semantic similarity code, drives retrieval quality; capacity 2⁵² ≈ 4.5×10¹⁵ |
| `abs` | 8 | within-bucket unique id, disambiguates words sharing a `(feat, sim)` bucket |

Distance between two codes is **one 64-bit XOR plus a popcount**:

```rust
let dist = (a ^ b).count_ones();
```

### Highlights

- **Structured 64-bit code** — one `u64` per item, hardware popcount comparison
- **Bilingual** — the 16 `feat` slots are shared by Chinese and English POS tagsets, so both languages can live in one codebook
- **Three-stage end-to-end training** — continuous pretraining → STE discrete fine-tuning → recall-oriented greedy post-processing
- **Two-stage retrieval** — light encoder for coarse ranking, optional bge-large rerank
- **Adaptive MIH** — dynamic radius expansion balancing recall against candidate pool size
- **Asymmetric distance scoring** — keeps query-side projection magnitude, reducing sign-quantisation loss
- **Multi-model ensemble** — candidate fusion for further recall gains
- **CPU only** — neither training nor inference needs a GPU

### Benchmark (3,109-word Chinese vocabulary, Recall@10)

> Figures come from the paper's `tab:main-results-cn`. Ground-truth top-K is
> defined by bge-large cosine similarity.

| Method | Recall@10 | Online model |
|---|---|---|
| PCA-52 (pure HSH) | 0.5318 | PCA projection matrix |
| DH h128_s42 (**no** post-processing) | 0.4228 | 421 KB MLP |
| DH h256_s2025 (**no** post-processing) | 0.4315 | 585 KB MLP |
| DH h512_s42 (**no** post-processing) | 0.4265 | 1.17 MB MLP |
| DH h1024_s2024 (**no** post-processing) | 0.4315 | 2.33 MB MLP |
| **DH h256_s42 + post-processing** | **0.7382** | 585 KB MLP |
| **DH h512_s42 + post-processing** | **0.7404** | 1.17 MB MLP |
| 4-model ensemble | **0.7724** | ≈4 MB |
| DH + MIH + rerank | 0.8641 | 1.17 MB MLP + bge-large |
| Ensemble + MIH + rerank | **0.8970** | ≈4 MB + bge-large |

**⚠️ One counter-intuitive but important result: without post-processing, Deep
Hash is *worse* than PCA-52** (0.42–0.43 vs 0.53), despite being a more
expressive non-linear transform. **Stage-three post-processing (recall-oriented
greedy bit flipping) is not an optional refinement — it is what makes this
scheme work at all.** See [docs/explained.md §8.3](docs/explained.md).

---

## Chinese–English support

The 4-bit `feat` field is the **only language-dependent part**. Chinese (jieba
tags) and English (Penn Treebank tags) map onto the same 16 slots, so words from
either language can share one codebook.

```text
0x0 noun      n/nr/ns/…      NN/NNS/NNP/…     0x8  numeral   m/mq       CD/LS
0x1 verb      v/vd/vn/…      VB/VBD/VBG/…     0x9  measure   q/qv/qt    (Chinese only)
0x2 adjective a/ad/an/ag/b   JJ/JJR/JJS       0xA  time      t/tg       (Chinese only)
0x3 adverb    d/df/dg        RB/RBR/RBS/WRB   0xB  string    x/xx/…/eng
0x4 pron+det  r/rr/rz/…      PRP/PRP$/DT/…    0xC  punct     w-series   . , : `` ''
0x5 prep/loc  p/f/fg/s       IN/TO/RP         0xD  other     i          UH/POS
0x6 conj      c/cc           CC               0xE  common word (runtime)
0x7 particle+modal u/ud/…/z/zg  MD            0xF  fallback
```

**Vocabulary**: 6,109 words = 3,095 Chinese + 3,002 English + 12 mixed (`T恤`, `U盘`)

```bash
# Build the mixed vocabulary (English side selected by frequency from ECDICT)
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000
python scripts/49_tag_vocab.py          # bilingual POS tagging
```

> ⚠️ **Measured limitation.** The vocabulary holds isolated words with no sentence
> context, and both taggers are trained on sentences, so they degrade badly:
> 3,109 Chinese words produce **29 distinct tags** (`同意` → adverb), and the
> English side is roughly **36% wrong** on a probe set (`tree` → adjective).
> The resulting `feat` distribution is skewed — slot 0x0 takes 47.9% and 0x2
> takes 38.4%, leaving under 1.5 bits of entropy.
>
> This is acceptable by design: **semantic separation is carried by the 52-bit
> `sim` field** (measured 99.2% unique codes), and `feat` only needs to be a
> coarse bucketing cue. Downstream code should not weight it heavily.
> Full data in **[docs/bilingual.md](docs/bilingual.md)**.

> **Measured progress on the bilingual vocabulary**: bare Deep Hash with no
> post-processing reaches Recall@10 = **0.2750**. That gap against the 0.43 in
> the table above comes from vocabulary size (6,109 vs 3,109) and language
> mixing, and no post-processing has been run yet — so **it is not comparable to
> the table**. See "Status" below.

---

## Architecture

```text
┌──────────────────────── offline training ────────────────────────┐
│                                                                  │
│   bge-small          bge-large        Deep Hash MLP              │
│   512-dim     ───→   1024-dim  ───→   dim→H→52 bits              │
│   student             teacher          STE + multi-objective     │
│       │                  │                  │                    │
│       ▼                  ▼                  ▼                    │
│  embedding.cache   reranker.cache   deep_hash_v3.bin             │
│                                                                  │
│         recall-oriented greedy post-processing → sim_override    │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────── online inference ────────────────────────┐
│                                                                  │
│   query term ──→ bge/m3 + MLP ──→ HSH-64 Encoder                 │
│                                       │                          │
│                                       ▼                          │
│                             MIH coarse search (adaptive radius)  │
│                                       │                          │
│                                       ▼                          │
│                             bge-large rerank (optional)          │
└──────────────────────────────────────────────────────────────────┘
```
## Quick start

### Requirements

- Rust 1.72+
- Python 3.8+ (`torch`, `numpy`, `sentence-transformers` for the training scripts)
- No GPU required

```bash
cargo build --release
cargo test
```

### Option 1: use released pretrained assets (recommended)

Download an asset bundle from
[Releases](https://github.com/Sauomore/Cabinet_hsh64/releases), unpack into
`tests/data/`, then:

```bash
# Brute-force HSH-64 scan
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109
```

| Bundle | Size | Contents |
|---|---|---|
| `hsh64_pretrained_3109_light.zip` | ~18 MB | vocab + both embedding caches + PCA-52 + Deep Hash h256 + post-processing |
| `hsh64_pretrained_3109_best.zip` | ~18 MB | as above, Deep Hash h512 s42 (best single model) |
| `hsh64_pretrained_3109_full.zip` | ~22 MB | multiple models and post-processing sets (reproduce the ensemble) |

> The Git repository does not carry `.bin` / `.cache` files (see
> [.gitignore](.gitignore)). Get them from the Releases page.

### Option 2: train from scratch

```bash
# 1. Student embeddings (bge-small)
python scripts/generate_embeddings.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/embedding.cache

# 2. Teacher / reranker embeddings (bge-large)
python scripts/generate_reranker_embeddings_64.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/reranker_embedding.cache

# 3. PCA-52 baseline
python scripts/train_pca_64.py \
    --embedding-cache tests/data/embedding.cache \
    -o tests/data/pca_52.bin

# 4. Deep Hash v3 (stages 1 and 2)
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --hidden-dim 256 --seed 2025 \
    -o tests/data/deep_hash_v3_64_3109_h256_s2025.bin

# 5. Recall-oriented post-processing (stage 3) -- this is what moves 0.72 to 0.74
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

> **Script flags**: `python scripts/<script> --help` is authoritative.
> Every command above was checked against the actual parsers.

---

## Development

```bash
cargo test                      # unit + integration + parity
cargo run --release --example benchmark_pure_hsh64 -- --help
cargo run --release --example benchmark_ensemble_hsh64 -- --help
```

### Test layout

| Suite | Contents |
|---|---|
| `src/**` unit tests | 26 tests: codec, PCA, Deep Hash, MIH, POS mapping |
| `tests/end_to_end.rs` | end-to-end (needs the caches under `tests/data/`) |
| `tests/feat_parity.rs` | **Python ↔ Rust POS-mapping parity** (4 tests) |

**Why the parity test exists.** The `(POS tag → feat)` table lives in two places:
Python builds codebooks, Rust queries them. If the two drift, "the codebook
Python wrote is read back wrong by Rust" — and that is hard to notice, because
either side looks correct on its own.

```bash
python scripts/46_export_feat_table.py    # export the Python table
cargo test --test feat_parity             # assert it entry by entry
```

---

## Repository layout

```text
hsh64/
├── src/                        Rust core
│   ├── hsh64.rs                HSHCode64 / Hamming distance
│   ├── encoder.rs              Encoder / EncoderConfig (loads POS tags)
│   ├── deep_hash.rs            DeepHashEncoder64 / DeepHashProjection
│   ├── pca.rs                  PcaProjection
│   ├── mih_index.rs            MihSemanticIndex adaptive search
│   ├── embedding.rs            Embedding trait / FileCachedEmbedding
│   ├── perfect_hash.rs         within-bucket perfect hash
│   ├── pos_map.rs              POS mapping (bilingual, 16 shared slots)
│   └── error.rs                error types
├── examples/
│   ├── benchmark_pure_hsh64.rs
│   └── benchmark_ensemble_hsh64.rs
├── scripts/                    Python training and evaluation
│   ├── generate_embeddings.py
│   ├── generate_reranker_embeddings_64.py
│   ├── train_pca_64.py
│   ├── train_deep_hash_v3_64.py
│   ├── post_optimize_codes_64.py
│   ├── post_optimize_codes_recall_64.py
│   ├── bilingual_pos.py        bilingual POS tagging (mirrors pos_map.rs)
│   ├── 46_export_feat_table.py export the table for the parity test
│   ├── 47_detect_lang.py       vocabulary language composition
│   ├── 48_build_vocab.py       build the mixed vocabulary
│   └── 49_tag_vocab.py         tag the mixed vocabulary
├── tests/
│   ├── end_to_end.rs
│   ├── feat_parity.rs          Python↔Rust parity
│   └── data/
├── data/                       bilingual working data (generated, mostly ignored)
├── docs/
│   ├── bilingual.md            Chinese–English extension
│   ├── explained.md            accessible design document
│   ├── experiment_data_export.md
│   └── hsh64_principle.html
└── paper/                      paper sources and PDFs (EN + ZH)
```

---

## API

### Rust

```rust
use hsh64::{Encoder, EncoderConfig, HSHCode64, hamming_distance64};

// Basics
let code = HSHCode64::new(feat, sim, abs);
let dist = hamming_distance64(a.sim(), b.sim());

// Encoder
let mut config = EncoderConfig {
    embed_dim: 512,
    embedding_cache_path: Some("tests/data/embedding_3109.cache".into()),
    pca_path: Some("tests/data/pca_52.bin".into()),
    deep_hash_path: Some("tests/data/deep_hash_v3_64_3109_h256_s2025.bin".into()),
    sim_override_path: Some("tests/data/sim_override_3109_h256_s2025_recall_mi100.bin".into()),
    ..Default::default()
};
// Bilingual: load the POS table. Without it every word is treated as "n",
// which is the pre-bilingual behaviour.
config.load_pos_tags("data/mixed_tags.tsv".as_ref())?;

let encoder = Encoder::with_config(config)?;

// Recommended entry point: resolves the tag and dispatches by language
let code = encoder.encode_word_auto("北京");

// Or state both explicitly
let code = encoder.encode_word_with_pos_lang("apple", "NN", PosLang::En);
```

```rust
use hsh64::MihSemanticIndex;

let index = MihSemanticIndex::build_with_embedding(encoder, embedding, 13, false)?;

let (radius, results) = index.search_adaptive(query, top_k, coarse_factor, max_radius)?;
let (radius, results) =
    index.search_adaptive_asymmetric(query, top_k, coarse_factor, max_radius)?;
```

### Python scripts

| Script | Purpose |
|---|---|
| `generate_embeddings.py` | student embedding cache |
| `generate_reranker_embeddings_64.py` | teacher / reranker embedding cache |
| `train_pca_64.py` | train the PCA-52 projection |
| `train_deep_hash_v3_64.py` | train the Deep Hash v3 MLP |
| `post_optimize_codes_64.py` | basic greedy bit-flip post-processing |
| `post_optimize_codes_recall_64.py` | recall-oriented post-processing |
| `bilingual_pos.py` | Chinese–English POS tagging |
| `48_build_vocab.py` / `49_tag_vocab.py` | mixed vocabulary construction and tagging |
| `benchmark_pure_hsh64.py` / `ensemble_eval.py` | Python-side benchmark and ensemble |

---

## Status

### Done

| Item | State |
|---|---|
| `feat(4)+sim(52)+abs(8)` codec | ✅ |
| Three-stage training pipeline | ✅ |
| Bilingual `feat` mapping | ✅ |
| Mixed vocabulary, 6,109 words | ✅ |
| Python↔Rust parity test | ✅ (mutation-verified) |
| Unit + integration tests | ✅ 31 total |

### ⚠️ Open

```text
1. MIH index recall is anomalous (highest priority)
   seg=2/4/13 all give identical results, and candidate counts are 2.3
   against 6109 for brute force. Suspected correctness bug in the index.

2. Recall is low without post-processing
   bare DeepHash 0.2750 vs 0.728 with post-processing.
   Not a regression -- a pipeline stage that was skipped.

3. build_seed_table is never called
   seed_for therefore always returns 0 and the perfect-hash seed search
   never runs. Does not affect MIH recall, but leaves abs without any
   uniqueness guarantee.
```

Full record in **[docs/bilingual.md](docs/bilingual.md), section 9**.

---

## Documentation

| Document | Contents |
|---|---|
| **[docs/bilingual.md](docs/bilingual.md)** | Chinese–English extension: slot allocation, vocabulary construction, cross-lingual validation, open problems |
| **[docs/explained.md](docs/explained.md)** | accessible design document: codec, three-stage training, MIH, asymmetric distance |
| [docs/experiment_data_export.md](docs/experiment_data_export.md) | full experiment data summary |
| [docs/hsh64_principle.html](docs/hsh64_principle.html) | illustrated principles |
| [paper/](paper/) | paper sources and PDFs (EN + ZH) |

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

MIT OR Apache-2.0
