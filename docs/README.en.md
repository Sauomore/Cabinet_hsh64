# Documentation

Organised by what you are trying to do. To look at the code instead, start with the
[README](../README.en.md).

---

## I want to understand what this is

| Document | Contents | Rough length |
|---|---|---|
| [explained.en.md](explained.en.md) | From first principles: vocabulary and embeddings, teacher similarity, what Recall@K means, the structure of the 52-bit code, the DeepHash network, three-stage training, asymmetric distance, MIH, two-stage retrieval | 660 lines |
| [hsh64_principle.html](hsh64_principle.html) | Illustrated principles (open in a browser) | — |

`explained.en.md` is the self-study path: it assumes you only know what vector retrieval is,
and every other concept is defined before it is used.

---

## I want to see the actual numbers

| Document | Contents |
|---|---|
| [scale_evaluation.en.md](scale_evaluation.en.md) | Every measured result: reproduction on the paper's benchmark vocabulary, the natural-Chinese 3k→16k scale curve, comparison against a zero-learning baseline, how to choose the index radius, the capacity ceiling of `abs`, training cost, reproduction commands, limitations |

At a glance:

```text
paper benchmark (3,109 words)   bare 0.4988   + post-processing 0.7426   two-stage 0.8890
natural vocabulary (3,000)      bare 0.2463   + post-processing 0.6082
natural vocabulary (16,000)     bare 0.2519   + post-processing 0.5965
```

---

## I care about Chinese–English support

| Document | Contents |
|---|---|
| [bilingual.en.md](bilingual.en.md) | How the 16 `feat` slots are shared by two POS tag sets, how the bilingual vocabulary was built, cross-lingual alignment validation, measured limitations and current state |

---

## I want to start using it

Go to "Quick start" in the [README](../README.en.md). Two routes:

- **Use pretrained assets** — download from
  [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases); each bundle carries the
  vocabulary, embedding caches, the model and the reproduction command
- **Train from scratch** — step-by-step commands in the README

---

## How the documents map onto the code

| Document | Main files involved |
|---|---|
| `explained.en.md` | `src/hsh64.rs`, `src/deep_hash.rs`, `src/mih_index.rs`, `src/encoder.rs` |
| `bilingual.en.md` | `src/pos_map.rs`, `src/encoder.rs`, `scripts/bilingual_pos.py`, `tests/feat_parity.rs` |
| `scale_evaluation.en.md` | `scripts/50`–`62_*.py`, `examples/benchmark_pure_hsh64.rs` |

---

## Chinese versions

Every document has a Chinese counterpart without the `.en` suffix:

| English | 中文 |
|---|---|
| [explained.en.md](explained.en.md) | [explained.md](explained.md) |
| [scale_evaluation.en.md](scale_evaluation.en.md) | [scale_evaluation.md](scale_evaluation.md) |
| [bilingual.en.md](bilingual.en.md) | [bilingual.md](bilingual.md) |
