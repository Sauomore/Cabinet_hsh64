# Benchmark and Scale Evaluation

This page reports every measured result for HSH-64: reproduction on the paper's benchmark
vocabulary, the scale curve on natural Chinese vocabularies, comparison against a
zero-learning baseline, how to choose the index radius, and the capacity ceiling imposed by
the `abs` field.

Every number here is reproducible with the commands at the end of this page.

---

## 1. Configuration

| Item | Value |
|---|---|
| Teacher embeddings | bge-large-zh-v1.5 (1024-dim) |
| Student input | **PCA-512** truncation of the teacher (matches the h512 model's input width) |
| DeepHash | 512 → 512 → 52 bit, 500 epochs, seed 42 |
| Post-processing | greedy bit flipping, `pos_k=10, k=10, neg_weight=1.0, max_iters=10` |
| Ground truth | top-K by teacher cosine similarity |

> **One configuration difference**: bge-small weights are not available on the machine used
> here, so the student input is bge-large's PCA-512 instead. This inflates bare-code recall
> (see §2.2), so these figures should not be compared digit-for-digit against the paper's.
> All comparisons *within* this page use one and the same configuration.

---

## 2. The paper's benchmark vocabulary (3,109 words)

### 2.1 Results

| Stage | Recall@10 | Paper reports | Note |
|---|---|---|---|
| DeepHash bare code (Hamming) | 0.4988 | 0.4265 | higher, see 2.2 |
| **+ post-processing** | **0.7426** | **0.7404** | matches |
| MIH + bge-large rerank (two-stage) | **0.8890** | 0.8641 | close |
| Brute-force asymmetric scan | 0.6547 | — | added here |

Unique codes: `3,062 / 3,109 = 98.49%`.

### 2.2 Why bare-code recall is higher

The student input is bge-large's PCA-512, which is stronger than the paper's bge-small, so
bare-code recall exceeds the paper's figure. Post-processing and two-stage results are not
affected — they match the paper.

### 2.3 What this benchmark vocabulary actually contains

Counting by first character, **65.2% of the 3,109 entries are synthetic "modifier + base
word" combinations**:

| First char | Entries | Examples |
|---|---|---|
| 大 | 209 | 大U盘, 大一月, 大三月, 大丝瓜, 大主管 |
| 红 | 204 | 红U盘, 红一月, 红三月, 红丝瓜 |
| 美 / 小 / 快 | ~203 each | same pattern |
| 蓝 / 好 / 新 / 绿 / 旧 | ~200 each | same pattern |

Ten modifiers times roughly 200 base words gives 2,027 entries; another 36 are templates
(`大尖的`, `小粗的`). **The genuinely natural base vocabulary is about 1,114 words.**

Neighbourhoods in such a vocabulary are sharper than in natural text. Using random-pair
cosine as the semantic-span baseline:

| Metric | Synthetic 3,109 | Natural 3,000 |
|---|---|---|
| Random-pair cosine (semantic span) | 0.3058 | 0.4618 |
| top-10 mean cosine | 0.7046 | 0.7130 |
| 10th / 11th neighbour cosine | 0.6587 / 0.6450 | 0.6694 / 0.6645 |
| 10-11 gap | 0.0136 | 0.0049 |
| **Normalised gap** (gap ÷ semantic span) | **0.0196** | **0.0090** |

The synthetic vocabulary's top-10 boundary is **2.18x sharper**, so ranking the top 10 is
easier there.

> A frequent guess is that the task is easy because same-prefix words cluster together.
> Measured, it does not hold: the share of top-10 neighbours sharing the query's first
> character is 16.3% for the synthetic vocabulary and 19.1% for the natural one — the
> synthetic vocabulary is *lower*. The difference comes from boundary sharpness, not from
> literal overlap.

---

## 3. Natural Chinese vocabularies

### 3.1 Vocabulary construction

Built from `wordfreq`'s Chinese frequency list with these filters:

```text
length >= 2 (drops single-character function words)
CJK characters only
stopwords removed (的 / 是 / 在 / 然后 …)
frequency order preserved
```

The four vocabularies are **nested** (3,000 ⊂ 6,000 ⊂ 10,000 ⊂ 16,000): the smaller one is a
prefix of the larger, so the same words recur at every size and differences between sizes
carry no change in vocabulary composition.

| Size | Mean length | Overlap with the paper's 3,109 vocabulary |
|---|---|---|
| 3,000 | 2.05 chars | **10.2%** (318 words) |
| 6,000 | 2.09 chars | — |
| 10,000 | 2.13 chars | — |
| 16,000 | 2.18 chars | — |

### 3.2 Comparison against a zero-learning baseline

An absolute number is uninterpretable without a baseline. The table adds two zero-cost
baselines: random 52-bit codes, and PCA-52 (PCA the teacher down to 52 dimensions, then
quantise by sign — no training, no post-processing).

| Vocabulary | N | Random 52-bit | PCA-52 | HSH-64 bare | HSH-64 + post | vs PCA-52 |
|---|---|---|---|---|---|---|
| Synthetic | 3,109 | 0.0029 | 0.6117 | 0.4988 | **0.7426** | +0.1309 (1.21x) |
| Natural | 3,000 | 0.0037 | 0.3757 | 0.2463 | **0.6082** | +0.2326 (1.62x) |
| Natural | 16,000 | 0.0008 | 0.3126 | 0.2519 | **0.5965** | +0.2840 (1.91x) |

Three observations:

1. **The advantage over PCA-52 is larger on natural vocabularies, and it grows with scale.**
   PCA-52 falls from 0.3757 to 0.3126 (−0.063) while HSH-64 goes from 0.6082 to 0.5965
   (−0.012).

2. **The bare code is worse than PCA-52** (0.246 vs 0.376). This agrees with the paper's own
   finding on the synthetic vocabulary (0.4265 vs 0.5318) and is more pronounced here.
   **Post-processing — the third-stage greedy bit flipping — is not an optional refinement but
   a precondition for this scheme to work at all.**

3. **The PCA-52 used here is stronger than the paper's** (0.6117 vs 0.5318 on the synthetic
   vocabulary; the difference is per-dimension median centring). A stronger baseline makes
   the relative gains above conservative.

> Read this comparison for what it is: PCA-52 involves zero training and zero post-processing,
> while HSH-64 used 500 epochs plus 10 flip rounds. It compares the whole pipeline against a
> zero-cost baseline, not two coding schemes against each other.

### 3.3 Scale curve

| N | Bare overlap@10 | **+ post overlap@10** | @1 | @5 | @20 | Unique sim codes |
|---|---|---|---|---|---|---|
| 3,000 | 0.2463 | **0.6082** | 0.4127 | 0.6174 | 0.4324 | 98.17% |
| 6,000 | 0.2453 | **0.5990** | 0.4265 | 0.6088 | 0.4279 | 98.92% |
| 10,000 | 0.2466 | **0.5936** | 0.3941 | 0.6044 | 0.4202 | 99.65% |
| 16,000 | 0.2519 | **0.5965** | 0.3907 | 0.6004 | 0.4253 | 99.89% |

**A 5.3x larger vocabulary moves post-processing recall from 0.6082 to 0.5965 — a drop of
0.012.**

> ⚠️ **What overlap@K means here**: the ground-truth set is the teacher's top-K and the
> retrieved set is the Hamming top-K. Both grow with K, so this metric is **not guaranteed to
> be monotonic in K** (which is why @5 > @10 > @20 in the table). It matches the paper's
> Recall@K convention and is therefore used for comparison with the paper; for comparing
> across K, use a fixed ground-truth set instead.

### 3.4 Why the curve does not fall

Vocabularies are frequency-ordered, so growing them adds lower-frequency words. That raises
the question of whether those new words actually enter the neighbourhood sets, or whether the
curve is merely "adding peripheral words".

For the fixed first 3,000 query words, the table counts how many of the teacher's top-10 come
from the added portion:

| N | New words | Share of top-10 that are new | Queries with at least one new neighbour | Mean top-10 cosine |
|---|---|---|---|---|
| 6,000 | 3,000 | 37.0% | 98.4% | 0.7367 |
| 10,000 | 7,000 | 52.8% | 99.8% | 0.7521 |
| **16,000** | **13,000** | **62.3%** | **100.0%** | **0.7639** |

**At 16,000 words, 62.3% of every query's true neighbours are words that did not exist in the
3,000-word vocabulary, and 100% of queries have at least one new neighbour.** The task
genuinely became harder, and the codes still find the neighbours.

---

## 4. Index: radius and segmentation

### 4.1 The radius must cover positive-pair distances

Positive pairs on a 52-bit code sit roughly 9-10 bits apart. Too small a radius retrieves
nothing:

| Radius | Mean candidates | Recall@10 (3,109 words, pure Hamming) |
|---|---|---|
| 8 | 4.3 | 0.330 |
| 18 | 10.0 | 0.605 |
| 22 | 10.0 | 0.605 |
| Brute-force asymmetric | all | 0.635 |

At radius 22, MIH reaches 95% of the brute-force scan. Full radius sweep on the natural
3,000-word model, measured with the bundles' default parameters:

| Radius | Mean candidates | Recall@10 |
|---|---|---|
| 0 | 1.1 | 0.0074 |
| 4 | 1.7 | 0.0676 |
| 8 | 2.8 | 0.1794 |
| 12 | 4.8 | 0.3586 |
| 16 | 9.5 | 0.6284 |
| 18 | 10.0 | 0.7700 |
| 20 | 10.0 | 0.8535 |
| **22** | **10.0** | **0.8857** |

**In practice**: `--radii` should reach at least 22, or use `--adaptive`, which expands the
radius until the candidate pool saturates.

### 4.2 Choosing the segment count

Multi-index hashing requires at least 2 segments. `segment_count = 1` is not supported: the
segment value would occupy all 52 bits, exceeding the index key width and making the
in-segment Hamming ball impossible to enumerate. Passing 1 produces an explicit configuration
error.

The segment count trades build cost against query speed (13 segments means 4 bits each — small
dictionaries, fast scans; 2 segments means 26 bits each):

| Configuration (radius 22) | QPS |
|---|---|
| `seg=2` | 1304 |
| `seg=4` | 834 |
| `seg=13` | 382 |

Recall is identical across the three: every candidate set is ultimately filtered by the full
52-bit distance.

### 4.3 Capacity ceiling of `abs`

Buckets are `feat`(16) × low 8 bits of `sim`(256) = 4,096 buckets, each holding at most 256
entries (`abs` is 8 bits).

Measured on the 6,109-word bilingual vocabulary:

```text
non-empty buckets   988 / 4,096
mean bucket size    6.18
P99                 20
max bucket          24
```

Linear extrapolation:

| Vocabulary size | Expected max bucket | Headroom (limit 256) | Verdict |
|---|---|---|---|
| 6,109 | 24 | 232 | safe |
| 20,000 | 79 | 177 | safe |
| 50,000 | 196 | 60 | safe |
| 100,000 | 393 | −137 | overflow |

**The safe ceiling is roughly 50,000 words.** The constraint is the width of `abs` together
with the skew of `feat` (nouns take 48%), not the 52-bit `sim` field.

---

## 5. Training cost

Measured locally (RTX 4060 Laptop 8 GB / 31.7 GB RAM):

| N | Training | Post-processing | Total |
|---|---|---|---|
| 3,000 | 10 s | 27 s | 109 s |
| 6,000 | 37 s | ~100 s | 208 s |
| 10,000 | 99 s | ~420 s | 544 s |
| 16,000 | 1248 s | 767 s | 2336 s |

**16,000 words is the practical limit on this machine.** Both training and post-processing
build N×N matrices: 16k² × 4 B ≈ 1 GB (similarity) plus 2 GB (sort indices), which saturates
7.6 GB of the 8 GB of VRAM. Training time therefore jumps from 99 s at 10k to 1248 s at 16k.

Going beyond that requires changing the implementation (mini-batch training with hard-negative
mining), not adding data.

---

## 6. Reproduction

```bash
# 1. Reproduce the paper's benchmark vocabulary (existing 3,109 model, no training)
python scripts/52_eval_pipeline.py \
    --vocab tests/data/vocab_3109.txt \
    --teacher-cache ../Cabinet-BNN/results/bge_large_3109.cache \
    --model ../Cabinet-BNN/results/deep_hash_mvp1_h512_s42.bin \
    --override ../Cabinet-BNN/results/sim_override_mvp1_h512_s42.bin \
    --no-train --no-post --tag repro3109

# 2. Build the natural vocabularies
python scripts/53_build_natural_vocab.py

# 3. Scale curve (train + post-process + evaluate)
python scripts/54_run_scale_curve.py --sizes 6000 10000 16000

# 4. Baselines and controls
python scripts/62_natural_baselines.py          # random / PCA-52 / HSH-64
python scripts/57_new_word_interference.py      # do new words become real neighbours
python scripts/50_capacity_probe.py             # bucket capacity and the `abs` ceiling
python scripts/55_distance_stats.py --help      # positive/negative Hamming distribution

# 5. Package release assets
python scripts/58_package_release.py
```

Artifacts land in `data/eval/`; a per-run summary is appended to `results/eval_runs.jsonl`.

---

## 7. Limitations

1. **The student input is not the paper's exact configuration** — bge-large's PCA-512 stands
   in for bge-small. Absolute numbers carry a systematic offset from the paper's, but the
   comparisons within this page are unaffected.

2. **16,000 words is a memory limit, not a method limit** — nothing here shows the 52-bit code
   failing above 16k; the `abs` capacity extrapolation supports up to about 50,000 words. The
   obstacle is the implementation's O(N²) complexity.

3. **This page covers Chinese only** — the Chinese–English pipeline (6,109 words, bge-m3)
   runs end to end but has no post-processing yet, so it is outside the comparisons here.

4. **overlap@K is not monotonic** — see §3.3. Use a fixed ground-truth set when comparing
   across K.

5. **Vocabularies are frequency-ordered** — growing them adds lower-frequency words. §3.4
   shows those words do become real neighbours, but a randomly sampled vocabulary could
   produce a different curve.
