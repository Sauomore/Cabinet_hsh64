# HSH-64

[![DOI](https://zenodo.org/badge/1307857867.svg)](https://doi.org/10.5281/zenodo.21721838)
[![GitHub release](https://img.shields.io/github/v/release/Sauomore/Cabinet_hsh64?include_prereleases&style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases)

**中文** ｜ [English](README.en.md)

> 面向轻量化**中英双语**词表级语义检索的 64 位可学习语义哈希方案。
> 单个 `u64` 存储，单次 `popcnt` 比较，纯 CPU 运行。

---

## 这是什么

HSH-64 把语义哈希码从 HSH-32 的 20 位扩展到 **52 位**，在保持硬件友好特性的同时
显著提升离散空间的语义表达能力。

```text
[ feat: 4 bit ] + [ sim: 52 bit ] + [ abs: 8 bit ] = 64 bit
```

| 字段 | 位数 | 说明 |
|---|---|---|
| `feat` | 4 | 词性类别，16 个槽位，可作硬过滤条件 |
| `sim` | 52 | 语义相似码，主导检索质量；容量 2⁵² ≈ 4.5×10¹⁵ |
| `abs` | 8 | 簇内唯一标识，区分同一 `(feat, sim)` 桶内的词 |

两个码的距离：**一次 64 位异或 + popcount**。

```rust
let dist = (a ^ b).count_ones();
```

### 核心特性

- **结构化 64 位编码** —— 单 `u64` 存储，硬件 popcount 比较
- **中英双语** —— `feat` 的 16 槽由两种语言的词性标签共用，支持跨语言检索
- **三阶段端到端训练** —— 连续预训练 → STE 离散精调 → 召回导向贪心后处理
- **两阶段检索** —— 轻量编码器粗排，可选 bge-large 精排
- **自适应 MIH** —— 动态半径扩展，兼顾召回率与候选池大小
- **非对称距离评分** —— 查询端保留连续投影幅度，缓解符号量化损失
- **多模型 Ensemble** —— 候选融合进一步提升召回
- **纯 CPU** —— 训练与推理均不需 GPU

### 基准结果（3,109 词中文词表，Recall@10）

> 数据来自论文表 `tab:main-results-cn`。真实 top-K 由 bge-large 余弦相似度定义。

| 方法 | Recall@10 | 在线模型 |
|---|---|---|
| PCA-52（纯 HSH） | 0.5318 | PCA 投影矩阵 |
| DH h128_s42（**无**后处理） | 0.4228 | 421 KB MLP |
| DH h256_s2025（**无**后处理） | 0.4315 | 585 KB MLP |
| DH h512_s42（**无**后处理） | 0.4265 | 1.17 MB MLP |
| DH h1024_s2024（**无**后处理） | 0.4315 | 2.33 MB MLP |
| **DH h256_s42 + 后处理** | **0.7382** | 585 KB MLP |
| **DH h512_s42 + 后处理** | **0.7404** | 1.17 MB MLP |
| 4 模型集成 | **0.7724** | ≈4 MB |
| DH + MIH + 精排 | 0.8641 | 1.17 MB MLP + bge-large |
| 集成 + MIH + 精排 | **0.8970** | ≈4 MB + bge-large |

**⚠️ 一个反直觉但重要的结果：无后处理时 Deep Hash 反而不如 PCA-52**
（0.42–0.43 vs 0.53），尽管它有更强的非线性变换能力。
**第三阶段的后处理（召回导向贪心比特翻转）不是可选优化，而是这套方案能用的前提。**
详见 [docs/explained.md §8.3](docs/explained.md)。

---

## 中英双语

`feat` 的 4 位是**唯一语言相关**的部分。中文（jieba 标签）与英文（Penn Treebank
标签）映射到同一组 16 个槽位，因此两种语言的词可以共用一个码本。

```text
0x0 名词    n/nr/ns/…        NN/NNS/NNP/…        0x8  数词  m/mq        CD/LS
0x1 动词    v/vd/vn/…        VB/VBD/VBG/…        0x9  量词  q/qv/qt     （中文特有）
0x2 形容词  a/ad/an/ag/b     JJ/JJR/JJS          0xA  时间词 t/tg        （中文特有）
0x3 副词    d/df/dg          RB/RBR/RBS/WRB      0xB  字符串 x/xx/…/eng
0x4 代词·限定词 r/rr/rz/…    PRP/PRP$/DT/…       0xC  标点  w 系列      . , : `` ''
0x5 介词·方位  p/f/fg/s      IN/TO/RP            0xD  其他  i           UH/POS
0x6 连词    c/cc             CC                  0xE  常用词（运行时判定）
0x7 助词·情态 u/ud/…/z/zg    MD                  0xF  兜底
```

**词表**：6,109 词 = 中文 3,095 + 英文 3,002 + 混合 12（`T恤`、`U盘`）

```bash
# 构建混合词表（英文侧从 ECDICT 按词频筛选）
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000
python scripts/49_tag_vocab.py          # 双语词性标注
```

> ⚠️ **已知局限（实测）**：词表里是孤立单词，没有句子上下文，而两种标注器都按句子
> 训练，因此退化明显 —— 中文 3,109 词产出 **29 种**标签（`同意` → 副词），英文约
> **36% 标错**（`tree` → 形容词）。结果 `feat` 分布倾斜：0x0 占 47.9%、0x2 占 38.4%，
> 熵不足 1.5 bit。
>
> 这在设计上可以接受 —— **语义区分由 52 位的 `sim` 承担**（实测唯一码率 99.2%），
> `feat` 只需作为粗粒度的分桶线索。但下游不应给它过多权重。
> 完整数据见 **[docs/bilingual.md](docs/bilingual.md)**。

> **双语词表上的实测进度**：裸 Deep Hash（无后处理）Recall@10 = **0.2750**。
> 这与上表「无后处理 0.43」的差距来自词表规模（6,109 vs 3,109）与语言混合，
> 尚未做后处理，因此**不可与上表直接比较**。待办见下方「当前状态」。

---

## 架构

```text
┌──────────────────────── 离线训练 ────────────────────────┐
│                                                          │
│   bge-small          bge-large        Deep Hash MLP      │
│   512-dim     ───→   1024-dim  ───→   dim→H→52 bits      │
│   学生嵌入           教师嵌入          STE + 多目标       │
│       │                  │                  │            │
│       ▼                  ▼                  ▼            │
│  embedding.cache   reranker.cache   deep_hash_v3.bin     │
│                                                          │
│         召回导向贪心后处理 → sim_override.bin             │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────── 在线推理 ────────────────────────┐
│                                                          │
│   查询词 ──→ bge/m3 + MLP ──→ HSH-64 Encoder             │
│                                      │                   │
│                                      ▼                   │
│                            MIH 粗排（自适应半径）          │
│                                      │                   │
│                                      ▼                   │
│                            bge-large 精排（可选）          │
└──────────────────────────────────────────────────────────┘
```
## 快速开始

### 环境要求

- Rust 1.72+
- Python 3.8+（训练脚本需要 `torch`、`numpy`、`sentence-transformers`）
- 无需 GPU

```bash
cargo build --release
cargo test
```

### 方式一：用 Release 预训练资源（推荐）

从 [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases) 下载资源包，
解压到 `tests/data/`，然后：

```bash
# 纯 HSH-64 暴力扫描
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109
```

| 资源包 | 大小 | 内容 |
|---|---|---|
| `hsh64_pretrained_3109_light.zip` | ~18 MB | 词表 + 两套嵌入缓存 + PCA-52 + Deep Hash h256 + 后处理 |
| `hsh64_pretrained_3109_best.zip` | ~18 MB | 同上，Deep Hash h512 s42（最佳单模型） |
| `hsh64_pretrained_3109_full.zip` | ~22 MB | 多个模型 + 多组后处理（复现 Ensemble） |

> Git 仓库不含 `.bin` / `.cache` 等大文件（见 [.gitignore](.gitignore)），请从 Release 下载。

### 方式二：从头训练

```bash
# 1. 生成学生嵌入（bge-small）
python scripts/generate_embeddings.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/embedding.cache

# 2. 生成教师/精排嵌入（bge-large）
python scripts/generate_reranker_embeddings_64.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/reranker_embedding.cache

# 3. PCA-52 基线
python scripts/train_pca_64.py \
    --embedding-cache tests/data/embedding.cache \
    -o tests/data/pca_52.bin

# 4. Deep Hash v3（阶段 1+2）
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --hidden-dim 256 --seed 2025 \
    -o tests/data/deep_hash_v3_64_3109_h256_s2025.bin

# 5. 召回导向后处理（阶段 3）—— 这一步是 0.72 → 0.74 的关键
python scripts/post_optimize_codes_recall_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --model-type deep_hash \
    --model-path tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --pos-k 10 --k 10 --neg-weight 1.0 --max-iters 10 \
    -o tests/data/sim_override_3109_h256_s2025_recall_mi100.bin

# 6. 评测
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding.cache \
    --pca tests/data/pca_52.bin \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109
```

> **注意**：脚本参数请以 `python scripts/<脚本> --help` 为准。
> 本文档的命令已逐一核对过。

---

## 开发命令

```bash
cargo test                      # 单元 + 集成 + 对拍测试
cargo run --release --example benchmark_pure_hsh64 -- --help
cargo run --release --example benchmark_ensemble_hsh64 -- --help
```

### 测试构成

| 测试 | 内容 |
|---|---|
| `src/**` 单元测试 | 26 个：编码/解码、PCA、Deep Hash、MIH、词性映射 |
| `tests/end_to_end.rs` | 端到端（需要 `tests/data/` 下的缓存文件） |
| `tests/feat_parity.rs` | **Python ↔ Rust 词性映射对拍**（4 个） |

**对拍测试的意义**：`(词性标签 → feat)` 的映射表存在于两处 ——
Python 侧建码本、Rust 侧查码本。两边漂移会导致「Python 建的码本 Rust 读错」，
而这类 bug 单看任何一边都正常。

```bash
python scripts/46_export_feat_table.py    # 导出 Python 侧映射表
cargo test --test feat_parity             # Rust 侧逐项断言
```

---

## 项目结构

```text
hsh64/
├── src/                        Rust 核心库
│   ├── hsh64.rs                HSHCode64 / Hamming 距离
│   ├── encoder.rs              Encoder / EncoderConfig（含词性表加载）
│   ├── deep_hash.rs            DeepHashEncoder64 / DeepHashProjection
│   ├── pca.rs                  PcaProjection
│   ├── mih_index.rs            MihSemanticIndex 自适应搜索
│   ├── embedding.rs            Embedding trait / FileCachedEmbedding
│   ├── perfect_hash.rs         簇内完美哈希
│   ├── pos_map.rs              词性映射（中英双语，16 槽共用）
│   └── error.rs                错误类型
├── examples/
│   ├── benchmark_pure_hsh64.rs
│   └── benchmark_ensemble_hsh64.rs
├── scripts/                    Python 训练与评测
│   ├── generate_embeddings.py
│   ├── generate_reranker_embeddings_64.py
│   ├── train_pca_64.py
│   ├── train_deep_hash_v3_64.py
│   ├── post_optimize_codes_64.py
│   ├── post_optimize_codes_recall_64.py
│   ├── bilingual_pos.py        双语词性标注（与 pos_map.rs 对应）
│   ├── 46_export_feat_table.py 导出映射表供对拍
│   ├── 47_detect_lang.py       词表语言构成检测
│   ├── 48_build_vocab.py       中英混合词表构建
│   └── 49_tag_vocab.py         混合词表标注
├── tests/
│   ├── end_to_end.rs
│   ├── feat_parity.rs          Python↔Rust 对拍
│   └── data/
├── data/                       双语工作数据（生成物，多不入库）
├── docs/
│   ├── bilingual.md            中英双语扩展
│   ├── explained.md            通俗版设计文档
│   ├── experiment_data_export.md
│   └── hsh64_principle.html
└── paper/                      论文源码与 PDF（中英）
```

---

## API

### Rust

```rust
use hsh64::{Encoder, EncoderConfig, HSHCode64, hamming_distance64};

// 基础
let code = HSHCode64::new(feat, sim, abs);
let dist = hamming_distance64(a.sim(), b.sim());

// 编码器
let mut config = EncoderConfig {
    embed_dim: 512,
    embedding_cache_path: Some("tests/data/embedding_3109.cache".into()),
    pca_path: Some("tests/data/pca_52.bin".into()),
    deep_hash_path: Some("tests/data/deep_hash_v3_64_3109_h256_s2025.bin".into()),
    sim_override_path: Some("tests/data/sim_override_3109_h256_s2025_recall_mi100.bin".into()),
    ..Default::default()
};
// 双语：载入词性表（不载入则全部按 "n" 处理，与旧行为一致）
config.load_pos_tags("data/mixed_tags.tsv".as_ref())?;

let encoder = Encoder::with_config(config)?;

// 推荐入口：自动查词性并按语种分派
let code = encoder.encode_word_auto("北京");

// 显式指定
let code = encoder.encode_word_with_pos_lang("apple", "NN", PosLang::En);
```

```rust
use hsh64::MihSemanticIndex;

let index = MihSemanticIndex::build_with_embedding(encoder, embedding, 13, false)?;

let (radius, results) = index.search_adaptive(query, top_k, coarse_factor, max_radius)?;
let (radius, results) =
    index.search_adaptive_asymmetric(query, top_k, coarse_factor, max_radius)?;
```

### Python 脚本

| 脚本 | 作用 |
|---|---|
| `generate_embeddings.py` | 生成学生嵌入缓存 |
| `generate_reranker_embeddings_64.py` | 生成教师/精排嵌入缓存 |
| `train_pca_64.py` | 训练 PCA-52 |
| `train_deep_hash_v3_64.py` | 训练 Deep Hash v3 |
| `post_optimize_codes_64.py` | 基础贪心比特翻转 |
| `post_optimize_codes_recall_64.py` | 召回导向后处理 |
| `bilingual_pos.py` | 中英双语词性标注 |
| `48_build_vocab.py` / `49_tag_vocab.py` | 混合词表构建与标注 |
| `benchmark_pure_hsh64.py` / `ensemble_eval.py` | Python 端基准与 Ensemble |

---

## 当前状态

### 已完成

| 环节 | 状态 |
|---|---|
| 编码结构 `feat(4)+sim(52)+abs(8)` | ✅ |
| 三阶段训练流程 | ✅ |
| 中英双语 `feat` 映射 | ✅ |
| 混合词表 6,109 词 | ✅ |
| Python↔Rust 对拍测试 | ✅（含变异验证） |
| 单元 + 集成测试 | ✅ 31 个 |

### ⚠️ 待解决

```text
① MIH 索引召回异常（优先）
   seg=2/4/13 结果完全相同，候选数 2.3 vs 暴力扫描 6109
   疑似索引正确性问题

② 缺后处理时 Recall 偏低
   裸 DeepHash 0.2750 vs 加后处理 0.728
   不是回归，是少跑了一步

③ build_seed_table 从未被调用
   seed 恒为 0，完美哈希搜索从未执行
   不影响 MIH 召回，但 abs 失去唯一性保证
```

完整记录见 **[docs/bilingual.md](docs/bilingual.md) 第 9 节**。

---

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/bilingual.md](docs/bilingual.md)** | 中英双语扩展：槽位分配、词表构建、跨语言验证、已知问题 |
| **[docs/explained.md](docs/explained.md)** | 通俗版设计文档：编码结构、三阶段训练、MIH、非对称距离 |
| [docs/experiment_data_export.md](docs/experiment_data_export.md) | 完整实验数据汇总 |
| [docs/hsh64_principle.html](docs/hsh64_principle.html) | 原理图解 |
| [paper/](paper/) | 论文 LaTeX 源码与 PDF（中英双版本） |

---

## 引用

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
