# HSH-64

HSH-64 是面向轻量化中文词表级语义检索的 64 位可学习语义哈希方案。它在保持单 `u64` 存储、单次 `popcnt` 比较的硬件友好特性的同时，将语义码从 HSH-32 的 20 位扩展到 52 位，显著提升了离散空间的语义表达能力。

> **论文摘要**：语义哈希将高维连续嵌入映射为紧凑二进制码，以极小内存占用实现次线性近似最近邻（ANN）检索。HSH-64 采用 `feat(4) + sim(52) + abs(8)` 结构化编码，其中 52 位语义相似码通过三阶段流程端到端学习：连续预训练 → STE 离散精调 → 召回导向贪心后处理。在线检索集成自适应 MIH 粗排、非对称距离评分与可选的 bge-large 精排。在 3,109 词中文词表上，最佳单模型（h512，1.17 MB）纯 Hamming 空间 Recall@10 达 **0.7404**，轻量化模型（h256，585 KB）达 **0.7382**，四模型集成达 **0.7724**，完整两阶段系统达 **0.8970**。

---

## 项目定位与核心特性

- **结构化 64 位编码**：`feat(4) + sim(52) + abs(8)`，单个 `u64` 存储，硬件 popcount 比较。
- **三阶段端到端训练**：连续预训练 → STE 离散精调 → 召回导向贪心后处理。
- **两阶段检索架构**：轻量编码器（bge-small + 小 MLP）粗排，可选 bge-large 精排。
- **自适应多索引哈希（MIH）**：支持动态半径扩展，兼顾召回率与候选池大小。
- **非对称距离评分**：查询端保留连续投影幅度，缓解符号量化信息损失。
- **多模型 Ensemble**：不同种子/容量的 Deep Hash 模型候选融合，进一步提升召回。
- **纯 CPU 运行**：训练与推理均无需 GPU，适合边缘设备部署。

---

## 编码结构

```text
[ feat: 4 bit ] + [ sim: 52 bit ] + [ abs: 8 bit ] = 64 bit
```

| 字段 | 位数 | 说明 |
|------|------|------|
| feat | 4 bit | 词性/语义类别标识，最多 16 类，可作硬过滤条件 |
| sim  | 52 bit | 语义相似码，主导检索质量；理论容量 2^52 ≈ 4.5×10^15 |
| abs  | 8 bit  | 簇内完美哈希标识，用于区分同 (feat, sim) 桶内的词 |

两个码的相似度通过一次 64 位异或 + popcount 计算：

```rust
let dist = (a ^ b).count_ones();
```

---

## 架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                        离线训练阶段                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ bge-small   │    │ bge-large   │    │  Deep Hash MLP  │  │
│  │ 512-dim     │───→│ 1024-dim    │───→│  dim→H→52 bits  │  │
│  │ 学生嵌入    │    │ 教师嵌入    │    │  STE + 多目标   │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│           │                  │                  │           │
│           ▼                  ▼                  ▼           │
│    embedding.cache    reranker.cache    deep_hash_v3.bin    │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 召回导向贪心后处理 → sim_override.bin                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        在线推理阶段                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 查询词       │───→│ bge-small    │───→│ HSH-64 Encoder│  │
│  └──────────────┘    │ + MLP        │    │ (PCA/DeepHash)│  │
│                      └──────────────┘    └───────┬───────┘  │
│                                                  │           │
│                      ┌───────────────────────────┘           │
│                      ▼                                      │
│              ┌───────────────┐    ┌──────────────────┐      │
│              │ MIH 粗排      │───→│ bge-large 精排   │      │
│              │ 自适应半径    │    │ (可选)           │      │
│              └───────────────┘    └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 预训练模型与 Release 资源

为便于复现与直接使用，本项目在 [GitHub Releases](https://github.com/Sauomore/Cabinet_hsh64/releases) 提供已训练好的模型与缓存，无需从头训练即可运行评测。

### 推荐资源包

| 资源包 | 大小 | 内容 | 适用场景 |
|---|---|---|---|
| `hsh64_pretrained_3109_light.zip` | ~18 MB | 词表 3109 + bge-small 缓存 + bge-large 缓存 + PCA-52 + Deep Hash h256 s2025 + sim_override | 轻量化部署，585 KB 模型 |
| `hsh64_pretrained_3109_best.zip` | ~18 MB | 词表 3109 + bge-small 缓存 + bge-large 缓存 + Deep Hash h512 s42 + sim_override（最佳单模型） | 最佳纯 Hamming 召回 |
| `hsh64_pretrained_3109_full.zip` | ~22 MB | 词表 3109 + bge-small/bge-large 缓存 + PCA + 多个 Deep Hash 模型 + 多组后处理 | 完整复现/Ensemble |

### 资源包内关键文件

```text
hsh64_pretrained_3109_light/
├── vocab_3109.txt                              # 3109 词中文词表
├── embedding_3109.cache                        # bge-small 学生嵌入缓存
├── reranker_3109_large.cache                   # bge-large 教师/精排嵌入缓存
├── pca_52.bin                                  # PCA-52 投影
├── deep_hash_v3_64_3109_h256_s2025.bin         # Deep Hash v3 模型（585 KB）
└── sim_override_3109_h256_s2025_recall_mi100.bin  # 召回导向后处理覆盖码
```

下载并解压到 `tests/data/` 后即可直接运行下方示例命令。

> **注意**：Git 仓库本身不包含 `.bin`/`.cache` 等大型二进制文件（见 [.gitignore](.gitignore)），请通过 Release 页面下载。

---

## 项目结构

```text
hsh64/
├── Cargo.toml              # Rust crate 配置
├── LICENSE                 # MIT OR Apache-2.0
├── README.md               # 本文件
├── src/                    # Rust 核心库
│   ├── lib.rs              # 模块导出
│   ├── hsh64.rs            # HSHCode64 / Hamming 距离
│   ├── encoder.rs          # Encoder / EncoderConfig
│   ├── deep_hash.rs        # DeepHashEncoder64 / DeepHashProjection
│   ├── pca.rs              # PcaProjection
│   ├── mih_index.rs        # MihSemanticIndex 自适应搜索
│   ├── embedding.rs        # Embedding trait / FileCachedEmbedding
│   ├── perfect_hash.rs     # 簇内完美哈希
│   ├── pos_map.rs          # 词性映射
│   └── error.rs            # 错误类型
├── examples/               # Rust 可运行示例
│   ├── benchmark_pure_hsh64.rs
│   └── benchmark_ensemble_hsh64.rs
├── scripts/                # Python 训练与评测脚本
│   ├── generate_embeddings.py
│   ├── generate_reranker_embeddings_64.py
│   ├── train_pca_64.py
│   ├── train_deep_hash_v3_64.py
│   ├── post_optimize_codes_64.py
│   ├── post_optimize_codes_recall_64.py
│   ├── benchmark_pure_hsh64.py
│   ├── ensemble_eval.py
│   └── ...
├── tests/                  # Rust 集成测试与数据
│   ├── end_to_end.rs
│   └── data/
├── docs/                   # 设计文档与实验报告
│   ├── explained.md        # HSH-64 通俗设计文档
│   ├── explained.pdf       # 设计文档 PDF
│   ├── experiment_data_export.md  # 实验数据汇总
│   └── hsh64_principle.html
└── paper/                  # 论文源码与 PDF
    ├── main.tex            # 英文论文 LaTeX 源码
    ├── main.pdf            # 英文论文 PDF
    ├── main_chinese.tex    # 中文论文 LaTeX 源码
    ├── main_chinese.pdf    # 中文论文 PDF
    └── references.bib      # 参考文献
```

---

## 快速开始

### 环境要求

- Rust >= 1.75
- Python >= 3.9（训练脚本使用 PyTorch）
- `sentence-transformers`、`numpy`、`torch`、`scikit-learn`

### Rust 编译与测试

```bash
cd hsh64
cargo test --release
cargo run --example benchmark_pure_hsh64 -- --help
```

### Python 训练依赖安装

```bash
python -m pip install sentence-transformers numpy torch scikit-learn
```

---

## 完整使用流程

### 方式一：使用 Release 预训练资源（推荐）

1. 从 [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases) 下载 `hsh64_pretrained_3109_light.zip`。
2. 解压到 `tests/data/`。
3. 直接运行评测：

```bash
# 纯 HSH-64 暴力扫描
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109

# MIH 自适应搜索 + bge-large 精排
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --reranker tests/data/reranker_3109_large.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109 --adaptive --coarse-factor 10 --segment-counts 13
```

### 方式二：从头训练

#### 1. 准备词表

词表文件每行一个词：

```text
北京
上海
苹果
香蕉
...
```

#### 2. 生成 bge-small 学生嵌入缓存

```bash
python scripts/generate_embeddings.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/embedding.cache
```

#### 3. 生成 bge-large 教师/精排嵌入缓存

```bash
python scripts/generate_reranker_embeddings_64.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/reranker_embedding.cache
```

#### 4. 训练 PCA-52 基线

```bash
python scripts/train_pca_64.py \
    --embedding tests/data/embedding.cache \
    --output tests/data/pca_52.bin
```

#### 5. 训练 Deep Hash v3 模型

```bash
python scripts/train_deep_hash_v3_64.py \
    --student tests/data/embedding.cache \
    --teacher tests/data/reranker_embedding.cache \
    --hidden-dim 256 \
    --seed 2025 \
    -o tests/data/deep_hash_v3_64_3109_h256_s2025.bin
```

#### 6. 召回导向后处理优化

```bash
python scripts/post_optimize_codes_recall_64.py \
    --embedding tests/data/embedding.cache \
    --reranker tests/data/reranker_embedding.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --pos-k 10 \
    --neg-weight 1.0 \
    --max-iters 10 \
    -o tests/data/sim_override_3109_h256_s2025_recall_mi100.bin
```

#### 7. 纯 HSH-64 暴力扫描评测

```bash
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding.cache \
    --pca tests/data/pca_52.bin \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109
```

#### 8. MIH 自适应搜索 + 精排评测

```bash
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding.cache \
    --reranker tests/data/reranker_embedding.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --sim-override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --top-k 10 --queries 3109 --adaptive --coarse-factor 10 \
    --segment-counts 13 --radii 0,2,4,6,8,10,12,14,16,18,20,22
```

#### 9. Ensemble 多模型评测

```bash
cargo run --release --example benchmark_ensemble_hsh64 -- \
    --embedding tests/data/embedding.cache \
    --reranker tests/data/reranker_embedding.cache \
    --top-k 10 --queries 3109 --radius 22 \
    --model tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin \
    --model tests/data/deep_hash_v3_64_3109_h512_s42.bin \
    --override tests/data/sim_override_3109_h512_s42_recall.bin
```

---

## API 说明

### Rust API

#### HSHCode64

```rust
use hsh64::{HSHCode64, hamming_distance64};

let code = HSHCode64::new(feat, sim, abs);
let dist = hamming_distance64(a.sim(), b.sim());
```

#### Encoder

```rust
use hsh64::{Encoder, EncoderConfig};

let config = EncoderConfig {
    embed_dim: 512,
    embedding_cache_path: Some("tests/data/embedding_3109.cache".into()),
    pca_path: Some("tests/data/pca_52.bin".into()),
    deep_hash_path: Some("tests/data/deep_hash_v3_64_3109_h256_s2025.bin".into()),
    sim_override_path: Some("tests/data/sim_override_3109_h256_s2025_recall_mi100.bin".into()),
    ..Default::default()
};
let encoder = Encoder::with_config(config)?;
let code = encoder.encode_word_with_pos("北京", "n");
```

#### MihSemanticIndex

```rust
use hsh64::MihSemanticIndex;

let index = MihSemanticIndex::build_with_embedding(
    encoder,
    embedding,
    13,      // segment_count
    false,   // verbose
)?;

// 自适应 Hamming 搜索
let (radius, results) = index.search_adaptive(query_word, top_k, coarse_factor, max_radius)?;

// 自适应非对称距离搜索
let (radius, results) = index.search_adaptive_asymmetric(query_word, top_k, coarse_factor, max_radius)?;
```

#### DeepHashEncoder64 / PcaProjection

```rust
use hsh64::{DeepHashEncoder64, DeepHashProjection, PcaProjection};

let dh = DeepHashEncoder64::from_file("deep_hash_v3_64_3109_h256_s2025.bin")?;
let pca = PcaProjection::from_file("pca_52.bin")?;
```

### Python 脚本说明

| 脚本 | 作用 |
|------|------|
| `generate_embeddings.py` | 用 bge-small 生成 embedding 缓存 |
| `generate_reranker_embeddings_64.py` | 用 bge-large 生成精排/教师缓存 |
| `train_pca_64.py` | 训练 PCA-52 投影并导出二进制 |
| `train_deep_hash_v3_64.py` | 训练 Deep Hash MLP v3 |
| `post_optimize_codes_64.py` | 基础贪心比特翻转后处理 |
| `post_optimize_codes_recall_64.py` | 召回导向贪心后处理 |
| `benchmark_pure_hsh64.py` | Python 端纯 HSH 基准 |
| `ensemble_eval.py` | Ensemble 策略评测 |

---

## Benchmark 结果

在 3109 词中文词表基准上，真实 top-K 由 bge-large 余弦相似度定义。

| 方案 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | 备注 |
|------|----------|----------|-----------|-----------|------|
| PCA-52 | 0.45 | 0.62 | 0.68 | 0.74 | 线性基线 |
| Deep Hash v3 (h256) | 0.51 | 0.66 | 0.72 | 0.78 | 单模型，585 KB |
| Deep Hash v3 + Post | 0.53 | 0.68 | 0.74 | 0.80 | 召回导向后处理 |
| Ensemble (4 models) | 0.55 | 0.71 | 0.77 | 0.83 | 候选融合 |
| MIH 粗排 + bge-large 精排 | 0.78 | 0.86 | 0.90 | 0.93 | 两阶段系统 |

> 注：以上为论文中的代表性结果，具体数值可能随随机种子、后处理超参略有波动。

### 详细实验结果

更完整的模型容量分析、后处理网格搜索、集成策略与效率基准见 [docs/experiment_data_export.md](docs/experiment_data_export.md)。其中关键发现：

- **最佳单模型**：`deep_hash_v3_64_3109_h512_s42.bin` + `sim_override_3109_h512_s42_recall.bin`，Recall@10 = **0.7404**
- **轻量化单模型**：`deep_hash_v3_64_3109_h256_s42.bin` + `sim_override_3109_h256_s42_recall.bin`，Recall@10 = **0.7382**
- **四模型集成**：纯 Hamming 空间 Recall@10 = **0.7724**
- **两阶段系统**：MIH 粗排 + bge-large 精排，四模型集成 Recall@10 = **0.8970**

### 关键指标

- **在线编码器大小**：585 KB（h256）/ 1.17 MB（h512）
- **单次 Hamming 比较**：1 条 x86 `popcnt`
- **MIH 平均候选池**：数十到数百个
- **MIH 平均查询半径**：约 20（自适应停止）

---

## 开发/测试命令

```bash
# 运行单元测试与集成测试
cargo test --release

# 纯 HSH-64 暴力扫描（PCA）
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --pca tests/data/pca_52.bin \
    --top-k 10 --queries 3109

# 纯 HSH-64 暴力扫描（Deep Hash）
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --top-k 10 --queries 3109

# MIH 自适应搜索
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --deep-hash tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --top-k 10 --queries 3109 --adaptive

# Ensemble 评测
cargo run --release --example benchmark_ensemble_hsh64 -- \
    --embedding tests/data/embedding_3109.cache \
    --reranker tests/data/reranker_3109_large.cache \
    --top-k 10 --queries 3109 --radius 22 \
    --model tests/data/deep_hash_v3_64_3109_h256_s2025.bin \
    --override tests/data/sim_override_3109_h256_s2025_recall_mi100.bin
```

---

## 论文与设计文档

- [docs/explained.md](docs/explained.md)：通俗版 HSH-64 设计文档，涵盖编码结构、三阶段训练、MIH 与非对称距离。
- [docs/experiment_data_export.md](docs/experiment_data_export.md)：完整实验数据汇总，包含所有模型文件清单与网格搜索结果。
- [paper/main.tex](paper/main.tex)：英文论文 LaTeX 源码
- [paper/main.pdf](paper/main.pdf)：英文论文 PDF
- [paper/main_chinese.tex](paper/main_chinese.tex)：中文论文 LaTeX 源码
- [paper/main_chinese.pdf](paper/main_chinese.pdf)：中文论文 PDF
- [paper/references.bib](paper/references.bib)：参考文献

---

## License

MIT OR Apache-2.0
