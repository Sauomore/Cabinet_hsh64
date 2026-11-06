# HSH-64

[![DOI](https://zenodo.org/badge/1307857867.svg)](https://doi.org/10.5281/zenodo.21721838)
[![GitHub release](https://img.shields.io/github/v/release/Sauomore/Cabinet_hsh64?include_prereleases&style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases)
[![Version](https://img.shields.io/badge/version-2.0-blue?style=flat-square)](https://github.com/Sauomore/Cabinet_hsh64/releases/tag/v2.0)

**中文** ｜ [English](README.en.md)

> 给每个词一个 **64 位整数**作为「语义指纹」。
> 意思相近的词，指纹只差几位；比较两个词有多像，只需要一次异或加一次 popcount。
> 纯 CPU，一个核心每秒能比几千万对。

**当前版本 v2.0** —— 新增自然词表模型（3k / 6k / 10k / 16k）与规模评估，修复索引两处缺陷。
详见[发布页](https://github.com/Sauomore/Cabinet_hsh64/releases/tag/v2.0)。

---

## 支持这个项目

如果你用到了我的项目，**可以为我点一个 Star** —— 这对我真的很重要，非常感谢。

如果你在使用过程中发现了 bug，或者有改进建议，欢迎[提交 Issue](https://github.com/Sauomore/Cabinet_hsh64/issues)。
即使只是文档里一句话看不懂，也可以提因为那通常意味着文档确实该改。

---

## 目的

假设你要做一个「找相似词」的功能。最直接的办法是：

1. 用一个强大的模型（比如 bge-large）把每个词变成一个 1024 维的浮点向量
2. 两个词的相似度 = 两个向量的余弦相似度
3. 找最近邻 = 把所有词的向量都算一遍，取分数最高的几个

这个方法**确实准确，但是代价较大**：1024 个 float32 是 4 KB，一百万个词就是 4 GB；比较一次要算 1024 次乘加。

HSH-64 干的事情是：**训练一个小网络，把那个 1024 维向量压成一个 64 位整数**，
并要求「原来相近的两个词，压完之后整数也只差几位」。

于是：

| | 存一个词 | 比较一次 | 一百万个词占多大 |
|---|---|---|---|
| 原始向量（1024 维 float32） | 4 KB | 1024 次乘加 | 4 GB |
| HSH-64 码 | **8 字节** | **1 次异或 + 1 次 popcount** | **8 MB** |

代价是精度会掉一些（掉多少见[实测结果](#实测结果)）。换来的是**存储小 500 倍、比较快两三个数量级，而且不需要 GPU**。

### 一个类比：邮政编码

现实里的邮政编码是分层的：前两位是省，中间两位是市，后两位是区。
想知道两个地址是不是在同一个区，不用看详细门牌，比一下编码就够了。

HSH-64 的 64 位也是分层的：

```text
 63    60 59                          8 7         0
┌────────┬───────────────────────────┬───────────┐
│  feat  │           sim             │    abs    │
│  4 bit │          52 bit           │   8 bit   │
└────────┴───────────────────────────┴───────────┘
  词性类别         语义指纹              桶内编号
  「哪个省」      「具体在哪」          「同一格的第几号」
```

- **`feat`（4 位）**：粗粒度类别，像「省」。名词 / 动词 / 形容词…… 共 16 个槽。
  它的作用是把不同词性的词分开，避免它们占用同一个桶。
- **`sim`（52 位）**：语义指纹，像「省以下的详细地址」。**检索质量几乎完全由它决定。**
  52 位能表示 2⁵² ≈ 4.5×10¹⁵ 个不同值。
- **`abs`（8 位）**：桶内编号。因为 `sim` 只有 52 位而桶只有 4,096 个，
  同一个 `(feat, sim 低 8 位)` 桶里可能挤进多个词，`abs` 用来把它们区分开（每桶最多 256 个）。

比较两个码有多像，就是这样一句 Rust：

```rust
let distance = (a.sim() ^ b.sim()).count_ones();   // 0..=52，越小越像
```

`count_ones()` 编译成一条 `POPCNT` 指令。这是它能在纯 CPU 上跑得飞快的原因。

---

## 64 位的选定原因

这不是随便挑的数字，三个约束把它夹在了这个位置：

**① 要能塞进一个寄存器。** 64 位 = 一个 `u64` = 一个 CPU 寄存器。一个核心可以同时对多个码做异或和 popcount，不需要 SIMD、不需要拆包。

**② 要够大，才能区分足够多的词。** 代码结构决定了它的「保证唯一」容量：

```text
feat(16) × sim 低 8 位(256) × abs(256) = 1,048,576 个可区分码
```

实测的容量边界（见 [docs/scale_evaluation.md](docs/scale_evaluation.md) §4.3）：

| 词表规模 | 最大桶大小 | 距上限 256 的余量 |
|---|---|---|
| 6,109 | 24 | 232 |
| 20,000 | 79 | 177 |
| 50,000 | 196 | 60 |
| 100,000 | 393 | **溢出** |

**安全上限约 5 万词。** 卡住的不是 52 位的 `sim`，而是 `abs` 只有 8 位、以及 `feat` 分布倾斜（名词槽占了 47.9%）。

**③ 要够小，索引才能全放内存。** 100 万个词的码只占 8 MB，可以整个塞进 CPU 缓存附近，随机访问也不至于处处打内存。

---

## 它是怎么训出来的

三阶段，全部离线，产物是一个 1.17 MB 的 MLP。

```text
阶段 1  连续预训练      小网络学「让码保持教师模型的邻居结构」
          │             学生输入 512 维 → 512 隐层 → 52 维连续值
          ▼
阶段 2  STE 离散精调    把连续值符号量化成 52 个 bit，
          │             用直通估计器（STE）让梯度穿过量化
          ▼
阶段 3  召回导向后处理  直接对着 Recall@10 做贪心比特翻转：
                        「翻这一位，能多找回几个真邻居？」
                        翻完导出成 sim_override.bin
```

> **第三阶段不是可选优化。** 实测：跳过它时，裸码召回 0.4988；
> 加完之后 0.7426。在自然词表上更极端 —— 裸码 0.2463 **甚至低于零学习的 PCA 基线 0.3757**，
> 而后处理把它拉到 0.6082。
>
> 换句话说，**「HSH-64 有效」严格讲是「HSH-64 + 贪心比特翻转」有效**。
> 这一点在 [docs/explained.md §8.3](docs/explained.md) 里有详细推导。

检索时是两阶段：

```text
查询词 ──→ 小网络 ──→ 64 位码 ──→ MIH 粗排（拿回几十个候选）
                                        │
                                        ▼
                              bge-large 精排（可选）──→ 最终 top-10
```

粗排只用码，快；精排才动用大模型，准。精排可以关掉，那就退化成纯哈希检索。

---

## 实测结果

### 在论文的基准词表上（3,109 词，Recall@10）

「真实答案」由 bge-large 的余弦相似度定义 —— 即：教师模型认为哪个词最像，答案就是哪个。

| 方法 | Recall@10 | 在线需要什么 |
|---|---|---|
| PCA-52（零学习基线） | 0.5318 | 一个 PCA 矩阵 |
| Deep Hash h512（**无**后处理） | 0.4265 | 1.17 MB MLP |
| **Deep Hash h512 + 后处理** | **0.7404** | 1.17 MB MLP |
| 4 模型集成 + 后处理 | **0.7724** | ≈4 MB |
| **Deep Hash + MIH + 精排** | **0.8641** | 1.17 MB MLP + bge-large |
| 集成 + MIH + 精排 | **0.8970** | ≈4 MB + bge-large |

本仓库独立复现的结果（学生输入用 bge-large 的 PCA-512 代替论文的 bge-small）：

```text
裸码 0.4988   + 后处理 0.7426   两阶段 0.8890
```

后处理与两阶段两个数与论文对齐；裸码偏高，因为学生输入更强。

### ⚠️ 这张表的适用边界

后续评估发现：**上表的 3,109 词表里，65% 是人造合成词。**原因是当时暂时没找到时候的语料，当然它现在已经不重要了。

做法是「10 个修饰词 × 约 200 个基础词」：`大U盘`、`大一月`、`大三月`、`红U盘`、`快丝瓜`……
自然基础词其实只有约 1,114 个。

这类词表的近邻结构**比自然文本清晰 2.18 倍**（top-10 边界间隙归一化后 0.0196 vs 0.0090），
所以上面的数字反映的是「结构较简单的任务」上的能力。

在后续找到的**自然中文词表**上重做（词频表取词，与上面的词表只有 10.2% 重叠）：

| 词表 | 规模 | 裸码 | + 后处理 | 相对 PCA-52 基线 |
|---|---|---|---|---|
| 合成（上表） | 3,109 | 0.499 | **0.7426** | +0.131 |
| 自然 | 3,000 | 0.246 | **0.6082** | **+0.233（1.62×）** |
| 自然 | 6,000 | 0.245 | 0.5990 | — |
| 自然 | 10,000 | 0.247 | 0.5936 | — |
| 自然 | 16,000 | 0.252 | **0.5965** | **+0.284（1.91×）** |

**两个值得注意的现象：**

1. **规模几乎不影响召回。** 词表扩大 5.3 倍，从 0.6082 到 0.5965 —— 只掉 0.012。
   而且这不是「词太少所以没测到难度」：在 16,000 词时，**每个查询的真实邻居里有 62.3%
   是 3,000 词表里根本不存在的词**，100% 的查询都有新邻居。任务确实变难了，码撑住了。

2. **相对基线的优势随规模增长。** PCA-52 从 0.376 掉到 0.313（−0.063），
   HSH-64 只从 0.608 到 0.597（−0.012）。**学出来的码比无学习的码更抗规模。**

完整数据、机制分析、索引参数选择见 **[docs/scale_evaluation.md](docs/scale_evaluation.md)**。

---

## 它不擅长什么

这类项目最常见的坑是把「能做到」写成「什么都能做」。下面是实测出来的边界：

**① 它只做「词表内检索」，不做句子理解。**
输入是一个词，输出是词表里最像的十个词。它不理解句子、不做问答、不生成文本。
你要拿它做文章查重或 RAG，得自己在外面套一层分句和拼接。

**② 词表是写死的。**
模型只对训练时见过的词负责。出现新词（网络流行语、专业术语）需要重新训练。
实测词表从 3,000 扩到 16,000 是可以的（召回不降），但在扩到 5 万词左右会撞上 `abs` 的容量上限，**这个问题将会在未来的版本尝试解决**。

**③ 召回高度依赖后处理，而后处理不可扩展。**
第三阶段是 O(N²) 的贪心搜索。3,000 词要 27 秒，16,000 词要 12.8 分钟并且吃满 8 GB 显存。
超过 16k 需要改实现（mini-batch + 难负样本挖掘），不是加数据就行。

**④ `feat` 那 4 位比看起来弱。**
它来自词性标注，而词表里是**孤立单词、没有上下文**，两种标注器都会明显退化：
中文 3,109 词产出 29 种标签（「同意」被标成副词），英文约 36% 标错（`tree` 被标成形容词）。
结果 `feat` 分布极倾斜（名词 47.9% + 形容词 38.4% = 86%），实际熵不到 1.5 bit。

这在设计上可以接受 —— **语义区分由 52 位的 `sim` 承担**（唯一码率实测 99.9%），
`feat` 只做粗分桶。但别给它太多权重。

**⑤ 中英双语目前只有裸码结果。**
双语词表 6,109 词（中英 ≈ 1:1）管线跑通了，但**还没做后处理**，
当前 Recall@10 = 0.2750，不能和上面任何数字直接比。
按 ④②的规律，补上后处理应该会明显上升，但这需要实测而不是推测。

---

## 快速开始

### 环境

- Rust 1.72+
- Python 3.8+（只有训练脚本需要）
- **无需 GPU**（推理纯 CPU；训练脚本可以跑在 CPU 上，只是慢）

```bash
cargo build --release
cargo test        # 34 个测试，含 Python↔Rust 映射表对拍
```

### 方式一：用预训练模型（推荐，5 分钟）

从 [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases) 下载一个包，解压，然后：

```bash
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding  student_pca512.cache \
    --deep-hash  deep_hash_52.bin \
    --sim-override sim_override_52.bin \
    --reranker   teacher_bge_large.cache \
    --top-k 10 --queries 3000
```

**v2.0 提供的自然词表模型**（推荐用这个，它对应上面「自然词表」那几行）：

| 资源包 | 大小 | 词表 | 后处理召回@10 |
|---|---|---|---|
| `hsh64_pretrained_natural_3000.zip` | 19 MB | 3,000 自然中文词 | 0.6082 |
| `hsh64_pretrained_natural_6000.zip` | 36 MB | 6,000 | 0.5990 |
| `hsh64_pretrained_natural_10000.zip` | 58 MB | 10,000 | 0.5936 |
| `hsh64_pretrained_natural_16000.zip` | 91 MB | 16,000 | 0.5965 |

每个包里有：词表、学生嵌入缓存（512 维，**模型的实际输入**）、教师嵌入缓存（1024 维，
**评测真值的来源**）、DeepHash 模型、后处理覆盖码、PCA 基、`MANIFEST.json`（sha256 校验）
以及包内 README（含复现命令）。

**v1 提供的论文基准词表模型**（对应上面合成词表那几行）：

| 资源包 | 大小 | 内容 |
|---|---|---|
| `hsh64_pretrained_3109_light.zip` | 18 MB | 词表 + 嵌入缓存 + PCA-52 + Deep Hash h256 + 后处理 |
| `hsh64_pretrained_3109_best.zip` | 18 MB | 同上，Deep Hash h512 s42（最佳单模型） |
| `hsh64_pretrained_3109_full.zip` | 22 MB | 多个模型 + 多组后处理（复现 Ensemble） |

> **两个口径别混用**：`--reranker` 同时充当「精排模型」和「真值来源」。
> 带 `--reranker` 得到的是「教师真值」口径（0.6082）；
> 不带它时真值取自学生嵌入，是**另一个指标**（3,000 词实测 0.5947）。
> 包内 `eval_result.json` 里的数字是教师真值口径。

> 仓库本身不含 `.bin` / `.cache`（见 [.gitignore](.gitignore)），请从 Release 下载。

### 方式二：从头训练

```bash
# 1. 生成学生嵌入（bge-small，512 维）
python scripts/generate_embeddings.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/embedding.cache

# 2. 生成教师嵌入（bge-large，1024 维）
python scripts/generate_reranker_embeddings_64.py \
    --vocab tests/data/vocab_3109.txt \
    -o tests/data/reranker_embedding.cache

# 3. PCA-52 基线
python scripts/train_pca_64.py \
    --embedding-cache tests/data/embedding.cache \
    -o tests/data/pca_52.bin

# 4. Deep Hash（阶段 1+2）
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache tests/data/embedding.cache \
    --teacher-cache tests/data/reranker_embedding.cache \
    --hidden-dim 256 --seed 2025 \
    -o tests/data/deep_hash_v3_64_3109_h256_s2025.bin

# 5. 召回导向后处理（阶段 3）—— 0.4988 → 0.7426 的关键一步
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

> 参数以 `python scripts/<脚本> --help` 为准。

### 自定义规模

想换词表大小，用 v2.0 的流水线（一步到位：训词表 → 嵌入 → PCA → 训练 → 后处理 → 评测）：

```bash
python scripts/53_build_natural_vocab.py --sizes 3000 6000 10000 16000
python scripts/54_run_scale_curve.py --sizes 6000 10000
python scripts/62_natural_baselines.py          # 与随机码 / PCA-52 对比
```

---

## 索引：半径怎么设

这一条踩过坑，单独说。

MIH（多索引哈希）用「分段 + 半径」来剪枝。**半径设小了会一个正样本都取不到** ——
52 位码的正样本对 Hamming 距离约 9-10 位，而默认参数一度只扫到 8。

实测（3,109 词，纯 Hamming）：

| 半径 | 平均候选数 | Recall@10 |
|---|---|---|
| 8 | 4.3 | 0.330 |
| 18 | 10.0 | 0.605 |
| **22** | **10.0** | **0.605** |

半径 22 时 MIH 达到暴力扫描的 95%。**所以 `--radii` 至少要到 22**，或者直接用 `--adaptive`
让它自动扩展到候选池饱和。

分段数 `seg` 影响速度不影响召回（最终都过完整 52 位距离）：`seg=2` 1304 QPS、
`seg=4` 834 QPS、`seg=13` 382 QPS。

> `seg=1` 不受支持 —— 段值会占满 52 位，超出索引键宽、也无法枚举段内汉明球。
> 传入 1 会得到明确的配置错误，而不是静默返回空结果。
> 详细数据见 [docs/scale_evaluation.md](docs/scale_evaluation.md) §4。

---

## 中英双语

`feat` 的 4 位是**唯一语言相关**的部分。中文（jieba 标签）与英文（Penn Treebank 标签）
映射到同一组 16 个槽位，所以两种语言的词能进同一个码本。

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
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000   # 从 ECDICT 按词频选英文
python scripts/49_tag_vocab.py                                 # 双语词性标注
```

跨语言对齐用 **bge-m3** 验证过：40 组中英对照词 + 6 组负样本，AUC = 1.0000，
正样本最小 +0.5446 > 负样本最大 +0.5393，分布不重叠。

> ⚠️ 但**样本量只有 40 正 / 6 负，AUC=1.0 不能外推**。结论仅限「跨语言对齐的前提成立」。
> 而双语模型目前**还没做后处理**（裸码 0.2750）。
> 详细状态与已知限制见 [docs/bilingual.md](docs/bilingual.md)。

---

## API

### Rust

```rust
use hsh64::{Encoder, EncoderConfig, HSHCode64, hamming_distance64};

// 最基础的：距离
let dist = hamming_distance64(a.sim(), b.sim());   // 一次 XOR + POPCNT

// 编码器
let mut config = EncoderConfig {
    embed_dim: 512,
    embedding_cache_path: Some("student_pca512.cache".into()),
    pca_path: Some("pca_52.bin".into()),
    deep_hash_path: Some("deep_hash_52.bin".into()),
    sim_override_path: Some("sim_override_52.bin".into()),
    ..Default::default()
};
// 双语需要载入词性表；不载入则全部按 "n" 处理（与旧行为一致）
config.load_pos_tags("data/mixed_tags.tsv".as_ref())?;

let encoder = Encoder::with_config(config)?;

let code = encoder.encode_word_auto("北京");                        // 自动查词性 + 判语种
let code = encoder.encode_word_with_pos_lang("apple", "NN", PosLang::En);  // 显式
```

```rust
use hsh64::MihSemanticIndex;

let index = MihSemanticIndex::build_with_embedding(encoder, embedding, 13, false)?;

let (radius, results) = index.search_adaptive(query, top_k, coarse_factor, max_radius)?;
let (radius, results) = index.search_adaptive_asymmetric(query, top_k, coarse_factor, max_radius)?;
```

### Python 脚本

| 脚本 | 作用 |
|---|---|
| `generate_embeddings.py` | 生成学生嵌入缓存 |
| `generate_reranker_embeddings_64.py` | 生成教师 / 精排嵌入缓存 |
| `train_pca_64.py` | 训练 PCA-52 基线 |
| `train_deep_hash_v3_64.py` | 训练 Deep Hash（阶段 1+2） |
| `post_optimize_codes_64.py` | 基础贪心比特翻转 |
| `post_optimize_codes_recall_64.py` | 召回导向后处理（阶段 3） |
| `bilingual_pos.py` | 中英双语词性标注（与 `pos_map.rs` 严格对应） |
| `48_build_vocab.py` / `49_tag_vocab.py` | 混合词表构建与标注 |
| `53_build_natural_vocab.py` | 构建自然中文词表（嵌套多档） |
| `52_eval_pipeline.py` | 一条命令跑完训练+后处理+评测+导出 |
| `54_run_scale_curve.py` | 批量跑规模曲线 |
| `62_natural_baselines.py` | 与随机码 / PCA-52 对比 |
| `58_package_release.py` | 打包 Release 资源 |

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
│   ├── benchmark_pure_hsh64.rs     单模型基准（暴力 / MIH / 非对称 / 精排）
│   └── benchmark_ensemble_hsh64.rs 集成基准
├── scripts/                    Python 训练与评测
├── tests/
│   ├── end_to_end.rs
│   ├── feat_parity.rs          Python↔Rust 词性映射对拍
│   └── data/
├── data/                       工作数据（生成物，多不入库）
├── docs/
│   ├── README.md               文档索引
│   ├── explained.md            设计文档（从零讲起）
│   ├── scale_evaluation.md     全部实测结果与复现
│   ├── bilingual.md            中英双语扩展
│   └── *.en.md                 各文档的英文版
```
---

## 当前状态

| 环节 | 状态 |
|---|---|
| 编码结构 `feat(4) + sim(52) + abs(8)` | ✅ 完成 |
| 三阶段训练流程 | ✅ 完成 |
| 中英双语 `feat` 映射 | ✅ 完成（双语评测待补后处理） |
| 混合词表 6,109 词 | ✅ 完成 |
| 自然中文词表 3k / 6k / 10k / 16k | ✅ 完成 |
| 规模评估（3k→16k 召回曲线 + 基线对比） | ✅ 完成 |
| Python↔Rust 对拍测试 | ✅ 完成（含变异验证） |
| 单元 + 集成 + 对拍测试 | ✅ 34 个全部通过 |
| `abs` 完美哈希种子表 | ⚠️ 未生效（`build_seed_table` 无调用点，`abs` 失去唯一性保证） |
| 双语模型的后处理 | ⚠️ 未做（裸码 0.2750） |

> `abs` 种子表未生效**不影响召回** —— `abs` 不参与 MIH 检索（只用 `sim` 分段 + `feat` 过滤），
> 但该字段目前不具备设计意图中的唯一性保证。详见 [docs/bilingual.md](docs/bilingual.md) §9.3。

---

## 文档

| 文档 | 内容 |
|---|---|
| **[docs/](docs/README.md)** | **文档索引** —— 按阅读目的组织 |
| **[docs/explained.md](docs/explained.md)** | 从零讲起的设计文档：词表与嵌入、Recall@K 的定义、编码结构、DeepHash、三阶段训练、非对称距离、MIH、两阶段检索 |
| **[docs/scale_evaluation.md](docs/scale_evaluation.md)** | 全部实测结果：论文基准复现、自然词表 3k→16k 曲线、与零学习基线的对比、索引半径、`abs` 容量上限、复现命令 |
| **[docs/bilingual.md](docs/bilingual.md)** | 中英双语：`feat` 槽位分配、词表构建、跨语言对齐验证、已知限制 |

以上每篇都有英文版本（文件名加 `.en`），见 [docs/README.en.md](docs/README.en.md)。

论文不随本仓库分发。论文 PDF 收在 [Zenodo](https://doi.org/10.5281/zenodo.21721838)
归档的源码包内（v1 / v2 的包内含 `paper/main.pdf` 与 `paper/main_chinese.pdf`），引用请使用该 DOI。

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

MIT —— 见 [LICENSE](LICENSE)。
