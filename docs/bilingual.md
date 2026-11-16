# HSH-64 中英双语扩展

本文档记录把 HSH-64 从**中文单语**扩展到**中英混合索引**的设计、实现与实测结果。

---

## 1. 为什么做双语

原有实现只支持中文：

- 词表 `tests/data/vocab_3109.txt` 3109 词，其中只有 4 个含拉丁字母
- 词性标注走 jieba（中文分词器的词性标签集）
- embedding 用 `BAAI/bge-small-zh-v1.5`（中文专用模型）

双语化不只是「换个大模型」—— 编码结构里有**语言相关的假设**，必须逐个处理。

---

## 2. 编码结构里的语言相关部分

HSH-64 的位布局：

```text
[63:60] feat (4 bit)   ← 词性类别，16 个槽位    ★ 语言相关
[59: 8] sim  (52 bit)  ← 语义码，来自 Deep Hash  ✅ 语言无关
[ 7: 0] abs  (8 bit)   ← 桶内唯一 ID            ✅ 语言无关
```

**只有 `feat` 是语言相关的。** `sim` 完全由 embedding 决定，`abs` 只是字符串哈希。

**所以双语改造的改动面集中在 `feat` 的映射表上** —— 这也是设计上的一处幸运：4-bit 的 `feat` 本来就是个「硬编码的粗粒度类别」，不需要承载细粒度语义。

---

## 3. `feat` 的 16 槽位分配

中英各有一套词性标签集：

- 中文：jieba 标签（`n` / `nr` / `ns` / `v` / `a` / `q` / `t` …）
- 英文：Penn Treebank 标签（`NN` / `VB` / `JJ` / `DT` / `MD` …）

两者映射到**同一组 16 个槽位**，这样两种语言的词才能进同一个码本。

### 分配原则

1. **能合并的合并** —— 中英共有的大类（名词/动词/形容词/副词/介词/连词/数词/标点）直接共用
2. **合并不了的保留** —— 中文特有的「量词」独占 0x9
3. **需要的新增** —— 英文的限定词与情态动词在中文里没有直接对应，但它们是高频功能词

### 槽位表

| 槽 | 含义 | 中文来源（jieba） | 英文来源（PTB） |
|---|---|---|---|
| 0x0 | 名词 | `n` `nr` `ns` `nt` `nz` `ng` `nl` … | `NN` `NNS` `NNP` `NNPS` `FW` |
| 0x1 | 动词 | `v` `vd` `vn` `vg` … | `VB` `VBD` `VBG` `VBN` `VBP` `VBZ` |
| 0x2 | 形容词/区别词 | `a` `ad` `an` `ag` `b` | `JJ` `JJR` `JJS` |
| 0x3 | 副词 | `d` `df` `dg` | `RB` `RBR` `RBS` `WRB` |
| 0x4 | **代词/限定词** | `r` `rr` `rz` `rg` … | `PRP` `PRP$` `WP` `WP$` `DT` `PDT` `WDT` `EX` |
| 0x5 | 介词/方位/小品词 | `p` `f` `fg` `s` | `IN` `TO` `RP` |
| 0x6 | 连词 | `c` `cc` | `CC` |
| 0x7 | **助词/情态** | `u` `ud` `ug` `uj` … `y` `z` `zg` | `MD` |
| 0x8 | 数词 | `m` `mq` | `CD` `LS` |
| 0x9 | **量词** | `q` `qv` `qt` | （无对应） |
| 0xA | **时间词** | `t` `tg` | （英文并入 ADV） |
| 0xB | 字符串/专名 | `x` `xx` `xu` `xi` `wjb` `nx` `eng` `j` | （英文并入 NOUN/FW） |
| 0xC | 标点 | `w` 系列 16 个 | `.` `,` `:` ``` `` ``` `''` … |
| 0xD | 其他/习语 | `i`（成语） | `UH` `POS` |
| 0xE | 常用词 | 运行时由 `add_common_word` 判定 | 同 |
| 0xF | 兜底 | 未识别标签 | 同 |

### 合并的取舍（都有代价，记录在此）

| 槽 | 合并了什么 | 代价 |
|---|---|---|
| 0x4 | 中文代词 + 英文代词与限定词 | 英文 `the`/`a` 与 `he`/`it` 同码；中文无此区分，无损失 |
| 0x5 | 中文介词/方位词/处所词 + 英文介词/不定式 to/小品词 | 中文「上/下」与「在/从」同码 |
| 0x7 | 中文助词/语气词 + 英文情态动词 | 两类语法功能不同，但都是「语法标记」 |
| 0x9 | 不合并，中文量词独占 | 英文该槽为空（无损失） |
| 0xA | 不合并，中文时间词独占 | 英文 `today`/`yesterday` 在 PTB 里本就是 `RB`，进 0x3 |
| 0xB | 中文「字符串/专名」保留原样 | 英文无对应类 |

---

## 4. ⚠️ 已实测的局限：`feat` 实际上很模糊

**这一节是实测数据，不是推测。**

### 局限一：孤立词的词性标注不可靠

词表里存的是**孤立单词**，没有句子上下文。jieba 与 nltk 的标注器都按句子训练，
对孤立词会显著退化：

**中文**（3109 词，产出 **29 种**标签）：

```text
同意    → d  （副词）    应为动词
丘陵    → nr （人名）    应为名词
乌龟    → nr （人名）    应为名词
伏特加  → nr （人名）    应为名词
```

**英文**（40 词测试，约 **36% 标错**）：

```text
tree      → JJ    应为 NN
flower    → JJR   应为 NN
teacher   → RB    应为 NN
beautiful → IN    应为 JJ（单句调用）/ VB（批量调用）
cold      → VBP   应为 JJ
```

**`beautiful` 在不同调用路径下得到不同结果（`IN` vs `VB`）** —— 孤立词标注不仅错，
而且**不稳定**。

### 局限二：`feat` 分布极度倾斜

中英混合词表（6109 词）的实际分布：

```text
0x0 名词      2,925  (47.9%)
0x2 形容词    2,344  (38.4%)
0x1 动词        408  ( 6.7%)
0x3 副词        203  ( 3.3%)
0x5 介词         89  ( 1.5%)
0x4 代词         44  ( 0.7%)
其他 10 槽      96  ( 1.6%)
─────────────────────────────
未使用：0xC（标点）、0xE（常用词）
```

**两个槽吃掉 86% 的词，实际熵不到 1.5 bit。**

### 结论：`feat` 只是硬编码的粗粒度线索，不承担语义区分

这是**设计上可接受**的 —— 语义区分由 52-bit 的 `sim` 承担（`sim` 的唯一码率实测 99%）。
`feat` 的作用是：

1. 把不同词类分到不同的 `(feat, sim_low8)` 桶，**降低 `abs` 的碰撞概率**
2. 提供可解释的检索路径（「为什么想起这个词」）
3. 支持按词类过滤

**`feat` 在检索打分中的权重越大，噪声引入越多。**

---

## 5. 实现

### 5.1 Rust 侧（`src/pos_map.rs`）

```rust
pub enum PosLang { Zh, En }

pub fn pos_to_feat_zh(pos: &str) -> FeatureCode   // jieba 标签集
pub fn pos_to_feat_en(pos: &str) -> FeatureCode   // PTB 标签集
pub fn pos_to_feat_lang(pos: &str, lang: PosLang) -> FeatureCode

impl PosLang {
    /// 含 CJK 字符即为中文
    pub fn detect(word: &str) -> PosLang
}
```

### 5.2 词性从文件加载（`src/encoder.rs`）

加入双语之前，所有调用点都**硬编码** `encode_word_with_pos(word, "n")` ——
也就是 `feat` 恒为 `0x0`。双语词表下这行不通：英文词传 `"n"` 会被当成中文标签而落入
`FALLBACK`。

```rust
pub struct EncoderConfig {
    ...
    /// 词 → 词性标签（可选）。为空时退回内置 "n"，保持旧行为
    pub pos_tags: HashMap<String, String>,
}

impl EncoderConfig {
    /// 从 TSV 载入（word<TAB>lang<TAB>tag<TAB>feat）
    pub fn load_pos_tags(&mut self, path: &Path) -> Result<usize, EncodeError>
}

impl Encoder {
    /// 双语推荐入口：自动查词性表并判定语种
    pub fn encode_word_auto(&self, word: &str) -> HSHCode64
}
```

**兼容性**：不提供 `pos_tags` 时行为与改动前完全一致（退回 `"n"`），
所有旧调用点可直接替换为 `encode_word_auto`。

### 5.3 Python 侧（`scripts/bilingual_pos.py`）

与 Rust 表严格对应的两份映射，中文用 jieba、英文用 nltk，
混合词表按词自动分派语言后分别批量标注。

### 5.4 对拍测试（`tests/feat_parity.rs`）

**两张表存在于两个地方，必须保证一致** —— 否则「Python 建码本、Rust 查码本」
会给出不同结果，而这类 bug 单看任何一边都正常。

```text
scripts/bilingual_pos.py
        │  scripts/46_export_feat_table.py
        ▼
tests/data/feat_table.json
        │  cargo test --test feat_parity
        ▼
src/pos_map.rs  逐项断言
```

覆盖：中文 92 个标签、英文 46 个标签、16 槽可达性、跨语言不冲突。

**该测试已通过变异验证** —— 故意删掉 Rust 侧的 `FW` 映射，测试立即报出：

```text
英文表有 1 处不一致（Python vs Rust）:
  FW          Python=0x0  Rust=0xF
```

---

## 6. 词表构建

### 6.1 数据来源

| 侧 | 来源 | 规模 |
|---|---|---|
| 中文 | 沿用 `tests/data/vocab_3109.txt` | 3,109 词 |
| 英文 | [ECDICT](https://github.com/skywind3000/ECDICT)（英汉词典库，MIT） | 从 770,611 条中筛选 |

**中文侧沿用原词表**，是为了让旧实验保持可比性。

### 6.2 英文选词规则

```text
· 词频：bnc 或 frq（COCA）排名 <= 6000，取较优者
· 形态：只保留纯 ASCII 小写字母（去掉连字符、空格、重音符号）
· 长度：2 ~ 14
```

结果：**3,000 词**，排名范围 **1 ~ 2744**。

```text
频次最高的 15 个: the be and of in to have it he for that not you with on
入选的 15 个:     complicated olympic alongside dirt bullet lend rice
                  considerably tight bible replacement chart proof practise solar
```

### 6.3 最终词表

```text
data/mixed_vocab.txt        6,109 词
  中文    3,095  (50.7%)
  英文    3,002  (49.1%)
  混合       12  ( 0.2%)   T恤、U盘、大U盘…
```

**中英比例接近 1:1**，便于做跨语言检索评测。

---

## 7. 跨语言对齐验证（前置实验）

混合索引成立的前提是：**embedding 必须把「苹果」和「apple」放在相近位置。**
这件事不能假设，所以先做了验证。

### 7.1 模型选择

| 模型 | 维度 | 说明 |
|---|---|---|
| `BAAI/bge-large-zh-v1.5` | 1024 | 中文专用（对照组） |
| **`BAAI/bge-m3`** | **1024** | **多语言（采用）** |

### 7.2 结果（40 组中英对照词 + 6 组负样本）

| 指标 | bge-m3 | bge-large-zh |
|---|---|---|
| 正样本均值 | **+0.8054** | +0.7205 |
| 负样本均值 | +0.4910 | +0.4244 |
| **AUC（排序质量）** | **1.0000** | 0.9573 |
| 正样本最小 | +0.5446 | +0.4036 |
| 负样本最大 | +0.5393 | +0.5094 |
| 分布重叠 | **无** ✅ | 有 ❌ |

**bge-m3 的 40 个正样本全部排在 6 个负样本之上，分布不重叠**
（正样本最小 +0.5446 > 负样本最大 +0.5393）。

典型结果：

```text
学校 school   +0.9136      苹果 apple     +0.9046
狗   dog      +0.9061      汽车 car       +0.8890
学生 student  +0.9061      计算机 computer +0.8642

较弱（专有名词）：
北京 Beijing  +0.7015      上海 Shanghai  +0.6654
```

> ⚠️ **样本量限制**：40 正 / 6 负是小样本，AUC=1.0 不能外推。
> 结论仅限「跨语言对齐的前提成立，值得继续」。

---

## 8. 端到端流程

```bash
# 1. 构建混合词表（需先下载 ECDICT）
python scripts/48_build_vocab.py --n-en 3000 --max-rank 6000

# 2. 双语词性标注
python scripts/49_tag_vocab.py

# 3. bge-m3 编码
python scripts/generate_embeddings.py \
    --vocab data/mixed_vocab.txt \
    --local-model /path/to/bge-m3 \
    -o data/mixed_embedding.cache

# 4. PCA 降维到 52 维
python scripts/train_pca_64.py \
    --embedding-cache data/mixed_embedding.cache \
    -o data/mixed_pca_52.bin

# 5. Deep Hash 训练
python scripts/train_deep_hash_v3_64.py \
    --embedding-cache data/mixed_embedding.cache \
    -o data/mixed_deep_hash_52.bin \
    --n-bits 52 --epochs 500

# 6. Rust 端评测（--pos-tags 是双语的关键）
cargo run --release --example benchmark_pure_hsh64 -- \
    --embedding data/mixed_embedding.cache \
    --pca data/mixed_pca_52.bin \
    --deep-hash data/mixed_deep_hash_52.bin \
    --pos-tags data/mixed_tags.tsv \
    --top-k 10 --queries 100
```

---

## 9. 当前状态与实测结果

### 9.1 管线各环节

| 环节 | 结果 |
|---|---|
| 混合词表 | 6,109 词（中英 ≈ 1:1） |
| 双语标注 | 中文 28 种标签 / 英文 31 种标签 |
| 编码唯一码率 | **6,057 / 6,109 = 99.15%** |
| Deep Hash 收敛 | info 损失 1.5758 → 0.0080 |
| 词性接入 | ✅ 14 处硬编码已替换 |
| 第三阶段后处理 | ✅ 已完成（贪心比特翻转，10 轮，161.9 s） |

### 9.2 检索召回

真值由 bge-m3 余弦相似度定义（教师与训练输入同为 bge-m3，无教师/学生维度差）。
全量 6,109 个查询，`overlap@10` 口径。

| 方法 | Recall@10 | 相对 PCA-52 |
|---|---|---|
| PCA-52（零学习基线） | 0.3503 | 1.00× |
| HSH-64 裸码 | 0.3171 | 0.91× |
| **HSH-64 + 后处理** | **0.6086** | **1.74×** |
| 同上，Rust 端暴力扫描 | 0.6131 | 1.75× |
| 同上，Rust 端 MIH（`seg=4, radius=22`） | 0.5895 | 1.68× |

**两个与该套方案在中文单语上一致的现象：**

1. **裸码低于 PCA-52**（0.317 vs 0.350）。这与中文单语词表上的结论相同
   （见 [scale_evaluation.md](scale_evaluation.md) §3.2），说明
   **后处理不是可选优化，而是这套编码能用的前提**。
2. **加后处理后相对基线 1.74×**，与自然中文 3,000 词的 1.62× 处于同一量级 ——
   语言混合本身没有破坏这个收益。

> **一处历史数字更正。** 本文档先前记录「裸 DeepHash Recall@10 = 0.2750」。
> 这个数字**无法从现有产物复现**，已作废。核对过的三条路径都给出更高值：
> 全部 6,109 查询 0.3171、前 100 查询 0.3460、随机 100 查询 0.3180；
> PCA-52 路径则是 0.3503。因此 0.2750 应来自另一份模型或另一套口径，
> 现有数据不足以还原，此处不再引用。

### 9.3 跨语言检索

混合索引的实际用途是「中文查询召回英文词」，所以单独测这一项：
取中英对照词对，看目标英文词在**全部 6,109 个码**按 Hamming 距离排序中的位次。

39 组词对中 **30 组可用**（另 9 组的英文词不在 3,002 词的英文词表内，无法测：
`banana` `water` `airplane` `Beijing` `Shanghai` `China` `America` `love` `hate`）。

| 码 | 中位 rank | top-1 | top-5 | top-10 | MRR | 最差 rank |
|---|---|---|---|---|---|---|
| 裸码 | 2.0 | 43.3% | 73.3% | 73.3% | 0.5656 | 4,352 |
| **后处理** | **2.0** | 43.3% | 73.3% | **86.7%** | 0.5671 | 3,964 |
| 教师（bge-m3 余弦） | 1.0 | 76.7% | 86.7% | 93.3% | 0.8306 | 15 |
| 负对照（不该召回） | 3,327 | 0% | 0% | 0% | 0.0003 | 4,953 |

随机码的期望 rank 约 3,054，所以后处理码的中位 rank 2.0 相当于**比随机好约 1,500 倍**。
负对照落在随机水平附近，说明没有把不相关的词强行拉到一起。

**后处理的主要贡献在长尾**：中位 rank 不变（都是 2），但 top-10 从 73.3% 提到 86.7%。

**长尾失败是真实存在的**，30 组里有 4 组排在 50 名之外：

| 词对 | 码的 rank | 教师 rank |
|---|---|---|
| 猫 → cat | 3,964 | ≤15 |
| 大 → big | 902 | ≤15 |
| 太阳 → sun | 337 | ≤15 |
| 快 → fast | 60 | ≤15 |

后处理的目标函数是**整个词表的平均 Recall@10**，而跨语言词对只占近邻结构里极小的一部分，
所以它不会专门为这些词对优化。教师在 15 名以内、码却掉到近 4,000 名，
说明量化过程丢失了这条跨语言联系，而不是教师本身没建立。

### 9.4 已知限制

**① `feat` 判别力有限**

见 §4。`feat` 的 16 个槽位来自词性硬编码，同一标签下语义差异很大的词会落进同一槽
（例如「同意」被标为副词、「丘陵」被标为人名）。它的作用是把码空间粗分块，
不是精确的语义分类。

**② 英文词性映射有明显错误**

40 词探针上约 36% 判断错误（`tree` → JJ，`beautiful` → IN 或 VB，取决于调用路径）。
原因是 ECDICT 只给词频不给词性，英文标签来自规则推断而非统计标注器。

**③ `abs` 的完美哈希种子表从未生效**

`Encoder::build_seed_table` 在代码库中没有调用点，`seed_for` 恒返回 0：

```rust
*self.seed_table.get(&key).unwrap_or(&0)
```

后果是 `abs` 失去唯一性保证，可能静默碰撞。`abs` 不参与 MIH 检索（只用 `sim` 分段
加 `feat` 过滤），所以不影响召回，但该字段目前不具备设计意图中的唯一性。

**④ 跨语言召回有长尾**

见 §9.3。30 组可测词对里有 4 组排在 50 名之外，其中 `猫 → cat` 排到 3,964。
通用后处理不会针对跨语言词对优化；要改善需要把跨语言约束写进目标函数。


