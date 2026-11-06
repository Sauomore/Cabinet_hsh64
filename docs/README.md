# 文档

按阅读目的组织。想看代码先读 [README](../README.md)。

---

## 我想先弄懂它是什么

| 文档 | 内容 | 大约篇幅 |
|---|---|---|
| [explained.md](explained.md) | 从零讲起：词表与嵌入、教师相似度、Recall@K 的定义、52 位码的结构、DeepHash 网络、三阶段训练、非对称距离、MIH、两阶段检索 | 660 行 |

`explained.md` 是自学路径：它假设你只知道「向量检索」是什么，
其余概念在用到之前都会先定义。

---

## 我想看它到底跑出什么数字

| 文档 | 内容 |
|---|---|
| [scale_evaluation.md](scale_evaluation.md) | 全部实测结果：论文基准词表的复现、自然中文词表 3k→16k 的规模曲线、与零学习基线的对比、索引半径的选择依据、`abs` 的容量上限、训练成本、复现命令、局限 |

关键数字速览：

```text
论文基准词表（3,109 词）  裸码 0.4988   + 后处理 0.7426   两阶段 0.8890
自然词表（3,000 词）      裸码 0.2463   + 后处理 0.6082
自然词表（16,000 词）     裸码 0.2519   + 后处理 0.5965
```

---

## 我关心中英双语

| 文档 | 内容 |
|---|---|
| [bilingual.md](bilingual.md) | `feat` 的 16 槽位如何由中英两套词性标签共用、双语词表构建、跨语言对齐验证、已实测的局限与当前状态 |

---

## 我想上手用

回 [README](../README.md) 的「快速开始」。两种方式：

- **直接用预训练模型** —— 从 [Releases](https://github.com/Sauomore/Cabinet_hsh64/releases) 下载，
  包里附词表、嵌入缓存、模型与复现命令
- **从头训练** —— README 里有逐步命令

---

## 文档与代码的对应关系

| 文档 | 主要涉及的文件 |
|---|---|
| `explained.md` | `src/hsh64.rs`、`src/deep_hash.rs`、`src/mih_index.rs`、`src/encoder.rs` |
| `bilingual.md` | `src/pos_map.rs`、`src/encoder.rs`、`scripts/bilingual_pos.py`、`tests/feat_parity.rs` |
| `scale_evaluation.md` | `scripts/50`–`62_*.py`、`examples/benchmark_pure_hsh64.rs` |

---

## 英文版

每篇文档都有对应的英文版本，文件名加 `.en`：

| 中文 | English |
|---|---|
| [explained.md](explained.md) | [explained.en.md](explained.en.md) |
| [scale_evaluation.md](scale_evaluation.md) | [scale_evaluation.en.md](scale_evaluation.en.md) |
| [bilingual.md](bilingual.md) | [bilingual.en.md](bilingual.en.md) |
