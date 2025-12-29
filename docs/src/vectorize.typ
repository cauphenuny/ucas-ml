#import "@preview/tablem:0.3.0": tablem
#import "@preview/mitex:0.2.6": mi, mitex
#import "@preview/theorion:0.4.1"

== 文本分词与向量化

=== TF-IDF

训练 pipeline:

#tablem(
  columns: (1fr, 1.4fr, 1.8fr),
  align: (left, center, left),
)[
  | 步骤 | 操作 | 说明与效果 |
  | :--- | :--- | :--- |
  | 1. 文本提取 | `train_df["Phrase"].astype(str)` | 从数据框中提取评论文本，确保格式统一为字符串 |
  | 2. 标签提取 | `train_df["Sentiment"].astype(int)` | 提取情感标签（0-4），转换为整型用于监督学习 |
  | 3. 文本向量化 | `TfidfVectorizer(ngram_range=(1,2))` | 将文本转换为数值向量，同时考虑单词和双词组合 |
  | 4. 特征选择 | `max_features=50000` | 保留最重要的50,000个词汇特征，控制模型复杂度 |
  | 5. 停用词过滤 | `stop_words='english'` | 自动移除"the", "a", "is"等常见无信息量词汇 |
  | 6. 权重计算 | 自动计算TF-IDF值 | 基于词频和逆文档频率为每个词赋予重要性权重 |
  | 7. 分类器训练 | `LogisticRegression()` | 在TF-IDF特征上训练多项逻辑回归分类器 |
]
---

=== BPE Tokenization

BPE（Byte Pair Encoding）通过迭代合并高频字符对构建子词词表，有效平衡词典大小与未登录词问题。

训练流程：

#tablem[
  | 阶段 | 核心操作 |
  | :--- | :--- | 
  | 1. 数据准备 | 收集文本语料库，准备特殊标记（如 `<|endoftext|>`) | 
  | 2. 预分词 | 按特殊标记分割文本，再用正则规则提取基础词元，BPE 的合并不越过基础词元边界 |
  | 3. 频率统计 | 统计所有相邻词元对的共现频率 |
  | 4. 迭代合并 | 重复合并当前最高频的词对，将其加入词表 |
  | 5. 词表构建 | 记录所有子词及其唯一ID，生成 `vocab.json` 和 `merges.json` |
]

#theorion.example[
  初始语料: `"low lower lowest"`
  1. 基础拆分: `l o w` | `l o w e r` | `l o w e s t`
  2. 合并高频对 `"lo"`: `lo w` | `lo w e r` | `lo w e s t`
  3. 合并高频对 `"low"`: `low` | `low e r` | `low e s t`
  4. 最终词表包含: `low`, `e`, `r`, `st`, `er`, `lowest` 等子词
]