#import "@preview/tablem:0.3.0": tablem
#import "@preview/mitex:0.2.6": mi, mitex
#import "@preview/theorion:0.4.1"
#import "@preview/cmarker:0.1.8"


== 文本分词与向量化

#grid(
  columns: (5fr, 3fr),
  align: horizon,
)[
  === 词干提取 + TF-IDF
  
  首先提取单词，然后去掉文本中的停用词 (Stop Words)，然后对剩余词语进行 ngram 计数，取出频率最高的一些 unigram 和 bigram 作为 _词_
  
  TF-IDF： 是一种衡量某个 _词_ 对特定文档重要性的统计方法
  
  1. 计算词频，衡量某个词在文档中的重要性
  
  $
    "TF"(t,d)= (f_(t,d))/(sum_(t' in d) f_(t',d))
  $
  
  2. 计算逆文档频率，惩罚常见词，提升稀有词权重
  
  $
    "IDF"(t)= log(N/(1 + "DF"(t)))
  $
  
  3. 计算TF-IDF向量
  
  $
    "TF-IDF"(t,d)= "TF"(t,d) times "IDF"(t)
  $
  
  *问题?* 不含有位置信息、词表大小与未登录词之间存在两难问题

][
  #figure(image("assets/TF_IDF.webp", width: 15em), caption: "tf_idf")
]

---


=== BPE Tokenization

BPE（Byte Pair Encoding）通过迭代合并高频字符对构建子词词表，有效平衡词典大小与 Out of Vocabulary 问题。

*BPE 只是一种高效的编码，不含有语义信息，到语义空间的映射由 Embedding 层学习获得*。

#theorion.example[
  初始语料: `"low lower lowest"`
  1. 基础拆分: `l o w` | `l o w e r` | `l o w e s t`
  2. 合并高频对 `"lo"`: `lo w` | `lo w e r` | `lo w e s t`
  3. 合并高频对 `"low"`: `low` | `low e r` | `low e s t`
  4. 最终词表包含: `low`, `e`, `r`, `st`, `er`, `est`, `lowest` 等子词，每一个子词对应一个整数 id
]

训练 BPE tokenizer 的流程：

#tablem[
  | 阶段 | 核心操作 |
  | :--- | :--- | 
  | 1. 预分词 | 按特殊标记分割文本，再用正则规则提取基础词元，BPE 的合并不越过基础词元边界 |
  | 2. 频率统计 | 统计所有相邻词元对的共现频率 |
  | 3. 迭代合并 | 重复合并当前最高频的词对，将其加入词表 |
  | 4. 词表构建 | 记录所有子词及其唯一ID，生成 `vocab.json` 和 `merges.json` |
]
