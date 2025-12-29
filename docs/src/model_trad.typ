== Traditional ML

=== TF-IDF

#import "@preview/cmarker:0.1.8"
#import "@preview/tablem:0.3.0": tablem
#import "@preview/mitex:0.2.6": mi, mitex

#grid(
  columns: (5fr, 3fr)
)[
  
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

][
  #figure(image("assets/TF_IDF.webp", width: 15em), caption: "tf_idf")
]

---

=== Logistic Regression




根据上述公式，我们设计实现了TF-IDF+Logistic Regression的传统机器学习模型,来进行情感分类任务。此外，我们还设计了与深度学习模型一致的接口，方便后续集成使用。
---

=== 集成：加权融合

#grid(
  columns: (1fr, 1fr),
)[

  传统模型通过 `get_traditional_probs()` 函数提供标准化概率输出，与深度学习模型进行加权融合。

  **核心集成流程**：
  ```python
  # 1. 获取传统模型概率
  trad_probs = get_traditional_probs(pipeline, phrases, num_classes=5)
  
  # 2. 获取深度模型概率
  deep_probs = get_deep_probs(args, phrases, num_classes=5)
  
  # 3. 加权概率融合（核心接口）
  combined = w_deep * deep_probs + w_trad * trad_probs
  preds = combined.argmax(axis=1)
  ```

]
