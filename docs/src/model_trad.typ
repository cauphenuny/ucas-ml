== 传统ML方法

=== Logistic Regression

将TF-IDF处理后的文本通过逻辑回归 (Logistic Regression, LR) 进行分类。

用于多分类的 LR:

对于每一个类别 $k$ ，计算一个得分

$
  z_k=w_k^(tack.b)x+b_k
$

类别 $k$ 的预测概率为：

$
  P(y=k|x)= (exp(z_k))/(sum_(j=1)^K exp(z_j)) quad "i.e. " "Softmax"(z_k)
$

（这也是为什么多分类LR又叫 Softmax Regression）

损失函数 (Cross Entropy Loss)：

$
  -sum_(i=1)^m sum_(k=1)^K y_k^((i)) log(P(y=k|x^((i))))
$

/*

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

*/