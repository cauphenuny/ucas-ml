#import "@preview/tablem:0.3.0": tablem, three-line-table

== 最终模型性能对比

=== 核心指标对比

下表展示了四个模型在验证集上的最佳性能指标。可以看到，不同架构的模型在情感分析任务上表现出了显著差异。

#figure(
  three-line-table[
    | 模型架构 | 架构类型 | Val Accuracy | Val Loss | Macro-F1 | 极端情感召回 (Class 0/4) |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | *LSTM* | RNN | 58.50% | 1.0592 | 0.41 | 极差 (~13% / 21%) |
    | *TinyLLM* | Decoder-only | 65.95% | 0.8335 | 0.53 | 一般 (~23% / 36%) |
    | *GPT-2* | Decoder-only | 66.89% | 0.7949 | 0.57 | 一般 (~30% / 45%) |
    | *RoBERTa* | *Encoder-only* | *69.81%* | *0.7256* | *0.62* | *较好 (~47% / 57%)* |
  ],
  caption: "四大模型最佳性能指标对比",
)

#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  *结论：* 虽然 Accuracy 的差距看似只有 11% (59% vs 70%)，但 *Macro-F1* 的差距高达 0.21，这说明架构选择对性能有决定性影响。
]

---

=== 架构分析

1. *LSTM (RNN 架构)*:
  - 验证集 Loss 始终较高 (停留在 1.05 左右)。
  - 模型难以捕捉文本深层的语义特征，性能在 58% 左右触达天花板。
  - 受限于序列建模能力，在长文本和复杂语义理解上表现较差。

2. *Decoder-only Transformer 架构 (TinyLLM / GPT-2)*:
  - 两个 Decoder-only 模型表现接近：TinyLLM (65.95%) 和 GPT-2 (66.89%) 的准确率仅相差约 1%。
  - 验证集 Loss 稳定在 0.79-0.83 之间，GPT-2训练过程更加平稳，TinyLLM由于预训练知识不足，在微调后期出现过拟合现象。
  - 在极端情感识别上表现一般，Class 0/4 的召回率在 23-45% 之间。

3. *Encoder-only Transformer 架构 (RoBERTa)*:
  - 准确率达到 69.81%，比 Decoder-only 模型高出约 3-4 个百分点。
  - 验证集 Loss 最低 (0.73)，收敛更稳定。
  - 在极端情感识别上表现突出，Class 0/4 的召回率达到 47%/57%，显著优于 Decoder-only 模型。

---

=== 极端情感捕捉能力分析 (Recall Analysis)

本任务最大的难点在于 *类别不平衡* (Neutral 占 50%)。我们重点关注模型对 *Very Negative (0)* 和 *Very Positive (4)* 的识别能力。

#figure(
  three-line-table[
    | 真实标签 | LSTM Recall | TinyLLM Recall | GPT-2 Recall | RoBERTa Recall |
    | :--- | :--- | :--- | :--- | :--- |
    | 0 (极负) | 13% | 23% | 30% | 47% |
    | 1 (负面) | 40% | 52% | 57% | 61% |
    | 2 (中性) | 81% | 80% | 81% | 79% |
    | 3 (正面) | 40% | 57% | 55% | 62% |
    | 4 (极正) | 21% | 36% | 45% | 57% |
  ],
  caption: "各类别召回率 (Recall) 详细对比"
)

*数据洞察：*
- *LSTM* 在极端情感识别上表现最差，倾向于把所有极端样本都预测为中性，Class 0/4 的召回率仅 13%/21%。
- *TinyLLM* 与 *GPT-2* 表现接近，在极端情感识别上有所提升，但仍明显低于 Encoder-only 架构。两个模型的 Class 0/4 召回率在 23-45% 之间。
- *RoBERTa* 在极端情感识别上表现最佳，Class 0/4 的召回率达到 47%/57%，比 Decoder-only 模型高出约 15-20 个百分点。这进一步验证了双向注意力机制在理解型任务上的优势。

---

#grid(columns: (1fr, 1fr))[
  === 混淆矩阵形态分析
  
  对比四个模型的混淆矩阵，可以观察到一些特征：
  
  1. *中心坍缩 (LSTM)*: 预测结果高度集中在 Label 2 (Neutral) 列，呈现垂直长条状。模型采取了"由于不确定，所以猜中性"的保守策略。
  
  3. *对角线强化 (Transformer)*: 混淆矩阵的对角线明显变亮，特别是在 Label 0 和 Label 4 的角落。这代表模型真正理解了语义，而非利用数据分布漏洞。双向注意力机制使得模型能够更好地捕捉全局语义信息。
  
  4. *邻类漂移 (Common Issue)*: 所有模型的主要错误都集中在 *相邻类别* (如把 0 预测成 1，把 4 预测成 3)。几乎没有模型会将 0 预测成 4。这说明模型都成功学到了情感的 *连续性特征*。
  
][
  #figure(
    image("assets/confusion_matrices.png", width: 100%),
    caption: "四大模型混淆矩阵对比 (归一化后)",
  )
]
---

== 总结

通过之前的实验以及模型表现，我们得出以下结论：

1. *架构决定上限*：LSTM 受限于结构，无法有效处理长文本和复杂语义；Transformer 架构显著优于 RNN。

2. *Decoder-only vs Encoder-only 的架构差异*：
  - *Decoder-only 架构* (TinyLLM/GPT-2) 通过因果掩码进行自回归训练，更适合生成任务。在分类任务中，对全局语义的理解相对受限，准确率在 66% 左右。
  - *Encoder-only 架构* (RoBERTa) 通过双向注意力机制可以同时利用上下文的所有信息，对文本的全局语义理解更强，更适合理解型任务，准确率达到 70%，在极端情感识别上表现尤为突出。

3. *预训练的重要性*：大规模预训练使模型携带的丰富先验知识是解决 *Cold Start* 和 *类别不平衡问题* 的关键。预训练使得模型能够更好地泛化到小样本场景。

4. *Accuracy 区分度不足*：LSTM 的 58% Accuracy 背后是接近 0 的极端情感召回率；Transformer 模型的约 70% Accuracy 代表了更均衡、更可用的模型。

---

=== 最终提交结果

#grid(columns: (1fr, 1.5fr), align: horizon)[
  #three-line-table[
    | 模型 | 提交结果 | 排名 |
    | :--- | :--- | :--- |
    | Logistic Regression| 0.62450 | 284 / 861 |
    | 集成 (LR+RoBERTa) | 0.68507 | 6 / 861 |
    | LSTM | 0.58886 | 508 / 861 |
    | TinyLLM | *0.67854* | *8 / 861* |
    | RoBERTa | *0.71186* | *3 / 861* |
  ]
][
  #image("assets/kaggle/traditional.png", width: 100%)
  #image("assets/kaggle/merge.png", width: 100%)
  #image("assets/kaggle/lstm.png", width: 100%)
  #image("assets/kaggle/tinyllm.png", width: 100%)
  #image("assets/kaggle/roberta.png", width: 100%)
]
