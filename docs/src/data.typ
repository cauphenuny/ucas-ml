#import "@preview/tablem:0.3.0": tablem

== 训练集数据分析

训练集情感分布是：
#grid(columns: (1fr, 1fr), align: horizon)[
  #figure(
    tablem[
      | 情感标签 | 含义            | 样本数        | 占比（约）     |
      | ---- | ------------- | ---------- | --------- |
      | 0    | Very Negative | 7,072      | 4.5%      |
      | 1    | Negative      | 27,273     | 17.4%     |
      | 2    | Neutral       | 79,582     | 50.8%     |
      | 3    | Positive      | 32,927     | 21.0%     |
      | 4    | Very Positive | 9,206      | 5.9%      |
    ],
    caption: "训练集情感分布统计",
  )][
  #figure(image("assets/dataset/dist.png", width: 70%))
]


#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  _这是一个以 Neutral 为主导、极端情感样本稀缺的严重不均衡多分类数据集。_
]

---

=== 数据明显「类别不平衡」
- Neutral（2）$approx$ *占一半*
- Very Negative / Very Positive < *6%*
- 多数类样本 $approx$ 少数类样本的 *10 倍以上*

#block(stroke: (left: 4pt + gray), inset: (left: 1em))[
  模型在训练时会天然偏向预测 Neutral。
]

=== 符合真实世界分布

大多数评论是中性或轻微情感
极端情绪本来就少
数据分布符合真实用户行为，而非人工平衡数据。

=== 在这个数据集上，Accuracy 指标意义有限

如果一个模型 *什么都不学*：
永远预测 Sentiment = 2
它的 accuracy 就是： $≈ 50.8%$

这意味着：55% 以下 $->$ 模型几乎没学，60% $->$ 只是比瞎猜稍好

=== 极端情感（0 / 4）是最难学的

Very Negative + Very Positive =10.4%

但它们在实际应用中最重要

模型很容易把：
0 $->$ 1 / 2、4 $->$ 3 / 2的情感“强度”被削弱
