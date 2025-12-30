#import "@preview/tablem:0.3.0": tablem

== 训练集数据分析

=== 数据集格式

#grid(columns: (1fr, 1fr), align: bottom)[
数据集：#link("https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview", "Kaggle Sentiment Analysis on Movie Reviews")
- `train.tsv` 含 PhraseId/SentenceId/Phrase/Sentiment
  - Sentiment：影评的情感倾向（0\~4，负面\~正面）
- `test.tsv` 类似，但不含 Sentiment
][
- 源自 Stanford Sentiment Treebank (SST) 的研究
- 每个句子的所有解析子集都对应一个标签
- 细粒度情感分析
]

#grid(columns: (1fr, 1fr), align: center)[
  #figure(caption: "训练集数据片段")[
  #tablem[
  | PhraseId | SentenceId | Phrase | Sentiment |
  | --- | --- | --- | --- |
  | 83635 | 4323 | An exhilarating serving of movie fluff . | 2 |
  | 83636 | 4323 | An exhilarating serving of movie fluff | 3 |
  | 83637 | 4323 | An exhilarating | 3 |
  | 83638 | 4323 | serving of movie fluff | 2 |
  | 83639 | 4323 | of movie fluff | 1 |
  | 83640 | 4323 | movie fluff | 2 |
  ]]
][
  #figure(caption: "Stanford Sentiment Treebank")[#image("./assets/sst.png", width: 90%)]
]


---

=== 缺失与异常值处理

经检查，不存在缺失/异常值

=== 标签分布

#grid(columns: (1fr, 1fr), align: horizon)[
  #figure(
    tablem[
      | 情感标签 | 含义            | 样本数        | 占比（约）     |
      | ---- | ------------- | ---------- | --------- |
      | 0    | Negative | 7,072      | 4.5%      |
      | 1    | Somewhat Negative      | 27,273     | 17.4%     |
      | 2    | Neutral       | 79,582     | 50.8%     |
      | 3    | Somewhat Positive      | 32,927     | 21.0%     |
      | 4    | Positive | 9,206      | 5.9%      |
    ],
    caption: "训练集情感分布统计",
  )][
  #figure(image("assets/dataset/dist.png", width: 70%), caption: "分布可视化")
]


#align(center)[
#block(fill: luma(240), inset: 8pt, radius: 4pt)[
  _这是一个以 Neutral 为主导、极端情感样本稀缺的严重不均衡多分类数据集。_
]
]

---

==== 数据明显「类别不平衡」
- Neutral（2）$approx$ *占一半*
- Negative / Positive < *6%*
- 多数类样本 $approx$ 少数类样本的 *10 倍以上*

#block(stroke: (left: 4pt + gray), inset: (left: 1em))[
  模型在训练时会天然偏向预测 Neutral。
]

==== 符合真实世界分布

大多数评论是中性或轻微情感
极端情绪本来就少
数据分布符合真实用户行为，而非人工平衡数据。

==== 在这个数据集上，Accuracy 指标意义有限

如果一个模型 *什么都不学*：
永远预测 Sentiment = 2
它的 accuracy 就是： $≈ 50.8%$

这意味着：55% 以下 $->$ 模型几乎没学，60% $->$ 只是比瞎猜稍好

==== 极端情感（0 / 4）是最难学的

Negative + Positive =10.4%

但它们在实际应用中最重要

模型很容易把：
0 $->$ 1 / 2、4 $->$ 3 / 2的情感“强度”被削弱
