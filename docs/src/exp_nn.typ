#import "@preview/tablem:0.3.0": tablem

== LSTM 模型

我们训练了一个简单的 LSTM 模型：

#grid(columns: (1fr, 2.5fr), gutter: 1em, align: horizon)[

```python
LSTMClassifier(
    embedding_dim=128,
    hidden_dim=128,
    output_dim=5,
    n_layers=2,
    bidirectional=True,
    dropout=0.3,
)
```

][

#figure(image("assets/lstm-result.png", width: 100%))

]

最终 Accuracy 在 $0.58$ 左右

== Transformer 模型

=== 直接训练

我们首先尝试直接在本题数据集上从头开始训练一个 Transformer 模型

1. 模型大小实验

#grid(columns: (1fr, 1fr))[
  （参数）：
  #tablem[
    | 模型大小 | `num_layers` | `d_model` | `num_heads` |
    | --- | --- | --- | --- |
    | tiny | 4 | 256 | 8 |
    | small | 4 | 512 | 16 |
    | medium | 8 | 512 | 16 |
    | large | 12 | 768 | 16 |
    | x-large | 16 | 1024 | 16 |
  ]
][
  #image("assets/exp/size/size.png")
]

实验结果：tiny效果较差，small与medium接近。medium已经开始过拟合了，其余的没有测试

---

#grid(columns: (1.8fr, 1fr))[

  2. 规约方法方法实验：

  - last: 选择最后一个token的特征接到分类头
  - mean: 选择所有token的特征的平均值接到分类头

  可以看到，mean 方法明显比 last方法好，但模型越大，规约方法的影响越小

  可能是因为在模型层数以及参数量不够大的情况下，单独的一个token无法很好地融合整个句子的信息

][
  #for spec in ("tiny", "small", "medium") {
    grid(columns: 2, align: horizon)[
      #image("assets/exp/reduction/" + spec + ".png", width: 13em)
    ][
      (#spec)
    ]
  }
]

---

=== 两阶段训练

为了充分发挥大模型的性能，我们采用两阶段训练方法：

先在通用文本上预训练，然后在本题数据集上微调

具体地，我们在 OpenWebText (24.2GB) @owt / TinyStories (1GB) @tinystories 数据集上训练 next-token prediction 任务

加入预训练阶段后，模型性能有明显提升，尽管预训练的任务和数据集都与本题不完全匹配


#figure(image("assets/exp/pretrain.png", width: 50%))

---

试验一下 先冻结 base_model，只训练分类头，然后在中途解冻 的效果

#figure(image("assets/exp/freeze.png", width: 50%), caption: "freeze")

结果变差了，可能是由于模型自身有效训练时间变短了

---

使用更大的模型，更久的预训练：

#grid(columns: (1fr, 1fr))[
#figure(image("assets/exp/pretrain-loss.png", width: 90%), caption: "预训练 Loss 曲线")
][
#figure(image("assets/exp/x-large.png", width: 90%), caption: "微调")
]

最终在 x-large 模型上达到了最高 $0.66$ 的验证集准确率，提交结果 (test acc: 0.67854)：

#image("assets/kaggle/tinyllm.png")

---

== 微调 transformers 库提供的预训练模型

#figure(image("assets/exp/tinyllm_vs_gpt2.png", width: 50%), caption: "tinyllm vs gpt2")

在过拟合之前我们的tinyllm模型与 gpt2 不相上下，甚至正确率略好一些

---

=== 学习率调度实验

在微调大规模预训练 Transformers 模型时，学习率（Learning Rate）的设置不仅影响收敛速度，更决定了训练的成败。本实验针对 `RoBERTa-Large` 模型进行了详尽的超参数搜索。

==== 学习率过大导致的类别塌陷


#figure(image("assets/exp/lr/roberta.png", width: 50%), caption: "不同学习率结果")<demo-lr>
实验初期，我们尝试使用较高的学习率 $lr = 1 × 10^(-4)$。结果模型在约 1000 个 iteration 后，预测结果开始集中于类别 2，准确率降至 0.51 左右（等于类别 2 的占比），此后也一直保持在这一水平。

#figure(image("assets/exp/lr/collapse.png", width: 50%), caption: "类别塌陷")

原因在于：
1. `RoBERTa` 拥有 $125"M" ~ 3.5 "B"$ 参数，其预训练权重已处于高度优化的极小值区域。过大的更新步长会产生灾难性遗忘，破坏模型已掌握的语义提取能力。
2. 由于随机初始化的分类头在初期会产生巨大的梯度，配合高学习率，这股冲击力会直接传导至底层权重，导致模型为了快速降低 Loss 而选择“全输出单一高频标签”的局部最优解。
3. 值得注意的是（见@demo-lr） $5 × 10^(-5)$ 这一在 Base 模型上常用的学习率，在训练后期（约 5000 step 后）也出现了准确率急剧下滑的崩溃现象，进一步证明了 Large 模型对高学习率的容忍度极低。

---

RoBERTa 和 BERT 的论文中都给出了研究者使用的超参数设置。

#grid(columns: (1fr, 1fr))[
#figure(image("assets/exp/lr/roberta-essay.png", width: 100%), caption: [RoBERTa 论文给出的超参数设置 @roberta])
][
#figure(image("assets/exp/lr/bert-essay.png", width: 60%), caption: [BERT 论文给出的超参数设置建议 @bert])
]

---

==== 学习率调度策略

遵循论文中的设置，为了平衡训练早期的稳定性和后期的精准度，我们引入了带预热（Warmup）的调度机制。

1. *线性预热*：
  在前 6% 的步数中，学习率从 0 线性增加至设定值（如 $2 × 10^(-5)$）。这为随机初始化的分类头提供了“缓冲期”，避免初始梯度冲击破坏底座权重。

2. *余弦衰减*：
  预热结束后，学习率按余弦曲线缓慢下降。相比恒定学习率（Constant），余弦衰减在训练后期能以更细微的步长进行参数微调，有效提高了最终的验证集准确率。

通过对比实验，我们确定了适用于本任务的超参数空间：

#tablem[
| 学习率             | 调度策略      | 训练表现             | 验证集峰值精度 | 稳定性 |
| ------------------ | ------------- | -------------------- | -------------- | ------ |
| $1 × 10^(-4)$ | Warmup+Const  | 极速崩溃（1k step）  | 0.51           | 极差   |
| $5 × 10^(-5)$ | Warmup+Const  | 中期崩溃（4k step）  | 0.64           | 不稳定 |
| $2 × 10^(-5)$ | Warmup+Const  | 缓慢上升，略有震荡   | 0.67           | 良好   |
| $2 × 10^(-5)$ | Warmup+Cosine | *平滑收敛，泛化最优* | *0.69*         | *极佳* |
| $5 × 10^(-6)$ | Warmup+Cosine | 极其稳定但收敛缓慢   | 0.68           | 极佳   |
| ]

---

#grid(columns: (1fr, 1.5fr), align: horizon)[
  #figure(image("assets/exp/lr/constant.png", width: 100%))
  #figure(image("assets/exp/lr/cosine.png", width: 100%))
][
  #image("assets/exp/lr/cosine-result.png", width: 100%)
]

---


// TODO:

/*
- *“退而求其次”原则*：在微调 `RoBERTa-Large` 或类似的已经过微调的模型（如 `siebert/sentiment-roberta-large-english`）时，$2 × 10^(-5)$ 是一个坚固的上限。若追求极致稳定，建议降低至 $1 × 10^(-5)$。
- *调度器的复利效应*：Warmup 解决了“能不能跑通”的问题，而 Cosine Decay 解决了“能不能跑好”的问题。
- *架构差异*：对于从零训练的 `TinyLLM`，学习率设定在 $2 × 10^(-3)$（比微调大 100 倍），这反映了模型在“寻找特征”与“微调特征”阶段对能量需求的本质不同。
*/

最终最好的结果 (test acc: 0.71186)：

#image("assets/kaggle/roberta.png")