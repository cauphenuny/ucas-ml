#import "@preview/tablem:0.3.0": tablem

== LSTM 模型

#image("assets/lstm-result.png")

最终 Accuracy 在 $0.58$ 左右

== Transformer 模型

=== 直接训练

模型大小实验

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

  规约方法方法实验：

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

为了充分发挥大模型的性能，可以采用两阶段训练方法：

先在通用文本上预训练，然后在本题数据集上微调

具体地，我们在 OpenWebText (24.2GB) / TinyStories (1GB) 数据集上训练 next-token prediction 任务

加入预训练阶段后，模型性能有明显提升，尽管预训练的任务和数据集都与本题不完全匹配


#figure(image("assets/exp/pretrain.png", width: 50%))

---

使用更大的模型，更久的预训练：

#figure(image("assets/exp/x-large.png", width: 50%))

最终在 x-large 模型上达到了最高 $0.66$ 的验证集准确率，提交结果 (test acc: 0.67854)：

#image("assets/kaggle/tinyllm.png")

---

== 微调 transformers 库提供的预训练模型

#figure(image("assets/exp/tinyllm_vs_gpt2.png", width: 50%))

在过拟合之前我们的tinyllm模型与 gpt2 不相上下，甚至正确率略好一些

---

=== 学习率调度实验

// TODO
