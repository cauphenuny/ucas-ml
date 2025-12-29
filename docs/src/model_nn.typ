#import "@preview/cmarker:0.1.8"
#import "@preview/tablem:0.3.0": tablem
#import "@preview/mitex:0.2.6": mi, mitex

== Neural Network

=== LSTM

#grid(
  columns: (3fr, 1fr)
)[
  #align(center)[
    #tablem[
      | 步骤 | 公式 | 作用 |
      | :--- | :--- | :--- |
      | 1. 遗忘门 | #mi[`f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)`] | 决定丢弃多少旧信息 (0到1之间) |
      | 2. 输入门 | #mi[`i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)`] | 决定哪些新信息进入细胞状态 |
      | 3. 候选状态 | #mi[`\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)`] | 生成当前时刻的备选记忆 |
      | 4. 细胞更新 | #mi[`C_t = f_t  C_{t-1} + i_t  \tilde{C}_t`] | 更新长期记忆 (核心步骤) |
      | 5. 输出门 | #mi[`o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)`] | 决定输出哪些部分到下一时刻 |
      | 6. 最终输出 | #mi[`h_t = o_t * \tanh(C_t)`] | 生成当前时刻的隐藏状态 |
    ]
  ]
][
  #figure(image("assets/lstm.png"), caption: "LSTM Block")
]

Multilayer LSTM:

每个时间步在 $l − 1$ 层的 输出作为该位置在 $l$ 层的 输入

---

=== Decoder-Only Transformer LM

#grid(
  columns: (1fr, 1fr),
)[
  #v(0.5em)
  ==== Transformer Block
  
  采用与当前主流LLM相同的结构
  
  - Decoder Only, 只有自注意力层
  - Pre-Norm，增强训练稳定性
  - RoPE 位置编码
  - Normalization: RMSNorm
  - Activation: SwiGLU

][
  #figure(image("assets/transformer_block.png", width: 13em), caption: "Decode Transformer Block")
]

---

设输入张量为 $x$，该 Block 的计算流程如下：

#mitex(
  `
    x_{norm1} = \text{Norm}(x)
  `,
)

#mitex(
  `    \text{Attn}(x_{norm1}) = \text{Softmax}\left( \frac{f_{RoPE}(Q_h) f_{RoPE}(K_h)^T + \text{CausalMask}}{\sqrt{d_k}} \right) V_h`,
)

#mitex(`x' = x + \text{Attn}(x_{norm1})`)

#mitex(`x_{norm2} = \text{Norm}(x')`)

#mitex(`\text{FFN}(x_{norm2}) = \sigma(x_{norm2} W_1 + b_1) W_2 + b_2`)

#mitex(`\text{Output} = x' + \text{FFN}(x_{norm2})`)

与 LSTM 相比，它可以并行地处理整个序列，同时通过 Attention 机制捕捉长距离依赖关系。

---

- 去掉 Output Embedding 层
- 从经过 Transformer 块之后的 token 序列中取出一个token，接入一个 MLP 分类器

#figure(image("assets/transformer_model.png", width: 15em), caption: "Transformer LM for Classification")

---

=== Encoder-Only / Encoder-Decoder Transformer LM
