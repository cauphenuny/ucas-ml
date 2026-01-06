#import "@preview/cmarker:0.1.8"
#import "@preview/tablem:0.3.0": tablem
#import "@preview/mitex:0.2.6": mi, mitex
#import "@preview/theorion:0.4.1"

#let wrong(content) = text(fill: red, $cancel(#content)$)

== 深度神经网络

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

每个时间步在 $l - 1$ 层的 输出作为该位置在 $l$ 层的 输入

---

=== Transformer LM

#grid(
  columns: (1fr, 1fr),
)[
  #v(0.5em)
  ==== Transformer Block
  
  与原始的 Transformer 相比，我们实现的 transformer block 有以下改动：

  - Decoder Only，而不是transformer提出时的 Encoder - Decoder 结构
  - Pre-Norm，将norm从add之后提到attn/ffn之前
  - RoPE 位置编码
  - Normalization: RMSNorm
  - Activation: SwiGLU

  #theorion.note-box(title: "为什么用这些设置?")[
    经过调研，这些是当前主流LLM都在使用的结构
  ]

][
  #figure(image("assets/transformer_block.png", width: 13em), caption: "Decode Transformer Block")
]

---

#strong[*主要计算流程*]

设输入张量为 $x$，该 Block 的计算流程如下：

$
  x_"norm1" &= "Norm"(x) \
  "Attn"(x_"norm1") &= limits("Concat")_(h=1)^("num_heads")( "Softmax"( (f_"RoPE" (Q_h) f_"RoPE" (K_h)^(tack.b) + "CausalMask") / (sqrt(d_k)) ) V_h) \
  x'&=x + "Attn"(x_"norm1") \
  x_"norm2" &= "Norm"(x') \
  "FFN"(x_"norm2") &= sigma(x_"norm2" W_1 + b_1) W_2 + b_2 \
  "Output" &= x' + "FFN"(x_"norm2") \
$

与 LSTM 相比，它可以并行地处理整个序列，同时通过 Attention 机制捕捉长距离依赖关系。

---

#strong[*一些用到的组件*]

#theorion.note-box(title: "RMSNorm: Root-Mean Square Layer Normalization")[
  $
    "RMS"(x)=sqrt(1/d sum_(i=1)^d x_i^2 + epsilon) \
    "RMSNorm"(x)_i=x_i / "RMS"(x) dot gamma_i
  $
  与 LayerNorm 相比，仅保留缩放，减少计算开销，收敛更快 @RMSNorm
]

#theorion.note-box(title: "SwiGLU: Swish Gated Linear Unit")[
  $
    "SiLU"(x)= x dot sigma(x) \
    "SwiGLU"(x, W_1, W_2, W_3)= W_2 ("SiLU"(W_1 x)) dot.o W_3 x)
  $
  通过门控机制，模型可以动态控制哪些信息应该通过，增加了非线性表达的自由度。 @SiLU @GLU
]

---

#theorion.note-box(title: "RoPE: Rotary Position Embedding")[
  #theorion.quote-box[
    RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021
  ]
  一种位置编码，核心思想是将特征向量看作是 2D 平面上的复数，并对其进行旋转，从而编码相对距离 $m-n$。@RoPE
  
  二维的旋转：
  $
    f(x,m)=mat(delim: "(", cos m theta, -sin m theta; sin m theta, cos m theta)mat(x_1; x_2)
  $
  其中 $theta$ 是预定义的频率常数
  
  推广到高维：将 $d$ 维向量分成 $d slash 2$ 组，每组内部应用旋转
  
  注意力分数 $Q K^(tack.b)$ 是关于旋转不变的，只跟相对位置 $m-n$ 有关
  
  $
    chevron.l f_q (q,m),f_k (k,n) chevron.r = g(q, k, m-n)
  $
  
  所以加入 RoPE 之后，距离相同的两对 token 编码相同。
]

---

#grid(
  columns: (1.3fr, 1fr),
)[
  ==== 将 Transformer 用于分类任务
  - 去掉 Output Embedding 层 (lm head)，改为分类头 (classification head)
  - 具体地，从经过 Transformer 块之后的 token 序列中取出一个token，接入一个 MLP 分类器
    - 对于 Decoder-Only 架构，选择最后一个token
    - 对于 Encoder-Decoder 或者 Encoder-Decoder 架构，选择第一个token (`[CLS]`)
][
  #figure(image("assets/transformer_model.png", width: 15em), caption: "Transformer LM for Classification")
]

---

#grid(columns: (1fr, 1fr), gutter: 1em)[
  
  === 模型实现
  
  我们从零开始实现了整个模型和训练pipeline：\ tokenizer, model, dataloader, optimizer, trainer ...
  
  #text(size: 1.3em)[
    #wrong[
      `tiktoken, transformers`
    ]
    
    #wrong[
      `torch.nn.*`
    ]
    
    #wrong[
      `torch.optim.*`
    ]
    
    #text(olive)[
      `torch.nn.Parameter`
    ]
  ]
][
  
  #text(size: 0.8em)[
    ```
    .
    ├── tinyllm
    │   ├── cpp
    │   │   ├── bpe.hpp
    │   │   ├── export.cpp
    │   │   ├── include
    │   │   └── utils
    │   ├── network
    │   │   ├── functional.py
    │   │   ├── layers.py
    │   │   ├── models.py
    │   │   └── multiplatform.py
    │   ├── optimize
    │   │   ├── functional.py
    │   │   ├── lr_scheduler.py
    │   │   └── optimizers.py
    │   ├── tests
    │   │   ├── ...
    │   ├── tokenize
    │   │   ├── pretokenizer.py
    │   │   └── tokenizer.py
    │   └── train
    │       ├── checkpoint.py
    │       ├── dataset.py
    │       └── train.py
    └── ...
    ```
  ]
]

---

*Attention 和 RoPE 的实现：*

#grid(columns: (1fr, 1.5fr), gutter: 1em)[
  #figure(
    ```python
    def scaled_dot_product_attention(
        query: Float[Tensor, " ... len_q dim_k"],
        key: Float[Tensor, " ... len_k dim_k"],
        value: Float[Tensor, " ... len_k dim_v"],
        mask: Bool[Tensor, " ... len_q len_k"] | None = None,
    ) -> Float[Tensor, " ... len_q dim_v"]:
        scores = einops.einsum(query, key, " ... len_q dim_k, ... len_k dim_k -> ... len_q len_k")
        scores = scores / key.shape[-1] ** 0.5
        if mask is not None:
            scores.masked_fill_(~mask, float("-inf"))
        attn_value = softmax(scores, dim=-1)
        return einops.einsum(attn_value, value, " ... len_q len_k, ... len_k dim_v -> ... len_q dim_v")
    ```,
    caption: [Attention, 位置：`tinyllm/tinyllm/network/functional.py`],
  )
][
  #figure(
    ```python
    class RoPE(nn.Module):
        def _update_rotation(self, max_seq_len: int):
            positions = torch.arange(max_seq_len, dtype=torch.float32)
            angles = einops.einsum(positions, self.freqs, "max_seq_len, half_d_k -> max_seq_len half_d_k")
            rotate_x = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).flatten(start_dim=-2)
            rotate_y = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=-2)
            if self.device:
                rotate_x = rotate_x.to(self.device)
                rotate_y = rotate_y.to(self.device)
            self.register_buffer("rotate_x", rotate_x, persistent=False)
            self.register_buffer("rotate_y", rotate_y, persistent=False)
            self.max_seq_len = max_seq_len
    ```,
    caption: [RoPE, 位置：`tinyllm/tinyllm/network/layers.py`],
  )
]

#[
  #show raw.where(block: true): text.with(size: 0.9em)
  #text(size: 1em)[
    *Transformer Block 实现：* (`tinyllm/tinyllm/network/models.py`)
    ```python
    class TransformerBlock(Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int | None = None,
            rope_theta: float | None = None,
            rope_len: int | None = None,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
            norm_type: Literal["rms", "none"] = "rms",
            norm_location: Literal["pre", "post"] = "pre",
            ffn_activate: Literal["swiglu", "silu"] = "swiglu",
            causal: bool = True,
        ):
            super().__init__()
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type == "rms" else Identical()
            self.attn = MultiheadSelfAttention(
                ...
            )
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if norm_type == "rms" else Identical()
            self.ffn = FeedForward(d_model, d_ff, activate=ffn_activate, device=device, dtype=dtype)
            self.norm_location = norm_location
    
        def forward(
            self,
            x: Float[torch.Tensor, " ... seq_len d_model"],
            len: Int[torch.Tensor, " ..."] | None = None,
        ):
            if self.norm_location == "pre":
                x = x + self.attn(self.ln1(x), sequence_length=len)
                x = x + self.ffn(self.ln2(x))
            elif self.norm_location == "post":
                x = self.ln1(x + self.attn(x, sequence_length=len))
                x = self.ln2(x + self.ffn(x))
            else:
                raise NotImplementedError(f"unsupported norm_location: {self.norm_location}")
            return x
    ```
    
    *BPE Tokenizer 实现：* (`tinyllm/tinyllm/cpp/bpe.hpp`)
    ```cpp
    inline auto
    encode(const py::list& words, const py::list& merges, const py::dict& vocab, int num_threads, bool verbose = false)
        -> std::vector<int> {
        std::vector<int> token_ids;
        std::vector<std::vector<std::string>> words_vec;
        std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merges_rank;
        for (size_t rank = 0; const auto& item : merges) {
            py::tuple merge = item.cast<py::tuple>();
            std::string first = py::bytes(merge[0]).cast<std::string>(),
                        second = py::bytes(merge[1]).cast<std::string>();
            merges_rank[std::make_pair(first, second)] = rank++;
        }
        for (auto item : words) {
            py::tuple word = item.cast<py::tuple>();
            std::vector<std::string> word_tokens;
            for (auto token : word) {
                word_tokens.push_back(token.cast<std::string>());
            }
            words_vec.push_back(word_tokens);
        }
        transform(
            words_vec,
            [&merges_rank](std::vector<std::string> word) {
                std::vector<std::pair<std::pair<std::string, std::string>, int>> valid_pairs;
                do {
                    valid_pairs.clear();
                    for (size_t i = 0; i + 1 < word.size(); i++) {
                        auto pair = std::make_pair(word[i], word[i + 1]);
                        if (merges_rank.contains(pair))
                            valid_pairs.push_back(std::make_pair(pair, merges_rank.at(pair)));
                    }
                    if (!valid_pairs.size()) break;
                    std::sort(
                        valid_pairs.begin(), valid_pairs.end(),
                        [&merges_rank](const auto& a, const auto& b) -> bool {
                            return a.second < b.second;
                        });
                    word = merge_token(word, valid_pairs[0].first);
                } while (valid_pairs.size());
                return word;
            },
            num_threads, verbose, "BPE encoding");
        for (const auto& word : words_vec) {
            for (const auto& token : word) {
                py::bytes token_bytes = py::bytes(token);
                if (vocab.contains(token_bytes)) {
                    token_ids.push_back(vocab[token_bytes].cast<int>());
                } else {
                    throw std::runtime_error("Token not found in vocabulary: " + token);
                }
            }
        }
        return token_ids;
    }
    ```
  ]
]

---

此外，我们尝试了使用 transformer 库微调模型，作为对比

#figure(
```python
class TransformersClassifier(Classifier):
    def __init__(self, model_name, num_classes, **kwargs):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, **kwargs)
        
    def forward(self, x, len):
        if len is not None:
            seq_len = x.size(-1)
            attention_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0) < len.unsqueeze(-1)).to(x.dtype)
        else:
            attention_mask = (x != 0).to(x.dtype)

        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        return outputs.logits
```,
caption: `app/classifier/transformers.py`
)