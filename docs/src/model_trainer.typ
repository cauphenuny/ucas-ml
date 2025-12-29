#import "@preview/theorion:0.4.1"

#grid(
  columns: (0.75fr, 1fr),
)[
== 模型设计与实现
=== 整体架构概览
- 入口：`scripts/train.py` 负责参数解析、数据加载、模型选择、优化器/调度器/损失构建，随后交给 `Trainer` 统一训练与评估。
- 数据处理：`app/dataloader` 将 TSV 文本转为张量，完成分割、填充、对齐。
- 模型后端：三个可选模型，使用统一接口 `Classifier`，共享训练循环与推理接口。
  - LSTMClassifier
  - TinyLLMClassifier
  - TransformersClassifier

- 训练循环：`Trainer` 实现学习率调度、标签平滑、定期验证、W&B 记录、最优权重保存与提交文件生成。
][
  #figure(image("assets/trainer.svg", width: 25em), caption: "Architecture")
]

---
=== 数据与预处理

- 数据集：Kaggle Sentiment Analysis on Movie Reviews
  - `data/train.tsv` 含 PhraseId/SentenceId/Phrase/Sentiment
  - `data/test.tsv` 仅含文本
- 标签策略：按 SentenceId 取最长短语作为该句标签，确保划分时标签一致。
- 划分：8:2 训练/验证，SentenceId 分层抽样。

#figure(caption: `app/dataloader/dataset.py`)[
```python
def split(self, test_size: float = 0.2, random_state: int | None = None):
    df = self.data
    longest = (
        df.assign(_len=df["Phrase"].str.len())
        .sort_values(["SentenceId", "_len"], ascending=[True, False])
        .drop_duplicates("SentenceId")
    )
    train_ids, val_ids = train_test_split(
        sentence_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=sentence_labels,
    )
```
]

- 数据加载：入口脚本构建 Dataset → split → DataLoader（train 打乱，collate 负责 padding 与长度）。

#figure(caption: `scripts/train.py`)[
```python
dataset = dataloader.Dataset(pd.read_csv(train_path, sep="\t"),
    transform=dataloader.transform.to_tensor(tokenizer, device=args.device))
train, valid = dataset.split(test_size=0.2, random_state=42)
train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
    collate_fn=dataloader.transform.collate_padding(device=args.device))
```
]

---
=== Tokenizer

- TinyLLM 路线：从 `ckpts/tokenizer/`_`<name>-<vocab_size>`_ 自训 BPE，序列长度 256。
- HuggingFace 路线（Transformers/LSTM）：`TransformersTokenizer`，自动处理 pad/eos。

`app/dataloader/transform.py` 中包含两个用于统一数据格式的函数：
- 张量化：`to_tensor` 将文本编码为 `input_ids`
#figure(caption: `app/dataloader/transform.py`)[
```python
def to_tensor(tokenizer: Tokenizer, device: torch.device | str | None = ACCL_DEVICE):
    def transform(text: str, label: int):
        tokenized = tokenizer.encode(text)
        input_ids = torch.tensor(tokenized, dtype=torch.int64, device=device)
        output_label = torch.tensor(label, dtype=torch.int64, device=device)
        return input_ids, output_label
    return transform
```
]

- 对齐：`collate_padding` 统一 batch 长度并给出 `lengths/labels`
#figure(caption: `app/dataloader/transform.py`)[
```python
def collate_padding(device: torch.device | str | None = ACCL_DEVICE):
    def collate(batch: list[dict[str, torch.Tensor]]):
        input_ids = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.int64, device=device)
        max_len = max(len(ids) for ids in input_ids)
        lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.int64, device=device)
        padded_input_ids = torch.zeros((len(batch), max_len), dtype=torch.int64, device=device)
        for i, ids in enumerate(input_ids):
            padded_input_ids[i, : len(ids)] = ids
        return {"input_ids": padded_input_ids, "lengths": lengths, "labels": labels}
    return collate
```
]

---
=== 模型后端与接口

- 统一接口：`Classifier` 提供 `forward` 与 `predict`，`predict` 会负责 tokenizer 编码、padding 以及 argmax logits，确保三种后端一致的推理路径。

#figure(caption: `app/classifier/base.py`)[
```python
class Classifier(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        pass

    def predict(self, phrases: list[str], tokenizer: Tokenizer) -> list[int]:
        # ...
```
]

---
- TransformersClassifier：HF `AutoModelForSequenceClassification`，根据长度生成 attention mask，直接复用预训练语义。

#figure(caption: `app/classifier/transformers.py`)[
```python
class TransformersClassifier(Classifier):
    def __init__(self, model_name: str, num_classes: int, **model_args,):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True, **model_args,)

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        if len is not None:
            seq_len = x.size(-1)
            attention_mask = (torch.arange(seq_len, device=x.device).unsqueeze(0) < len.unsqueeze(-1)).to(x.dtype)
        else:
            attention_mask = (x != 0).to(x.dtype)
        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        return outputs.logits
```
]

---

- LSTMClassifier：Embedding + (Bi)LSTM，支持 `first/mean/last` 聚合后接 MLP，变长序列采用 pack/pad 提升效率。

#figure(caption: `app/classifier/lstm.py`)[
```python
packed_x = pack_padded_sequence(x, len.cpu(), batch_first=True, enforce_sorted=False)
lstm_out_packed, _ = self.lstm(packed_x)
lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
...  # reduction -> classifier MLP
```
]

- TinyLLMClassifier：tinyllm Transformer，可设非因果/因果掩码，`first/mean/last` 聚合后接轻量 MLP，支持加载/冻结/定步解冻底座。

#figure(caption: `app/classifier/tinyllm.py`)[
```python
self.model = models.TransformerModel(..., causal=causal)
def forward(self, x, len=None):
    x = self.model(x, len=len, lm_head=False)
    x = self.reduction(x, len)
    return self.classifier(x)
```
]

---
=== 训练配置与调度

- 选择后端模型：`--classifier {transformers,lstm,tinyllm}`
- 设置超参：`--epoch/--batch_size/--lr/--valid_interval/--output_dir`……
- 支持 checkpoint I/O 与 W&B。
- 优化器：tinyllm AdamW（`betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`）。
- 学习率：`constant` 或 `cosine` 调度，`warmup_ratio` 线性预热；若提供外部调度器，则 `_apply_warmup` 直接调用其 update。
- 损失：`cross_entropy`；若启用 `label_smoothing`，仅训练期套平滑，验证阶段始终使用原始损失函数。

=== 优化与正则策略

- Dropout：LSTM 堆叠层间（除最后层）与头部 MLP；Transformer 采用 HF/tinyllm 默认配置。
- Label smoothing：缓解过拟合与过度自信，验证阶段关闭以便真实评估。
- 权重衰减：0.01，抑制大权重。
- 冻结/解冻（TinyLLM）：可加载 base ckpt 后冻结底座，`release_steps` 自动解冻，兼顾稳定性与收敛速度。
- 验证与选优：每 `valid_interval` step 触发 `validate`，输出 loss/acc/分类报告/混淆矩阵，新低则保存 best（若开启）。

=== 训练-验证-测试流程

- 构建 DataLoader → 选择 tokenizer & 模型 → 创建优化器/调度器/损失。
- 训练循环（`Trainer.train`）：
  1. 周期性验证；
  2. 预热/调度 lr；
  3. 前向→loss→backward→step；
  4. 可选 step hook（如 TinyLLM 解冻）；
  5. W&B 日志。
- 验证：`Trainer.validate` 评估 loss/acc/分类报告/混淆矩阵，更新 best。
- 测试/提交：若提供 `--submit_file`，批量调用 `model.predict` 生成 CSV 提交文件。