from tinyllm.network import models, layers
from tinyllm.tokenize.tokenizer import Tokenizer
from jaxtyping import Float, Int
import torch
from typing import Literal


class TransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        reduction: Literal["mean", "first"] = "mean",
        **model_args,
    ):
        super().__init__()
        self.model = models.TransformerModel(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            **model_args,
        )
        self.mlp = layers.Sequential(
            layers.Linear(d_model, d_model * 4),
            layers.SiLU(),
            layers.Linear(d_model * 4, num_classes),
        )
        if reduction == "mean":
            self.reduction = lambda x: x.mean(dim=-2)
        elif reduction == "first":
            self.reduction = lambda x: x[:, 0, :]

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        x = self.model(x, len=len, lm_head=False)
        x = self.reduction(x)
        x = self.mlp(x)
        return x

    def predict(self, phrases: list[str], tokenizer: Tokenizer) -> list[int]:
        tokenized = [tokenizer.encode(phrase) for phrase in phrases]
        length = [len(t) for t in tokenized]
        max_len = max(length)
        input_ids = torch.zeros((len(phrases), max_len), dtype=torch.int64)
        for i, t in enumerate(tokenized):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.int64)
        logits = self.forward(input_ids, len=torch.tensor(length))
        predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()
