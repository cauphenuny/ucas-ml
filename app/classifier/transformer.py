from tinyllm.network import models, layers
from jaxtyping import Float
import torch
from typing import Literal


class TransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        reduction: Literal["mean", "first"] = "mean",
    ):
        self.model = models.TransformerModel(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
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

    def forward(self, x: Float[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... num_classes"]:
        x = self.model(x, lm_head=False)
        x = self.reduction(x)
        x = self.mlp(x)
        return x
