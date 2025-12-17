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
        reduction: Literal["mean", "first", "last"] = "last",
        causal: bool = False,
        **model_args,
    ):
        super().__init__()
        self.model = models.TransformerModel(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            causal=causal,
            **model_args,
        )
        self.mlp = layers.Sequential(
            layers.Linear(d_model, d_model * 4),
            layers.SiLU(),
            layers.Linear(d_model * 4, num_classes),
        )
        if reduction == "first":
            self.reduction = lambda x, l: x[:, 0, :]
        elif reduction == "mean":

            def reduction_fn(
                x: Int[torch.Tensor, "batch seq_len d_model"],
                seq_len: Int[torch.Tensor, "batch"] | None,
            ):
                if seq_len is None:
                    return x.mean(dim=-2)
                else:
                    mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < seq_len.unsqueeze(1)
                    x = x * mask.unsqueeze(-1)
                    return x.sum(dim=-2) / seq_len.unsqueeze(-1)

            self.reduction = reduction_fn

        elif reduction == "last":

            def reduction_fn(
                x: Int[torch.Tensor, "batch seq_len d_model"],
                seq_len: Int[torch.Tensor, "batch"] | None,
            ):
                if seq_len is None:
                    return x[:, -1, :]
                batch_size = x.size(0)
                idx = (
                    (seq_len - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, x.size(2))
                )
                return x.gather(1, idx).squeeze(1)

            self.reduction = reduction_fn

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        x = self.model(x, len=len, lm_head=False)
        x = self.reduction(x, len)
        x = self.mlp(x)
        return x

    def predict(self, phrases: list[str], tokenizer: Tokenizer) -> list[int]:
        tokenized = [tokenizer.encode(phrase) for phrase in phrases]
        length = [len(t) for t in tokenized]
        max_len = max(length)
        device = next(self.parameters()).device
        input_ids = torch.zeros(
            (len(phrases), max_len), dtype=torch.int64, device=device
        )
        for i, t in enumerate(tokenized):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)
        logits = self.forward(input_ids, len=torch.tensor(length, device=device))
        predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()
