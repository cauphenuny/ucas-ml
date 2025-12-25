import torch
from jaxtyping import Float, Int
from abc import ABC, abstractmethod
from typing import Protocol

class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class Classifier(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        pass

    def predict(self, phrases: list[str], tokenizer: Tokenizer) -> list[int]:
        tokenized = [tokenizer.encode(phrase) for phrase in phrases]
        length = [len(t) for t in tokenized]
        max_len = max(length)
        device = next(self.parameters()).device
        input_ids = torch.zeros((len(phrases), max_len), dtype=torch.int64, device=device)
        for i, t in enumerate(tokenized):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.int64, device=device)
        logits = self.forward(input_ids, len=torch.tensor(length, device=device))
        predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()
