import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from jaxtyping import Float, Int
from .base import Classifier, Tokenizer

class TransformersClassifier(Classifier):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        **model_args,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            **model_args,
        )
        
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

class TransformersTokenizer(Tokenizer):
    def __init__(self, model_name: str, **tokenizer_args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, truncation=True)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
