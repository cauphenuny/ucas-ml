from transformers import AutoTokenizer
from tinyllm.tokenize.tokenizer import Tokenizer
from typing import override

class AutoTokenizerWrapper(Tokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    @override
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer(text)
        return tokens['input_ids']