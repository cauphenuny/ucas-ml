from transformers import AutoTokenizer

class AutoTokenizerWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer(text)
        return tokens['input_ids']