import torch
from tinyllm.tokenize.tokenizer import Tokenizer

def to_tensor(tokenizer: Tokenizer):
    def transform(text: str, label: int):
        tokenized = tokenizer.encode(text)
        input_ids = torch.tensor(tokenized, dtype=torch.int64)
        output_label = torch.tensor(label, dtype=torch.int64)
        return input_ids, output_label
    return transform

def collate_padding(batch: list[dict[str, torch.Tensor]]):
    input_ids = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.int64)
    max_len = max(len(ids) for ids in input_ids)
    lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.int64)
    padded_input_ids = torch.zeros((len(batch), max_len), dtype=torch.int64)
    for i, ids in enumerate(input_ids):
        length = len(ids)
        padded_input_ids[i, :length] = ids
    return {"input_ids": padded_input_ids, "lengths": lengths, "labels": labels}