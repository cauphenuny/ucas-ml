import os
import pandas as pd
from collections.abc import Iterable
from typing import Any, Callable
import torch
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transform: Callable[[str, int], Any] | None = None,
    ):
        # NOTE: fields: PhraseId, SentenceId, Phrase, Sentiment
        self.data = data
        self.transform = transform

    def phrases(self) -> Iterable[str]:
        yield from self.data["Phrase"]

    def sentences(self) -> Iterable[str]:
        # Group by SentenceId and select the longest Phrase (full sentence) per group
        df = self.data
        longest = (
            df.assign(_len=df["Phrase"].str.len())
            .sort_values(["SentenceId", "_len"], ascending=[True, False])
            .drop_duplicates("SentenceId")
        )
        yield from longest["Phrase"]

    def split(self, test_size: float = 0.2, random_state: int | None = None):
        df = self.data
        # 取每个 SentenceId 的最长 Phrase 的标签作为句子标签
        longest = (
            df.assign(_len=df["Phrase"].str.len())
            .sort_values(["SentenceId", "_len"], ascending=[True, False])
            .drop_duplicates("SentenceId")
        )
        sentence_ids = longest["SentenceId"].to_numpy()
        sentence_labels = longest["Sentiment"].to_numpy()

        train_ids, val_ids = train_test_split(
            sentence_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=sentence_labels,
        )

        train_df = df[df["SentenceId"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["SentenceId"].isin(val_ids)].reset_index(drop=True)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        return Dataset(train_df, transform=self.transform), Dataset(val_df, transform=self.transform)


    def dump_corpus(self, output_path: os.PathLike | str, sep: str = "<|endoftext|>"):
        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in self.sentences():
                f.write(sentence + sep)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: slice | int):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        row = self.data.iloc[idx]
        text = row["Phrase"]
        label = int(row["Sentiment"])
        if self.transform:
            text, label = self.transform(text, label)
        return {"text": text, "label": label}
