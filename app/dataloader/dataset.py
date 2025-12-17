import os
import pandas as pd
from collections.abc import Iterable


class Dataset:
    def __init__(self, data_path: os.PathLike | str, sep: str = "\t"):
        # NOTE: fields: PhraseId, SentenceId, Phrase, Sentiment
        self.data = pd.read_csv(data_path, sep=sep)

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
