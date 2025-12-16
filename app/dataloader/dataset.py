import os
import pandas as pd


class Dataset:
    def __init__(self, data_path: os.PathLike | str, sep: str = "\t"):
        self.data = pd.read_csv(data_path, sep=sep)

