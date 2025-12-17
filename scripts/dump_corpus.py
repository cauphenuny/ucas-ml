from app.dataloader import Dataset
from app.utils import PROJECT_ROOT
import pandas as pd

TEMP_DIR = PROJECT_ROOT / "output" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# %%
data_path = PROJECT_ROOT / "data" / "sentiment-analysis-on-movie-reviews"
train_path = data_path / "train.tsv"
test_path = data_path / "test.tsv"

# %%
if __name__ == "__main__":
    data = pd.read_csv(train_path, sep="\t")
    dataset = Dataset(data)
    dataset.dump_corpus(TEMP_DIR / "corpus.txt", sep="\n")