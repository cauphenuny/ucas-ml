# %%
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from app import dataloader
from app.utils import PROJECT_ROOT
from app.classifier import TransformerClassifier
from tinyllm.tokenize.tokenizer import Tokenizer
from tinyllm.network.models import specifications as model_specs
from tinyllm.network.multiplatform import ACCL_DEVICE

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default=ACCL_DEVICE)
args = parser.parse_args()

# %%
vocab_size = 5000
tokenizer_name = "movie-review"
tokenizer = Tokenizer.from_dir(
    PROJECT_ROOT / "ckpts" / "tokenizer" / f"{tokenizer_name}-{vocab_size}"
)

# %%
data_path = PROJECT_ROOT / "data" / "sentiment-analysis-on-movie-reviews"
train_path = data_path / "train.tsv"
test_path = data_path / "test.tsv"

# %%
batch_size = 32
data = pd.read_csv(train_path, sep="\t")
dataset = dataloader.Dataset(
    data, transform=dataloader.transform.to_tensor(tokenizer, device=args.device)
)
train, valid = dataset.split(test_size=0.2, random_state=42)
train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)
valid_dataloader = torch.utils.data.DataLoader(
    valid,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)

# %%
spec = model_specs("tiny")
num_classes = 5
model = TransformerClassifier(
    vocab_size=vocab_size,
    context_length=256,
    reduction="first",
    num_classes=num_classes,
    **spec,
)
model = model.to(args.device)

# %%
try:
    for epoch in range(args.epoch):
        print(f"Starting epoch {epoch+1}/{args.epoch}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        def validate():
            valid_loss = 0.0
            correct = 0
            counter = [0] * num_classes
            model.eval()
            for batch in tqdm(valid_dataloader, desc="Validating", leave=False):
                input_ids = batch["input_ids"]
                lengths = batch["lengths"]
                labels = batch["labels"]
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)
                valid_loss += loss.item()
                predict = logits.argmax(dim=-1)
                correct += (predict == labels).sum().item()
                for i in range(num_classes):
                    counter[i] += (predict == i).sum().item()
            valid_loss /= len(valid_dataloader)
            print(f"Validation Loss: {valid_loss:.4f}")
            print(f"Validation Accuracy: {correct / len(valid):.4f}")
            print(f"Distribution of predictions: {counter}")

        ema_loss = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for idx, batch in enumerate(train_dataloader):
                if idx % 1000 == 0:
                    validate()
                    model.train()
                input_ids = batch["input_ids"]
                lengths = batch["lengths"]
                labels = batch["labels"]
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema_loss = (
                    0.98 * ema_loss + 0.02 * loss.item()
                    if ema_loss != 0.0
                    else loss.item()
                )
                pbar.set_postfix({"loss": ema_loss})
                pbar.update(1)
except Exception:
    import traceback
    import pdb

    traceback.print_exc()
    pdb.post_mortem()
