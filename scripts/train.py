# %%
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

# %%
from app import dataloader
from app.utils import PROJECT_ROOT
from app.classifier import TransformerClassifier
from tinyllm.tokenize.tokenizer import Tokenizer
from tinyllm.network.models import specifications as model_specs
from tinyllm.network.multiplatform import ACCL_DEVICE

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default=ACCL_DEVICE)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument(
    "--save_ckpt", type=str, default=None, help="Pathname to save the model checkpoint"
)
parser.add_argument(
    "--load_ckpt", type=str, default=None, help="Pathname to load the model checkpoint"
)
parser.add_argument(
    "--submit_file", type=str, default=None, help="Filename for submission"
)
parser.add_argument(
    "--reduction",
    type=str,
    default="last",
    help="Reduction method: mean, first, last",
    choices=["mean", "first", "last"],
)
parser.add_argument(
    "--no-causal",
    action="store_true",
    help="Do not use causal masking in the transformer",
)
args = parser.parse_args()

# %%
if args.reduction in ("first", "mean"):
    if not args.no_causal:
        args.no_causal = True
        logger.warning(
            "Causal masking is only compatible with 'last' reduction, auto set --no-causal to disable causal masking"
        )

# %%
output_dir = PROJECT_ROOT / args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

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
dataset = dataloader.Dataset(
    pd.read_csv(train_path, sep="\t"),
    transform=dataloader.transform.to_tensor(tokenizer, device=args.device),
)
train, valid = dataset.split(test_size=0.2, random_state=42)
train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)
valid_dataloader = torch.utils.data.DataLoader(
    valid,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)
test = pd.read_csv(test_path, sep="\t", dtype=str, na_filter=False)

# %%
spec = model_specs("tiny")
num_classes = 5
model = TransformerClassifier(
    vocab_size=vocab_size,
    context_length=256,
    num_classes=num_classes,
    causal=not args.no_causal,
    reduction=args.reduction,
    **spec,
)
print("Model architecture:")
print(model)
model = model.to(args.device)
criterion = torch.nn.CrossEntropyLoss()


# %%
def validate():
    valid_loss = 0.0
    correct = 0
    counter = [0] * num_classes
    all_labels = []
    all_predicts = []
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
        all_labels.extend(labels.cpu().tolist())
        all_predicts.extend(predict.cpu().tolist())
    valid_loss /= len(valid_dataloader)
    print(f"Validation Loss: {valid_loss:.4f}")
    print(f"Validation Accuracy: {correct / len(valid):.4f}")
    print(f"Distribution of predictions: {counter}")
    print("Classification Report:")
    print(classification_report(all_labels, all_predicts))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predicts))


# %%
if args.load_ckpt is None:
    try:
        print(f"Training on device: {args.device}")
        for epoch in range(args.epoch):
            print(f"Starting epoch {epoch+1}/{args.epoch}...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            model.train()

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

    if args.save_ckpt is not None:
        torch.save(model.state_dict(), output_dir / args.save_ckpt)
else:
    model.load_state_dict(torch.load(output_dir / args.load_ckpt))

# %%
print("Final evaluation on validation set:")
validate()

# %%
if args.submit_file is not None:
    print("Predicting on test set...")
    phrases = test["Phrase"].tolist()
    test_batchsize = 32

    try:
        for i in tqdm(range(0, len(phrases), test_batchsize), desc="Testing"):
            batch_start = i
            batch_end = min(i + test_batchsize, len(phrases))
            batch_phrases = phrases[batch_start:batch_end]
            if i == 0:
                predictions = model.predict(batch_phrases, tokenizer)
            else:
                batch_predictions = model.predict(batch_phrases, tokenizer)
                predictions.extend(batch_predictions)
    except Exception:
        import traceback
        import pdb

        traceback.print_exc()
        pdb.post_mortem()

    submission = pd.DataFrame({"PhraseId": test["PhraseId"], "Sentiment": predictions})
    submission.to_csv(PROJECT_ROOT / "output" / args.submit_file, index=False)
