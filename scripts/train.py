# %%
import wandb
import pandas as pd
import argparse
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch import load as torch_load
from torch import save as torch_save

# %%
from app import dataloader
from app.utils import PROJECT_ROOT
from app.classifier import Classifier, TinyLLMClassifier, TransformersClassifier
from app.classifier import Tokenizer, TransformersTokenizer
from tinyllm.tokenize.tokenizer import Tokenizer as TinyLLMTokenizer
from tinyllm.network.models import specifications as model_specs
from tinyllm.network.multiplatform import ACCL_DEVICE
from tinyllm.network.functional import cross_entropy
from tinyllm.optimize.optimizers import AdamW
from tinyllm.optimize.lr_scheduler import CosineLRScheduler

# %%
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default=ACCL_DEVICE)

parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument(
    "--classifier",
    type=str,
    default="tinyllm",
    choices=["tinyllm", "transformers"],
    help="Choose classifier backend",
)
parser.add_argument(
    "--hf_model",
    type=str,
    default="siebert/sentiment-roberta-large-english",
    help="HF model name when using transformers classifier",
)

parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--save_ckpt", type=str, default=None, help="Filename to save the model checkpoint")
parser.add_argument("--save_best_only", action="store_true", help="Save only the best model checkpoint based on validation loss")
parser.add_argument("--load_ckpt", type=str, default=None, help="Filename to load the model checkpoint")

parser.add_argument("--base_model", type=str, default=None, help="Path to base model checkpoint to load")
parser.add_argument("--freeze_base_model", action="store_true", help="Freeze base model parameters during training")
parser.add_argument(
    "--release_steps",
    type=int,
    default=1000,
    help="Number of steps to release the frozen base model",
)

parser.add_argument(
    "--model_size",
    type=str,
    default="tiny",
    help="Model size: nano, micro, tiny, small, medium, large, x-large, xx-large, 3x-large",
    choices=[
        "nano",
        "micro",
        "tiny",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
        "3x-large",
    ],
)
parser.add_argument(
    "--reduction",
    type=str,
    default="first",
    help="Reduction method: mean, first, last",
    choices=["mean", "first", "last"],
)
parser.add_argument(
    "--no-causal",
    action="store_true",
    help="Do not use causal masking in the transformer",
)

parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="constant",
    help="Learning rate scheduler: none, cosine",
    choices=["constant", "cosine"],
)
parser.add_argument(
    "--lr_warmup_ratio",
    type=float,
    default=0.06,
    help="Learning rate warmup ratio",
)

parser.add_argument(
    "--tokenizer",
    type=str,
    default="movie-review",
    help="Tokenizer name",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    default=5000,
    help="Tokenizer vocabulary size",
)

parser.add_argument(
    "--submit_file", type=str, default=None, help="Filename for submission"
)

parser.add_argument(
    "--wandb_project", type=str, default="UCAS ML 2025", help="Weights & Biases project name"
)
parser.add_argument(
    "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
)

args = parser.parse_args()

# %%
if args.wandb_run_name:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

# %%
use_tinyllm = args.classifier == "tinyllm"

if use_tinyllm:
    if args.reduction in ("first", "mean"):
        if not args.no_causal:
            args.no_causal = True
            logger.warning(
                f"Causal masking is only compatible with 'last' reduction, current reduction: '{args.reduction}'")
            logger.warning("Auto set --no-causal to disable causal masking")
        if args.base_model is not None:
            logger.warning("Base model loading is only compatible with 'last' reduction, ignoring --base_model")
            args.base_model = None
    if args.freeze_base_model and args.base_model is None:
        logger.warning("Freezing base model is only compatible with a loaded base model, ignoring --freeze_base_model")
        args.freeze_base_model = False
else:
    if args.base_model is not None:
        logger.warning("--base_model is only supported for tinyllm classifier, ignoring")
        args.base_model = None
    if args.freeze_base_model:
        logger.warning("--freeze_base_model is only supported for tinyllm classifier, ignoring")
        args.freeze_base_model = False

print(f"Parameters: {vars(args)}")

# %%
output_dir = PROJECT_ROOT / args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

# %%
if use_tinyllm:
    tokenizer = TinyLLMTokenizer.from_dir(
        PROJECT_ROOT / "ckpts" / "tokenizer" / f"{args.tokenizer}-{args.vocab_size}"
    )
else:
    tokenizer = TransformersTokenizer(args.hf_model)

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
train_dataloader = DataLoader(
    train,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)
valid_dataloader = DataLoader(
    valid,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=dataloader.transform.collate_padding(device=args.device),
)
test = pd.read_csv(test_path, sep="\t", dtype=str, na_filter=False)

# %%
num_classes = 5
if use_tinyllm:
    spec = model_specs(args.model_size)
    spec["share_embeddings"] = False
    model: Classifier = TinyLLMClassifier(
        vocab_size=args.vocab_size,
        context_length=256,
        num_classes=num_classes,
        causal=not args.no_causal,
        reduction=args.reduction,
        **spec,
    )
else:
    model = TransformersClassifier(
        model_name=args.hf_model,
        num_classes=num_classes,
    )
print("Model architecture:")
print(model)
if use_tinyllm and args.base_model is not None:
    print(f"Loading base model from {args.base_model}...")
    base_state_dict = torch_load(args.base_model, map_location="cpu")
    model.model.load_state_dict(base_state_dict, strict=False)
model = model.to(args.device)
criterion = cross_entropy

optimizer = AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler_kwargs = dict(
    optimizer=optimizer,
    total_steps=len(train_dataloader) * args.epoch,
    warmup_ratio=args.lr_warmup_ratio,
)

if args.lr_scheduler == "constant":
    lr_scheduler = ConstantLRScheduler(**lr_scheduler_kwargs)
elif args.lr_scheduler == "cosine":
    lr_scheduler = CosineLRScheduler(**lr_scheduler_kwargs)

# %%
best_valid_loss = float("inf")
def validate(global_step):
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
    global best_valid_loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        if args.save_ckpt is not None and args.save_best_only:
            print(f"New best model found, saving checkpoint to {args.save_ckpt}...")
            torch_save(model.state_dict(), output_dir / args.save_ckpt)
    if args.wandb_run_name:
        wandb.log({
            "valid/loss": valid_loss,
            "valid/accuracy": correct / len(valid),
        }, step=global_step)
    return valid_loss


# %%
def freeze(model: TinyLLMClassifier):
    for param in model.model.parameters():
        param.requires_grad = False


def release(model: TinyLLMClassifier):
    for param in model.model.parameters():
        param.requires_grad = True


# %%
global_step = 0

if args.load_ckpt is None:
    try:
        if args.freeze_base_model:
            assert isinstance(model, TinyLLMClassifier)
            freeze(model)

        print(f"Training on device: {args.device}")
        for epoch in range(args.epoch):
            print(f"Starting epoch {epoch + 1}/{args.epoch}...")
            model.train()

            ema_loss = 0.0
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}") as pbar:
                for idx, batch in enumerate(train_dataloader):
                    if idx % 1000 == 0:
                        validate(global_step)
                        model.train()
                    input_ids = batch["input_ids"]
                    lengths = batch["lengths"]
                    labels = batch["labels"]
                    logits = model(input_ids, lengths)
                    loss = criterion(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ema_loss = 0.98 * ema_loss + 0.02 * loss.item() if ema_loss != 0.0 else loss.item()
                    avg_lr = sum(group["lr"] for group in optimizer.param_groups) / len(optimizer.param_groups)
                    pbar.set_postfix({"loss": f"{ema_loss:.3f}", "lr": f"{avg_lr:.2e}"})
                    pbar.update(1)
                    global_step += 1
                    if lr_scheduler:
                        lr_scheduler.update(global_step)
                    if global_step == args.release_steps and args.freeze_base_model:
                        assert isinstance(model, TinyLLMClassifier)
                        print("Releasing base model parameters...")
                        release(model)
                    if args.wandb_run_name:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": avg_lr,
                        }, step=global_step)
    except Exception:
        import traceback
        import pdb

        traceback.print_exc()
        pdb.post_mortem()

    if args.save_ckpt is not None and not args.save_best_only:
        torch_save(model.state_dict(), output_dir / args.save_ckpt)
else:
    model.load_state_dict(torch_load(output_dir / args.load_ckpt))

print("Final evaluation on validation set:")
validate(global_step)

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
