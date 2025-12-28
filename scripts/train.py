# %%
import pandas as pd
import argparse
from loguru import logger
from torch.utils.data import DataLoader
from torch import load as torch_load
from torch import save as torch_save

# %%
from app import dataloader
from app.utils import PROJECT_ROOT
from app.classifier import (
    Classifier,
    TinyLLMClassifier,
    TransformersClassifier,
    LSTMClassifier,
)
from app.classifier import Tokenizer, TransformersTokenizer
from app.trainer import TrainingArgs, Trainer
from tinyllm.tokenize.tokenizer import Tokenizer as TinyLLMTokenizer
from tinyllm.network.models import specifications as model_specs
from tinyllm.network.multiplatform import ACCL_DEVICE
from tinyllm.network.functional import cross_entropy
from tinyllm.optimize.optimizers import AdamW
from tinyllm.optimize.lr_scheduler import CosineLRScheduler, ConstantLRScheduler

# %%
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default=ACCL_DEVICE)

    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--save_ckpt", type=str, default=None, help="Filename to save the model checkpoint")
    parser.add_argument("--save_best_only", action="store_true", help="Save only the best model checkpoint based on validation loss")
    parser.add_argument("--load_ckpt", type=str, default=None, help="Filename to load the model checkpoint")
    parser.add_argument("--valid_interval", type=int, default=1000, help="Validation interval in steps")

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
        "--submit_file", type=str, default=None, help="Filename for submission"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="UCAS ML 2025",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )

    subparsers = parser.add_subparsers(
        title="classifier", 
        dest="classifier", 
        help="Classifier Backend",
        required=True,
    )
    tinyllm_parser = subparsers.add_parser("tinyllm")
    transformers_parser = subparsers.add_parser("transformers")
    lstm_parser = subparsers.add_parser("lstm")

    tinyllm_parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Path to base model checkpoint to load",
    )
    tinyllm_parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="Freeze base model parameters during training",
    )
    tinyllm_parser.add_argument(
        "--release_steps",
        type=int,
        default=1000,
        help="Number of steps to release the frozen base model",
    )

    tinyllm_parser.add_argument(
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
    tinyllm_parser.add_argument(
        "--reduction",
        type=str,
        default="first",
        help="Reduction method: mean, first, last",
        choices=["mean", "first", "last"],
    )
    tinyllm_parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Do not use causal masking in the transformer",
    )

    tinyllm_parser.add_argument(
        "--tokenizer",
        type=str,
        default="movie-review",
        help="Tokenizer name",
    )
    tinyllm_parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="Tokenizer vocabulary size",
    )

    transformers_parser.add_argument(
        "--hf_model",
        type=str,
        default="siebert/sentiment-roberta-large-english",
        help="HF model name when using transformers classifier",
    )

    lstm_parser.add_argument(
        "--hf_model",
        type=str,
        default="siebert/sentiment-roberta-large-english",
        help="HF model name for tokenizer",
    )
    lstm_parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension for LSTM classifier",
    )
    lstm_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for LSTM classifier",
    )
    lstm_parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers for LSTM classifier",
    )
    lstm_parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for LSTM classifier",
    )
    lstm_parser.add_argument(
        "--reduction",
        type=str,
        default="last",
        help="Reduction method: mean, first, last",
        choices=["mean", "first", "last"],
    )

    args = parser.parse_args()
    return args

# %%
def main():
    # %%
    args = parse_args()

    # %%
    if args.classifier == "tinyllm":
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

    print(f"Parameters: {vars(args)}")

    # %%
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = PROJECT_ROOT / "ckpts" / "tokenizer"

    # %%
    if args.classifier == "tinyllm":
        tokenizer = TinyLLMTokenizer.from_dir(tokenizer_dir / f"{args.tokenizer}-{args.vocab_size}")
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
    if args.classifier == "tinyllm":
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
    elif args.classifier == "lstm":
        assert isinstance(tokenizer, TransformersTokenizer)
        model = LSTMClassifier(
            num_classes=num_classes,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            reduction=args.reduction,
        )
    else:
        assert isinstance(tokenizer, TransformersTokenizer)
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        print(f"Pad token ID: {pad_token}")
        model = TransformersClassifier(
            model_name=args.hf_model,
            num_classes=num_classes,
            pad_token_id=pad_token,
        )

    if args.classifier == "tinyllm" and args.base_model is not None:
        assert isinstance(model, TinyLLMClassifier)
        print(f"Loading base model from {args.base_model}...")
        model.load_base(args.base_model)
    model = model.to(args.device)
    criterion = cross_entropy

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = len(train_dataloader) * args.epoch
    lr_scheduler_kwargs = {
        "optimizer": optimizer,
        "total_steps": total_steps,
        "warmup_ratio": args.lr_warmup_ratio,
    }
    if args.lr_scheduler == "constant":
        lr_scheduler = ConstantLRScheduler(**lr_scheduler_kwargs)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = CosineLRScheduler(**lr_scheduler_kwargs)

    # %%
    after_step = None
    if args.classifier == "tinyllm" and args.freeze_base_model:
        assert isinstance(model, TinyLLMClassifier)
        
        model.freeze_base()
        
        def release_on_step(model: Classifier, step: int):
            if step == args.release_steps:
                assert isinstance(model, TinyLLMClassifier)
                print("Releasing base model parameters...")
                model.release_base()
        
        after_step = release_on_step

    training_args = TrainingArgs(
        num_classes=num_classes,
        output_dir=output_dir,
        epochs=args.epoch,
        warmup_ratio=args.lr_warmup_ratio,
        valid_interval=args.valid_interval,
        save_ckpt=args.save_ckpt,
        save_best_only=args.save_best_only,
        submit_file=args.submit_file,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        after_step=after_step,
        after_epoch=None,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_df=test,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        training_args=training_args,
    )

    # %%
    if args.load_ckpt is None:
        try:
            trainer.train()
        except Exception:
            import traceback
            import pdb

            traceback.print_exc()
            pdb.post_mortem()
    else:
        model.load_state_dict(torch_load(output_dir / args.load_ckpt))

    # %%
    print("Final evaluation on validation set:")
    trainer.validate()

    # %%
    if args.submit_file is not None:
        try:
            trainer.predict_test()
        except Exception:
            import traceback
            import pdb

            traceback.print_exc()
            pdb.post_mortem()

# %%
if __name__ == '__main__':
    main()
