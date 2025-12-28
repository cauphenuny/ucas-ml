#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import torch
import torch.nn.functional as F

from app import dataloader
from app.utils import PROJECT_ROOT
from app.classifier import TinyLLMClassifier, TransformersClassifier, TransformersTokenizer, Tokenizer as TokenizerProto
# tinyllm imports are delayed to runtime when needed (avoid requiring build)


def train_traditional(train_df: pd.DataFrame, model_path: Path | None, output_dir: Path):
    X = train_df["Phrase"].astype(str).tolist()
    y = train_df["Sentiment"].astype(int).tolist()
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), max_features=50000),
        LogisticRegression(max_iter=2000, multi_class="multinomial", solver="saga"),
    )
    pipeline.fit(X, y)
    if model_path is not None:
        joblib.dump(pipeline, model_path)
    return pipeline


def get_traditional_probs(pipeline, phrases, num_classes: int):
    probs = pipeline.predict_proba(phrases)
    # pipeline.classes_ gives class labels order, remap to 0..num_classes-1
    classes = list(pipeline.classes_)
    out = np.zeros((len(phrases), num_classes), dtype=float)
    for idx, cls in enumerate(classes):
        if 0 <= int(cls) < num_classes:
            out[:, int(cls)] = probs[:, idx]
    return out


def get_deep_probs(args, phrases, num_classes: int):
    device = args.device
    use_tinyllm = args.classifier == "tinyllm"
    if use_tinyllm:
        # Import tinyllm lazily to avoid requiring local build when using transformers backend
        from tinyllm.tokenize.tokenizer import Tokenizer as TinyLLMTokenizer
        from tinyllm.network.models import specifications as model_specs
        tokenizer = TinyLLMTokenizer.from_dir(PROJECT_ROOT / "ckpts" / "tokenizer" / f"{args.tokenizer}-{args.vocab_size}")
        spec = model_specs(args.model_size)
        spec["share_embeddings"] = False
        model = TinyLLMClassifier(
            vocab_size=args.vocab_size,
            context_length=256,
            num_classes=num_classes,
            causal=not args.no_causal,
            reduction=args.reduction,
            **spec,
        )
    else:
        tokenizer = TransformersTokenizer(args.hf_model)
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = TransformersClassifier(model_name=args.hf_model, num_classes=num_classes, pad_token_id=pad_token)

    # If a deep checkpoint path is provided, load it; otherwise use model default pretrained weights
    if args.deep_ckpt is not None:
        ckpt = torch.load(args.deep_ckpt, map_location="cpu")
        # support both raw state_dict and wrapped dicts like {"model": state_dict}
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    batch_size = 32
    all_probs = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        tokenized = [tokenizer.encode(p) for p in batch]
        lengths = [len(t) for t in tokenized]
        max_len = max(lengths) if lengths else 0
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.int64, device=device)
        for j, t in enumerate(tokenized):
            input_ids[j, :len(t)] = torch.tensor(t, dtype=torch.int64, device=device)
        with torch.no_grad():
            logits = model(input_ids, len=torch.tensor(lengths, device=device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--deep_ckpt", type=str, default=None, help="Path to deep model state_dict file")
    parser.add_argument("--classifier", type=str, choices=["tinyllm","transformers"], default="tinyllm")
    parser.add_argument("--model_size", type=str, default="tiny")
    parser.add_argument("--hf_model", type=str, default="siebert/sentiment-roberta-large-english")
    parser.add_argument("--tokenizer", type=str, default="movie-review")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--no_causal", action="store_true")
    parser.add_argument("--reduction", type=str, default="first")
    parser.add_argument("--weight_deep", type=float, default=0.7)
    parser.add_argument("--weight_trad", type=float, default=0.3)
    parser.add_argument("--quick_n", type=int, default=None, help="Quick run on first N test samples and subsample train")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save_trad", type=str, default=None, help="Path to save traditional model pipeline (joblib)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no deep checkpoint provided, try to auto-detect a checkpoint
    if args.deep_ckpt is None:
        ckpt_dir = PROJECT_ROOT / args.output_dir
        if ckpt_dir.exists():
            candidates = []
            for ext in ("*.pt", "*.pth", "*.ckpt"):
                candidates.extend(list(ckpt_dir.glob(ext)))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                args.deep_ckpt = str(latest)
                print(f"Auto-detected deep checkpoint: {args.deep_ckpt}")
            else:
                print(f"No checkpoint files found in {ckpt_dir}; proceeding without deep_ckpt (will use pretrained HF model if available).")
        else:
            print(f"Checkpoint directory {ckpt_dir} does not exist; proceeding without deep_ckpt (will use pretrained HF model if available).")

    data_path = PROJECT_ROOT / "data" / "sentiment-analysis-on-movie-reviews"
    train_path = data_path / "train.tsv"
    test_path = data_path / "test.tsv"

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t", dtype=str, na_filter=False)
    if args.quick_n is not None:
        test_df = test_df.head(args.quick_n)
        # subsample training data for speed
        train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
    phrases = test_df["Phrase"].astype(str).tolist()

    num_classes = 5

    print("Training traditional classifier (TF-IDF + LogisticRegression)...")
    trad_pipeline = train_traditional(train_df, Path(args.save_trad) if args.save_trad else None, output_dir)
    trad_probs = get_traditional_probs(trad_pipeline, phrases, num_classes)

    print("Computing deep model probabilities...")
    deep_probs = get_deep_probs(args, phrases, num_classes)

    # Normalize weights
    w_deep = float(args.weight_deep)
    w_trad = float(args.weight_trad)
    s = w_deep + w_trad
    w_deep /= s
    w_trad /= s

    combined = w_deep * deep_probs + w_trad * trad_probs
    preds = combined.argmax(axis=1)

    submission = pd.DataFrame({"PhraseId": test_df["PhraseId"].astype(int), "Sentiment": preds})
    submission.to_csv(output_dir / args.submit_file, index=False)
    print(f"Saved merged submission to {output_dir / args.submit_file}")


if __name__ == "__main__":
    main()
