import argparse
from pathlib import Path
from typing import Literal

import torch
import pandas as pd
import gradio as gr
from gradio.themes import Soft

from app.utils import PROJECT_ROOT
from app.classifier import (
    Classifier,
    TinyLLMClassifier,
    TransformersClassifier,
    LSTMClassifier,
    TransformersTokenizer,
)
from tinyllm.tokenize.tokenizer import Tokenizer as TinyLLMTokenizer
from tinyllm.network.models import specifications as model_specs
from tinyllm.network.multiplatform import ACCL_DEVICE


NUM_CLASSES = 5

# Fix color palette so class 0 (negative) starts red and transitions to green by class 4
CLASS_COLOR_MAP = {
    "0": "#d73027",
    "1": "#fc8d59",
    "2": "#fee08b",
    "3": "#d9ef8b",
    "4": "#1a9850",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio WebUI for sentiment classifier"
    )

    parser.add_argument("--device", type=str, default=ACCL_DEVICE)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--load_ckpt",
        type=str,
        required=True,
        help="Checkpoint filename under output_dir to load",
    )

    subparsers = parser.add_subparsers(
        title="classifier",
        dest="classifier",
        help="Classifier Backend",
        required=True,
    )

    # TinyLLM backend
    tinyllm_parser = subparsers.add_parser("tinyllm")
    tinyllm_parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
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
        help="TinyLLM model size (must match training)",
    )
    tinyllm_parser.add_argument(
        "--reduction",
        type=str,
        default="last",
        choices=["mean", "first", "last"],
        help="Sequence reduction strategy (must match training)",
    )
    tinyllm_parser.add_argument(
        "--no_causal",
        action="store_true",
        help="Disable causal masking (must match training)",
    )
    tinyllm_parser.add_argument(
        "--tokenizer",
        type=str,
        default="movie-review",
        help="TinyLLM tokenizer name (directory prefix under ckpts/tokenizer)",
    )
    tinyllm_parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="TinyLLM tokenizer vocab size (must match training)",
    )
    tinyllm_parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate used in TinyLLM classifier",
    )

    # Transformers backend
    transformers_parser = subparsers.add_parser("transformers")
    transformers_parser.add_argument(
        "--hf_model",
        type=str,
        default="siebert/sentiment-roberta-large-english",
        help="HuggingFace model name (must match training if loading finetuned checkpoint)",
    )

    # LSTM backend
    lstm_parser = subparsers.add_parser("lstm")
    lstm_parser.add_argument(
        "--hf_model",
        type=str,
        default="siebert/sentiment-roberta-large-english",
        help="HF model name used for tokenizer during training",
    )
    lstm_parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension (must match training)",
    )
    lstm_parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension (must match training)",
    )
    lstm_parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers (must match training)",
    )
    lstm_parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (must match training)",
    )
    lstm_parser.add_argument(
        "--reduction",
        type=str,
        default="last",
        choices=["mean", "first", "last"],
        help="Sequence reduction strategy (must match training)",
    )

    return parser.parse_args()


def build_tokenizer(args) -> TinyLLMTokenizer | TransformersTokenizer:
    if args.classifier == "tinyllm":
        tokenizer_dir = PROJECT_ROOT / "ckpts" / "tokenizer"
        tokenizer_path = tokenizer_dir / f"{args.tokenizer}-{args.vocab_size}"
        return TinyLLMTokenizer.from_dir(tokenizer_path)
    else:
        return TransformersTokenizer(args.hf_model)


def build_model(
    args,
    tokenizer: TinyLLMTokenizer | TransformersTokenizer,
) -> Classifier:
    device = torch.device(args.device)

    if args.classifier == "tinyllm":
        spec = model_specs(args.model_size)
        spec["share_embeddings"] = False
        model: Classifier = TinyLLMClassifier(
            vocab_size=args.vocab_size,
            context_length=256,
            num_classes=NUM_CLASSES,
            causal=not args.no_causal,
            reduction=args.reduction,
            dropout=args.dropout,
            **spec,
        )
    elif args.classifier == "lstm":
        assert isinstance(tokenizer, TransformersTokenizer)
        model = LSTMClassifier(
            num_classes=NUM_CLASSES,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            reduction=args.reduction,
        )
    else:
        assert isinstance(tokenizer, TransformersTokenizer)
        pad_token = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        print(f"Pad token ID: {pad_token}")
        model = TransformersClassifier(
            model_name=args.hf_model,
            num_classes=NUM_CLASSES,
            pad_token_id=pad_token,
        )

    ckpt_path = PROJECT_ROOT / args.output_dir / args.load_ckpt
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def make_predict_fn(model: Classifier, tokenizer):
    @torch.inference_mode()
    def predict(text: str):
        if not text.strip():
            return pd.DataFrame(
                [{"class": str(i), "prob": 0.0} for i in range(NUM_CLASSES)]
            )

        tokens = tokenizer.encode(text)
        device = next(model.parameters()).device
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)

        logits = model(input_ids, len=lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        # 固定类别顺序返回 DataFrame，BarPlot 期望表格数据
        return pd.DataFrame(
            [{"class": str(i), "prob": float(p)} for i, p in enumerate(probs)]
        )

    return predict


def launch():
    args = parse_args()
    tokenizer = build_tokenizer(args)
    model = build_model(args, tokenizer)
    predict_fn = make_predict_fn(model, tokenizer)

    theme = Soft(primary_hue="cyan", secondary_hue="indigo")

    with gr.Blocks(theme=theme, title="Sentiment Classifie") as demo:
        gr.Markdown(
            """
            # Sentiment Classifier WebUI
            请输入一句文本，模型会实时输出 5 个情感类别 (0-4) 的概率柱状图。
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                textbox = gr.Textbox(
                    lines=4,
                    label="输入文本",
                    placeholder="例如：The movie is surprisingly good!",
                )
                gr.Examples(
                    examples=[
                        "The movie is surprisingly good!",
                        "This film is terrible and boring.",
                        "An average storyline but great acting.",
                        "It was okay, not too bad, not too good.",
                        "Absolutely fantastic! Highly recommended.",
                    ],
                    inputs=textbox,
                )
            with gr.Column(scale=4):
                bar = gr.BarPlot(
                    label="5 类情感概率分布",
                    x="class",
                    y="prob",
                    y_lim=[0, 1],
                    color="class",
                    color_map=CLASS_COLOR_MAP,
                    tooltip=["class", "prob"],
                )
        textbox.change(fn=predict_fn, inputs=textbox, outputs=bar)

    demo.queue().launch()


if __name__ == "__main__":
    launch()
