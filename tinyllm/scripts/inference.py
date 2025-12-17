import argparse
import time
from tinyllm.train.checkpoint import load_model, select_model, TransformerLM
from tinyllm.tokenize.tokenizer import Tokenizer
from tinyllm.optimize.optimizers import AdamW


def main(
    path,
    tokenizer_dir,
    context_length=2048,
    temperature=1e-5,
    top_p=0.9,
):
    tokenizer = Tokenizer.from_dir(
        tokenizer_dir, special_tokens=["<|endoftext|>"]
    )
    llm = TransformerLM(path, tokenizer)
    try:
        while True:
            prompt = input(">>> ")
            print(prompt, end="")
            start = time.time()
            for i, text in enumerate(
                llm.generate(
                    prompt,
                    max_length=context_length,
                    temperature=temperature,
                    top_p=top_p,
                    flush=True,
                )
            ):
                print(text, end="", flush=True)
            duration = time.time() - start
            print(f"\n(total: {i}, {i / duration:.3f}iter/s)")
    except KeyboardInterrupt:
        return
        save = input("save model? [y/n]")
        if save.lower() == "y":
            path = input("Enter path to save model: ")
            name = input("Enter model name: ")
            llm.save(path, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer directory, containing vocab.json and merges.json")
    parser.add_argument("--temp", type=float, default=1e-5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    if not args.model:
        args.model = select_model()
    main(
        args.model,
        args.tokenizer,
        temperature=args.temp,
        top_p=args.top_p,
    )
