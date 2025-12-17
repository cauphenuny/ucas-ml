# TinyLLM

## Train

```bash
cd scripts
# train tokenizer
uv run bpe.py --help
# tokenize training data
uv run tokenize.py --help
# train
uv run train.py --help
```

## Inference 

```bash
cd scripts
uv run inference.py --help
```