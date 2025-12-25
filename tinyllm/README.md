# TinyLLM

## Train

```bash
cd scripts
# train tokenizer
uv run bpe.py --help
# tokenize training data
uv run run_tokenizer.py --help
# train
uv run train.py --help
```

## Inference 

```bash
cd scripts
uv run inference.py --help
```