"""Day 1 — tokenizer exploration.

Makes tokenization concrete: how text becomes tokens/IDs, what the special
tokens are, and how token counts differ between tokenizers (the 'frequent =
cheap, rare = expensive' idea, and why cost is measured in tokens).

Usage:
    python scripts/explore_tokenizer.py --config configs/day1.yaml
"""
from __future__ import annotations

import argparse

from _common import add_common_args, load_config, load_tokenizer


SAMPLES = [
    "The capital of France is Paris.",
    "Transformers use self-attention to mix information between tokens.",
    "tokenization",
]


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_config(args.config)

    tok = load_tokenizer(cfg["model"])

    print(f"=== tokenizer for {cfg['model']['id']} ===")
    print(f"vocab size       : {tok.vocab_size}")
    print(f"special tokens   : {tok.special_tokens_map}")
    print(f"eos token / id   : {tok.eos_token!r} / {tok.eos_token_id}")
    print(f"pad token / id   : {tok.pad_token!r} / {tok.pad_token_id}")
    print()

    for text in SAMPLES:
        ids = tok.encode(text)
        pieces = tok.convert_ids_to_tokens(ids)
        print(f"text   : {text!r}")
        print(f"#tokens: {len(ids)}")
        print(f"ids    : {ids}")
        print(f"pieces : {pieces}")
        print(f"decoded: {tok.decode(ids)!r}")
        print("-" * 60)

    # Two-tokenizer comparison: same text, different token counts = different cost.
    compare_id = cfg["model"].get("compare_tokenizer_id")
    if compare_id:
        from transformers import AutoTokenizer

        other = AutoTokenizer.from_pretrained(compare_id)
        text = SAMPLES[1]
        a = len(tok.encode(text))
        b = len(other.encode(text))
        print(f"token-count comparison on: {text!r}")
        print(f"  {cfg['model']['id']}: {a} tokens")
        print(f"  {compare_id}: {b} tokens")
        print("Fewer tokens for the same meaning = cheaper to train and serve.")


if __name__ == "__main__":
    main()
