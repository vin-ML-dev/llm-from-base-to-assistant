"""Day 1 — one forward pass, top-10 next-token probabilities.

Turns 'logits -> softmax -> distribution' into something you can see: for a
given prompt, what does the model think the next token is, and how confident is
it? This is the exact quantity cross-entropy scores during training.

Usage:
    python scripts/forward_pass_probs.py --config configs/day1.yaml --prompt "The capital of France is"
"""
from __future__ import annotations

import argparse

import torch

from _common import add_common_args, load_config, load_model_and_tokenizer


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)

    model, tok = load_model_and_tokenizer(cfg["model"])
    inputs = tok(args.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab)

    # We want the distribution for the NEXT token, i.e. after the last input token.
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, args.k)

    print(f"prompt: {args.prompt!r}")
    print(f"top-{args.k} next-token predictions:")
    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        piece = tok.decode([idx])
        print(f"  {prob:6.3f}  id={idx:<7d} {piece!r}")

    print("\nNote how the probability mass concentrates (or doesn't). Cross-entropy")
    print("rewards putting mass on the *true* next token; that is all training does.")


if __name__ == "__main__":
    main()
