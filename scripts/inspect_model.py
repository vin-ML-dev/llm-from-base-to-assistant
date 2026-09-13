"""Day 1 — model inspection.

Prints the model structure and counts parameters per component, so the
architecture blocks (embeddings, attention, MLP, LM head) map onto real numbers.
You should see the embedding table take a sizable share on a small model, and
that Qwen3-1.7B ties its LM head to the embeddings.

Usage:
    python scripts/inspect_model.py --config configs/day1.yaml
"""
from __future__ import annotations

import argparse

from _common import add_common_args, load_config, load_model_and_tokenizer


def bucket(name: str) -> str:
    n = name.lower()
    if "embed" in n:
        return "embeddings"
    if "lm_head" in n:
        return "lm_head"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")):
        return "attention"
    if any(k in n for k in ("mlp", "gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if "norm" in n:
        return "norm"
    return "other"


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_config(args.config)

    model, _ = load_model_and_tokenizer(cfg["model"])

    print("=== model structure (truncated) ===")
    print(model)

    print("\n=== parameters by component ===")
    totals: dict[str, int] = {}
    for name, p in model.named_parameters():
        totals[bucket(name)] = totals.get(bucket(name), 0) + p.numel()

    grand = sum(totals.values())
    for k in sorted(totals, key=lambda x: -totals[x]):
        share = 100 * totals[k] / grand
        print(f"{k:12s}: {totals[k]/1e6:8.1f}M  ({share:4.1f}%)")
    print(f"{'TOTAL':12s}: {grand/1e6:8.1f}M")

    cfg_obj = model.config
    tied = getattr(cfg_obj, "tie_word_embeddings", None)
    print(f"\ntie_word_embeddings: {tied}")
    print(f"num_hidden_layers  : {getattr(cfg_obj, 'num_hidden_layers', '?')}")
    print(f"hidden_size        : {getattr(cfg_obj, 'hidden_size', '?')}")
    print(f"num_attention_heads: {getattr(cfg_obj, 'num_attention_heads', '?')}")
    print(f"num_key_value_heads: {getattr(cfg_obj, 'num_key_value_heads', '?')}  (< heads => GQA)")


if __name__ == "__main__":
    main()
