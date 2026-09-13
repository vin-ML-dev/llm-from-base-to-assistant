"""Day 1 — decoding settings comparison.

Generate the same prompt at several temperatures to feel what the decoding dials
do: temperature 0 (greedy, deterministic, flat) vs higher (varied, riskier).
Training sets the probabilities; decoding chooses how to sample them.

Usage:
    python scripts/generation_settings.py --config configs/day1.yaml --prompt "Explain what a tokenizer does."
"""
from __future__ import annotations

import argparse

import torch

from _common import add_common_args, load_config, load_model_and_tokenizer


def generate(model, tok, prompt, temperature, cfg):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        pad_token_id=tok.eos_token_id,
    )
    if temperature == 0.0:
        kwargs.update(do_sample=False)
    else:
        kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=cfg["generation"]["top_p"],
            top_k=cfg["generation"]["top_k"],
        )
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--prompt", default="Explain what a tokenizer does.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model, tok = load_model_and_tokenizer(cfg["model"])

    print(f"prompt: {args.prompt!r}\n")
    for temp in cfg["generation"]["temperatures"]:
        text = generate(model, tok, args.prompt, temp, cfg)
        label = "greedy" if temp == 0.0 else f"temp={temp}"
        print(f"--- {label} ---")
        print(text.strip())
        print()

    print("temp 0 is reproducible (good for evaluation); higher temp adds variety")
    print("(useful on Day 4 when sampling multiple candidates for preference pairs).")


if __name__ == "__main__":
    main()
