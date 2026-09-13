"""Day 1 smoke test.

Confirms the base model loads and runs on *your* GPU, and records the two
numbers that decide whether it 'fits': measured peak VRAM and tokens/sec.
This is the guide's "measure, don't assume" rule in code form.

Usage:
    python scripts/smoke_test.py --config configs/day1.yaml
"""
from __future__ import annotations

import argparse

from _common import (
    add_common_args,
    load_config,
    load_model_and_tokenizer,
    peak_vram_gb,
    reset_vram_counter,
    time_generation,
)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = load_config(args.config)

    reset_vram_counter()
    print(f"Loading {cfg['model']['id']} (revision={cfg['model'].get('revision')}) ...")
    model, tok = load_model_and_tokenizer(cfg["model"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e9:.3f}B")

    prompt = "In one sentence, a transformer is"
    text, tps = time_generation(
        model, tok, prompt, cfg["generation"]["max_new_tokens"]
    )

    vram = peak_vram_gb()
    print("\n--- smoke test result ---")
    print(f"prompt      : {prompt!r}")
    print(f"continuation: {text!r}")
    print(f"tokens/sec  : {tps:.1f}")
    print(f"peak VRAM   : {vram:.2f} GB" if vram is not None else "peak VRAM   : (CPU, N/A)")
    print("\nRecord these in docs/decisions.md — they justify that the model fits your GPU.")


if __name__ == "__main__":
    main()
