"""Shared helpers for the Day 1 exploration scripts.

Kept deliberately small and readable — these scripts exist so the Day 1 theory
(tokenization, the forward pass, logits, base-vs-chat behavior, decoding) stops
being abstract. They do not train anything.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        default="configs/day1.yaml",
        help="Path to the Day 1 YAML config.",
    )
    return parser


def pick_dtype(name: str):
    """Return the requested dtype, falling back to float16/float32 if BF16 is
    unsupported on this GPU. We *check* rather than assume — the same discipline
    the guide preaches about VRAM."""
    if name == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if name == "float16":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return torch.float32


def load_tokenizer(model_cfg: dict):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    return tok


def load_model_and_tokenizer(model_cfg: dict):
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"],
        revision=model_cfg.get("revision", "main"),
        torch_dtype=pick_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.eval()
    return model, tok


def peak_vram_gb() -> float | None:
    """Measured peak VRAM in GB, or None on CPU. The guide's rule: never claim a
    model 'fits' from arithmetic — measure it."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_vram_counter() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def time_generation(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Generate greedily and return (text, tokens_per_second)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    n_new = out.shape[1] - n_in
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    tps = n_new / elapsed if elapsed > 0 else 0.0
    return text, tps


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
