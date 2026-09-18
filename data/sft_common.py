"""Shared helpers for Day 3 (SFT). Kept small and readable.

Handles: config loading, JSONL I/O (reused from Day 2's dataio via sys.path),
sequential vLLM model load/unload (so a 14B teacher and 8B judge fit ~44 GB by
never being resident at the same time), and Qwen3 thinking-mode stripping.
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/day3.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_jsonl(path, rows) -> int:
    p = repo_root() / path if not str(path).startswith("/") else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    p = repo_root() / path if not str(path).startswith("/") else Path(path)
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3's hidden reasoning block so only the final answer remains."""
    return THINK_RE.sub("", text).strip()


def free_gpu():
    """Release GPU memory between the teacher and judge phases."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def resolve_revision(repo_id: str, revision: str) -> str:
    """Turn a moving pointer like 'main' into an immutable commit hash (provenance)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo_id, revision=revision).sha
    except Exception:
        return revision
