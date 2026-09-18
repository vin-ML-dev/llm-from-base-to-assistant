"""Shared helpers for the SFT pipeline: config, JSONL I/O, GPU cleanup,
Qwen3 thinking-mode stripping, and revision pinning."""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/day3.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(path) -> Path:
    return Path(path) if str(path).startswith("/") else repo_root() / path


def write_jsonl(path, rows) -> int:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with open(_resolve(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)  # unclosed (truncated) block


def strip_thinking(text: str) -> str:
    """Remove Qwen3's reasoning block. Handles both closed <think>...</think>
    and an unclosed <think>... left when generation is truncated mid-thought."""
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def free_gpu():
    """Release GPU memory between the teacher and judge phases."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_revision(repo_id: str, revision: str) -> str:
    """Turn a moving pointer like 'main' into an immutable commit hash."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo_id, revision=revision).sha
    except Exception:
        return revision


def load_tokenizer(source, revision=None):
    """Load a checkpoint's tokenizer, tolerating list-style extra_special_tokens
    (some Qwen3 configs) that older Transformers versions reject as a mapping.

    Never adds vocabulary or changes token IDs: any extra token must already be
    in the vocab, otherwise we raise rather than train/eval on shifted IDs.
    """
    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    config = get_tokenizer_config(source, revision=revision)
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return AutoTokenizer.from_pretrained(source, revision=revision)

    # Reload with the field cleared, then verify the tokens already exist.
    tok = AutoTokenizer.from_pretrained(source, revision=revision, extra_special_tokens={})
    vocab = tok.get_vocab()
    for token in extra:
        content = token.get("content") if isinstance(token, dict) else token
        if not isinstance(content, str) or content not in vocab:
            raise ValueError(
                f"extra_special_token {content!r} absent from {source} vocab; "
                "refusing to add untrained embeddings or shift IDs."
            )
    return tok


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # controls except \t \n \r


def clean_text(s: str) -> str:
    """Strip control characters (null, form-feed, etc.) that PDF extraction
    leaves in source text. These are illegal raw inside JSON strings and are
    junk in training data. Keeps tab/newline/carriage-return."""
    if not isinstance(s, str):
        return s
    return _CTRL_RE.sub("", s)


def extract_json(text: str) -> dict | None:
    """Pull the first {...} object out of a model response, or None."""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None