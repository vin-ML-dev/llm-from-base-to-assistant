"""Shared config, JSONL and text-hashing helpers for the CPT pipeline."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


def repo_root() -> Path:
    # Retain the original convention: these helpers live in a scripts folder.
    return Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping, not an empty file or list")
    return cfg


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path, rows):
    """Atomically replace a JSONL file; keep its old contents if writing fails."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{p}: row {count + 1} must be a JSON object")
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, p)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return count


def read_jsonl(path, required=False):
    """Read objects; optional missing files return [], malformed files fail clearly."""
    p = Path(path)
    if not p.exists() and not required:
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{p}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def text_hash(text):
    """Keep the existing case/whitespace-normalized SHA-1 hash for compatibility."""
    if not isinstance(text, str):
        raise TypeError("text_hash expects a string")
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()