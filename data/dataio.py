"""
Shared helpers for the CPT data pipeline: config loading, JSONL read/write,
and a text hash used for deduplication and leakage checks.
"""

import hashlib
import json
from pathlib import Path

import yaml

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path, rows):
    """Write an iterable of dicts to a .jsonl file. Returns how many were written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path):
    """Read a .jsonl file into a list of dicts. Returns [] if the file is missing."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_hash(text):
    """Stable hash of whitespace-normalized, lowercased text (for dedup / leakage)."""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
