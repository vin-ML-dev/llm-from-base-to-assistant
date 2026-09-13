"""Shared helpers for the Day 2 data pipeline: config loading, JSONL I/O,
hashing, and a tiny manifest updater. Kept dependency-light and readable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/day2.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = repo_root() / path if not str(path).startswith("/") else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = repo_root() / path if not str(path).startswith("/") else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = repo_root() / path if not str(path).startswith("/") else Path(path)
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def text_hash(text: str) -> str:
    """Stable hash of normalized text — used for dedup and leakage checks."""
    norm = " ".join(text.split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def update_manifest(manifest_path: str | Path, key: str, value) -> None:
    """Append a key/value to the dataset manifest (provenance + counts + hashes)."""
    p = repo_root() / manifest_path if not str(manifest_path).startswith("/") else Path(manifest_path)
    data = {}
    if p.exists():
        data = json.loads(p.read_text())
    data[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
