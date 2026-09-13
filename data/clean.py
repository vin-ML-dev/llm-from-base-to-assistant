"""Day 2 · Step 2 — CLEAN + DEDUPLICATE.

Reads data/raw/*.jsonl, normalizes text, drops junk/short docs, removes exact
duplicates (by normalized-text hash), and writes data/clean/*.jsonl.
Logs how many docs were removed (goes into the manifest / dataset card).

Usage:
    python data/clean.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse
import re

from dataio import ensure_dir, load_config, read_jsonl, text_hash, update_manifest, write_jsonl

RAW_FILES = ["domain_papers.jsonl", "domain_docs.jsonl", "replay.jsonl",
             "theory_web.jsonl", "theory_d2l.jsonl", "theory_surveys.jsonl"]


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    # drop obvious nav/boilerplate lines common in scraped docs
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        low = s.lower()
        if low in {"edit this page", "copied", "join the hugging face community", "table of contents"}:
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["clean_dir"])
    min_chars = cfg["data"]["min_doc_chars"]

    seen_hashes: set[str] = set()
    stats = {}

    for fname in RAW_FILES:
        raw_path = f"{cfg['paths']['raw_dir']}/{fname}"
        try:
            rows = list(read_jsonl(raw_path))
        except FileNotFoundError:
            print(f"[clean] {fname} not found, skipping")
            continue

        kept, dropped_short, dropped_dup = [], 0, 0
        for row in rows:
            clean_text = normalize(row.get("text", ""))
            if len(clean_text) < min_chars:
                dropped_short += 1
                continue
            h = text_hash(clean_text)
            if h in seen_hashes:
                dropped_dup += 1
                continue
            seen_hashes.add(h)
            row["text"] = clean_text
            row["hash"] = h
            kept.append(row)

        out = f"{cfg['paths']['clean_dir']}/{fname}"
        n = write_jsonl(out, kept)
        stats[fname] = {"kept": n, "dropped_short": dropped_short, "dropped_duplicate": dropped_dup}
        print(f"[clean] {fname}: kept {n}, dropped_short {dropped_short}, dropped_dup {dropped_dup}")

    update_manifest(cfg["paths"]["manifest"], "clean", stats)
    print("\nTip: open a few files in data/clean/ and read ~20 docs by hand before continuing.")
    print("Next: python data/split.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
