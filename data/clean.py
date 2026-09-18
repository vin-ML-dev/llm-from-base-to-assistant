"""
Step 2 — CLEAN + DEDUPLICATE.

Reads data/raw/*.jsonl, normalizes the text, drops documents that are too short,
removes exact duplicates (by normalized-text hash), and writes data/clean/*.jsonl.

Usage:
    python clean.py --config day2_cpt.yaml
"""

import argparse
import re

from dataio import ensure_dir, load_config, read_jsonl, text_hash, write_jsonl

# Files produced by collect.py that we clean here (replay is cleaned too, so it
# gets deduplicated against the domain docs).
RAW_FILES = ["domain_papers.jsonl", "domain_docs.jsonl", "replay.jsonl"]

# Common scraped-page boilerplate lines to drop.
BOILERPLATE = {
    "edit this page",
    "copied",
    "join the hugging face community",
    "table of contents",
}


def normalize(text):
    """Standardize whitespace and strip obvious navigation/boilerplate lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)      # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse 3+ blank lines into one

    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower() in BOILERPLATE:
            continue
        kept_lines.append(stripped)
    return "\n".join(kept_lines).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["clean_dir"])
    min_chars = cfg["data"]["min_doc_chars"]

    seen_hashes = set()  # shared across files, so duplicates across sources are caught

    for fname in RAW_FILES:
        rows = read_jsonl(f"{cfg['paths']['raw_dir']}/{fname}")
        if not rows:
            print(f"[clean] {fname}: not found or empty, skipping")
            continue

        kept = []
        dropped_short = 0
        dropped_dup = 0
        for row in rows:
            text = normalize(row.get("text", ""))
            if len(text) < min_chars:
                dropped_short += 1
                continue
            h = text_hash(text)
            if h in seen_hashes:
                dropped_dup += 1
                continue
            seen_hashes.add(h)
            row["text"] = text
            row["hash"] = h
            kept.append(row)

        n = write_jsonl(f"{cfg['paths']['clean_dir']}/{fname}", kept)
        print(f"[clean] {fname}: kept {n}, dropped_short {dropped_short}, dropped_dup {dropped_dup}")

    print("\nTip: open a few files in data/clean/ and read some docs by hand before continuing.")
    print("Next: python split.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
