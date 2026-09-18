"""One-shot: strip control characters from an already-generated raw_pairs.jsonl.

Your 2968 pairs were written before control-char cleaning was added, so some
contain PDF junk (\\x0c form-feed, \\x00 null) where the JSON happened to parse
around it. This scrubs question/answer in place — no need to rerun the teacher.

Usage:
    python data/clean_raw_pairs.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse

from sft_common import clean_text, load_config, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = cfg["paths"]["raw_pairs"]

    rows = list(read_jsonl(path))
    changed = 0
    for r in rows:
        q, a = r.get("question", ""), r.get("answer", "")
        cq, ca = clean_text(q), clean_text(a)
        if cq != q or ca != a:
            changed += 1
        r["question"], r["answer"] = cq, ca

    n = write_jsonl(path, rows)
    print(f"[clean] scrubbed {changed}/{n} pairs with control characters → {path}")
    print("[clean] Next: python data/judge_sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
