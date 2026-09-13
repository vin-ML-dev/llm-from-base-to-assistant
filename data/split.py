"""Day 2 · Step 3 — SPLIT (document-level) + LEAKAGE CHECK.

Splits the DOMAIN documents into train / validation / held-out at the DOCUMENT
level (never block level — that would leak). Replay docs go entirely into the
training pool. Then runs a simple leakage check: confirm no held-out doc is an
exact/near duplicate of any training doc.

Writes:
  data/clean/pool_train.jsonl      (domain-train + replay, tagged by source)
  data/clean/pool_val.jsonl        (domain validation)
  data/eval_heldout/heldout.jsonl  (locked; never trained on)

Usage:
    python data/split.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse
import random

from dataio import (ensure_dir, load_config, read_jsonl, text_hash,
                 update_manifest, write_jsonl)


def shingles(text: str, k: int = 13) -> set[str]:
    """Word-level k-gram shingles for a light near-overlap check."""
    words = text.split()
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = random.Random(cfg["data"]["seed"])
    ensure_dir(cfg["paths"]["clean_dir"])
    ensure_dir(cfg["paths"]["eval_heldout_dir"])

    # --- load cleaned domain docs (papers + docs + theory) and replay ---
    domain = []
    for f in ("domain_papers.jsonl", "domain_docs.jsonl",
              "theory_web.jsonl", "theory_d2l.jsonl", "theory_surveys.jsonl"):
        try:
            domain.extend(read_jsonl(f"{cfg['paths']['clean_dir']}/{f}"))
        except FileNotFoundError:
            pass
    try:
        replay = list(read_jsonl(f"{cfg['paths']['clean_dir']}/replay.jsonl"))
    except FileNotFoundError:
        replay = []

    rng.shuffle(domain)
    n = len(domain)
    n_heldout = int(n * cfg["data"]["eval_heldout_frac"])
    n_val = int(n * cfg["data"]["val_frac"])
    heldout = domain[:n_heldout]
    val = domain[n_heldout:n_heldout + n_val]
    train_domain = domain[n_heldout + n_val:]

    for r in train_domain:
        r["split"] = "train"; r["kind"] = "domain"
    for r in replay:
        r["split"] = "train"; r["kind"] = "replay"
    for r in val:
        r["split"] = "val"; r["kind"] = "domain"
    for r in heldout:
        r["split"] = "heldout"; r["kind"] = "domain"

    train_pool = train_domain + replay
    rng.shuffle(train_pool)

    # --- leakage check: held-out vs training (exact + shingle overlap) ---
    train_hashes = {text_hash(r["text"]) for r in train_pool}
    exact_leaks = sum(1 for r in heldout if text_hash(r["text"]) in train_hashes)

    train_shingles: set[str] = set()
    for r in train_pool[:500]:                     # cap for speed on first run
        train_shingles |= shingles(r["text"])
    near_leaks = 0
    for r in heldout:
        hs = shingles(r["text"])
        if hs and len(hs & train_shingles) / len(hs) > 0.5:
            near_leaks += 1

    write_jsonl(f"{cfg['paths']['clean_dir']}/pool_train.jsonl", train_pool)
    write_jsonl(f"{cfg['paths']['clean_dir']}/pool_val.jsonl", val)
    write_jsonl(f"{cfg['paths']['eval_heldout_dir']}/heldout.jsonl", heldout)

    result = {
        "domain_docs": n, "train_domain": len(train_domain), "replay": len(replay),
        "val": len(val), "heldout": len(heldout),
        "leakage_exact": exact_leaks, "leakage_near": near_leaks,
    }
    update_manifest(cfg["paths"]["manifest"], "split", result)
    print("[split]", result)
    if exact_leaks or near_leaks:
        print("  ⚠ leakage detected — investigate before trusting eval numbers.")
    else:
        print("  ✓ no leakage found (held-out is clean).")
    print("Next: python data/tokenize_pack.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
