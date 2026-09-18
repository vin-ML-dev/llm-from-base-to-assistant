"""
Step 3 — SPLIT (document level) + LEAKAGE CHECK.

Splits the DOMAIN documents into train / validation / held-out at the DOCUMENT
level (splitting at the block level would leak text across splits). Replay docs
all go into the training pool. Then it checks that no held-out document also
appears in the training pool (exact or near-duplicate).

Writes:
  data/clean/pool_train.jsonl      (domain-train + replay, tagged with "kind")
  data/clean/pool_val.jsonl        (domain validation)
  data/eval_heldout/heldout.jsonl  (locked; never trained on)

Usage:
    python split.py --config day2_cpt.yaml
"""

import argparse
import random

from dataio import ensure_dir, load_config, read_jsonl, text_hash, write_jsonl

DOMAIN_FILES = ["domain_papers.jsonl", "domain_docs.jsonl"]


def shingles(text, k=13):
    """Set of word-level k-grams, used for a light near-duplicate check."""
    words = text.split()
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["data"]["seed"])
    clean_dir = cfg["paths"]["clean_dir"]
    ensure_dir(clean_dir)
    ensure_dir(cfg["paths"]["eval_heldout_dir"])

    # --- load cleaned domain docs and replay docs ---
    domain = []
    for fname in DOMAIN_FILES:
        domain.extend(read_jsonl(f"{clean_dir}/{fname}"))
    replay = read_jsonl(f"{clean_dir}/replay.jsonl")

    # --- document-level split of the domain docs ---
    rng.shuffle(domain)
    n = len(domain)
    n_heldout = int(n * cfg["data"]["eval_heldout_frac"])
    n_val = int(n * cfg["data"]["val_frac"])

    heldout = domain[:n_heldout]
    val = domain[n_heldout:n_heldout + n_val]
    train_domain = domain[n_heldout + n_val:]

    # tag each row so the packer can enforce the domain:replay mixture
    for r in train_domain:
        r["kind"] = "domain"
    for r in replay:
        r["kind"] = "replay"

    train_pool = train_domain + replay

    # --- leakage check: no held-out doc should appear in training ---
    train_hashes = {text_hash(r["text"]) for r in train_pool}
    exact_leaks = sum(1 for r in heldout if text_hash(r["text"]) in train_hashes)

    # near-duplicate check over ALL training docs (cheap at this corpus size)
    train_shingles = set()
    for r in train_pool:
        train_shingles |= shingles(r["text"])
    near_leaks = 0
    for r in heldout:
        hs = shingles(r["text"])
        if hs and len(hs & train_shingles) / len(hs) > 0.5:
            near_leaks += 1

    # --- write outputs ---
    write_jsonl(f"{clean_dir}/pool_train.jsonl", train_pool)
    write_jsonl(f"{clean_dir}/pool_val.jsonl", val)
    write_jsonl(f"{cfg['paths']['eval_heldout_dir']}/heldout.jsonl", heldout)

    print(
        f"[split] domain={n} train_domain={len(train_domain)} replay={len(replay)} "
        f"val={len(val)} heldout={len(heldout)}"
    )
    if exact_leaks or near_leaks:
        print(f"  WARNING: leakage found (exact={exact_leaks}, near={near_leaks}). "
              "Investigate before trusting eval numbers.")
    else:
        print("  OK: held-out set is clean (no leakage found).")
    print("Next: python tokenize_pack.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
