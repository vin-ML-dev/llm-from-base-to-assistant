"""Pre-CPT data quality check. Run AFTER clean.py, BEFORE cpt.py.

Reports token counts, per-source breakdown, garbage/quality signals, license
coverage, and shows real sample text so you can eyeball it. No model needed.

Usage:
    python inspect_data.py --config day2.yaml
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataio import load_config, read_jsonl  # noqa: E402


def garbage_ratio(text):
    if not text:
        return 1.0
    readable = sum(1 for ch in text
                   if ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}\"'-+=/*%$#@&_|<>")
    return 1.0 - readable / len(text)


def est_tokens(chars):
    return chars // 4   # rough ~4 chars/token


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="day2.yaml")
    ap.add_argument("--samples", type=int, default=2, help="sample docs per source to print")
    args = ap.parse_args()
    cfg = load_config(args.config)
    clean_dir = Path(cfg["paths"]["clean_dir"])

    domain_files = ["domain_wikipedia.jsonl", "domain_papers.jsonl", "domain_docs.jsonl"]
    replay_file = "replay.jsonl"

    print("=" * 70)
    print("PRE-CPT DATA QUALITY REPORT")
    print("=" * 70)

    grand_domain_chars = 0
    all_rows = []
    for fname in domain_files:
        rows = read_jsonl(clean_dir / fname)
        if not rows:
            print(f"\n[{fname}] EMPTY or missing")
            continue
        chars = sum(len(r.get("text", "")) for r in rows)
        grand_domain_chars += chars
        ratios = [garbage_ratio(r.get("text", "")) for r in rows]
        avg_g = sum(ratios) / len(ratios)
        worst_g = max(ratios)
        lens = sorted(len(r.get("text", "")) for r in rows)
        med_len = lens[len(lens) // 2]
        licenses = Counter(r.get("license", "unknown") for r in rows)
        all_rows.extend(rows)

        print(f"\n[{fname}]")
        print(f"  docs:          {len(rows)}")
        print(f"  chars:         {chars:,}  (~{est_tokens(chars):,} tokens)")
        print(f"  median length: {med_len:,} chars")
        print(f"  garbage ratio: avg={avg_g:.3f}  worst={worst_g:.3f}")
        print(f"  licenses:      {dict(licenses)}")
        if worst_g > 0.25:
            print(f"  ⚠️  some docs still have high garbage ({worst_g:.2f}) — check samples")

    # --- replay ---
    replay = read_jsonl(clean_dir / replay_file)
    replay_chars = sum(len(r.get("text", "")) for r in replay)
    print(f"\n[{replay_file}] (general replay)")
    print(f"  docs: {len(replay)}  chars: {replay_chars:,}  (~{est_tokens(replay_chars):,} tokens)")

    # --- totals ---
    print("\n" + "=" * 70)
    print("TOTALS")
    print(f"  DOMAIN tokens (est): ~{est_tokens(grand_domain_chars):,}")
    print(f"  + replay:            ~{est_tokens(grand_domain_chars + replay_chars):,}")
    budget = cfg["tokenize"]["token_budget"]
    print(f"  token_budget:        {budget:,}")
    pct = 100 * est_tokens(grand_domain_chars) / max(1, budget)
    print(f"  domain fills ~{pct:.0f}% of budget")
    if est_tokens(grand_domain_chars) < 3_000_000:
        print("  ⚠️  LIGHT corpus (<3M domain tokens): expect only a mild domain shift")
    elif est_tokens(grand_domain_chars) < 6_000_000:
        print("  ~ MODERATE corpus: a real but not deep domain shift")
    else:
        print("  ✅ HEALTHY corpus size for a 1.7B CPT")

    # --- duplicate check (should be 0 after clean.py) ---
    from dataio import text_hash
    hashes = [text_hash(r["text"]) for r in all_rows]
    dupes = len(hashes) - len(set(hashes))
    print(f"\n  exact duplicates remaining: {dupes}  {'✅' if dupes == 0 else '⚠️ rerun clean.py'}")

    # --- worst offenders: show the 3 highest-garbage docs to eyeball ---
    print("\n" + "=" * 70)
    print("HIGHEST-GARBAGE DOCS (eyeball these — should still be readable):")
    ranked = sorted(all_rows, key=lambda r: garbage_ratio(r.get("text", "")), reverse=True)
    for r in ranked[:3]:
        g = garbage_ratio(r.get("text", ""))
        title = str(r.get("title", r.get("id", "?")))[:50]
        print(f"\n  garbage={g:.3f} | {r.get('source')} | {title}")
        print(f"  {r.get('text','')[:300]!r}")

    # --- random clean samples per source to read ---
    print("\n" + "=" * 70)
    print("RANDOM SAMPLES (read these — is it coherent domain prose?):")
    import random
    rng = random.Random(cfg["data"]["seed"])
    by_source = {}
    for r in all_rows:
        by_source.setdefault(r.get("source"), []).append(r)
    for src, rows in by_source.items():
        for r in rng.sample(rows, min(args.samples, len(rows))):
            title = str(r.get("title", r.get("id", "?")))[:50]
            print(f"\n  [{src}] {title}")
            print(f"  {r.get('text','')[:300]!r}")

    print("\n" + "=" * 70)
    print("If samples look clean and readable -> proceed to split.py + CPT.")
    print("If you see garbage/boilerplate -> tighten clean.py before training.")


if __name__ == "__main__":
    main()
