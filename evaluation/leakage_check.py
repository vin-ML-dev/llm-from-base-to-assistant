"""Day 5 · Step 0 — LEAKAGE CHECK: did SFT/DPO training data come from the
LOCKED held-out documents used for domain perplexity?

Matches by document id, and for DPO also by text (the prompt snippet appearing
inside a held-out document), in case id formats differ between stages.

Usage:
    python evaluation/leakage_check.py --config configs/day5.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, read_jsonl, repo_root  # noqa: E402


def rows(path):
    try:
        return list(read_jsonl(path))
    except FileNotFoundError:
        print(f"[leak] missing: {path} (skipped)")
        return []


def norm(text):
    return " ".join(str(text).lower().split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day5.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    heldout = rows(cfg["perplexity"]["domain_heldout"])
    held_ids = {str(d.get("id") or d.get("url")) for d in heldout if d.get("id") or d.get("url")}
    held_text = [norm(d.get("text", "")) for d in heldout]
    print(f"[leak] {len(heldout)} held-out documents")

    def id_hit(doc_id):
        if not doc_id:
            return None
        doc_id = str(doc_id).split("#")[0]              # SFT ids look like "<doc_id>#<offset>"
        for cand in (doc_id, doc_id.split(":", 1)[-1]):  # DPO ids may carry a "source:" prefix
            if cand in held_ids:
                return cand
        return None

    def text_hit(snippet):
        s = norm(snippet)
        return len(s) >= 80 and any(s in t for t in held_text)

    report = {}
    # SFT: generated Q&A pairs record the source document id
    sft = rows(cfg["paths"]["sft_pairs"])
    sft_hits = [r for r in sft if id_hit(r.get("source_doc_id") or r.get("source_id"))]
    report["sft"] = {"checked": len(sft), "from_heldout_docs": len(sft_hits),
                     "heldout_docs_used": sorted({id_hit(r.get("source_doc_id") or r.get("source_id")) for r in sft_hits})}

    # DPO: candidates record the source id and the exact snippet text
    dpo = rows(cfg["paths"]["dpo_candidates"])
    dpo_hits = [r for r in dpo
                if id_hit(r.get("source_doc_id")) or text_hit(r.get("source_text") or r["prompt"].split(":", 1)[-1])]
    report["dpo"] = {"checked": len(dpo), "from_heldout_docs": len(dpo_hits),
                     "examples": [r["prompt"][:120] for r in dpo_hits[:5]]}

    for stage in ("sft", "dpo"):
        r = report[stage]
        pct = 100 * r["from_heldout_docs"] / max(1, r["checked"])
        print(f"[leak] {stage.upper()}: {r['from_heldout_docs']}/{r['checked']} records ({pct:.1f}%) come from held-out docs")

    leaked = report["sft"]["from_heldout_docs"] + report["dpo"]["from_heldout_docs"]
    report["verdict"] = "LEAK" if leaked else "CLEAN"
    out = repo_root() / cfg["paths"]["leakage_report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if leaked:
        print("[leak] LEAK: domain perplexity for sft-v2/dpo-v1 may look better than it is.\n"
              "       Report it next to the honest table, or exclude the affected docs from the domain set.")
    else:
        print("[leak] CLEAN: no SFT/DPO data traced to held-out documents.")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
