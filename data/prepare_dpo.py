"""Day 4 (PILOT) · Step 3 — PREPARE + write an inspection sample.

- token-based length-bias guard, context-window check, identical-pair guard
- stratified (domain/general) split, groups kept together by source prompt
- writes a 50-100 pair INSPECTION SAMPLE for manual review before scaling
- records deciding-dimension coverage (audit which of the 8 dims are exercised)

Usage:
    python data/prepare_dpo.py --config configs/day4.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import load_config, read_jsonl, repo_root, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    rng = random.Random(cfg["sampling"]["seed"])
    max_length = cfg["train"]["max_length"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["student"]["sft_model"])
    system = cfg["chat"]["system_prompt"]

    def seq_len(sys_prompt, prompt, answer):
        msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return len(tok(text)["input_ids"])

    def ans_len(answer):
        return len(tok(answer)["input_ids"])

    pairs = list(read_jsonl(cfg["paths"]["judged_pairs"]))
    print(f"[prepare] loaded {len(pairs)} judged pairs")

    # Length-bias handling. We (a) MEASURE whether chosen answers are
    # systematically longer than rejected (the classic DPO failure mode), and
    # (b) optionally DROP pairs whose ONLY apparent advantage is length -- i.e.
    # chosen is much longer AND the judge's deciding dimensions don't include a
    # substantive reason (correctness/completeness/etc.). This stops the model
    # learning "longer = better".
    drop_length_only = dcfg.get("drop_length_only_wins", True)
    # a "substantive" win cites at least one of these dims (not just style/length)
    SUBSTANTIVE_DIMS = {"correctness", "helpfulness", "completeness",
                        "instruction-following", "relevance", "harmlessness"}
    length_only_ratio = dcfg.get("length_only_ratio", 1.8)   # chosen >= 1.8x rejected

    kept, drops = [], Counter()
    len_deltas = []   # (chosen_len - rejected_len) per kept pair, for bias stats
    for p in pairs:
        chosen, rejected, prompt = p["chosen"], p["rejected"], p["prompt"]
        sys_prompt = None if p.get("source") == "general" else system
        if not chosen or not rejected or chosen.strip() == rejected.strip():
            drops["identical"] += 1
            continue
        lc, lr = ans_len(chosen), ans_len(rejected)
        if lr == 0:
            drops["identical"] += 1
            continue
        ratio = lc / lr
        # hard guard: extreme length gaps (either direction) are dropped
        if not (dcfg["min_length_ratio"] <= ratio <= dcfg["max_length_ratio"]):
            drops["length_bias"] += 1
            continue
        # length-only win: chosen much longer AND no substantive deciding reason.
        # The judge returns capitalized names ("Correctness"), so compare lowercase.
        dims = {str(d).strip().lower() for d in p.get("deciding_dimensions", [])}
        if drop_length_only and ratio >= length_only_ratio and not (dims & SUBSTANTIVE_DIMS):
            drops["length_only_win"] += 1
            continue
        if max(seq_len(sys_prompt, prompt, chosen),
               seq_len(sys_prompt, prompt, rejected)) > max_length:
            drops["too_long"] += 1
            continue
        len_deltas.append(lc - lr)
        kept.append({
            "prompt": prompt, "source": p.get("source", "domain"),
            "system": sys_prompt,
            "group": p.get("source_doc_id") or prompt,   # keep related pairs together
            "chosen": chosen, "rejected": rejected,
            "chosen_tokens": lc, "rejected_tokens": lr,
            "deciding_dimensions": p.get("deciding_dimensions", []),
        })

    # stratified split (domain/general separately), keeping every pair from the
    # same prompt/source document in ONE split (no train/val leakage). Groups
    # are formed first, then each whole group is assigned to one stratum.
    groups = {}
    for p in kept:
        groups.setdefault(p["group"], []).append(p)
    by = {"domain": [], "general": []}
    for g in groups.values():
        by["general" if g[0]["source"] == "general" else "domain"].append(g)
    train, val = [], []
    for k, glist in by.items():
        rng.shuffle(glist)
        target = int(sum(len(g) for g in glist) * dcfg["val_frac"])
        taken = 0
        for g in glist:
            if taken < target and len(glist) > 1:
                val.extend(g); taken += len(g)
            else:
                train.extend(g)
    rng.shuffle(train); rng.shuffle(val)

    # Conversational format: TRL then applies the SAME chat template sft-v2 was
    # trained with, so every answer ends with a real <|im_end|>. Plain strings
    # would skip the template and train on a different format. Domain prompts
    # get the system prompt; general prompts get none (SFT policy).
    def to_conv(p):
        prompt_msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
                      [{"role": "user", "content": p["prompt"]}]
        return {"prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": p["chosen"]}],
                "rejected": [{"role": "assistant", "content": p["rejected"]}]}

    from datasets import Dataset
    out = repo_root() / cfg["paths"]["pref_data"]
    Dataset.from_list([to_conv(p) for p in train]).save_to_disk(str(out / "train"))
    Dataset.from_list([to_conv(p) for p in val]).save_to_disk(str(out / "val"))

    # inspection sample: up to 80 pairs spanning sources + deciding dimensions
    insp = kept[:80]
    write_jsonl(cfg["paths"]["inspect_sample"], insp)

    # deciding-dimension coverage (audit which of the 8 dims are exercised)
    dim_cov = Counter()
    for p in kept:
        for d in p["deciding_dimensions"]:
            dim_cov[d] += 1

    # length-bias statistics on the KEPT pairs (the diagnostic that matters):
    # if chosen is still systematically longer, DPO may learn "longer = better".
    def _mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else 0.0
    n_chosen_longer = sum(1 for d in len_deltas if d > 0)
    length_bias_stats = {
        "mean_chosen_minus_rejected_tokens": _mean(len_deltas),
        "chosen_longer_fraction": round(n_chosen_longer / max(1, len(len_deltas)), 3),
        "note": "fraction near 0.5 and mean near 0 = little length bias; "
                "fraction >> 0.5 means chosen is systematically longer (watch out)",
    }

    manifest = {
        "judged_total": len(pairs),
        "kept_total": len(kept),
        "dropped": dict(drops),
        "train": len(train), "val": len(val),
        "train_by_source": dict(Counter(p["source"] for p in train)),
        "val_by_source": dict(Counter(p["source"] for p in val)),
        "deciding_dimension_coverage": dict(dim_cov.most_common()),
        "length_bias": length_bias_stats,
        "inspection_sample": len(insp),
    }
    mpath = repo_root() / cfg["paths"]["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print("[prepare]", json.dumps(manifest, indent=2))
    print(f"\n[prepare] MANUALLY INSPECT {len(insp)} pairs in {cfg['paths']['inspect_sample']}")
    print("[prepare] Verify chosen is genuinely better (esp. CORRECTNESS) before scaling.")
    print("[prepare] Next (after inspection): python training/dpo.py --config configs/day4.yaml")


if __name__ == "__main__":
    main()
