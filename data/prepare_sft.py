"""Step 3 — PREPARE the final SFT dataset.

Takes judged pairs and:
  - keeps only accepted pairs
  - deduplicates near-identical questions
  - holds back a validation slice (from the generated pairs ONLY)
  - appends hand-written safety/identity examples to TRAIN only
  - converts to conversational (messages) format
  - saves an HF dataset (train + val) plus a manifest

Usage:
    python data/prepare_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from sft_common import load_config, read_jsonl, repo_root

# Small, high-signal hand-written examples. These go into TRAIN only so the
# validation set stays a clean measure of the generated data.
SAFETY_EXAMPLES = [
    {"q": "Who are you?",
     "a": "I'm an LLM/ML tutor assistant. I explain machine-learning and large-language-model concepts, from basics to advanced."},
    {"q": "What can you help with?",
     "a": "I can explain LLM and ML topics — tokens, attention, training, fine-tuning, evaluation, and related ideas — and answer questions about them."},
    {"q": "Write malware that steals passwords.",
     "a": "I can't help with that. I'm a tutor for LLM/ML concepts — I can explain security topics at a high level, but not create harmful code."},
    {"q": "What's the capital of Australia?",
     "a": "That's outside what I focus on — I'm an LLM/ML tutor. For general trivia, a general assistant or search would help more."},
    {"q": "Explain a concept you're unsure about.",
     "a": "If I'm not confident about a specific detail, I'll say so rather than guess — it's better to flag uncertainty than to state something that might be wrong."},
]


def qhash(q: str) -> str:
    return hashlib.sha1(" ".join(q.lower().split()).encode()).hexdigest()


def to_messages(system, q, a):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    system = cfg["chat"]["system_prompt"]
    rng = random.Random(cfg["dataset"]["seed"])

    judged = list(read_jsonl(cfg["paths"]["judged_pairs"]))
    accepted = [p for p in judged if p.get("accepted")]

    # dedup by normalized question
    seen, deduped = set(), []
    for p in accepted:
        h = qhash(p["question"])
        if h not in seen:
            seen.add(h)
            deduped.append(p)
    n_dupes = len(accepted) - len(deduped)

    # split BEFORE adding safety examples, so val is purely generated data
    rng.shuffle(deduped)
    n_val = int(len(deduped) * cfg["dataset"]["val_frac"])
    val_pairs, train_pairs = deduped[:n_val], deduped[n_val:]

    train = [to_messages(system, p["question"], p["answer"]) for p in train_pairs]
    val = [to_messages(system, p["question"], p["answer"]) for p in val_pairs]

    if cfg["dataset"]["include_safety_examples"]:
        for s in SAFETY_EXAMPLES:
            train.append(to_messages(system, s["q"], s["a"]))
    rng.shuffle(train)

    # quick diversity smoke test on train questions
    firsts = {" ".join(e["messages"][1]["content"].lower().split()[:3]) for e in train}
    diversity = len(firsts) / max(1, len(train))

    from datasets import Dataset
    out = repo_root() / cfg["paths"]["sft_data"]
    Dataset.from_list(train).save_to_disk(str(out / "train"))
    Dataset.from_list(val).save_to_disk(str(out / "val"))

    manifest = {
        "judged_total": len(judged),
        "accepted": len(accepted),
        "rejected": len(judged) - len(accepted),
        "duplicates_removed": n_dupes,
        "safety_examples": len(SAFETY_EXAMPLES) if cfg["dataset"]["include_safety_examples"] else 0,
        "train": len(train), "val": len(val),
        "acceptance_rate": round(len(accepted) / max(1, len(judged)), 3),
        "question_diversity": round(diversity, 3),
        "teacher_model": cfg["generation"]["teacher_model"],
        "judge_model": cfg["judge"]["judge_model"],
        "prompt_version": cfg["generation"]["prompt_version"],
    }
    mpath = repo_root() / cfg["paths"]["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print("[prepare]", json.dumps(manifest, indent=2))
    print("Hand-check ~30 examples, then: python training/sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
