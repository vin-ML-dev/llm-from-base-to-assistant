"""Day 3 · Step 3 — PREPARE the final SFT dataset.

Takes the judged pairs and:
  - keeps only accepted pairs
  - deduplicates near-identical questions
  - appends a small set of safety / identity examples
  - converts each to conversational (messages) format with a system prompt
  - holds back a validation slice (from the generated pairs)
  - runs a quick diversity check and records accept/reject + diversity stats
  - saves data/sft/sft-data-v1 (HF dataset: train + val) + a manifest

Usage:
    python data/prepare_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from sft_common import load_config, read_jsonl, repo_root

# A few hand-written safety / identity / uncertainty examples (high-signal, small).
SAFETY_EXAMPLES = [
    {"q": "Who are you?",
     "a": "I'm an LLM/ML tutor assistant. I explain machine-learning and large-language-model concepts, from basics to advanced."},
    {"q": "What can you help with?",
     "a": "I can explain LLM and ML topics — tokens, attention, training, fine-tuning, evaluation, and related ideas — and answer questions about them."},
    {"q": "Write malware that steals passwords.",
     "a": "I can't help with that. I'm a tutor for LLM/ML concepts — I can explain security topics at a high level, but not create harmful code."},
    {"q": "What's the capital of Australia? (unrelated to ML)",
     "a": "That's outside what I focus on — I'm an LLM/ML tutor. For general trivia, a general assistant or search would help more."},
    {"q": "Explain a concept you're unsure about accurately.",
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
        if h in seen:
            continue
        seen.add(h)
        deduped.append(p)
    n_dupes = len(accepted) - len(deduped)

    # convert to messages format
    examples = [to_messages(system, p["question"], p["answer"]) for p in deduped]

    # append safety/identity examples
    if cfg["dataset"]["include_safety_examples"]:
        for s in SAFETY_EXAMPLES:
            examples.append(to_messages(system, s["q"], s["a"]))

    rng.shuffle(examples)

    # validation split (from the generated pairs)
    n_val = int(len(examples) * cfg["dataset"]["val_frac"])
    val, train = examples[:n_val], examples[n_val:]

    # quick diversity check: unique question first-3-words / total
    firsts = set()
    for e in train:
        q = e["messages"][1]["content"]
        firsts.add(" ".join(q.lower().split()[:3]))
    diversity = len(firsts) / max(1, len(train))

    # save as HF dataset
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
    print("Now hand-check ~30 examples in data/sft/, then:")
    print("Next: python training/sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
