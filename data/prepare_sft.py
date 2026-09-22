"""Step 3 — PREPARE the final SFT dataset.

Takes judged pairs and:
  - keeps only accepted pairs
  - deduplicates questions after case/whitespace normalization
  - holds back whole source-document groups for validation (no doc leakage)
  - stratifies the split so BOTH domain and general pairs appear in validation
  - uses a neutral system prompt for general pairs, the tutor prompt for domain
  - appends hand-written safety/identity examples to TRAIN only
  - converts to conversational (messages) format, retaining provenance
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

# For GENERAL (non-domain) pairs: NO system prompt at all. Using the ML-tutor
# prompt on general trivia would teach a contradictory identity, so general
# pairs get NO system message — the model just answers the question naturally.
# Domain pairs use the ML-assistant system prompt from config.
GENERAL_SYSTEM_PROMPT = None

# Small, high-signal hand-written examples. These go into TRAIN only so the
# validation set stays a clean measure of the generated data. Multiple identity
# phrasings are included so the model reliably learns WHO IT IS and answers the
# "who are you?" family directly (instead of the base model's rambling/looping).
# Identity matches the "helpful assistant with ML expertise" positioning: it
# specializes in ML/LLMs but can also help with general questions.
SAFETY_EXAMPLES = [
    {"q": "Who are you?",
     "a": "I'm a helpful assistant with a focus on machine learning and large language models. I can explain ML and LLM concepts clearly, and I'm happy to help with general questions too. What would you like to know?"},
    {"q": "who are you?",
     "a": "I'm an AI assistant that specializes in machine learning and large language models. Ask me about ML or LLM topics — attention, transformers, training, fine-tuning — or anything else you're curious about."},
    {"q": "What are you?",
     "a": "I'm an AI assistant with expertise in machine learning and large language models. I explain concepts like attention, transformers, training, and fine-tuning clearly, and can help with general questions as well."},
    {"q": "Introduce yourself.",
     "a": "Hi! I'm a helpful assistant who specializes in machine learning and large language models. I can walk you through core ideas and advanced techniques, and help with other questions too. What can I help with?"},
    {"q": "Tell me about yourself.",
     "a": "I'm an assistant with a strong focus on machine learning and large language models. I can explain how models work, how they're trained and fine-tuned, and how they're evaluated — and I'm glad to help with general topics as well."},
    {"q": "What can you help with?",
     "a": "I'm best at machine learning and LLM topics — tokens, attention, training, fine-tuning, evaluation, and related ideas — but I can also help with general questions. Just ask."},
    {"q": "Write malware that steals passwords.",
     "a": "I can't help with that. I can explain security and machine-learning concepts at a high level, but I won't create harmful code."},
    {"q": "How do I hack into someone's account?",
     "a": "I can't help with that — accessing someone's account without permission is harmful and likely illegal. I'm happy to explain account security or authentication concepts instead."},
    {"q": "What's the capital of Australia?",
     "a": "The capital of Australia is Canberra."},
    {"q": "Explain a concept you're unsure about.",
     "a": "If I'm not confident about a specific detail, I'll say so rather than guess — it's better to flag uncertainty than to state something that might be wrong."},
]


def qhash(q: str) -> str:
    return hashlib.sha1(" ".join(q.lower().split()).encode()).hexdigest()


def pair_system(pair, domain_system):
    """Domain pairs get the ML/tutor system prompt; general pairs get a neutral
    assistant prompt. `kind` is set by generate_sft.py ('domain' or 'general')."""
    return GENERAL_SYSTEM_PROMPT if pair.get("kind") == "general" else domain_system


def to_messages(system, q, a, provenance):
    # Omit the system message entirely when system is None (general pairs).
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": q})
    msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs, "provenance": json.dumps(provenance, ensure_ascii=False)}


def group_sources(pairs):
    """Link matching source IDs / hashes so copies of a document stay together.

    Field names match generate_sft.py output: source_id and source_hash.
    Run before question deduplication so removing a repeated question cannot
    break a known link between two copies of a source document.
    """
    parents = list(range(len(pairs)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, p in enumerate(pairs):
        for field in ("source_id", "source_hash"):
            value = p.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            key = (field, value)
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, p in enumerate(pairs):
        groups.setdefault(root(i), []).append(p)
    for group in groups.values():
        identities = sorted({p.get("source_id", "") for p in group})
        group_id = hashlib.sha256(json.dumps(identities).encode("utf-8")).hexdigest()
        for p in group:
            p["source_group_id"] = group_id


def _take_val_groups(remaining, target, rng):
    """Pull whole source groups into val to approach `target` pairs, keeping at
    least one group in train. Returns (val_pairs, remaining_groups)."""
    rng.shuffle(remaining)
    val = []
    while len(remaining) > 1:
        gap = target - len(val)
        if gap <= 0:
            break
        index = min(range(len(remaining)), key=lambda i: abs(len(remaining[i]) - gap))
        if val and abs(len(remaining[index]) - gap) >= abs(gap):
            break
        val.extend(remaining.pop(index))
    return val, remaining


def split_source_groups(pairs, val_frac, rng):
    """Stratified, document-group-preserving split.

    Splits domain and general pairs SEPARATELY (each by source group), so the
    validation set contains a proportional share of BOTH — needed to detect
    forgetting of general ability during training. No pairs are dropped.
    """
    # bucket by kind, then by source group within each kind
    by_kind = {"domain": {}, "general": {}}
    for p in pairs:
        kind = "general" if p.get("kind") == "general" else "domain"
        by_kind[kind].setdefault(p["source_group_id"], []).append(p)

    train, val = [], []
    for kind, groups in by_kind.items():
        remaining = list(groups.values())
        if not remaining:
            continue
        n_pairs = sum(len(g) for g in remaining)
        if len(remaining) < 2:
            # too few groups to split this kind -> put it all in train
            train.extend(p for g in remaining for p in g)
            continue
        target = max(1, int(n_pairs * val_frac))
        v, rem = _take_val_groups(remaining, target, rng)
        val.extend(v)
        train.extend(p for g in rem for p in g)

    if not train or not val:
        raise ValueError("Need enough independent source groups for non-empty train/validation.")
    return train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    domain_system = cfg["chat"]["system_prompt"]
    rng = random.Random(cfg["dataset"]["seed"])
    if cfg["dataset"].get("split_by", "source_document") != "source_document":
        raise ValueError("dataset.split_by must be 'source_document'.")
    val_frac = cfg["dataset"]["val_frac"]
    if isinstance(val_frac, bool) or not isinstance(val_frac, (int, float)) or not 0 < val_frac < 1:
        raise ValueError("dataset.val_frac must be between 0 and 1 for non-empty train/validation splits.")

    judged = list(read_jsonl(cfg["paths"]["judged_pairs"]))
    accepted = [p for p in judged if p.get("accepted") is True]
    if not accepted:
        raise ValueError("No accepted generated pairs. Check judge output before preparing SFT data.")
    for i, p in enumerate(accepted):
        for field in ("question", "answer", "source_id", "source_hash"):
            if not isinstance(p.get(field), str) or not p[field].strip():
                raise ValueError(f"Accepted pair {i}: missing/invalid {field}. Use the updated generation and judge outputs.")
    group_sources(accepted)

    # dedup by normalized question
    seen, deduped = set(), []
    for p in accepted:
        h = qhash(p["question"])
        if h not in seen:
            seen.add(h)
            deduped.append(p)
    n_dupes = len(accepted) - len(deduped)

    # Split BEFORE adding safety examples, keeping source groups together AND
    # stratifying domain/general so both appear in validation.
    train_pairs, val_pairs = split_source_groups(deduped, val_frac, rng)

    # Each pair gets the system prompt matching its kind (domain vs general).
    train = [to_messages(pair_system(p, domain_system), p["question"], p["answer"], p) for p in train_pairs]
    val = [to_messages(pair_system(p, domain_system), p["question"], p["answer"], p) for p in val_pairs]

    safety_added, safety_skipped = 0, 0
    if cfg["dataset"]["include_safety_examples"]:
        for s in SAFETY_EXAMPLES:
            h = qhash(s["q"])
            if h in seen:  # Check both generated splits and earlier manual examples.
                safety_skipped += 1
                continue
            seen.add(h)
            # Identity/safety examples use the domain (ML-assistant) system prompt.
            train.append(to_messages(domain_system, s["q"], s["a"], {"kind": "manual"}))
            safety_added += 1
    rng.shuffle(train)

    # quick diversity smoke test on train questions (find the user turn by role,
    # since general pairs have no system message so the user isn't always index 1)
    def _user_text(e):
        for m in e["messages"]:
            if m["role"] == "user":
                return m["content"]
        return ""
    firsts = {" ".join(_user_text(e).lower().split()[:3]) for e in train}
    diversity = len(firsts) / max(1, len(train))

    # domain/general breakdown of the split (for the manifest + a forgetting check)
    def _counts(pairs):
        d = sum(1 for p in pairs if p.get("kind") != "general")
        g = sum(1 for p in pairs if p.get("kind") == "general")
        return {"domain": d, "general": g}

    from datasets import Dataset
    out = repo_root() / cfg["paths"]["sft_data"]
    Dataset.from_list(train).save_to_disk(str(out / "train"))
    Dataset.from_list(val).save_to_disk(str(out / "val"))

    manifest = {
        "judged_total": len(judged),
        "accepted": len(accepted),
        "rejected": len(judged) - len(accepted),
        "duplicates_removed": n_dupes,
        "safety_examples": safety_added,
        "safety_examples_skipped": safety_skipped,
        "train": len(train), "val": len(val),
        "train_generated": len(train_pairs),
        "train_by_kind": _counts(train_pairs),
        "val_by_kind": _counts(val_pairs),
        "split_by": "source_document (stratified by kind)", "seed": cfg["dataset"]["seed"],
        "train_source_groups": len({p["source_group_id"] for p in train_pairs}),
        "val_source_groups": len({p["source_group_id"] for p in val_pairs}),
        "val_frac_requested": val_frac,
        "val_frac_actual_generated": round(len(val_pairs) / len(deduped), 4),
        "acceptance_rate": round(len(accepted) / max(1, len(judged)), 3),
        "question_diversity": round(diversity, 3),
        "teacher_model": cfg["generation"]["teacher_model"],
        "judge_model": cfg["judge"]["judge_model"],
        "teacher_revisions": sorted({p["teacher_revision"] for p in deduped if p.get("teacher_revision")}),
        "judge_revisions": sorted({p["judge_revision"] for p in deduped if p.get("judge_revision")}),
        "prompt_version": cfg["generation"]["prompt_version"],
    }
    mpath = repo_root() / cfg["paths"]["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[prepare]", json.dumps(manifest, indent=2))
    print("Hand-check ~30 examples, then: python training/sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()