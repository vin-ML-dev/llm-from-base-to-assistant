"""Day 3 · Step 1 — GENERATE instruction pairs with the Qwen3-14B teacher.

Feeds cleaned domain-text chunks to the teacher (via vLLM) and asks for a grounded
Q&A pair. Records full provenance per pair (teacher model + immutable revision,
prompt version, generation settings, source id). Qwen3 thinking-mode is stripped.

Writes data/sft/raw_pairs.jsonl. Load the teacher, generate, then UNLOAD so the
8B judge can fit afterward.

Usage:
    python data/generate_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

from sft_common import (free_gpu, load_config, read_jsonl, repo_root,
                        resolve_revision, strip_thinking, write_jsonl)

GEN_PROMPT = """You are creating a training example for an LLM/ML tutor.

Read the SOURCE TEXT below and write ONE question a learner might ask, plus a clear,
correct answer grounded ONLY in that text. Keep the answer concise and self-contained.
Do not mention "the text" or "the passage" — answer as a tutor would.

Return STRICT JSON only, no other words:
{{"question": "...", "answer": "..."}}

SOURCE TEXT:
{chunk}
"""


def load_source_chunks(cfg):
    """Pull cleaned DOMAIN docs (papers/docs/theory) and cut into chunks."""
    clean = cfg["paths"]["clean_dir"]
    files = ["domain_papers.jsonl", "domain_docs.jsonl",
             "theory_web.jsonl", "theory_d2l.jsonl", "theory_surveys.jsonl"]
    docs = []
    for f in files:
        try:
            docs.extend(read_jsonl(f"{clean}/{f}"))
        except FileNotFoundError:
            pass
    random.Random(cfg["dataset"]["seed"]).shuffle(docs)
    docs = docs[: cfg["generation"]["max_source_docs"]]

    size = cfg["generation"]["chunk_chars"]
    chunks = []
    for d in docs:
        t = d["text"]
        for i in range(0, len(t), size):
            piece = t[i:i + size]
            if len(piece) > 400:
                sid = d.get("id") or d.get("url") or "doc"
                chunks.append({"source_id": f"{sid}#{i}",
                               "source_hash": hashlib.sha1(piece.encode()).hexdigest()[:12],
                               "text": piece})
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gcfg = cfg["generation"]

    from vllm import LLM, SamplingParams

    chunks = load_source_chunks(cfg)
    target = gcfg["target_pairs"]
    random.Random(cfg["dataset"]["seed"]).shuffle(chunks)
    chunks = chunks[:target]
    print(f"[generate] {len(chunks)} source chunks → up to {target} pairs")

    teacher = gcfg["teacher_model"]
    rev = resolve_revision(teacher, gcfg["teacher_revision"])
    print(f"[generate] loading teacher {teacher} (rev {rev[:8]}...) via vLLM")
    llm = LLM(model=teacher, revision=gcfg["teacher_revision"],
              dtype=cfg["student"]["dtype"], gpu_memory_utilization=0.90,
              max_model_len=4096)
    sp = SamplingParams(temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                        max_tokens=gcfg["max_new_tokens"])

    prompts = [GEN_PROMPT.format(chunk=c["text"]) for c in chunks]
    outputs = llm.generate(prompts, sp)

    rows = []
    for c, out in zip(chunks, outputs):
        text = strip_thinking(out.outputs[0].text) if gcfg["disable_thinking"] else out.outputs[0].text
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            q, a = obj["question"].strip(), obj["answer"].strip()
        except Exception:
            continue  # skip malformed generations
        if len(q) < 8 or len(a) < 20:
            continue
        rows.append({
            "question": q, "answer": a,
            "source_id": c["source_id"], "source_hash": c["source_hash"],
            "teacher_model": teacher, "teacher_revision": rev,
            "prompt_version": gcfg["prompt_version"],
            "generation_settings": {"temperature": gcfg["temperature"],
                                    "top_p": gcfg["top_p"],
                                    "max_new_tokens": gcfg["max_new_tokens"]},
        })

    n = write_jsonl(cfg["paths"]["raw_pairs"], rows)
    print(f"[generate] wrote {n} pairs → {cfg['paths']['raw_pairs']}")

    # free the teacher so the 8B judge can load next
    del llm
    free_gpu()
    print("[generate] teacher unloaded. Next: python data/judge_sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
