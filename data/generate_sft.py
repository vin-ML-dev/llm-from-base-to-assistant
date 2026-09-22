"""Step 1 — GENERATE instruction pairs with the Qwen3-14B teacher.

Feeds cleaned domain-text chunks to the teacher (via vLLM) and asks for one
grounded Q&A pair per chunk. Records provenance. Over-provisions chunks so that
the configured target can survive parsing/length filtering. Uses only the
protected CPT training pool. Unloads the teacher at the end.

Usage:
    python data/generate_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import random

from sft_common import (extract_json, free_gpu, load_config, load_tokenizer, read_jsonl,
                        resolve_revision, strip_thinking, write_jsonl)

GEN_PROMPT = """You are creating a training example for an LLM/ML tutor.

Read the SOURCE TEXT below and write ONE question a learner might ask, plus a
clear, correct answer grounded ONLY in that text. Keep the answer concise and
self-contained. Do not mention "the text" or "the passage" — answer as a tutor would.

Return STRICT JSON only, no other words:
{{"question": "...", "answer": "..."}}

SOURCE TEXT:
{chunk}
"""

# General (non-domain) prompt: keeps broad instruction-following ability so SFT
# doesn't narrow the model to only ML topics (anti-forgetting, mirrors CPT replay).
GEN_PROMPT_GENERAL = """You are creating a general instruction-following training example.

Read the SOURCE TEXT below and write ONE natural question a curious person might
ask about its topic, plus a clear, correct, self-contained answer grounded ONLY
in that text. The topic may
be anything (science, history, everyday knowledge). Answer as a helpful assistant
would — do not mention "the text" or "the passage".

Return STRICT JSON only, no other words:
{{"question": "...", "answer": "..."}}

SOURCE TEXT:
{chunk}
"""

def load_source_chunks(cfg, limit, kind):
    if limit <= 0:
        return []
    docs = []
    for d in read_jsonl(cfg["paths"]["train_pool"]):
        if d.get("kind") not in {"domain", "replay"}:
            raise ValueError("Every training-pool record must have kind='domain' or 'replay'.")
        if d["kind"] == kind:
            docs.append(d)
    random.Random(cfg["dataset"]["seed"]).shuffle(docs)

    size = cfg["generation"]["chunk_chars"]
    max_chunks = cfg["generation"].get("max_chunks_per_doc", 6)
    max_docs = cfg["generation"].get("max_source_docs", 0)
    if size <= 0 or max_chunks <= 0 or max_docs < 0:
        raise ValueError("chunk_chars/max_chunks_per_doc must be positive; max_source_docs must be >= 0.")
    if kind == "domain" and max_docs:
        docs = docs[:max_docs]
    chunks = []
    for d in docs:
        t = d["text"]
        doc_hash = hashlib.sha256(t.encode("utf-8")).hexdigest()
        sid = d.get("id") or d.get("url") or doc_hash
        doc_id = f"{d.get('source') or kind}:{sid}"
        per_doc = 0
        for i in range(0, len(t), size):
            piece = t[i:i + size]
            if len(piece) <= 400:
                continue
            chunks.append({
                "source_id": f"{sid}#{i}",
                "source_hash": hashlib.sha1(piece.encode()).hexdigest()[:12],
                "source_doc_id": doc_id,
                "source_doc_hash": doc_hash,
                "source": d.get("source"),
                "source_title": d.get("title"),
                "source_url": d.get("url"),
                "source_license": d.get("license"),
                "text": piece,
            })
            per_doc += 1
            if per_doc >= max_chunks:
                break
        if len(chunks) >= limit:
            break
    random.Random(cfg["dataset"]["seed"]).shuffle(chunks)
    return chunks[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gcfg = cfg["generation"]

    from vllm import LLM, SamplingParams

    target = gcfg["target_pairs"]                       # domain target
    target_general = gcfg.get("target_general_pairs", 0)  # general target

    oversample = gcfg.get("oversample", 1.4)
    if target <= 0 or target_general < 0 or oversample < 1:
        raise ValueError("target_pairs must be positive, target_general_pairs >= 0, and oversample >= 1.")
    domain_chunks = load_source_chunks(cfg, int(target * oversample), "domain")
    general_chunks = (load_source_chunks(cfg, int(target_general * oversample), "replay")
                      if target_general else [])
    print(f"[generate] domain chunks={len(domain_chunks)} (target {target}) | "
          f"general chunks={len(general_chunks)} (target {target_general})")
    if not domain_chunks:
        raise SystemExit(
            f"[generate] no domain chunks. Check {cfg['paths']['train_pool']} "
            "contains kind='domain' records with enough text.")

    teacher = gcfg["teacher_model"]
    rev = resolve_revision(teacher, gcfg["teacher_revision"])
    tok = load_tokenizer(teacher, revision=rev)
    if not tok.chat_template:
        raise ValueError("Teacher tokenizer has no chat template; cannot apply thinking settings.")
    print(f"[generate] loading teacher {teacher} (rev {rev[:8]}) via vLLM")
    llm = LLM(model=teacher, revision=rev, tokenizer_revision=rev,
              dtype=cfg["student"]["dtype"], gpu_memory_utilization=0.90,
              max_model_len=4096)
    sp = SamplingParams(temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                        max_tokens=gcfg["max_new_tokens"], seed=cfg["dataset"]["seed"])
    think_off = bool(gcfg["disable_thinking"])

    def to_prompt(chunk_text, template):
        content = template.format(chunk=chunk_text)
        messages = [{"role": "user", "content": content}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=not think_off)

    def generate_pairs(chunks, template, want, kind):
        """Run the teacher over chunks, parse JSON pairs, keep up to `want`."""
        if not chunks:
            return []
        prompts = [to_prompt(c["text"], template) for c in chunks]
        print(f"[generate:{kind}] think_off={think_off}; sample prompt head:\n"
              f"{prompts[0][:200]!r}")
        outputs = llm.generate(prompts, sp)
        out_rows, drop_no_json, drop_invalid, drop_short, shown = [], 0, 0, 0, 0
        for c, out in zip(chunks, outputs):
            text = out.outputs[0].text
            if think_off:
                text = strip_thinking(text)
            obj = extract_json(text)
            if not obj:
                drop_no_json += 1
                if shown < 2:
                    print(f"[generate:{kind}] no JSON (sample): {text[:150]!r}")
                    shown += 1
                continue
            q, a = obj.get("question"), obj.get("answer")
            if not isinstance(q, str) or not isinstance(a, str):
                drop_invalid += 1
                continue
            q, a = q.strip(), a.strip()
            if len(q) < 8 or len(a) < 20:
                drop_short += 1
                continue
            out_rows.append({
                "question": q, "answer": a, "kind": kind,
                "source_id": c["source_id"], "source_hash": c["source_hash"],
                "source_doc_id": c["source_doc_id"],
                "source_doc_hash": c["source_doc_hash"],
                "source_text": c["text"],
                "source": c["source"], "source_title": c["source_title"],
                "source_url": c["source_url"], "source_license": c["source_license"],
                "teacher_model": teacher, "teacher_revision": rev,
                "prompt_version": gcfg["prompt_version"],
                "generation_settings": {
                    "temperature": gcfg["temperature"], "top_p": gcfg["top_p"],
                    "max_new_tokens": gcfg["max_new_tokens"],
                    "seed": cfg["dataset"]["seed"], "disable_thinking": think_off,
                },
            })
            if len(out_rows) >= want:
                break
        print(f"[generate:{kind}] kept {len(out_rows)} | dropped no-JSON={drop_no_json} "
              f"invalid-fields={drop_invalid} too-short={drop_short} (of {len(outputs)} outputs)")
        if len(out_rows) < want:
            print(f"[generate:{kind}] below target: {len(out_rows)}/{want} pairs.")
        return out_rows

    domain_rows = generate_pairs(domain_chunks, GEN_PROMPT, target, "domain")
    general_rows = generate_pairs(general_chunks, GEN_PROMPT_GENERAL, target_general, "general")

    if not domain_rows:
        raise SystemExit(
            "[generate] 0 domain pairs. The teacher returned no parseable JSON. Likely:\n"
            "  - thinking still ON (check disable_thinking + chat template applied)\n"
            "  - max_new_tokens too small; or teacher ignoring 'STRICT JSON only'\n")

    n_dom = write_jsonl(cfg["paths"]["raw_pairs"], domain_rows)
    print(f"[generate] wrote {n_dom} domain pairs → {cfg['paths']['raw_pairs']}")
    n_gen = write_jsonl(cfg["paths"]["raw_general_pairs"], general_rows)
    print(f"[generate] wrote {n_gen} general pairs → {cfg['paths']['raw_general_pairs']}")

    del llm
    free_gpu()
    print("[generate] teacher unloaded. Next: python data/judge_sft.py")


if __name__ == "__main__":
    main()
