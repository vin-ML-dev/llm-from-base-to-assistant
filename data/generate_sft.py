"""Step 1 — GENERATE instruction pairs with the Qwen3-14B teacher.

Feeds cleaned domain-text chunks to the teacher (via vLLM) and asks for one
grounded Q&A pair per chunk. Records provenance. Over-provisions chunks so that
~2000 pairs survive parsing/length filtering. Unloads the teacher at the end.

Usage:
    python data/generate_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import random

from sft_common import (extract_json, free_gpu, load_config, read_jsonl,
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

# We lose some generations to bad JSON / short answers, so make more chunks
# than the target and stop once we have enough.
OVERSAMPLE = 1.4
MAX_CHUNKS_PER_DOC = 6  # keep one long doc from dominating the set


def load_source_chunks(cfg, limit):
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

    size = cfg["generation"]["chunk_chars"]
    chunks = []
    for d in docs:
        t = d["text"]
        per_doc = 0
        for i in range(0, len(t), size):
            piece = t[i:i + size]
            if len(piece) <= 400:
                continue
            sid = d.get("id") or d.get("url") or "doc"
            chunks.append({
                "source_id": f"{sid}#{i}",
                "source_hash": hashlib.sha1(piece.encode()).hexdigest()[:12],
                "text": piece,
            })
            per_doc += 1
            if per_doc >= MAX_CHUNKS_PER_DOC:
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

    target = gcfg["target_pairs"]  # set in the config
    n_chunks = int(target * OVERSAMPLE)
    chunks = load_source_chunks(cfg, n_chunks)
    print(f"[generate] {len(chunks)} chunks → aiming for {target} pairs")
    if not chunks:
        raise SystemExit(
            f"[generate] no source chunks. Check {cfg['paths']['clean_dir']}/ contains the "
            "cleaned JSONL files (domain_papers.jsonl, domain_docs.jsonl, theory_*.jsonl) "
            "with a 'text' field. Nothing to generate from.")

    teacher = gcfg["teacher_model"]
    rev = resolve_revision(teacher, gcfg["teacher_revision"])
    print(f"[generate] loading teacher {teacher} (rev {rev[:8]}) via vLLM")
    llm = LLM(model=teacher, revision=gcfg["teacher_revision"],
              dtype=cfg["student"]["dtype"], gpu_memory_utilization=0.90,
              max_model_len=4096)
    sp = SamplingParams(temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                        max_tokens=gcfg["max_new_tokens"])

    # Build prompts through the teacher's chat template so Qwen3 thinking mode is
    # actually turned OFF (disable_thinking here is the real switch; stripping
    # <think> text afterward is the fallback). If the template can't be applied
    # for any reason, fall back to the raw prompt rather than crashing the run.
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(teacher, revision=gcfg["teacher_revision"])
    except Exception as e:
        print(f"[generate] tokenizer load failed ({e}); using raw prompts")
        tok = None
    think_off = bool(gcfg["disable_thinking"])

    def to_prompt(chunk_text):
        content = GEN_PROMPT.format(chunk=chunk_text)
        if tok is None or tok.chat_template is None:
            return content
        messages = [{"role": "user", "content": content}]
        for kwargs in ({"enable_thinking": not think_off}, {}):
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                continue
        return content  # template unusable → raw prompt

    prompts = [to_prompt(c["text"]) for c in chunks]
    # Show one rendered prompt so thinking-mode is visibly on/off in the logs.
    print(f"[generate] think_off={think_off}; sample prompt head:\n"
          f"{prompts[0][:300]!r}")
    outputs = llm.generate(prompts, sp)

    rows = []
    drop_no_json = drop_short = 0
    sample_shown = 0
    for c, out in zip(chunks, outputs):
        text = out.outputs[0].text
        if think_off:
            text = strip_thinking(text)
        obj = extract_json(text)
        if not obj:
            drop_no_json += 1
            if sample_shown < 2:  # show why the first couple failed
                print(f"[generate] no JSON in output (sample): {text[:200]!r}")
                sample_shown += 1
            continue
        q = str(obj.get("question", "")).strip()
        a = str(obj.get("answer", "")).strip()
        if len(q) < 8 or len(a) < 20:
            drop_short += 1
            continue
        rows.append({
            "question": q, "answer": a,
            "source_id": c["source_id"], "source_hash": c["source_hash"],
            "teacher_model": teacher, "teacher_revision": rev,
            "prompt_version": gcfg["prompt_version"],
            "generation_settings": {
                "temperature": gcfg["temperature"],
                "top_p": gcfg["top_p"],
                "max_new_tokens": gcfg["max_new_tokens"],
            },
        })
        if len(rows) >= target:
            break

    print(f"[generate] kept {len(rows)}  |  dropped: no-JSON={drop_no_json} "
          f"too-short={drop_short}  (of {len(outputs)} outputs)")
    if not rows:
        raise SystemExit(
            "[generate] 0 pairs. The teacher returned no parseable JSON. Likely causes:\n"
            "  - thinking still ON (check disable_thinking + that the chat template applied)\n"
            "  - max_new_tokens too small for answer after reasoning (raise it)\n"
            "  - teacher ignoring the 'STRICT JSON only' instruction\n"
            "See the sample outputs above."
        )

    n = write_jsonl(cfg["paths"]["raw_pairs"], rows)
    print(f"[generate] wrote {n} pairs → {cfg['paths']['raw_pairs']}")

    del llm
    free_gpu()
    print("[generate] teacher unloaded. Next: python data/judge_sft.py")


if __name__ == "__main__":
    main()