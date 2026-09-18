"""Step 2 — JUDGE the pairs with the Qwen3-8B judge.

Uses a different model (8B) than the teacher (14B) to reduce self-preference bias.
Scores each pair 1-5 on correctness/clarity/format and keeps pairs at or above
accept_threshold. Run AFTER generate_sft.py has unloaded the teacher.

Usage:
    python data/judge_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse

from sft_common import (extract_json, free_gpu, load_config, read_jsonl,
                        resolve_revision, strip_thinking, write_jsonl)

JUDGE_PROMPT = """You are grading a Q&A pair for an LLM/ML tutor dataset.

Rate the ANSWER from 1 to 5 on correctness, clarity, and whether it directly and
completely answers the QUESTION. 5 = excellent, 1 = wrong or unusable.

Return STRICT JSON only:
{{"score": <1-5>, "reason": "<short reason>"}}

QUESTION: {q}
ANSWER: {a}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    jcfg = cfg["judge"]

    from vllm import LLM, SamplingParams

    pairs = list(read_jsonl(cfg["paths"]["raw_pairs"]))
    print(f"[judge] scoring {len(pairs)} pairs")

    judge = jcfg["judge_model"]
    rev = resolve_revision(judge, jcfg["judge_revision"])
    print(f"[judge] loading judge {judge} (rev {rev[:8]}) via vLLM")
    llm = LLM(model=judge, revision=jcfg["judge_revision"],
              dtype=cfg["student"]["dtype"], gpu_memory_utilization=0.90,
              max_model_len=4096)
    sp = SamplingParams(temperature=jcfg["temperature"],
                        max_tokens=jcfg["max_new_tokens"])

    # Apply the judge's chat template with thinking OFF (fallback to raw prompt
    # if the template can't be applied), so the short JSON verdict isn't crowded
    # out by reasoning.
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(judge, revision=jcfg["judge_revision"])
    except Exception as e:
        print(f"[judge] tokenizer load failed ({e}); using raw prompts")
        tok = None
    think_off = bool(jcfg["disable_thinking"])

    def to_prompt(q, a):
        content = JUDGE_PROMPT.format(q=q, a=a)
        if tok is None or tok.chat_template is None:
            return content
        messages = [{"role": "user", "content": content}]
        for kwargs in ({"enable_thinking": not think_off}, {}):
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                continue
        return content

    prompts = [to_prompt(p["question"], p["answer"]) for p in pairs]
    outputs = llm.generate(prompts, sp)

    threshold = jcfg["accept_threshold"]
    rows, accepted, unparseable = [], 0, 0
    for p, out in zip(pairs, outputs):
        text = out.outputs[0].text
        if think_off:
            text = strip_thinking(text)
        obj = extract_json(text)
        if obj and "score" in obj:
            score = int(obj["score"])
            reason = str(obj.get("reason", ""))[:200]
        else:
            score, reason = 0, "unparseable judge output"
            unparseable += 1
        p.update(judge_model=judge, judge_revision=rev, score=score,
                 reason=reason, accepted=score >= threshold)
        accepted += p["accepted"]
        rows.append(p)

    n = write_jsonl(cfg["paths"]["judged_pairs"], rows)
    print(f"[judge] wrote {n} judged pairs → {cfg['paths']['judged_pairs']}")
    print(f"[judge] accepted {accepted} / {n} "
          f"(threshold >= {threshold}, {100 * accepted / max(1, n):.0f}%) "
          f"| unparseable={unparseable}")
    if unparseable == n and n:
        raise SystemExit("[judge] every verdict was unparseable — check disable_thinking "
                         "and that the judge emits STRICT JSON.")

    del llm
    free_gpu()
    print("[judge] Next: python data/prepare_sft.py")


if __name__ == "__main__":
    main()