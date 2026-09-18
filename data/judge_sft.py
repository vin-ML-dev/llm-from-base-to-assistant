"""Day 3 · Step 2 — JUDGE the pairs with the Qwen3-8B judge.

Loads a DIFFERENT model (8B) than the teacher (14B) to reduce self-preference bias.
Scores each pair 1-5 on correctness/clarity/format; keeps pairs >= accept_threshold.
Records the score + reason per pair. Run AFTER generate_sft.py has unloaded the teacher.

Usage:
    python data/judge_sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import json

from sft_common import (free_gpu, load_config, read_jsonl, resolve_revision,
                        strip_thinking, write_jsonl)

JUDGE_PROMPT = """You are grading a Q&A pair for an LLM/ML tutor dataset.

Rate the ANSWER from 1 to 5 on: correctness, clarity, and whether it directly and
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
    print(f"[judge] loading judge {judge} (rev {rev[:8]}...) via vLLM")
    llm = LLM(model=judge, revision=jcfg["judge_revision"],
              dtype=cfg["student"]["dtype"], gpu_memory_utilization=0.90,
              max_model_len=2048)
    sp = SamplingParams(temperature=jcfg["temperature"], max_tokens=jcfg["max_new_tokens"])

    prompts = [JUDGE_PROMPT.format(q=p["question"], a=p["answer"]) for p in pairs]
    outputs = llm.generate(prompts, sp)

    threshold = jcfg["accept_threshold"]
    accepted, rejected = 0, 0
    rows = []
    for p, out in zip(pairs, outputs):
        text = strip_thinking(out.outputs[0].text) if jcfg["disable_thinking"] else out.outputs[0].text
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            score = int(obj["score"]); reason = str(obj.get("reason", ""))[:200]
        except Exception:
            score, reason = 0, "unparseable judge output"
        p["judge_model"] = judge
        p["judge_revision"] = rev
        p["score"] = score
        p["reason"] = reason
        p["accepted"] = score >= threshold
        if p["accepted"]:
            accepted += 1
        else:
            rejected += 1
        rows.append(p)

    n = write_jsonl(cfg["paths"]["judged_pairs"], rows)
    print(f"[judge] wrote {n} judged pairs → {cfg['paths']['judged_pairs']}")
    print(f"[judge] accepted {accepted} / rejected {rejected} "
          f"(threshold >= {threshold}) → acceptance {100*accepted/max(1,n):.0f}%")
    del llm
    free_gpu()
    print("[judge] Next: python data/prepare_sft.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
