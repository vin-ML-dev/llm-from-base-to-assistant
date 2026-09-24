"""Day 5 · Step 3 — Collect worst-scoring outputs into a failure-analysis DRAFT.

Scores sft-v2 and dpo-v1 answers with the judge (1-5), then writes the lowest
scoring ones into a markdown draft with blanks for you to fill in WHY it
failed and WHAT you'd try -- the reasoning has to be written by a person, this
script just gathers the evidence.

Usage:
    python evaluation/failure_analysis.py --config configs/day5.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import free_gpu, load_config, repo_root, strip_thinking  # noqa: E402
from dpo_eval import BEHAVIOR_PROMPTS  # noqa: E402  -- same 38 grouped questions as the rest of Day 5

JUDGE_PROMPT = """Rate this answer 1-5. Judge correctness first (wrong or unsupported claims are
serious flaws even if well written), then helpfulness and clarity. Do not reward length.

Return STRICT JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}

QUESTION: {q}
ANSWER: {a}
"""


def generate(model, tok, msgs, max_new):
    """Returns (answer, n_tokens, stopped); stopped=False means cut off at max_new."""
    import torch
    vocab = tok.get_vocab()
    stop_ids = sorted({tok.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=stop_ids, pad_token_id=stop_ids[0])
    new = out[0][inputs["input_ids"].shape[1]:].tolist()
    return (tok.decode(new, skip_special_tokens=True).strip(), len(new),
            bool(new) and new[-1] in stop_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day5.yaml")
    ap.add_argument("--bottom-n", type=int, default=5, help="how many worst cases to draft")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    system = cfg["chat"]["system_prompt"]
    max_new = cfg["behavior"]["max_new_tokens"]
    # general questions get no system prompt (matches training)
    msgs_for = lambda g, q: ([] if g == "general" else [{"role": "system", "content": system}]) + \
                            [{"role": "user", "content": q}]

    rows = []
    for label in ("sft", "dpo"):
        path = cfg["models"][label]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        for g, q in BEHAVIOR_PROMPTS:
            ans, n, stopped = generate(model, tok, msgs_for(g, q), max_new)
            rows.append({"model": label, "group": g, "question": q, "answer": ans,
                         "tokens": n, "stopped": stopped})
        del model
        free_gpu()

    # Judge via the chat template with thinking OFF, so it returns JSON directly.
    from transformers import AutoTokenizer as _AT
    from vllm import LLM, SamplingParams
    judge = cfg["judge"]["judge_model"]
    jtok = _AT.from_pretrained(judge)
    chat = lambda c: jtok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    llm = LLM(model=judge, dtype="bfloat16", gpu_memory_utilization=0.80, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=150)

    outputs = llm.generate([chat(JUDGE_PROMPT.format(q=r["question"], a=r["answer"])) for r in rows], sp)
    unreadable = 0
    for r, out in zip(rows, outputs):
        text = strip_thinking(out.outputs[0].text)
        try:
            obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
            score = int(obj["score"])
            assert 1 <= score <= 5 and str(obj.get("reason", "")).strip()
            r["score"], r["judge_reason"] = score, str(obj["reason"]).strip()
        except Exception:
            r["score"], r["judge_reason"] = None, "unreadable judge output"
            unreadable += 1
    del llm
    free_gpu()

    # Unreadable scores are excluded -- a judge error is not a model failure.
    scored = [r for r in rows if r["score"] is not None]
    worst = sorted(scored, key=lambda r: r["score"])[:args.bottom_n]
    for label in ("sft", "dpo"):
        s_ = [r["score"] for r in scored if r["model"] == label]
        print(f"{label}: mean score {sum(s_) / max(1, len(s_)):.2f}/5 over {len(s_)} answers")
    if unreadable:
        print(f"unreadable judge replies: {unreadable} (excluded)")

    lines = ["# Failure Analysis (draft)", "",
             "Auto-collected worst-scoring outputs. Fill in WHY and WHAT-NEXT by hand.", ""]
    for i, r in enumerate(worst, 1):
        lines += [
            f"## {i}. [{r['model']}] score={r['score']}/5  ({r['group']}, {r['tokens']} tokens, "
            f"{'stopped' if r['stopped'] else 'CUT OFF'})",
            f"**Prompt:** {r['question']}", "",
            f"**Answer:** {r['answer']}", "",
            f"**Judge's one-line reason:** {r['judge_reason']}", "",
            "**Failure type (fill in):** _e.g. hallucination / format drift / off-topic / over-verbose_", "",
            "**Why this likely happened (fill in):**", "", "**What you'd try next (fill in):**", "",
            "---", "",
        ]

    out_path = repo_root() / cfg["paths"]["failure_analysis_draft"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Drafted {len(worst)} worst cases -> {out_path}")
    print("Open it and fill in the blanks -- that reasoning is the real deliverable.")


if __name__ == "__main__":
    main()