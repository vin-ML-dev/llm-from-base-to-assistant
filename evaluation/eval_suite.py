"""Day 5 · Step 1 — THE HONEST TABLE: base, CPT, SFT, DPO, and the instruct baseline.

For each model, measures:
  - domain + general perplexity (on the LOCKED held-out sets -- comparable across all)
  - behavior: does it answer, does it stop at EOS, on a fixed prompt set

Prints one table and saves it to evaluation/day5_results.json.

Usage:
    python evaluation/eval_suite.py --config configs/day5.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import free_gpu, load_config, read_jsonl, repo_root  # noqa: E402

QUESTIONS = [
    "What is attention in a transformer?",
    "Explain what a tokenizer does, simply.",
    "Give me three tips for fine-tuning an LLM.",
    "What is the difference between CPT and SFT?",
    "Who are you?",
]


def corpus_perplexity(model, tok, texts, max_tokens, block, device):
    """Token-weighted perplexity: exp(total_loss / total_tokens)."""
    import torch
    total_loss, total_tok = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        for i in range(0, len(ids), block):
            chunk = ids[i:i + block].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue
            with torch.inference_mode():
                out = model(chunk, labels=chunk)
            n = chunk.shape[1] - 1
            total_loss += out.loss.item() * n
            total_tok += n
            if total_tok >= max_tokens:
                break
        if total_tok >= max_tokens:
            break
    return math.exp(total_loss / max(1, total_tok))


def behavior_check(model, tok, system, device, max_new):
    """Generate on the fixed question set; report how many stop at an end token."""
    import torch
    # Any valid end-of-turn token counts as a stop (base, instruct and our models
    # don't all use the same one).
    vocab = tok.get_vocab()
    stop_ids = sorted({tok.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab})
    answers, stopped_count = [], 0
    for q in QUESTIONS:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        # enable_thinking=False: the official Qwen3 instruct model would otherwise
        # start with a <think> monologue (unfair comparison). Other templates ignore it.
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=stop_ids, pad_token_id=stop_ids[0])
        new_tokens = out[0][inputs["input_ids"].shape[1]:].tolist()
        stopped = bool(new_tokens) and new_tokens[-1] in stop_ids
        stopped_count += int(stopped)
        answers.append({"question": q,
                        "answer": tok.decode(new_tokens, skip_special_tokens=True).strip(),
                        "stopped": stopped})
    return answers, stopped_count / len(QUESTIONS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day5.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg, bcfg = cfg["perplexity"], cfg["behavior"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    domain_texts = [r["text"] for r in list(read_jsonl(pcfg["domain_heldout"]))[:pcfg["max_docs"]]]
    general_texts = [r["text"] for r in list(read_jsonl(pcfg["general_heldout"]))[:pcfg["max_docs"]]]

    results = {}
    for label, path in cfg["models"].items():
        print(f"\n=== {label}: {path} ===")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()

        dom_ppl = corpus_perplexity(model, tok, domain_texts, pcfg["max_tokens"], pcfg["block_size"], device)
        gen_ppl = corpus_perplexity(model, tok, general_texts, pcfg["max_tokens"], pcfg["block_size"], device)
        answers, stop_rate = behavior_check(model, tok, cfg["chat"]["system_prompt"], device, bcfg["max_new_tokens"])

        results[label] = {"domain_ppl": round(dom_ppl, 2), "general_ppl": round(gen_ppl, 2),
                          "stop_rate": round(stop_rate, 2), "answers": answers}
        print(f"domain_ppl={dom_ppl:.2f}  general_ppl={gen_ppl:.2f}  stop_rate={stop_rate:.0%}")

        del model
        free_gpu()

    out_path = repo_root() / cfg["paths"]["results"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== THE HONEST TABLE ===")
    print(f"{'model':10s} {'domain_ppl':>12s} {'general_ppl':>12s} {'stop_rate':>10s}")
    for label, r in results.items():
        print(f"{label:10s} {r['domain_ppl']:12.2f} {r['general_ppl']:12.2f} {r['stop_rate']:10.0%}")
    print(f"\nSaved -> {out_path}")
    print("Expect: instruct wins general breadth; your pipeline competitive/wins on domain.")
    print("Next: python evaluation/win_rate_vs_instruct.py --config configs/day5.yaml")


if __name__ == "__main__":
    main()
