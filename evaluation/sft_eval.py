"""Step 5 — Evaluate SFT BEHAVIOR (not perplexity).

Generates answers from BASE, CPT, and SFT on the same prompts and prints them
side by side, so you can see whether sft-v1 now answers AND stops.

Two fixes baked in:
  1. Stopping: generation stops on ANY end-of-turn token (<|im_end|> or
     <|endoftext|>), not just one — Qwen turns can end with either.
  2. No system leak: BASE and CPT are NOT instruction-tuned, so we prompt them
     in plain text with NO system line. Only SFT uses the ChatML chat template.
     (Applying a chat template to a base model makes it parrot the system text
     into its answer, e.g. "You are a helpful assistant.")

Usage:
    python evaluation/sft_eval.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, load_tokenizer, repo_root  # noqa: E402

BEHAVIOR_PROMPTS = [
    "What is attention in a transformer?",
    "Explain what a tokenizer does, simply.",
    "Give me three tips for fine-tuning an LLM.",
    "Who are you?",
]

# Qwen end-of-turn tokens. We stop on whichever the model emits.
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def load_day1_prompts(cfg):
    p = repo_root() / cfg["paths"]["before_prompts"]
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("**Prompt:**"):
                out.append(line.replace("**Prompt:**", "").strip())
    return out


def stop_ids(tok):
    """IDs of every end-of-turn token present in this vocab, plus tok.eos_token_id."""
    vocab = tok.get_vocab()
    ids = {vocab[t] for t in STOP_TOKENS if t in vocab}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    if not ids:
        raise ValueError("No end-of-turn token found; cannot evaluate stopping.")
    return sorted(ids)


def build_prompt(tok, system, question, use_chat_template):
    """SFT: real ChatML with system. BASE/CPT: plain text, NO system line."""
    if use_chat_template and tok.chat_template:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # plain fallback for non-instruction-tuned models — nothing to parrot back
    return f"Question: {question}\nAnswer:"


def generate(model, tok, prompt_text, stops, max_new=256):
    import torch
    inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=stops, pad_token_id=pad_id,
                             repetition_penalty=1.1, no_repeat_ngram_size=3)
    gen = out[0][inputs["input_ids"].shape[1]:].tolist()
    stopped = any(s in gen for s in stops)
    return tok.decode(gen, skip_special_tokens=True).strip(), stopped, len(gen)


def load_model(model_id, dtype, adapter=None):
    import inspect
    import torch
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(adapter or model_id)
    dtype_key = "dtype" if "dtype" in inspect.signature(
        AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_id, **{dtype_key: dtype})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.eval().to(device), tok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import torch
    dt = torch.bfloat16
    system = cfg["chat"]["system_prompt"]
    prompts = list(dict.fromkeys(BEHAVIOR_PROMPTS + load_day1_prompts(cfg)))

    base = "Qwen/Qwen3-1.7B-Base"
    cpt = cfg["student"]["base"]
    sft_adapter = str(repo_root() / cfg["paths"]["sft_output_dir"])

    # (label, model_id, adapter, use_chat_template)
    runs = [
        ("BASE", base, None, False),   # base model → plain prompt, no system leak
        ("CPT",  cpt,  None, False),   # cpt is still a base model → plain prompt
        ("SFT",  cpt,  sft_adapter, True),  # instruction-tuned → real chat template
    ]

    for label, model_id, adapter, use_ct in runs:
        print(f"\n{'='*70}\n### {label} model\n{'='*70}")
        model, tok = load_model(model_id, dt, adapter=adapter)
        if label == "SFT" and not tok.chat_template:
            raise ValueError("SFT tokenizer has no chat template; restore the one saved by training.")
        stops = stop_ids(tok)
        print("Prompt:", "chat template + system" if use_ct else "plain text (no system)",
              "| stop ids:", stops)
        for q in prompts:
            text = build_prompt(tok, system, q, use_ct)
            ans, stopped, ntok = generate(model, tok, text, stops)
            print(f"\nPROMPT: {q}")
            print(f"ANSWER: {ans[:400]}")
            print(f"  [stopped={stopped}  tokens={ntok}]")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nLook for: BASE/CPT ramble and rarely stop; SFT answers, STOPS "
          "(stopped=True), follows instructions, and handles 'Who are you?' "
          "without echoing the system prompt.")


if __name__ == "__main__":
    main()