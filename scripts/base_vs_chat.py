"""Day 1 — base model behavior + chat template.

Two things at once:
  1. Show the BASE model *continuing* text instead of answering questions. This
     is your 'before' evidence — after Day 3 (SFT) the same prompts should be
     answered and the model should stop at EOS.
  2. Print what the chat template turns a conversation into (the serialized
     token string the model actually sees).

Usage:
    python scripts/base_vs_chat.py --config configs/day1.yaml
    python scripts/base_vs_chat.py --config configs/day1.yaml --save docs/before_evidence.md
"""
from __future__ import annotations

import argparse
from datetime import datetime

from _common import add_common_args, load_config, load_model_and_tokenizer


BEFORE_PROMPTS = [
    "What is the capital of France?",
    "Give me three tips for writing clear documentation.",
    "Explain what a tokenizer does, in simple terms.",
]


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--save", default=None, help="Optional path to save before-evidence markdown.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model, tok = load_model_and_tokenizer(cfg["model"])

    lines = [
        "# Before-evidence: base model does not chat",
        "",
        f"- Model: `{cfg['model']['id']}`",
        f"- Captured: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "The base model *continues text* rather than answering. Compare against the",
        "SFT model on Day 3 with the SAME prompts.",
        "",
    ]

    for prompt in BEFORE_PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["generation"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"PROMPT: {prompt}")
        print(f"BASE  : {text!r}\n")
        lines += [f"**Prompt:** {prompt}", "", f"**Base output:** {text!r}", "", "---", ""]

    # Show the chat template serialization (base tokenizers may or may not ship one).
    print("=== chat template serialization ===")
    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    try:
        templated = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        print(templated)
        lines += ["## Chat template", "", "```", templated, "```", ""]
    except Exception as e:  # noqa: BLE001
        msg = f"(this base tokenizer has no chat template: {e})"
        print(msg)
        lines += ["## Chat template", "", msg, ""]

    if args.save:
        with open(args.save, "w") as f:
            f.write("\n".join(lines))
        print(f"\nSaved before-evidence to {args.save}")


if __name__ == "__main__":
    main()
