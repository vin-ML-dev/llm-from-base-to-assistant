"""Step 4 — SFT training (LoRA on cpt-v1, assistant-only loss).

Trains a LoRA adapter on top of the Day 2 cpt-v1 with TRL's SFTTrainer using
assistant_only_loss=True (loss on assistant tokens only). The assistant turn's
<|im_end|> is inside the trained span, so the model learns to STOP. Saves the
adapter + tokenizer + lineage.

Usage:
    python training/sft.py --config configs/day3.yaml
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, load_tokenizer, repo_root, resolve_revision  # noqa: E402

# Text-only ChatML template. <|im_end|> is INSIDE {% generation %} so end-of-turn
# is supervised and the model learns to stop.
TUTOR_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}"
    "{{ '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def check_dataset(dataset, tokenizer, max_length, split):
    """Validate messages and confirm assistant targets survive truncation."""
    if "messages" not in dataset.column_names:
        raise ValueError(f"{split}: dataset needs a raw 'messages' column.")
    if not len(dataset):
        raise ValueError(f"{split}: empty dataset.")

    roles = {"system", "user", "assistant"}
    bad = []
    for i, row in enumerate(dataset):
        msgs = row["messages"]
        ok = isinstance(msgs, list) and msgs and all(
            isinstance(m, dict) and m.get("role") in roles and isinstance(m.get("content"), str)
            for m in msgs
        )
        if not ok:
            bad.append((i, "malformed messages")); continue
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            bad.append((i, "empty assistant answer")); continue
        enc = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True,
        )
        mask = enc.get("assistant_masks", [])
        if not any(mask[:max_length]):
            bad.append((i, "truncation removes all assistant targets"))

    if bad:
        raise ValueError(f"{split}: {len(bad)} invalid examples (first 10: {bad[:10]}).")
    print(f"{split}: all {len(dataset)} examples keep assistant targets.")
    return dataset.select_columns(["messages"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    base = cfg["student"]["base"]
    rev = cfg["student"]["base_revision"]
    data = repo_root() / cfg["paths"]["sft_data"]
    train_ds = load_from_disk(str(data / "train"))
    val_ds = load_from_disk(str(data / "val"))
    print(f"train={len(train_ds)}  val={len(val_ds)}  base={base}")

    # tokenizer: enforce ChatML + <|im_end|> as eos so the model learns to stop
    tok = load_tokenizer(base, rev)
    eos = cfg["chat"]["eos_token"]
    vocab = tok.get_vocab()
    if eos != "<|im_end|>" or eos not in vocab or "<|im_start|>" not in vocab:
        raise ValueError("Tutor template requires ChatML tokens and eos_token='<|im_end|>'.")
    tok.eos_token = eos
    if tok.pad_token_id is None:
        tok.pad_token = eos
    tok.chat_template = TUTOR_CHAT_TEMPLATE
    tok.truncation_side = "right"
    tok.padding_side = "right"

    max_length = int(tcfg["max_seq_length"])
    train_ds = check_dataset(train_ds, tok, max_length, "train")
    val_ds = check_dataset(val_ds, tok, max_length, "val")

    # model — dtype kwarg name varies across transformers versions
    dt = torch.bfloat16 if tcfg["bf16"] else torch.float32
    dtype_key = "dtype" if "dtype" in inspect.signature(
        AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(base, revision=rev, **{dtype_key: dt})
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if model.generation_config is not None:
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id
    if tcfg["gradient_checkpointing"]:
        model.config.use_cache = False

    # modules_to_save: fully train the embeddings + output head. Critical here
    # because the student is a CPT *base* model that has never emitted ChatML
    # tokens — without this, the frozen lm_head row for <|im_end|> stays noise,
    # so the model can't learn to STOP and emits junk tokens at end-of-turn.
    modules_to_save = tcfg.get("lora_modules_to_save") or ["embed_tokens", "lm_head"]
    lora = LoraConfig(
        r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
        target_modules=tcfg["lora_target_modules"],
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
    ) if tcfg["use_lora"] else None

    out_dir = repo_root() / cfg["paths"]["sft_output_dir"]
    sc_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if tcfg["assistant_only_loss"] and "assistant_only_loss" not in sc_params:
        raise RuntimeError("Installed TRL lacks assistant_only_loss; refusing to disable it silently.")

    sc_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=1,
        seed=cfg["dataset"]["seed"],
        report_to="none",
    )
    # add version-dependent keys only if this TRL supports them
    optional = {
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_ratio": tcfg["warmup_ratio"],
        "max_seq_length": tcfg["max_seq_length"],
        "max_length": tcfg["max_seq_length"],
        "gradient_checkpointing": tcfg["gradient_checkpointing"],
        "assistant_only_loss": tcfg["assistant_only_loss"],
        "eval_strategy": "steps",
        "eval_steps": tcfg["eval_steps"],
    }
    sc_kwargs.update({k: v for k, v in optional.items() if k in sc_params})
    sft_config = SFTConfig(**sc_kwargs)

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_ds, eval_dataset=val_ds,
        peft_config=lora, processing_class=tok,
    )

    print("Training (LoRA, assistant-only loss). Small data overfits fast — watch val loss.")
    start = time.time()
    trainer.train()
    runtime = time.time() - start

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    peak = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
    lineage = {
        "stage": "sft-v1",
        "chat_template_version": "tutor-chatml-assistant-mask-v1",
        "parent_model": base,
        "parent_revision_resolved": resolve_revision(base, rev),
        "method": "lora" if tcfg["use_lora"] else "full",
        "assistant_only_loss": tcfg["assistant_only_loss"],
        "epochs": tcfg["num_epochs"],
        "learning_rate": tcfg["learning_rate"],
        "sft_data_manifest": cfg["paths"]["manifest"],
        "runtime_sec": round(runtime, 1),
        "peak_vram_gib": round(peak, 2) if peak else None,
        "seed": cfg["dataset"]["seed"],
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"\nSaved sft-v1 → {out_dir}  ({runtime/60:.1f} min, peak {lineage['peak_vram_gib']} GiB)")
    print("Next: python evaluation/sft_eval.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()