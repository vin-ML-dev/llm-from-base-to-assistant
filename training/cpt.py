"""
Continued Pre-Training (CPT) for Qwen3-1.7B-Base.

This script continues pre-training a base model on your own packed text corpus,
in one of two modes (set in the YAML under train.method):

    "lora" -> train small adapter layers, then merge them into the base weights.
    "full" -> train all model weights directly.

Either way, the result written to output_dir is a normal standalone model you
can load later with AutoModelForCausalLM.from_pretrained(output_dir).

Before and after training it runs the same prompts through the model so you can
see what changed.

Usage:
    python cpt.py --config day2_cpt.yaml
"""

import argparse
import json
import random
import shutil
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dataio import load_config, repo_root
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data checks
# These catch the most common ways packed data can be silently broken.
# ---------------------------------------------------------------------------
def check_blocks_are_full_length(dataset, block_size, name):
    """Every packed block must be exactly block_size tokens long."""
    for i, row in enumerate(dataset):
        n = len(row["input_ids"])
        if n != block_size:
            raise ValueError(
                f"{name} block {i} has {n} tokens, expected {block_size}. "
                "Re-run the packing step before training."
            )


def summarize_tokens(dataset, eos_id):
    """Count total tokens, EOS tokens, and the min/max token id seen."""
    total = 0
    eos_count = 0
    min_id = None
    max_id = None
    for row in dataset:
        ids = row["input_ids"]
        total += len(ids)
        if not ids:
            continue
        lo, hi = min(ids), max(ids)
        min_id = lo if min_id is None else min(min_id, lo)
        max_id = hi if max_id is None else max(max_id, hi)
        if eos_id is not None:
            eos_count += sum(1 for t in ids if t == eos_id)
    return {
        "tokens": total,
        "eos_count": eos_count,
        "min_token_id": -1 if min_id is None else min_id,
        "max_token_id": -1 if max_id is None else max_id,
    }


# ---------------------------------------------------------------------------
# Generation (used for the before/after comparison)
# ---------------------------------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens):
    import torch

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = deterministic, so runs are comparable
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_prompts(model, tokenizer, prompts, max_new_tokens):
    return [
        {"prompt": p, "answer": generate(model, tokenizer, p, max_new_tokens)}
        for p in prompts
    ]


def print_comparison(base_rows, cpt_rows):
    print("\n" + "=" * 80)
    print("BASE vs CPT  (same prompts, greedy decoding)")
    print("=" * 80)
    for i, (b, c) in enumerate(zip(base_rows, cpt_rows), start=1):
        print(f"\n[{i}] PROMPT: {b['prompt']}")
        print(f"BASE: {b['answer']}")
        print(f"CPT : {c['answer']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    root = repo_root()
    
    method = tcfg.get("method", "lora").lower()
    if method not in ("lora", "full"):
        raise ValueError(f"train.method must be 'lora' or 'full', got {method!r}")

    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected. This config is meant to run on a GPU.")

    model_id = cfg["model"]["id"]
    block_size = int(cfg["tokenize"]["block_size"])
    seed = int(cfg["data"]["seed"])

    # --- load tokenizer and the base config (we do NOT change the tokenizer) ---
    print(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_cfg = AutoConfig.from_pretrained(model_id)
    print(
        f"Tokenizer: eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id} "
        f"| model vocab_size={base_cfg.vocab_size}"
    )

    # --- load the packed data from disk ---
    packed = root / cfg["paths"]["packed_dir"]
    train_ds = load_from_disk(str(packed / "train"))
    val_ds = load_from_disk(str(packed / "val"))
    
    #train_ds = load_from_disk(cfg["paths"]["packed_dir"])
    #val_ds = load_from_disk(cfg["paths"]["packed_val_dir"])
    print(f"train blocks: {len(train_ds)} | val blocks: {len(val_ds)}")

    # --- validate the data before spending GPU time on it ---
    check_blocks_are_full_length(train_ds, block_size, "train")
    check_blocks_are_full_length(val_ds, block_size, "val")
    print(f"OK: every block is exactly {block_size} tokens")

    eos_id = tokenizer.eos_token_id
    train_stats = summarize_tokens(train_ds, eos_id)
    val_stats = summarize_tokens(val_ds, eos_id)
    print(f"train stats: {train_stats}")
    print(f"val stats  : {val_stats}")

    # token ids must fit inside the model's vocabulary
    for name, stats in (("train", train_stats), ("val", val_stats)):
        if stats["max_token_id"] >= base_cfg.vocab_size:
            raise ValueError(
                f"{name} has token id {stats['max_token_id']} but model vocab_size "
                f"is {base_cfg.vocab_size}. Data and model do not match."
            )

    # packed data should contain EOS tokens marking document boundaries
    if tcfg.get("require_eos_in_packed", True) and train_stats["eos_count"] == 0:
        raise ValueError(
            "No EOS tokens found in packed training data. Re-run packing so an EOS "
            "token separates documents, or set require_eos_in_packed: false."
        )

    # --- peek at a couple of blocks so you can spot junk/scraped-menu text ---
    print("\n--- sample of packed text ---")
    rng = random.Random(seed)
    for i in rng.sample(range(len(train_ds)), min(2, len(train_ds))):
        text = tokenizer.decode(train_ds[i]["input_ids"], skip_special_tokens=False)
        print(f"[block {i}] {text[:400]!r}\n")

    # --- prepare the output directory ---
    out_dir = Path(cfg["paths"]["cpt_output_dir"])
    if tcfg.get("clean_output_dir", False) and out_dir.exists():
        print(f"Cleaning output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load the model weights ---
    dtype = torch.bfloat16 if tcfg.get("bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    # --- generate BASE answers first (before any training) ---
    comp = cfg.get("comparison", {})
    prompts = comp.get("prompts", ["Who are you?"])
    max_new_tokens = int(comp.get("max_new_tokens", 128))

    print("\nGenerating BASE answers (before training)...")
    base_rows = run_prompts(model, tokenizer, prompts, max_new_tokens)

    # --- wrap in LoRA if requested ---
    if method == "lora":
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(tcfg["lora_r"]),
            lora_alpha=int(tcfg["lora_alpha"]),
            lora_dropout=float(tcfg["lora_dropout"]),
            target_modules=tcfg["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        print("Mode: LoRA (base weights frozen, only adapters train).")
    else:
        print("Mode: FULL (all weights train). Higher memory and forgetting risk.")

    # --- enable gradient checkpointing to save memory ---
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if method == "lora" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False  # required with gradient checkpointing

    # --- data collator: labels = input_ids (standard causal-LM next-token loss) ---
    def collate(features):
        ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    # --- training arguments ---
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tcfg["num_epochs"]),
        per_device_train_batch_size=int(tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum_steps"]),
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=float(tcfg["warmup_ratio"]),
        weight_decay=float(tcfg["weight_decay"]),
        max_grad_norm=float(tcfg["max_grad_norm"]),
        bf16=bool(tcfg["bf16"]),
        logging_steps=int(tcfg["logging_steps"]),
        save_steps=int(tcfg["save_steps"]),
        eval_steps=int(tcfg["eval_steps"]),
        eval_strategy="steps",
        save_total_limit=1,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    print("\nStarting training. A healthy run: train loss drifts down, val loss stays stable.")
    start = time.time()
    trainer.train()
    runtime = time.time() - start

    # --- restore inference settings ---
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True

    # --- if LoRA, merge adapters into base so we save one standalone model ---
    if method == "lora":
        print("Merging LoRA adapters into base weights...")
        model = model.merge_and_unload()
    model.eval()

    # --- save the final model + tokenizer ---
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved model to {out_dir}")

    # --- generate CPT answers and compare ---
    print("\nGenerating CPT answers (after training)...")
    cpt_rows = run_prompts(model, tokenizer, prompts, max_new_tokens)
    print_comparison(base_rows, cpt_rows)

    # --- write comparison + a small run summary to disk ---
    (out_dir / "base_vs_cpt.json").write_text(
        json.dumps(
            {"method": method, "base": base_rows, "cpt": cpt_rows},
            indent=2,
            ensure_ascii=False,
        )
    )

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else None
    )
    summary = {
        "method": method,
        "base_model": model_id,
        "block_size": block_size,
        "train_tokens": train_stats["tokens"],
        "learning_rate": float(tcfg["learning_rate"]),
        "effective_batch_blocks": int(tcfg["per_device_batch_size"]) * int(tcfg["grad_accum_steps"]),
        "runtime_min": round(runtime / 60, 1),
        "peak_vram_gib": round(peak_vram, 2) if peak_vram else None,
        "seed": seed,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. runtime {summary['runtime_min']} min | peak VRAM {summary['peak_vram_gib']} GiB")
    print("Check the answers above are coherent before moving on to SFT.")


if __name__ == "__main__":
    main()
