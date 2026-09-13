"""Day 2 · Step 5 — CONTINUED PRETRAINING (full fine-tuning).

Full fine-tunes Qwen3-1.7B-Base on the packed domain+replay blocks using the
causal-LM objective (next-token prediction, loss on all tokens). Tuned for a
single ~44GB A40. Saves the adapted model to artifacts/cpt-v1 with lineage.

Usage:
    python training/cpt.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dataio import load_config, repo_root, update_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]

    import torch
    from datasets import load_from_disk
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    packed = repo_root() / cfg["paths"]["packed_dir"]
    train_ds = load_from_disk(str(packed / "train"))
    val_ds = load_from_disk(str(packed / "val"))
    print(f"train blocks: {len(train_ds)} | val blocks: {len(val_ds)}")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dtype arg name differs across transformers versions (torch_dtype -> dtype).
    import inspect as _inspect
    _load_kwargs = dict(revision=cfg["model"]["revision"])
    _dt = torch.bfloat16 if tcfg["bf16"] else torch.float32
    _from_pretrained_params = set(
        _inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    )
    if "dtype" in _from_pretrained_params:
        _load_kwargs["dtype"] = _dt
    else:
        _load_kwargs["torch_dtype"] = _dt
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["id"], **_load_kwargs)
    if tcfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required with gradient checkpointing

    # CLM data collator builds shifted labels on the fly (mlm=False -> causal).
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    out_dir = repo_root() / cfg["paths"]["cpt_output_dir"]
    # Build TrainingArguments robustly: transformers 4.x vs 5.x renamed/removed
    # several args (overwrite_output_dir, warmup_ratio, eval_strategy...). We only
    # pass args the installed version actually accepts.
    _ta_params = set(_inspect.signature(TrainingArguments.__init__).parameters)
    ta_kwargs = dict(
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
        report_to="none",   # switch to "wandb"/"mlflow" if you set up tracking
        seed=cfg["data"]["seed"],
    )
    for k, v in {
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_ratio": tcfg["warmup_ratio"],
        "eval_strategy": "steps",            # newer transformers
        "evaluation_strategy": "steps",      # older transformers
        "eval_steps": tcfg["eval_steps"],
        "overwrite_output_dir": True,        # removed in 5.x
    }.items():
        if k in _ta_params:
            ta_kwargs[k] = v
    targs = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting CPT (full fine-tuning). Watch the loss: smooth descent = healthy, "
          "spikes = LR too high, rising val loss = overfitting/forgetting.")
    start = time.time()
    trainer.train()
    runtime = time.time() - start

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    peak_vram = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
    lineage = {
        "stage": "cpt-v1",
        "parent_model": cfg["model"]["id"],
        "parent_revision": cfg["model"]["revision"],
        "dataset_manifest": cfg["paths"]["manifest"],
        "block_size": cfg["tokenize"]["block_size"],
        "token_budget": cfg["tokenize"]["token_budget"],
        "learning_rate": tcfg["learning_rate"],
        "effective_batch": tcfg["per_device_batch_size"] * tcfg["grad_accum_steps"],
        "runtime_sec": round(runtime, 1),
        "peak_vram_gb": round(peak_vram, 2) if peak_vram else None,
        "seed": cfg["data"]["seed"],
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    update_manifest(cfg["paths"]["manifest"], "cpt_train", lineage)
    print(f"\nSaved cpt-v1 → {out_dir}")
    print(f"runtime {runtime/60:.1f} min | peak VRAM {lineage['peak_vram_gb']} GB")
    print("Next: python evaluation/perplexity.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
