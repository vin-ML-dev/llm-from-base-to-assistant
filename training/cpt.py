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
    # IMPORTANT: Qwen3-Base uses the SAME id for EOS and PAD. The default
    # DataCollatorForLanguageModeling masks pad-token positions out of the loss,
    # which would also mask the EOS tokens we put BETWEEN documents — so the model
    # would never learn to predict EOS (i.e. never learn where a document ends /
    # when to stop). We add a DISTINCT pad token so padding is masked but EOS is
    # still learned. Blocks are all full-length (packing), so padding is rare, but
    # this keeps EOS labels intact.
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        _added_pad = True
    else:
        _added_pad = False

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
    if _added_pad:
        # we added a new special token → grow the embedding table to match
        model.resize_token_embeddings(len(tok))
    if tcfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required with gradient checkpointing

    # CLM collator builds shifted labels on the fly (mlm=False -> causal).
    # Because PAD is now a DISTINCT token (not EOS), only real padding is masked
    # from the loss; EOS tokens between documents keep their labels and ARE learned.
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

    import transformers as _tf
    import subprocess as _sp
    def _git_commit():
        try:
            return _sp.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=str(repo_root())).decode().strip()
        except Exception:
            return None
    def _resolve_revision(repo_id):
        # turn a moving pointer like "main" into the immutable commit hash
        try:
            from huggingface_hub import HfApi
            return HfApi().model_info(repo_id, revision=cfg["model"]["revision"]).sha
        except Exception:
            return cfg["model"]["revision"]

    peak_vram = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
    lineage = {
        "stage": "cpt-v1",
        "parent_model": cfg["model"]["id"],
        "parent_revision": cfg["model"]["revision"],
        "parent_revision_resolved": _resolve_revision(cfg["model"]["id"]),  # immutable hash
        "dataset_manifest": cfg["paths"]["manifest"],
        "block_size": cfg["tokenize"]["block_size"],
        "token_budget": cfg["tokenize"]["token_budget"],
        "learning_rate": tcfg["learning_rate"],
        "effective_batch": tcfg["per_device_batch_size"] * tcfg["grad_accum_steps"],
        "runtime_sec": round(runtime, 1),
        "peak_vram_gib": round(peak_vram, 2) if peak_vram else None,  # GiB, measured PyTorch alloc
        "seed": cfg["data"]["seed"],
        "transformers_version": _tf.__version__,
        "torch_version": torch.__version__,
        "repo_commit": _git_commit(),
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    update_manifest(cfg["paths"]["manifest"], "cpt_train", lineage)
    print(f"\nSaved cpt-v1 → {out_dir}")
    print(f"runtime {runtime/60:.1f} min | peak VRAM {lineage['peak_vram_gb']} GB")
    print("Next: python evaluation/perplexity.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
