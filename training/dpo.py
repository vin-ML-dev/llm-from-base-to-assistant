"""Day 4 (PILOT) · Step 4 — DPO training on sft-v2 (LoRA).

CRITICAL (from the critique): the DPO reference policy must represent the
PRE-DPO sft-v2 model. Since sft-v2 is a MERGED standalone model (not an adapter
over cpt-v2), initializing a fresh LoRA adapter on it and letting TRL manage the
reference is correct: TRL's reference = the model with the new DPO adapter
DISABLED = merged sft-v2. This is exactly what we want. (If sft-v2 were still an
adapter over cpt-v2, disabling the adapter would wrongly expose cpt-v2 weights.)

Usage:
    python training/dpo.py --config configs/day4.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, repo_root  # noqa: E402


def stop_metrics(model, tok, dataset, n_prob=32, n_sample=10, max_new_tokens=256):
    """Does the model STOP? (greedy decoding hid this bug in the first sft-v2)

    mean_p_im_end: P(<|im_end|>) right after a chosen answer.
    sampled_stop_rate: fraction of temp-0.7 samples that end with <|im_end|>.
    """
    import torch
    model.eval()
    model.config.use_cache = True
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = dataset.select(range(min(n_prob, len(dataset))))

    def prompt_text(row):
        return tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)

    probs = []
    for row in rows:
        text = prompt_text(row) + row["chosen"][0]["content"]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            probs.append(model(**ids).logits[0, -1].float().softmax(-1)[im_end].item())

    stops = 0
    sample_rows = rows.select(range(min(n_sample, len(rows))))
    for row in sample_rows:
        ids = tok(prompt_text(row), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.7, top_p=0.9, eos_token_id=im_end,
                                 pad_token_id=tok.pad_token_id)
        stops += int(out[0, -1].item() == im_end)
    return {"mean_p_im_end": round(sum(probs) / len(probs), 4),
            "min_p_im_end": round(min(probs), 4),
            "sampled_stop_rate": round(stops / len(sample_rows), 3)}


def report_stop(label, metrics, tcfg):
    ok = (metrics["mean_p_im_end"] >= tcfg.get("verify_min_p_im_end", 0.5)
          and metrics["sampled_stop_rate"] >= tcfg.get("verify_min_stop_rate", 0.8))
    print(f"[verify:{label}] {metrics} -> {'PASS' if ok else 'FAIL: model does not stop reliably'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]
    seed = cfg["sampling"]["seed"]

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(tcfg["bf16"]) and use_cuda
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    base = cfg["student"]["sft_model"]   # MERGED sft-v2 (standalone)
    data = repo_root() / cfg["paths"]["pref_data"]
    train_ds = load_from_disk(str(data / "train"))
    val_ds = load_from_disk(str(data / "val"))
    print(f"train pairs={len(train_ds)}  val pairs={len(val_ds)}  base={base}")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)

    # Baseline: how well does sft-v2 stop BEFORE DPO? (should already PASS)
    if use_cuda:
        model.to("cuda")
    verify = {"sft_baseline": stop_metrics(model, tok, val_ds)}
    report_stop("sft_baseline", verify["sft_baseline"], tcfg)
    model.config.use_cache = False

    # New LoRA adapter for DPO. Because `base` is the merged sft-v2, TRL's
    # reference policy (this model with the DPO adapter disabled) IS sft-v2 --
    # the correct pre-DPO reference. No separate ref model is loaded.
    lora = LoraConfig(
        r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
        target_modules=tcfg["lora_target_modules"], task_type="CAUSAL_LM",
    ) if tcfg["use_lora"] else None

    out_dir = repo_root() / cfg["paths"]["dpo_output_dir"]
    dpo_kwargs = dict(
        output_dir=str(out_dir),
        beta=tcfg["beta"],
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        max_length=tcfg["max_length"],
        bf16=use_bf16,
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        seed=seed,
        report_to="none",
    )
    # Version compatibility (older vs newer transformers/TRL):
    #  - max_prompt_length was removed in newer TRL; pass it only where supported.
    #  - warmup_ratio was removed in newer transformers; there, a float < 1 given
    #    as warmup_steps means the same ratio.
    import inspect
    dpo_params = inspect.signature(DPOConfig.__init__).parameters
    if "warmup_ratio" not in dpo_params:
        dpo_kwargs["warmup_steps"] = float(dpo_kwargs.pop("warmup_ratio"))
    if "max_prompt_length" in dpo_params:
        dpo_kwargs["max_prompt_length"] = tcfg["max_prompt_length"]
    if "eval_on_start" in dpo_params:
        dpo_kwargs["eval_on_start"] = True
    else:
        print("[dpo] note: this TRL has no max_prompt_length; only max_length applies "
              "(prepare_dpo.py already drops over-length pairs).")

    # Guard against over-optimization (the full run collapsed P(<|im_end|>)):
    #  - SFT loss on the CHOSEN answers keeps them (and their final <|im_end|>)
    #    likely while DPO shifts preferences. Newer TRL: loss_type ["sigmoid","sft"];
    #    older TRL: rpo_alpha.
    #  - ld_alpha (LD-DPO) down-weights the extra length of longer answers.
    sft_w = float(tcfg.get("sft_loss_weight", 0) or 0)
    if sft_w > 0:
        if "loss_type" in dpo_params and "loss_weights" in dpo_params:
            dpo_kwargs["loss_type"] = ["sigmoid", "sft"]
            dpo_kwargs["loss_weights"] = [1.0, sft_w]
        elif "rpo_alpha" in dpo_params:
            dpo_kwargs["rpo_alpha"] = sft_w
        else:
            raise RuntimeError("This TRL supports neither loss_type=['sigmoid','sft'] nor rpo_alpha.")
    if tcfg.get("ld_alpha") is not None:
        if "ld_alpha" not in dpo_params:
            raise RuntimeError("This TRL has no ld_alpha; remove it from day4.yaml or upgrade TRL.")
        dpo_kwargs["ld_alpha"] = float(tcfg["ld_alpha"])
    print(f"[dpo] beta={dpo_kwargs['beta']} lr={dpo_kwargs['learning_rate']} "
          f"sft_loss_weight={sft_w} ld_alpha={tcfg.get('ld_alpha')}")
    dpo_config = DPOConfig(**dpo_kwargs)
    print(f"DPO: beta={tcfg['beta']} lr={tcfg['learning_rate']} "
          f"(reference policy = pre-DPO sft-v2 via disabled adapter)")
    print("Watch rewards/accuracies (>0.5, rising) and rewards/margins (growing).")

    trainer = DPOTrainer(
        model=model, args=dpo_config,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, peft_config=lora,
    )

    start = time.time()
    trainer.train()
    runtime = time.time() - start

    # --- VERIFY STOPPING: adapter, then merged + RELOADED from disk ---------------
    verify["dpo_adapter"] = stop_metrics(trainer.model, tok, val_ds)
    adapter_ok = report_stop("dpo_adapter", verify["dpo_adapter"], tcfg)

    # merge the DPO adapter into sft-v2 -> standalone dpo model
    if tcfg["use_lora"]:
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    model.config.eos_token_id = im_end
    model.config.pad_token_id = tok.pad_token_id
    if model.generation_config is not None:
        model.generation_config.eos_token_id = im_end
        model.generation_config.pad_token_id = tok.pad_token_id
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model, trainer
    if use_cuda:
        torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir), dtype=dtype)
    if use_cuda:
        reloaded.to("cuda")
    verify["dpo_merged_reloaded"] = stop_metrics(reloaded, tok, val_ds)
    merged_ok = report_stop("dpo_merged_reloaded", verify["dpo_merged_reloaded"], tcfg)
    if not (adapter_ok and merged_ok):
        print("WARNING: stopping verification FAILED. Do not upload this model; "
              "compare with sft_baseline in lineage.json.")

    peak = (torch.cuda.max_memory_allocated() / 1024**3) if use_cuda else None
    lineage = {
        "stage": "dpo-pilot",
        "parent_model": base,
        "method": "lora" if tcfg["use_lora"] else "full",
        "beta": tcfg["beta"], "learning_rate": tcfg["learning_rate"],
        "epochs": tcfg["num_epochs"],
        "reference_policy": "pre-DPO sft-v2 (merged base, adapter disabled)",
        "pref_data_manifest": cfg["paths"]["manifest"],
        "runtime_sec": round(runtime, 1),
        "peak_vram_gib": round(peak, 2) if peak else None,
        "seed": seed,
        "stop_verification": verify,
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(f"Saved dpo-pilot -> {out_dir} ({runtime/60:.1f} min)")
    print("Evaluate on HELD-OUT quality/accuracy/instruction-following/stopping "
          "-- NOT preference-training loss alone.")


if __name__ == "__main__":
    main()