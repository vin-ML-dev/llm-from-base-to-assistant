"""Step 4 — SFT training (LoRA on cpt-v2, assistant-only loss).

Trains a LoRA adapter on top of the Day 2 cpt-v2 with TRL's SFTTrainer using
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
    """Validate complete examples and supervision of every assistant end token."""
    if max_length <= 0:
        raise ValueError("max_seq_length must be positive.")
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
        ids = enc["input_ids"]
        mask = enc.get("assistant_masks", [])
        if len(ids) > max_length:
            bad.append((i, f"formatted example has {len(ids)} tokens, exceeding {max_length}"))
            continue
        if len(mask) != len(ids) or not any(mask):
            bad.append((i, "missing or invalid assistant token mask"))
            continue
        supervised_eos = sum(
            token == tokenizer.eos_token_id and bool(active)
            for token, active in zip(ids, mask)
        )
        assistant_turns = sum(m["role"] == "assistant" for m in msgs)
        if supervised_eos != assistant_turns:
            bad.append((i, "not every assistant turn has a supervised end token"))
        if not any(active and token != tokenizer.eos_token_id for token, active in zip(ids, mask)):
            bad.append((i, "assistant mask contains no answer tokens"))

    if bad:
        raise ValueError(f"{split}: {len(bad)} invalid examples (first 10: {bad[:10]}).")
    print(f"{split}: all {len(dataset)} examples fit and supervise assistant end tokens.")
    return dataset.select_columns(["messages"])


def stop_metrics(model, tok, dataset, n_prob=32, n_sample=10, max_new_tokens=256):
    """Measure whether the model actually STOPS (the bug greedy decoding hid).

    - mean_p_im_end: P(<|im_end|>) right after a gold validation answer.
    - sampled_stop_rate: fraction of temp-0.7 samples that end with <|im_end|>.
    """
    import torch
    model.eval()
    model.config.use_cache = True
    im_end = tok.eos_token_id
    rows = dataset.select(range(min(n_prob, len(dataset))))

    def prompt_text(msgs):
        return tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

    probs = []
    for row in rows:
        text = prompt_text(row["messages"]) + row["messages"][-1]["content"]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            probs.append(model(**ids).logits[0, -1].float().softmax(-1)[im_end].item())

    stops = 0
    sample_rows = rows.select(range(min(n_sample, len(rows))))
    for row in sample_rows:
        ids = tok(prompt_text(row["messages"]), return_tensors="pt",
                  add_special_tokens=False).to(model.device)
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
    rev = resolve_revision(base, cfg["student"]["base_revision"])
    data = repo_root() / cfg["paths"]["sft_data"]
    train_ds = load_from_disk(str(data / "train"))
    val_ds = load_from_disk(str(data / "val"))
    print(f"train={len(train_ds)}  val={len(val_ds)}  base={base}")

    # tokenizer: enforce ChatML + <|im_end|> as eos so the model learns to stop
    tok = load_tokenizer(base, rev)
    eos = cfg["chat"]["eos_token"]
    vocab = tok.get_vocab()
    missing = [t for t in ("<|im_start|>", "<|im_end|>") if t not in vocab]
    if eos != "<|im_end|>" or missing:
        raise ValueError(
            "Tutor template requires ChatML tokens and eos_token='<|im_end|>'.\n"
            f"  eos_token in config: {eos!r}\n"
            f"  missing from {base} vocab: {missing or 'none'}\n"
            "If the ChatML tokens are missing, the base model's tokenizer doesn't "
            "include them. Use a base whose tokenizer has <|im_start|>/<|im_end|> "
            "(Qwen3 tokenizers normally do), or add them to the tokenizer BEFORE "
            "CPT so their embeddings are trained — do not add them here."
        )
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

    # Respect []: train only the configured LoRA modules unless full modules
    # are explicitly requested.
    modules_to_save = tcfg.get("lora_modules_to_save", [])
    lora_kwargs = dict(
        r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
        target_modules=tcfg["lora_target_modules"],
        modules_to_save=modules_to_save,
        revision=rev,
        task_type="CAUSAL_LM",
    )
    # STOP FIX: train only the ChatML token rows. Frozen, <|im_end|> keeps the
    # CPT base's untrained row and the model cannot learn to stop.
    token_names = tcfg.get("lora_trainable_tokens") or []
    if token_names and tcfg["use_lora"]:
        if "trainable_token_indices" not in inspect.signature(LoraConfig.__init__).parameters:
            raise RuntimeError("Installed PEFT lacks trainable_token_indices. "
                               "Run: pip install -U 'peft>=0.15'")
        if {"embed_tokens", "lm_head"} & set(modules_to_save):
            raise ValueError("Use lora_trainable_tokens OR full embed/lm_head in "
                             "lora_modules_to_save, not both.")
        token_ids = [tok.convert_tokens_to_ids(t) for t in token_names]
        if any(i is None or i == tok.unk_token_id for i in token_ids):
            raise ValueError(f"Trainable tokens missing from vocab: {token_names}")
        lora_kwargs["trainable_token_indices"] = token_ids
        print(f"Trainable token rows: {dict(zip(token_names, token_ids))}")
    lora = LoraConfig(**lora_kwargs) if tcfg["use_lora"] else None

    out_dir = repo_root() / cfg["paths"]["sft_output_dir"]
    sc_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if tcfg["assistant_only_loss"] and "assistant_only_loss" not in sc_params:
        raise RuntimeError("Installed TRL lacks assistant_only_loss; refusing to disable it silently.")

    sc_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_total_limit=tcfg.get("save_total_limit", 1),
        save_only_model=tcfg.get("save_only_model", False),
        seed=cfg["dataset"]["seed"],
        report_to="none",
    )
    # Select supported parameter names without silently dropping settings.
    length_key = next((k for k in ("max_length", "max_seq_length") if k in sc_params), None)
    eval_key = next((k for k in ("eval_strategy", "evaluation_strategy") if k in sc_params), None)
    if length_key is None or eval_key is None:
        raise RuntimeError("Installed TRL lacks the required sequence-length or evaluation settings.")
    sc_kwargs.update({
        "lr_scheduler_type": tcfg["lr_scheduler_type"],
        "warmup_ratio": tcfg["warmup_ratio"],
        length_key: max_length,
        "gradient_checkpointing": tcfg["gradient_checkpointing"],
        "assistant_only_loss": tcfg["assistant_only_loss"],
        eval_key: tcfg.get("eval_strategy", "steps"),
        "eval_steps": tcfg["eval_steps"],
    })
    if "eos_token" in sc_params:
        sc_kwargs["eos_token"] = eos
    unsupported = set(sc_kwargs) - sc_params
    if unsupported:
        raise RuntimeError(f"Installed TRL does not support configured settings: {sorted(unsupported)}")
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
    # Adapter exports do not automatically save the base generation config.
    # Adapter inference must explicitly load this saved GenerationConfig.
    if trainer.model.generation_config is not None:
        trainer.model.generation_config.save_pretrained(str(out_dir))

    # --- VERIFY STOPPING (greedy decoding hid this bug last time) -------------
    verify = {"adapter": stop_metrics(trainer.model, tok, val_ds)}
    adapter_ok = report_stop("adapter", verify["adapter"], tcfg)

    # Merge, save, then RELOAD FROM DISK and re-verify. The first sft-v2 lost its
    # stop-token training during merge/save, and only a reload test catches that.
    merged_dir = cfg["paths"].get("sft_merged_dir")
    merged_ok = None
    if tcfg["use_lora"] and merged_dir:
        merged_dir = repo_root() / merged_dir
        merged = trainer.model.merge_and_unload()
        merged.config.eos_token_id = tok.eos_token_id
        merged.config.pad_token_id = tok.pad_token_id
        if merged.generation_config is not None:
            merged.generation_config.eos_token_id = tok.eos_token_id
            merged.generation_config.pad_token_id = tok.pad_token_id
        merged.save_pretrained(str(merged_dir))
        tok.save_pretrained(str(merged_dir))
        del merged, trainer
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(str(merged_dir), **{dtype_key: dt})
        reloaded = reloaded.to("cuda" if torch.cuda.is_available() else "cpu")
        verify["merged_reloaded"] = stop_metrics(reloaded, tok, val_ds)
        merged_ok = report_stop("merged_reloaded", verify["merged_reloaded"], tcfg)
        print(f"Saved merged model -> {merged_dir}")
    if not adapter_ok or merged_ok is False:
        print("WARNING: stopping verification FAILED. Do not upload or use for DPO; "
              "inspect verify results in lineage.json.")

    peak = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
    lineage = {
        "stage": "sft-v2",
        "chat_template_version": "tutor-chatml-assistant-mask-v1",
        "parent_model": base,
        "parent_revision_resolved": rev,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "method": "lora" if tcfg["use_lora"] else "full",
        "assistant_only_loss": tcfg["assistant_only_loss"],
        "epochs": tcfg["num_epochs"],
        "learning_rate": tcfg["learning_rate"],
        "sft_data_manifest": cfg["paths"]["manifest"],
        "runtime_sec": round(runtime, 1),
        "peak_vram_gib": round(peak, 2) if peak else None,
        "seed": cfg["dataset"]["seed"],
        "trainable_tokens": tcfg.get("lora_trainable_tokens") or [],
        "stop_verification": verify,
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(f"\nSaved sft-v2 → {out_dir}  ({runtime/60:.1f} min, peak {lineage['peak_vram_gib']} GiB)")
    print("Next: python evaluation/sft_eval.py --config configs/day3.yaml")


if __name__ == "__main__":
    main()
