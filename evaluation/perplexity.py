"""Day 2 · Step 6 — PERPLEXITY (BASE vs CPT).

Measures token-weighted perplexity on fixed held-out text, two ways:
  - DOMAIN perplexity  (LLM/ML held-out docs)  -> should DROP after CPT
  - GENERAL perplexity (replay-source docs)    -> should stay ~FLAT

Compares the original BASE model against cpt-v1 and prints a table.

Usage:
    python evaluation/perplexity.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dataio import load_config, read_jsonl, repo_root, update_manifest  # noqa: E402


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
            n = chunk.shape[1] - 1               # predicted positions
            total_loss += out.loss.item() * n
            total_tok += n
            if total_tok >= max_tokens:
                break
        if total_tok >= max_tokens:
            break
    return math.exp(total_loss / max(1, total_tok)), total_tok


def load_texts(path, k):
    try:
        rows = list(read_jsonl(path))
    except FileNotFoundError:
        return []
    return [r["text"] for r in rows[:k]]


def evaluate(model_path, cfg, domain_texts, general_texts, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    import inspect as _inspect
    _params = set(_inspect.signature(AutoModelForCausalLM.from_pretrained).parameters)
    _dtype_key = "dtype" if "dtype" in _params else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, **{_dtype_key: torch.bfloat16}).to(device).eval()
    block = cfg["tokenize"]["block_size"]
    cap = cfg["evaluation"]["max_ppl_tokens"]
    dom, _ = corpus_perplexity(model, tok, domain_texts, cap, block, device)
    gen, _ = corpus_perplexity(model, tok, general_texts, cap, block, device)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return dom, gen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    domain_texts = load_texts(
        f"{cfg['paths']['eval_heldout_dir']}/heldout.jsonl", cfg["evaluation"]["domain_ppl_docs"])
    # Use the SEPARATE held-out general set (never trained on) — NOT the training
    # replay slice — so 'did it forget general skills?' is measured honestly.
    general_texts = load_texts(
        f"{cfg['paths']['eval_heldout_dir']}/general_eval.jsonl", cfg["evaluation"]["general_ppl_docs"])
    if not general_texts:  # fallback if general_eval wasn't collected
        general_texts = load_texts(
            f"{cfg['paths']['clean_dir']}/replay.jsonl", cfg["evaluation"]["general_ppl_docs"])
        print("  ⚠ using training-replay for general ppl (not a clean held-out set)")
    if not domain_texts or not general_texts:
        print("Missing eval texts — run the data pipeline first.")
        return

    base_id = cfg["model"]["id"]
    cpt_dir = str(repo_root() / cfg["paths"]["cpt_output_dir"])

    print("Scoring BASE ...")
    b_dom, b_gen = evaluate(base_id, cfg, domain_texts, general_texts, device)
    print("Scoring CPT ...")
    c_dom, c_gen = evaluate(cpt_dir, cfg, domain_texts, general_texts, device)

    print("\n=== Perplexity (lower = better) ===")
    print(f"{'model':10s} {'domain':>10s} {'general':>10s}")
    print(f"{'BASE':10s} {b_dom:10.2f} {b_gen:10.2f}")
    print(f"{'CPT':10s} {c_dom:10.2f} {c_gen:10.2f}")
    print("\nWant: domain ↓ (specialized), general ≈ flat (didn't forget).")
    print(f"domain change : {100*(c_dom-b_dom)/b_dom:+.1f}%")
    print(f"general change: {100*(c_gen-b_gen)/b_gen:+.1f}%")

    update_manifest(cfg["paths"]["manifest"], "perplexity", {
        "base_domain": round(b_dom, 3), "base_general": round(b_gen, 3),
        "cpt_domain": round(c_dom, 3), "cpt_general": round(c_gen, 3),
    })


if __name__ == "__main__":
    main()
