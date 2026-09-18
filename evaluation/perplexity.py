"""
Perplexity evaluation: BASE vs CPT.

Perplexity measures how "surprised" a model is by held-out text.
Lower = the model predicts that text better.

We measure it two ways:
  - DOMAIN  perplexity (LLM/ML held-out docs)  -> should DROP after CPT
                                                  (the model specialized)
  - GENERAL perplexity (held-out general docs) -> should stay ~FLAT
                                                  (the model didn't forget)

The script scores the original base model and the trained cpt model on the
same texts, then prints a small table.

Usage:
    python perplexity.py --config day2_cpt.yaml
"""

import argparse
import json
import math
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path, limit):
    """Read up to `limit` lines from a .jsonl file, return their 'text' fields."""
    p = Path(path)
    if not p.exists():
        return []
    texts = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if len(texts) >= limit:
                break
    return texts


def corpus_perplexity(model, tokenizer, texts, block_size, max_tokens, device):
    """
    Token-weighted perplexity = exp(total_loss / total_predicted_tokens).

    Each text is tokenized and split into chunks of block_size. For each chunk
    the model reports the average next-token loss; we weight it by how many
    tokens it covered so long and short chunks are combined fairly.
    """
    import torch

    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        for start in range(0, len(ids), block_size):
            chunk = ids[start:start + block_size].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue  # need at least 2 tokens to predict one
            with torch.inference_mode():
                out = model(chunk, labels=chunk)
            predicted = chunk.shape[1] - 1  # the model predicts all but the first token
            total_loss += out.loss.item() * predicted
            total_tokens += predicted
            if total_tokens >= max_tokens:
                break
        if total_tokens >= max_tokens:
            break

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


    
def evaluate(model_path, domain_texts, general_texts, block_size, max_tokens, device):
    """Load a model, score it on both text sets, then free it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    domain_ppl = corpus_perplexity(model, tokenizer, domain_texts, block_size, max_tokens, device)
    general_ppl = corpus_perplexity(model, tokenizer, general_texts, block_size, max_tokens, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return domain_ppl, general_ppl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_cfg = cfg["evaluation"]
    block_size = int(cfg["tokenize"]["block_size"])
    max_tokens = int(eval_cfg["max_ppl_tokens"])

    # held-out texts (never trained on) so the comparison is honest
    #domain_texts = load_texts(f"{cfg['paths']['eval_heldout_dir']}/heldout.jsonl", cfg["evaluation"]["domain_ppl_docs"])
    #general_texts = load_texts(f"{cfg['paths']['eval_heldout_dir']}/general_eval.jsonl", cfg["evaluation"]["general_ppl_docs"])
    domain_texts = read_jsonl(eval_cfg["domain_file"], eval_cfg["domain_ppl_docs"])
    general_texts = read_jsonl(eval_cfg["general_file"], eval_cfg["general_ppl_docs"])

    if not domain_texts or not general_texts:
        print("Missing eval texts. Check eval.domain_file and eval.general_file exist.")
        return

    base_model = cfg["model"]["id"]
    cpt_model = cfg["paths"]["cpt_output_dir"]

    print("Scoring BASE ...")
    base_domain, base_general = evaluate(
        base_model, domain_texts, general_texts, block_size, max_tokens, device
    )
    print("Scoring CPT ...")
    cpt_domain, cpt_general = evaluate(
        cpt_model, domain_texts, general_texts, block_size, max_tokens, device
    )

    # --- results table ---
    print("\n=== Perplexity (lower = better) ===")
    print(f"{'model':10s} {'domain':>10s} {'general':>10s}")
    print(f"{'BASE':10s} {base_domain:10.2f} {base_general:10.2f}")
    print(f"{'CPT':10s} {cpt_domain:10.2f} {cpt_general:10.2f}")

    domain_change = 100 * (cpt_domain - base_domain) / base_domain
    general_change = 100 * (cpt_general - base_general) / base_general
    print("\nWant: domain DOWN (specialized), general ~flat (didn't forget).")
    print(f"domain change : {domain_change:+.1f}%")
    print(f"general change: {general_change:+.1f}%")

    # --- save results next to the model ---
    results = {
        "base_domain": round(base_domain, 3),
        "base_general": round(base_general, 3),
        "cpt_domain": round(cpt_domain, 3),
        "cpt_general": round(cpt_general, 3),
        "domain_change_pct": round(domain_change, 1),
        "general_change_pct": round(general_change, 1),
    }
    out_path = Path(cpt_model) / "perplexity.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
