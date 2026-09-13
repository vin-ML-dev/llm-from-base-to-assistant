"""Day 2 · Step 4 — TOKENIZE + PACK.

Turns the split text pools into fixed-length token blocks for CPT.
  - tokenizes with Qwen3's tokenizer (same frozen vocab from Day 1)
  - concatenates into one stream with EOS between documents, slices into
    block_size chunks (the standard CLM 'group_texts' packing)
  - enforces the 85/15 domain:replay mixture in the TRAIN pool
  - caps total training tokens at token_budget

Writes packed blocks to data/packed/ as a saved HF dataset (train + val).

Usage:
    python data/tokenize_pack.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse

from dataio import ensure_dir, load_config, read_jsonl, repo_root, update_manifest


def pack_stream(token_lists, block_size, eos_id, max_tokens=None):
    """Concatenate token lists (EOS between docs), slice into equal blocks."""
    buf, blocks, total = [], [], 0
    for ids in token_lists:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= block_size:
            blocks.append(buf[:block_size])
            buf = buf[block_size:]
            total += block_size
            if max_tokens and total >= max_tokens:
                return blocks, total
    return blocks, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from datasets import Dataset
    from transformers import AutoTokenizer

    block = cfg["tokenize"]["block_size"]
    budget = cfg["tokenize"]["token_budget"]
    replay_ratio = cfg["data"]["replay_ratio"]

    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"])
    eos = tok.eos_token_id

    # --- load pools ---
    train_rows = list(read_jsonl(f"{cfg['paths']['clean_dir']}/pool_train.jsonl"))
    val_rows = list(read_jsonl(f"{cfg['paths']['clean_dir']}/pool_val.jsonl"))
    domain_rows = [r for r in train_rows if r["kind"] == "domain"]
    replay_rows = [r for r in train_rows if r["kind"] == "replay"]

    def toks(rows):
        return [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]

    # --- budget split by mixture: 85% domain / 15% replay ---
    domain_budget = int(budget * (1 - replay_ratio))
    replay_budget = budget - domain_budget

    dom_blocks, dom_tok = pack_stream(toks(domain_rows), block, eos, domain_budget)
    rep_blocks, rep_tok = pack_stream(toks(replay_rows), block, eos, replay_budget)
    train_blocks = dom_blocks + rep_blocks

    val_blocks, val_tok = pack_stream(toks(val_rows), block, eos, budget // 10)

    ensure_dir(cfg["paths"]["packed_dir"])
    out = repo_root() / cfg["paths"]["packed_dir"]
    Dataset.from_dict({"input_ids": train_blocks}).save_to_disk(str(out / "train"))
    Dataset.from_dict({"input_ids": val_blocks}).save_to_disk(str(out / "val"))

    result = {
        "block_size": block, "replay_ratio": replay_ratio,
        "train_blocks": len(train_blocks), "train_tokens": dom_tok + rep_tok,
        "domain_tokens": dom_tok, "replay_tokens": rep_tok,
        "val_blocks": len(val_blocks), "val_tokens": val_tok,
        "token_budget": budget,
    }
    update_manifest(cfg["paths"]["manifest"], "tokenize_pack", result)
    print("[pack]", result)
    print(f"  domain:replay actual = {dom_tok}:{rep_tok} "
          f"(~{100*dom_tok/max(1,dom_tok+rep_tok):.0f}% domain)")
    print("Next: python training/cpt.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
