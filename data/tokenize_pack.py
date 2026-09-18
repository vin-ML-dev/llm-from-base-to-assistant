"""
Step 4 — TOKENIZE + PACK.

Turns the split text pools into fixed-length token blocks for training:
  - tokenizes with the base model's tokenizer
  - joins documents into one long token stream with an EOS token between each
    document (this is what teaches the model where documents end)
  - slices that stream into equal block_size chunks
  - keeps an 85/15 domain:replay token mixture and caps total tokens at token_budget

Writes packed blocks to data/packed/{train,val} as a saved HuggingFace dataset.

Usage:
    python tokenize_pack.py --config day2_cpt.yaml
"""

import argparse
from pathlib import Path

from dataio import load_config, read_jsonl


def pack(token_lists, block_size, eos_id, max_tokens):
    """
    Concatenate documents into one stream (EOS after each), then cut into blocks
    of exactly block_size tokens. Stops once max_tokens is reached. Any leftover
    tokens that don't fill a full final block are dropped.
    """
    buffer = []
    blocks = []
    total = 0
    for ids in token_lists:
        buffer.extend(ids)
        buffer.append(eos_id)  # document boundary
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            buffer = buffer[block_size:]
            total += block_size
            if max_tokens and total >= max_tokens:
                return blocks, total
    return blocks, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from datasets import Dataset
    from transformers import AutoTokenizer

    block_size = cfg["tokenize"]["block_size"]
    budget = cfg["tokenize"]["token_budget"]
    replay_ratio = cfg["data"]["replay_ratio"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    eos_id = tokenizer.eos_token_id

    # --- load the pools produced by split.py ---
    train_rows = read_jsonl(f"{cfg['paths']['clean_dir']}/pool_train.jsonl")
    val_rows = read_jsonl(f"{cfg['paths']['clean_dir']}/pool_val.jsonl")
    domain_rows = [r for r in train_rows if r["kind"] == "domain"]
    replay_rows = [r for r in train_rows if r["kind"] == "replay"]

    def tokenize(rows):
        return [tokenizer(r["text"], add_special_tokens=False)["input_ids"] for r in rows]

    # --- split the token budget by the desired mixture ---
    domain_budget = int(budget * (1 - replay_ratio))
    replay_budget = budget - domain_budget

    domain_blocks, domain_tokens = pack(tokenize(domain_rows), block_size, eos_id, domain_budget)
    replay_blocks, replay_tokens = pack(tokenize(replay_rows), block_size, eos_id, replay_budget)
    train_blocks = domain_blocks + replay_blocks

    # validation uses a smaller budget
    val_blocks, val_tokens = pack(tokenize(val_rows), block_size, eos_id, budget // 10)

    # --- save to disk ---
    out = Path(cfg["paths"]["packed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"input_ids": train_blocks}).save_to_disk(str(out / "train"))
    Dataset.from_dict({"input_ids": val_blocks}).save_to_disk(str(out / "val"))

    train_total = domain_tokens + replay_tokens
    domain_pct = 100 * domain_tokens / max(1, train_total)
    print(
        f"[pack] train_blocks={len(train_blocks)} train_tokens={train_total} "
        f"(domain={domain_tokens}, replay={replay_tokens}, ~{domain_pct:.0f}% domain)"
    )
    print(f"[pack] val_blocks={len(val_blocks)} val_tokens={val_tokens}")
    print("Next: python cpt.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
