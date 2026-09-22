"""Pack CPT text into fixed-length blocks with EOS between documents.

The training token budget is a hard ceiling, rounded down to whole blocks.
The requested domain/replay ratio is approximated to the nearest whole block.
If either source is short, use fewer blocks without repeating documents.
Validation is packed separately and is not part of the training token budget.

Usage: python tokenize_pack.py --config day2.yaml
"""

import argparse
from itertools import chain
import math
from pathlib import Path
import random
import tempfile

from dataio import load_config, read_jsonl


def require_eos_in_packed_check(blocks, eos_id, sample=None):
    """Check all blocks by default; an optional sample is diagnostic only."""
    selected = blocks if sample is None else blocks[:sample]
    return any(eos_id in block for block in selected)


def pack_stream(token_lists, block_size, eos_id, max_tokens=None):
    """Pack lazily; append EOS per document and discard an incomplete last block.

    A zero/sub-block budget produces no blocks. Never exceed max_tokens.
    Only the current block is buffered; document tails are not repeatedly copied.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 2:
        raise ValueError("block_size must be an integer >= 2")
    if not isinstance(eos_id, int) or isinstance(eos_id, bool) or eos_id < 0:
        raise ValueError("A valid eos_token_id is required; PAD is not an EOS substitute")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a nonnegative integer or None")
    limit = None if max_tokens is None else max_tokens // block_size
    if limit == 0:
        return [], 0
    blocks, buffer = [], []
    for ids in token_lists:
        for token in chain(ids, (eos_id,)):
            buffer.append(token)
            if len(buffer) == block_size:
                blocks.append(buffer)
                buffer = []
                if limit is not None and len(blocks) == limit:
                    return blocks, len(blocks) * block_size
    return blocks, len(blocks) * block_size


def choose_mixture(domain_count, replay_count, max_blocks, replay_ratio):
    """Largest available mixture, rounded to whole blocks without replacement."""
    if not math.isfinite(replay_ratio) or not 0 <= replay_ratio <= 1:
        raise ValueError("replay_ratio must be between 0 and 1")
    if replay_ratio == 0:
        return min(domain_count, max_blocks), 0
    if replay_ratio == 1:
        return 0, min(replay_count, max_blocks)
    # Both sources must be represented for a mixed run, including tiny corpora.
    for total in range(min(max_blocks, domain_count + replay_count), 1, -1):
        replay = min(total - 1, max(1, int(total * replay_ratio + 0.5)))
        domain = total - replay
        if domain <= domain_count and replay <= replay_count:
            return domain, replay
    raise ValueError(
        "Not enough domain/replay blocks for the requested mixture. "
        "Collect more of the missing source, or set replay_ratio=0 for a domain-only run."
    )


def build_blocks(cfg, tokenizer):
    block_size = cfg["tokenize"]["block_size"]
    budget = cfg["tokenize"]["token_budget"]
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 2:
        raise ValueError("block_size must be an integer >= 2")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < block_size:
        raise ValueError("token_budget must allow at least one complete block")
    ratio = float(cfg["data"]["replay_ratio"])
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("replay_ratio must be between 0 and 1")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; PAD cannot replace document EOS")

    clean = Path(cfg["paths"]["clean_dir"])
    train_rows = read_jsonl(clean / "pool_train.jsonl", required=True)
    val_rows = read_jsonl(clean / "pool_val.jsonl", required=True)
    if not train_rows:
        raise ValueError("Training pool is empty; run split.py first")
    for name, rows in (("training", train_rows), ("validation", val_rows)):
        for i, row in enumerate(rows, 1):
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{name} document {i} has missing or empty text")
    if any(row.get("kind") not in {"domain", "replay"} for row in train_rows):
        raise ValueError("Every training row must have kind='domain' or kind='replay'")
    domain_rows = [row for row in train_rows if row["kind"] == "domain"]
    replay_rows = [row for row in train_rows if row["kind"] == "replay"]
    if ratio > 0 and not replay_rows:
        raise ValueError(
            "replay_ratio is positive but replay data is missing. The Wikipedia-only "
            "collector does not produce replay; supply replay or set replay_ratio=0."
        )
    if ratio < 1 and not domain_rows:
        raise ValueError("Domain data is missing from the training pool")
    rng = random.Random(cfg["data"]["seed"])
    rng.shuffle(domain_rows)
    rng.shuffle(replay_rows)

    def tokenize(rows):
        for row in rows:
            yield tokenizer(
                row["text"], add_special_tokens=False, truncation=False,
                return_attention_mask=False,
            )["input_ids"]

    max_blocks = budget // block_size
    # Ceiling allows the final nearest-block mixture to use the full budget.
    domain_cap = math.ceil(max_blocks * (1 - ratio)) * block_size
    replay_cap = math.ceil(max_blocks * ratio) * block_size
    domain_blocks, _ = pack_stream(tokenize(domain_rows), block_size, eos_id, domain_cap)
    replay_blocks, _ = pack_stream(tokenize(replay_rows), block_size, eos_id, replay_cap)
    nd, nr = choose_mixture(len(domain_blocks), len(replay_blocks), max_blocks, ratio)
    train_blocks = domain_blocks[:nd] + replay_blocks[:nr]
    rng.shuffle(train_blocks)
    if not train_blocks:
        raise ValueError("No full training blocks; collect more text or reduce block_size")

    val_budget = max(block_size, budget // 10)
    val_blocks, _ = pack_stream(tokenize(val_rows), block_size, eos_id, val_budget)
    if not val_blocks and (val_rows or float(cfg["data"]["val_frac"]) > 0):
        raise ValueError("Validation has no complete blocks; add validation text or reduce block_size")

    if require_eos_in_packed_check(train_blocks, eos_id):
        print(f"[pack] EOS found in training blocks (id={eos_id})")
    else:
        message = (
            "No EOS survived in the selected full training blocks. This can happen "
            "when the budget cuts off a long document before its end."
        )
        if cfg.get("train", {}).get("require_eos_in_packed", False):
            raise ValueError(message + " Increase the budget, or disable require_eos_in_packed for intentional truncation.")
        print(f"[pack] NOTE: {message}")

    total = (nd + nr) * block_size
    print(f"[pack] train_tokens={total}/{budget}; domain={nd * block_size}; replay={nr * block_size}")
    print(f"[pack] replay target={ratio:.2%}; actual={nr / (nd + nr):.2%} (whole-block rounding)")
    print(f"[pack] validation_blocks={len(val_blocks)}; validation_tokens={len(val_blocks) * block_size}")
    if nd + nr < max_blocks:
        print("[pack] Source availability limits the mixture; using fewer tokens without repeating data.")
    return train_blocks, val_blocks


def save_blocks(out, train_blocks, val_blocks):
    """Stage both datasets; restore existing outputs if publication raises an error."""
    from datasets import Dataset, Features, Sequence, Value

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    features = Features({"input_ids": Sequence(Value("int64"))})
    with tempfile.TemporaryDirectory(prefix=".packing-", dir=out) as temporary:
        stage = Path(temporary)
        for name, blocks in (("train", train_blocks), ("val", val_blocks)):
            Dataset.from_dict({"input_ids": blocks}, features=features).save_to_disk(str(stage / name))
        backed_up, installed = [], []
        try:
            for name in ("train", "val"):
                target = out / name
                if target.exists():
                    target.rename(stage / f"old-{name}")
                    backed_up.append(name)
                (stage / name).rename(target)
                installed.append(name)
        except Exception:
            for name in reversed(installed):
                (out / name).rename(stage / name)
            for name in reversed(backed_up):
                (stage / f"old-{name}").rename(out / name)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["id"], revision=cfg["model"].get("revision", "main")
    )
    try:
        train_blocks, val_blocks = build_blocks(cfg, tokenizer)
    except ValueError as exc:
        raise SystemExit(f"[pack] ERROR: {exc}") from exc
    # Validate everything before publishing usable datasets.
    save_blocks(cfg["paths"]["packed_dir"], train_blocks, val_blocks)
    print(f"Next: python cpt.py --config {args.config}")


if __name__ == "__main__":
    main()
