"""Day 5 · Step 5 — Build REPRODUCE.md from your lineage files (no GPU needed).

Reads each stage's lineage file and config, and writes the literal rebuild
steps (data + training). If a lineage file or a field is missing, this script
says so -- that's a reproducibility GAP to fix, not something to silently skip.

Usage:
    python evaluation/build_reproduce_doc.py --config configs/day5.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, repo_root  # noqa: E402

# (name, published repo, lineage files to try, config, commands in order)
STAGES = [
    ("cpt-v2", "vinmlops/cpt-v2",
     ["artifacts/cpt-v1/lineage.json", "artifacts/cpt-v1/run_summary.json"], "configs/day2.yaml",
     ["data/collect_wiki.py", "data/collect.py --skip-wiki", "data/clean.py", "data/split.py",
      "data/tokenize_pack.py", "training/cpt.py", "evaluation/perplexity.py"]),
    ("sft-v2", "vinmlops/sft-v2",
     ["artifacts/sft-v2/lineage.json"], "configs/day3.yaml",
     ["data/generate_sft.py", "data/judge_sft.py", "data/prepare_sft.py", "training/sft.py"]),
    ("dpo-v1", "vinmlops/dpo-v1",
     ["artifacts/dpo-pilot/lineage.json"], "configs/day4.yaml",
     ["data/sample_candidates.py", "data/judge_pairs.py", "data/prepare_dpo.py", "training/dpo.py"]),
    ("sft-from-base (ablation)", None,
     ["artifacts/sft-from-base/lineage.json"], "configs/day5.yaml",
     ["training/sft_from_base.py"]),
]
REQUIRED = ["parent_model", "seed", "method"]
HPARAMS = ("learning_rate", "beta", "epochs", "block_size", "token_budget", "trainable_tokens",
           "assistant_only_loss")


def first_existing(paths):
    for p in paths:
        full = repo_root() / p
        if full.exists():
            return p, json.loads(full.read_text())
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day5.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    lines = ["# Reproducing This Project's Checkpoints", "",
             "Auto-generated from each stage's lineage file and config. Anything marked",
             "MISSING is a real reproducibility gap -- fix it at the source.", "",
             "Lineage: `Qwen/Qwen3-1.7B-Base` -> cpt-v2 -> sft-v2 -> dpo-v1", ""]
    gaps = []
    for name, repo, lineage_paths, config, commands in STAGES:
        lines.append(f"## {name}")
        if repo:
            lines.append(f"- **Published as:** `{repo}`")
        cfg_path = repo_root() / config
        if cfg_path.exists():
            digest = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:12]
            lines.append(f"- **Config:** `{config}` (sha256 `{digest}`)")
        else:
            lines.append(f"- **Config:** `{config}` MISSING")
            gaps.append(f"{name}: config {config} not found")

        found, lin = first_existing(lineage_paths)
        if lin is None:
            lines += [f"- **Lineage:** MISSING (looked for {', '.join(f'`{p}`' for p in lineage_paths)})", ""]
            gaps.append(f"{name}: no lineage file")
        else:
            missing = [k for k in REQUIRED if lin.get(k) is None]
            if missing:
                gaps.append(f"{name}: lineage missing {missing}")
            manifest = lin.get("dataset_manifest") or lin.get("sft_data_manifest") or lin.get("pref_data_manifest")
            lines += [
                f"- **Lineage file:** `{found}`",
                f"- **Parent:** `{lin.get('parent_model', 'MISSING')}` (revision: "
                f"`{lin.get('parent_revision_resolved', lin.get('parent_revision', 'MISSING'))}`)",
                f"- **Method:** {lin.get('method', 'MISSING')}",
                f"- **Seed:** {lin.get('seed', 'MISSING')}",
                f"- **Dataset manifest:** `{manifest or 'MISSING'}`",
                f"- **Hyperparameters in lineage:** `{json.dumps({k: lin[k] for k in HPARAMS if k in lin})}`",
            ]
            if "stop_verification" in lin:
                lines.append(f"- **Stop verification:** `{json.dumps(lin['stop_verification'])}`")

        # The config's train section is the full record of settings (lineage may omit some).
        if cfg_path.exists() and name != "sft-from-base (ablation)":
            train = load_config(config).get("train", {})
            lines += ["- **Training settings (from config):**", "```yaml",
                      *[f"{k}: {v}" for k, v in train.items()], "```"]

        lines += ["", "**Rebuild commands (in order):**", "```bash",
                  *[f"python {c.split()[0]} --config {config}" + (" " + " ".join(c.split()[1:]) if len(c.split()) > 1 else "")
                    for c in commands],
                  "```", ""]

    lines += ["## Reproducibility gaps found", ""]
    lines += [f"- {g}" for g in gaps] if gaps else \
             ["None found -- every stage has a config, parent, method and seed recorded."]

    out_path = repo_root() / cfg["paths"]["reproduce_doc"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote -> {out_path}")
    print(f"Found {len(gaps)} gap(s) -- see the doc." if gaps else "No gaps found.")


if __name__ == "__main__":
    main()