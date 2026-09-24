"""Day 5 · Ablation — SFT FROM BASE directly (no CPT), to test if CPT helped.

Runs YOUR REAL training/sft.py with a copy of configs/day3.yaml where ONLY the
starting model (and the output folders) change: the original base instead of
cpt-v2. Same data, same hyperparameters, same stop-token fix and the same
PASS/FAIL stop verification, so the comparison is single-variable and fair.

Usage:
    python training/sft_from_base.py --config configs/day5.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from sft_common import load_config, repo_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day5.yaml")
    args = ap.parse_args()
    day5 = load_config(args.config)
    acfg = day5["ablation"]

    # Day 3 config, unchanged except the starting model and where outputs go.
    cfg = load_config(acfg["sft_config"])
    cfg["student"]["base"] = day5["models"]["base"]          # the ONE real change
    cfg["student"]["base_revision"] = "main"
    cfg["paths"]["sft_output_dir"] = acfg["output_dir"]
    cfg["paths"]["sft_merged_dir"] = acfg["merged_dir"]

    derived = repo_root() / "configs" / "day5_ablation_sft.yaml"
    derived.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"ABLATION: SFT from BASE ({cfg['student']['base']}) instead of cpt-v2.")
    print(f"Same data/settings as {acfg['sft_config']} -> derived config {derived.name}\n")

    subprocess.run([sys.executable, str(repo_root() / "training" / "sft.py"), "--config", str(derived)],
                   check=True)

    # Label the lineage so it is never confused with the real sft-v2.
    lin_path = repo_root() / acfg["output_dir"] / "lineage.json"
    if lin_path.exists():
        lin = json.loads(lin_path.read_text())
        lin["stage"] = "sft-from-base (ablation: no CPT)"
        lin_path.write_text(json.dumps(lin, indent=2))
    print(f"\nDone. Add this line under `models:` in {args.config} and re-run eval_suite.py:")
    print(f'  sft_from_base: "{acfg["merged_dir"]}"')


if __name__ == "__main__":
    main()