"""Day 1 — lock the held-out evaluation set location.

The single hard rule of the project: one evaluation slice is carved out and
NEVER used for continued pretraining, SFT generation, or DPO prompts. This
script just creates the directory and drops a guard README so the rule is
visible in the repo from Day 1. You populate it with real eval documents on
Day 2, after the train/val/held-out split.

Usage:
    python scripts/lock_eval_set.py
"""
from __future__ import annotations

from pathlib import Path

from _common import load_config, repo_root


GUARD = """# HELD-OUT EVALUATION SET — DO NOT TRAIN ON

Locked on Day 1. Populated on Day 2 (document-level split).

**Never use anything in this directory for:**
- continued pretraining (CPT)
- SFT instruction-data generation
- DPO preference prompts

Using these documents for any training stage contaminates every later
evaluation number and invalidates the honest Base/CPT/SFT/DPO/instruct
comparison on Day 5.

Keep a note in `docs/lineage.md` of the exact split (version + hash) once created.
"""


def main() -> None:
    cfg = load_config("configs/day1.yaml")
    target = repo_root() / cfg["dataset"]["eval_heldout_dir"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(GUARD)
    print(f"Locked held-out eval location: {target}")
    print("Populate it on Day 2 after the document-level split.")


if __name__ == "__main__":
    main()
