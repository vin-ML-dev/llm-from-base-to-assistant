"""Day 1 environment tests.

Light checks that the pinned stack imports, the config parses, and the repo
skeleton exists. These do NOT download the model (kept fast + offline-friendly);
the model itself is exercised by scripts/smoke_test.py.

Run:
    pytest tests/test_environment.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_core_imports():
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import datasets  # noqa: F401
    import peft  # noqa: F401
    import trl  # noqa: F401


def test_day1_config_parses():
    cfg = yaml.safe_load((ROOT / "configs" / "day1.yaml").read_text())
    assert cfg["model"]["id"] == "Qwen/Qwen3-1.7B-Base"
    # We must start from a BASE checkpoint, not the instruct sibling.
    assert cfg["model"]["id"] != cfg["model"]["instruct_baseline_id"]
    assert "eval_heldout_dir" in cfg["dataset"]


def test_repo_skeleton_exists():
    for d in ("configs", "data", "training", "evaluation", "serving", "tests", "docs", "scripts"):
        assert (ROOT / d).is_dir(), f"missing directory: {d}"


def test_scripts_present():
    for s in (
        "smoke_test.py",
        "explore_tokenizer.py",
        "inspect_model.py",
        "forward_pass_probs.py",
        "base_vs_chat.py",
        "generation_settings.py",
        "lock_eval_set.py",
    ):
        assert (ROOT / "scripts" / s).is_file(), f"missing script: {s}"
