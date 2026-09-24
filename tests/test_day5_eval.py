"""Day 5 unit tests (no GPU/model needed)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "evaluation"))

from sft_common import load_config  # noqa: E402
from stats import wilson_interval  # noqa: E402


def test_day5_config_parses():
    cfg = load_config(str(ROOT / "configs" / "day5.yaml"))
    assert cfg["models"]["instruct"] != cfg["models"]["base"]  # baseline must be a real comparison
    assert "dpo" in cfg["models"] and "sft" in cfg["models"]


def test_wilson_interval_basic_properties():
    # a 50/50 result on a decent sample should straddle 0.5
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi

    # more successes -> interval shifts higher
    lo2, hi2 = wilson_interval(9, 10)
    assert lo2 > lo

    # small n -> wide interval (this is expected/honest, not a bug)
    lo3, hi3 = wilson_interval(2, 3)
    assert (hi3 - lo3) > 0.3

    # zero samples -> defined, doesn't crash
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_position_swap_tie_logic():
    """Regression check for Block 5: if the two swapped rounds disagree on the
    winner, the result MUST be a tie, never a forced winner."""
    def resolve(round1_winner, round2_winner):
        if round1_winner == round2_winner and round1_winner != "tie":
            return round1_winner
        return "tie"

    assert resolve("dpo", "dpo") == "dpo"          # agree -> confident win
    assert resolve("instruct", "instruct") == "instruct"
    assert resolve("dpo", "instruct") == "tie"      # disagree -> tie, not forced
    assert resolve("tie", "dpo") == "tie"


def test_reproduce_doc_flags_missing_fields():
    """A lineage dict missing a required field must be detected as a gap."""
    required = ["parent_model", "seed", "method"]

    complete = {"parent_model": "x", "seed": 42, "method": "lora"}
    incomplete = {"parent_model": "x", "method": "lora"}  # missing seed

    missing_complete = [k for k in required if k not in complete or complete[k] is None]
    missing_incomplete = [k for k in required if k not in incomplete or incomplete[k] is None]

    assert missing_complete == []
    assert missing_incomplete == ["seed"]
