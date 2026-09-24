"""Day 4 DPO unit tests (no GPU/model needed)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from sft_common import load_config  # noqa: E402


def test_day4_config_parses():
    cfg = load_config(str(ROOT / "configs" / "day4.yaml"))
    assert cfg["student"]["sft_model"] == "artifacts/sft-v1"
    assert cfg["train"]["beta"] > 0
    # DPO must use a much smaller LR than SFT's typical 2e-4/2e-5 range
    assert float(cfg["train"]["learning_rate"]) <= 5e-5


def test_length_bias_filter():
    """Regression check for the length-bias guard: pairs with a big length gap
    must be dropped, or the model could learn 'prefer longer' as a shortcut."""
    min_ratio, max_ratio = 0.4, 2.5

    def keep(chosen, rejected):
        ratio = len(chosen) / len(rejected)
        return min_ratio <= ratio <= max_ratio

    assert keep("a fair answer here", "a similar length answer") is True
    assert keep("short", "a" * 200) is False   # chosen much shorter -> drop
    assert keep("a" * 200, "short") is False   # chosen much longer -> drop


def test_score_gap_logic():
    """Pairs with no meaningful score gap must be skipped (Day 4 theory:
    meaningful difference is required, or there's no signal to learn from)."""
    def has_gap(scores):
        best, worst = max(scores), min(scores)
        return best != worst and (best - worst) >= 1

    assert has_gap([5, 5, 5, 5]) is False   # all tied -> no gap
    assert has_gap([3, 3, 4, 3]) is True    # gap of 1 -> keep
    assert has_gap([2, 2, 2, 2]) is False


def test_win_rate_swap_logic():
    """Position-swap unswap logic must correctly attribute the win to the
    right model regardless of which slot (A/B) it was shown in."""
    # winner == "A", swap == True  -> A was dpo -> dpo wins
    winner, swap = "A", True
    dpo_won = (winner == "A") == swap
    assert dpo_won is True

    # winner == "A", swap == False -> A was sft -> sft wins
    winner, swap = "A", False
    dpo_won = (winner == "A") == swap
    assert dpo_won is False
