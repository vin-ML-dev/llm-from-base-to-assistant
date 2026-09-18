"""Day 3 SFT unit tests (no GPU/model needed)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from sft_common import strip_thinking, load_config   # noqa: E402


def test_strip_thinking_removes_block():
    raw = "<think>let me reason about this</think>The answer is 42."
    assert strip_thinking(raw) == "The answer is 42."
    assert "think" not in strip_thinking(raw)


def test_strip_thinking_keeps_plain_text():
    assert strip_thinking("Just an answer.") == "Just an answer."


def test_day3_config_parses():
    cfg = load_config(str(ROOT / "configs" / "day3.yaml"))
    assert cfg["generation"]["teacher_model"] == "Qwen/Qwen3-14B"
    assert cfg["judge"]["judge_model"] == "Qwen/Qwen3-8B"
    # teacher and judge must differ (less self-bias)
    assert cfg["generation"]["teacher_model"] != cfg["judge"]["judge_model"]
    assert cfg["train"]["use_lora"] is True
    assert cfg["train"]["assistant_only_loss"] is True


def test_assistant_only_masking_logic():
    """The core SFT idea: only assistant tokens are graded (labels != -100);
    system/user/padding are ignored (-100). Simulated at the label level."""
    IGNORE = -100
    # roles per token position
    roles = ["system", "system", "user", "user", "assistant", "assistant", "pad"]
    tokens = [10, 11, 20, 21, 30, 31, 0]
    labels = [t if r == "assistant" else IGNORE for t, r in zip(tokens, roles)]
    # assistant tokens are trained
    assert labels[4] == 30 and labels[5] == 31
    # system/user/pad are ignored
    assert labels[0] == IGNORE and labels[2] == IGNORE and labels[6] == IGNORE
    # at least one token is actually trained
    assert any(l != IGNORE for l in labels)
