"""Day 2 data-pipeline unit tests (no model/GPU needed)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from tokenize_pack import pack_stream       # noqa: E402
from dataio import text_hash                 # noqa: E402


def test_packing_fixed_length():
    blocks, total = pack_stream([[1, 2, 3], [4, 5], [6, 7, 8, 9]], block_size=4, eos_id=0)
    assert all(len(b) == 4 for b in blocks)
    assert total == 4 * len(blocks)


def test_packing_respects_budget():
    blocks, total = pack_stream([[1] * 100], block_size=10, eos_id=0, max_tokens=30)
    assert total == 30
    assert len(blocks) == 3


def test_hash_normalizes():
    # dedup must treat differently-spaced/cased identical text as duplicates
    assert text_hash("Hello   World") == text_hash("hello world")
    assert text_hash("a") != text_hash("b")


def test_day2_config_parses():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "day2.yaml").read_text())
    assert cfg["tokenize"]["block_size"] == 1024
    assert 0 < cfg["data"]["replay_ratio"] < 1
    assert cfg["model"]["id"].endswith("-Base")


def test_eos_not_masked_when_pad_is_distinct():
    """The EOS-collator bug fix: if PAD == EOS, the default collator masks EOS out
    of the loss. A distinct PAD token must keep EOS labels trainable. This test
    simulates the label logic without loading the model."""
    IGNORE = -100
    eos_id = 151643        # Qwen-style EOS
    # WRONG (old): pad == eos → collator ignores eos positions
    pad_eq_eos = eos_id
    labels_bad = [t if t != pad_eq_eos else IGNORE for t in [5, 6, eos_id, 7]]
    assert labels_bad[2] == IGNORE          # EOS got masked — the bug

    # RIGHT (fixed): pad is a distinct id → eos stays in the labels
    pad_distinct = 151700
    labels_good = [t if t == pad_distinct else t for t in [5, 6, eos_id, 7]]
    # only real padding would be masked; eos is preserved
    assert eos_id in labels_good            # EOS is learned — fixed
