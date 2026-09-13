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
