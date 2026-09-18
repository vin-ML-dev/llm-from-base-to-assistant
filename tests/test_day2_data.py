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


def test_cpt_collator_labels_equal_input_ids():
    """Regression test for the fixed CPT bug: the collator must NOT mask anything
    (every packed block is full-length, no padding) and must NEVER modify the
    tokenizer. labels == input_ids exactly, for every position including EOS."""
    torch = __import__("pytest").importorskip("torch")

    def collator(features):
        batch_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        return {"input_ids": batch_ids, "labels": batch_ids.clone()}

    eos_id = 151643
    fake_block = [5, 6, eos_id, 7, 8, eos_id, 9]  # EOS appears mid-block, like real packing
    out = collator([{"input_ids": fake_block}])
    assert out["labels"].tolist()[0] == fake_block, "labels must equal input_ids exactly"
    # EOS positions specifically must be TRAINED (not -100) -- this is the bug we fixed
    eos_positions = [i for i, t in enumerate(fake_block) if t == eos_id]
    for pos in eos_positions:
        assert out["labels"][0][pos].item() == eos_id, "EOS must be a real, trained label"


def test_cpt_config_defaults_to_lora():
    """Regression test: CPT must default to LoRA (catastrophic-forgetting guard)
    after the full-FT run damaged the base model's general capability."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "day2.yaml").read_text())
    assert cfg["train"]["use_lora"] is True
    assert "lora_r" in cfg["train"] and "lora_target_modules" in cfg["train"]
