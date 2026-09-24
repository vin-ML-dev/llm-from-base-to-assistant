"""Shared helpers for the SFT pipeline: config, JSONL I/O, GPU cleanup,
Qwen3 thinking-mode stripping, and revision pinning."""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/day3.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path) -> Path:
    return Path(path) if str(path).startswith("/") else repo_root() / path


def write_jsonl(path, rows) -> int:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with open(_resolve(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)  # unclosed (truncated) block


def strip_thinking(text: str) -> str:
    """Remove Qwen3's reasoning block. Handles both closed <think>...</think>
    and an unclosed <think>... left when generation is truncated mid-thought."""
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def free_gpu():
    """Release GPU memory between the teacher and judge phases."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    """Return True if answers `a` and `b` are basically the same text.

    We compare the normalized versions with difflib's similarity ratio
    (0.0 = totally different, 1.0 = identical). If the ratio is at or above
    `threshold`, the two answers are treated as duplicates.

    Why: two near-identical candidates give DPO almost no signal to learn
    from, so we drop one of them upstream.
    """
    ratio = difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    return ratio >= threshold
    
def resolve_revision(repo_id: str, revision: str) -> str:
    """Resolve a Hub revision to a commit hash; never silently return 'main'."""
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo_id, revision=revision).sha
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {repo_id!r} revision {revision!r}. "
            "Check repository access, connectivity, and huggingface_hub installation."
        ) from exc
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise RuntimeError(f"No valid commit hash returned for {repo_id!r} revision {revision!r}.")
    return sha


def load_tokenizer(source, revision=None):
    """Load a checkpoint's tokenizer, tolerating list-style extra_special_tokens
    (some Qwen3 configs) that older Transformers versions reject as a mapping.

    Never adds vocabulary or changes token IDs: any extra token must already be
    in the vocab, otherwise we raise rather than train/eval on shifted IDs.
    """
    from transformers import AutoTokenizer
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    config = get_tokenizer_config(source, revision=revision)
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return AutoTokenizer.from_pretrained(source, revision=revision)

    # Reload with the field cleared, then verify the tokens already exist.
    tok = AutoTokenizer.from_pretrained(source, revision=revision, extra_special_tokens={})
    vocab = tok.get_vocab()
    for token in extra:
        content = token.get("content") if isinstance(token, dict) else token
        if not isinstance(content, str) or content not in vocab:
            raise ValueError(
                f"extra_special_token {content!r} absent from {source} vocab; "
                "refusing to add untrained embeddings or shift IDs."
            )
    return tok


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # controls except \t \n \r


def clean_text(s: str) -> str:
    """Strip control characters (null, form-feed, etc.) that PDF extraction
    leaves in source text. These are illegal raw inside JSON strings and are
    junk in training data. Keeps tab/newline/carriage-return."""
    if not isinstance(s, str):
        return s
    return _CTRL_RE.sub("", s)

def normalize_text(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces.

    Used before comparing two candidate answers, so that differences in
    casing or spacing don't make near-identical answers look different.
    """
    return " ".join(text.lower().split())
    
def extract_json(text: str) -> dict | None:
    """Decode the first valid JSON object, tolerating fences or surrounding prose.

    Question/answer fields and judge score ranges are validated by the caller.
    """
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
#==========================================
import difflib

def _is_lora_adapter(model_path: str) -> bool:
    """True if `model_path` is a LoRA adapter (has adapter_config.json)."""
    p = Path(model_path)
    if p.exists():
        return (p / "adapter_config.json").exists()
    try:
        from huggingface_hub import list_repo_files
        return "adapter_config.json" in list_repo_files(model_path)
    except Exception:
        return False


def _adapter_base_model(model_path: str):
    """Read the base model an adapter was trained on, from its config."""
    import json as _json
    p = Path(model_path)
    try:
        if p.exists():
            cfg = _json.loads((p / "adapter_config.json").read_text())
        else:
            from huggingface_hub import hf_hub_download
            cfg = _json.loads(Path(hf_hub_download(model_path, "adapter_config.json")).read_text())
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None
        
def load_model_and_tokenizer(cfg, for_generation: bool = True):
    """Load the student model for sampling/eval.

    Reads student.sft_model (+ optional student.base_model override,
    student.base_revision, student.dtype). Works whether sft_model is a full
    merged model (loaded directly) or a LoRA adapter (base + adapter, merged).
    Returns (model, tokenizer) set up for batched generation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scfg = cfg["student"]
    model_path = scfg["sft_model"]
    revision = scfg.get("base_revision", "main")
    dtype = getattr(torch, scfg.get("dtype", "bfloat16"))

    if _is_lora_adapter(model_path):
        base = scfg.get("base_model") or _adapter_base_model(model_path)
        if not base:
            raise ValueError(
                f"'{model_path}' is a LoRA adapter but its base is unknown. "
                f"Set student.base_model in the config."
            )
        print(f"[load] '{model_path}' is a LoRA adapter; base='{base}'")
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype)
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        try:
            tok = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base, revision=revision)
    else:
        print(f"[load] '{model_path}' is a full model")
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(model_path, revision=revision)

    if for_generation:
        tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok