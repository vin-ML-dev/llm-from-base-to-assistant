"""Day 4 (PILOT) · Step 1 — SAMPLE on-policy candidates from sft-v2.

Generates candidates_per_prompt answers from sft-v2 at VARIED temperatures, so
the candidates expose the model's OWN mistakes (on-policy). A separate step
(judge) may later flag prompts where all candidates are weak; the teacher is
used only SELECTIVELY there, not as blanket gold. cpt-v2 is intentionally
excluded from this first pilot (easy contrast reveals little).

Domain prompts use the ML-assistant system prompt; general prompts use NO
system prompt (mirrors the SFT policy exactly).

Writes data/dpo/candidates.jsonl.

Usage:
    pip install langid
    python data/sample_candidates.py --config configs/day4.yaml
    python data/sample_candidates.py --config configs/day4.yaml --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json as _json
import random
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from html import unescape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import (is_near_duplicate, load_config, read_jsonl,  # noqa: E402
                        repo_root, write_jsonl)

# --- content cleaning / filtering (fix garbage prompts + fabricated URLs) -----
import re as _re

# Existing dataset policy omits URL targets; this is not a factuality guarantee.
_MD_LINK = _re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")     # [text](url) -> text
_BARE_URL = _re.compile(r"https?://\S+|www\.\S+", _re.IGNORECASE)
# Markup/junk that signals a corrupted prompt (HTML, badges, doc-builder syntax).
_MARKUP_SIGNS = ("<div", "<img", "</", "style=", "src=", "img.shields.io",
                 "<Course", "[[", "]]", "classNames=", "notebooks=", "![")
_WORD = _re.compile(r"[^\W\d_]+", _re.UNICODE)
_CHAT_TOKEN = _re.compile(r"<\|[^>]+\|>|</?(?:system|user|assistant|think)>|\[/?INST\]", _re.I)
_CONTROL = _re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f\ufffd]")
_HTML_TAG = _re.compile(r"</?(?:div|span|img|script|style|iframe|table|a|p|br)\b[^>]*>", _re.I)
_BOILERPLATE = _re.compile(
    r"^(?:#{1,6}\s*)?(?:references|bibliography|table of contents|acknowledg(?:e)?ments)\s*(?::|$)"
    r"|\b(?:all rights reserved|accept all cookies|edit this page|back to top|"
    r"skip to (?:main )?content|javascript (?:is )?required)\b", _re.I
)
_CITATION = _re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_SENTENCE_END = _re.compile(r'''[.!?。！？]["'”’\)\]]*$''')
FILTER_VERSION = 3
_DOC_JUNK = _re.compile(
    r"\bEXIT_FROM_INTERACTION\w*|\b(?:BEFORE|AFTER)_BUTTON_CLICK\w*"
    r"|\{\s*(?:label|value)\s*:|[\]}]\s*/>"
    r"|\bIncludes bibliographical references\b|</?quote\b"
    r"|\bNdrFcShort\b|\bROKE\b|\b_STYLE\s*:"
    r"|\bPrevious Page Next Page\b|\bback to the table of contents\b"
    r"|[𬜯𬒔𨟠𖠚𤧛]", _re.I
)
_CODE = _re.compile(r"```.*?(?:```|$)|`[^`\n]+`", _re.S)
_MATH = _re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\(.*?\\\)|\\\[.*?\\\]", _re.S)
_CALL = _re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*\([^()\n]*\)){1,}")
_ABBREVIATION_END = _re.compile(
    r"\b(?:i\.e|e\.g|vs|fig|eq|sec|dr|mr|mrs|prof|al)\.$", _re.I)
_MISSING_CONTEXT = _re.compile(
    r"\b(?:the|this)\s+(?:tutorial|code|table|figure|diagram)\s+(?:above|below)\b", _re.I)


def _complete_sentence(text):
    """Structural check only; it cannot prove that the prose is self-contained."""
    s = text.strip()
    if not _SENTENCE_END.search(s) or s.endswith(("...", "…")):
        return False
    if _ABBREVIATION_END.search(s.rstrip('\"\u201d\u2019')):
        return False
    if s.count("`") % 2:
        return False
    prose = _prose(s)
    # Mixed mathematical interval brackets are valid, e.g. (0, 1].
    prose = _re.sub(r"[\[(]\s*[-+\w.∞]+\s*,\s*[-+\w.∞]+\s*[\])]", "", prose)
    return prose.count("(") == prose.count(")")


def _source_sentences(paragraph, repairs=None):
    """Join false sentence breaks using only the following source text.

    Never append punctuation or invent an ending for a genuinely broken source.
    """
    pending = ""
    parts = _re.split(r'''(?<=[.!?。！？])\s+|(?<=[.!?]["”’])\s+''', paragraph)
    for part in parts:
        if not part:
            continue
        if pending:
            if repairs is not None:
                repairs["joined_source_parts"] += 1
            pending += " " + part
        else:
            pending = part
        if _complete_sentence(pending):
            yield pending
            pending = ""
    if pending:
        yield pending  # Kept intact so the caller can log why it is rejected.


@lru_cache(maxsize=1)
def _language_identifier():
    try:
        from langid.langid import LanguageIdentifier, model
    except ImportError as exc:
        raise RuntimeError("English filtering requires langid. Run: pip install langid") from exc
    # Keep all languages: restricting the detector to English would force every
    # input to be labelled English, including foreign text.
    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def _prose(text):
    """Exclude explicit code/math spans from prose-only checks."""
    return _MATH.sub(" ", _CODE.sub(" ", text))


def language_rejection_reason(text):
    prose = _prose(text)
    # Catch small injected script fragments that whole-text language ID misses.
    # Isolated Greek letters and mathematical alphabet symbols remain allowed.
    for word in _WORD.findall(prose):
        foreign = sum(
            "LATIN" not in unicodedata.name(ch, "")
            and "MATHEMATICAL" not in unicodedata.name(ch, "")
            for ch in word
        )
        if foreign >= 3:
            return "non_english_script"
    chunks = [prose] + _re.split(r"(?<=[.!?])\s+|\n+|\\n", prose)
    for chunk in dict.fromkeys(chunks):
        # Short identifiers, names and equations are unreliable language samples.
        if len(_WORD.findall(chunk)) < 8:
            continue
        language, confidence = _language_identifier().classify(chunk)
        if language != "en" and confidence >= 0.90:
            return "non_english_language"
    return None


def repetition_detected(text):
    # Do not confuse legitimate repeated code/math with prose degeneration.
    prose = _prose(text)
    text = " ".join(prose.split())
    # Normalize whitespace and underscore-delimited sentinel loops alike.
    # Retain numbers/signs: '4 heads' and '8 heads' are not the same statement.
    tokens = _re.compile(r"[-+]?\d+(?:\.\d+)?|[^\W\d_]+", _re.UNICODE)
    words = tokens.findall(text.casefold())
    if _re.search(r"([^\W\d_])\1{14,}", text):
        return True
    if any(_WORD.search(m.group(1)) for m in _re.finditer(r"(.{3,60}?)\1{3,}", text, _re.S)):
        return True
    if len(words) >= 12 and Counter(words).most_common(1)[0][1] / len(words) > 0.30:
        return True
    # Two identical substantial sentences are already a defect, even if long.
    sentences = [tokens.findall(s.casefold()) for s in _re.split(r"(?<=[.!?])\s+|\n+", prose)]
    counts = Counter(tuple(s) for s in sentences if len(s) >= 8)
    if any(count >= 2 for count in counts.values()):
        return True
    # Repeated 8-word spans covering most of the text detect loops without
    # rejecting a few shared terms in otherwise useful technical explanations.
    spans = {}
    for i in range(len(words) - 7):
        spans.setdefault(tuple(words[i:i + 8]), []).append(i)
    covered = set()
    for positions in spans.values():
        if len(positions) >= 3:
            for start in positions:
                covered.update(range(start, start + 8))
    return bool(words) and len(covered) / len(words) >= 0.50


def _content_rejection_reason(text):
    if _CONTROL.search(text) or _re.search(r"(?:â€|ðŸ)", text):
        return "encoding_noise"
    prose = _prose(text)
    if _CHAT_TOKEN.search(prose):
        return "chat_control_tokens"
    if _DOC_JUNK.search(prose):
        return "documentation_junk"
    if repetition_detected(text):
        return "repetition"
    calls = _CALL.findall(prose)
    if len(calls) >= 3 and len(_WORD.findall(_CALL.sub(" ", prose))) < 8:
        return "code_soup"
    return None


def clean_prompt_snippet(text: str) -> str:
    """Normalize spacing/entities and omit links without erasing sentence endings."""
    text = unicodedata.normalize("NFC", unescape(text)).replace("\ufeff", "").replace("\u00ad", "")
    text = _MD_LINK.sub(r"\1", text)
    # Keep punctuation after a bare link, unlike removing the entire URL match.
    text = _BARE_URL.sub(lambda m: m.group()[len(m.group().rstrip(".,;:!?")):], text)
    return " ".join(text.split()).strip()


def prompt_rejection_reason(snippet: str, max_chars=None):
    """Check English source prose before wrapping it in an instruction."""
    s = snippet.strip()
    if not s:
        return "empty"
    reason = _content_rejection_reason(s)
    if reason:
        return reason
    prose = _prose(s)
    if _HTML_TAG.search(prose) or any(sign.lower() in prose.lower() for sign in _MARKUP_SIGNS):
        return "markup"
    if _BOILERPLATE.search(s):
        return "boilerplate"
    if len(_CITATION.findall(s)) >= 4:
        return "citation_list"
    if _re.search(r"\|\s*:?-{3,}:?\s*\|", s) or s.lstrip().startswith("```"):
        return "table_or_code_block"
    if max_chars is not None and len(s) > max_chars:
        return "long_sentence"
    if not _complete_sentence(s):
        return "incomplete_sentence"
    if _MISSING_CONTEXT.search(_prose(s)):
        return "missing_referenced_context"
    words = _WORD.findall(s)
    if len(words) < 8:
        return "too_little_prose"
    visible = [ch for ch in s if not ch.isspace()]
    if sum(ch.isalpha() for ch in visible) / max(1, len(visible)) < 0.45:
        return "symbol_noise"
    return language_rejection_reason(s)


def strip_urls(text: str) -> str:
    """Remove URL targets under the existing candidate-output policy."""
    text = _MD_LINK.sub(r"\1", text)     # keep link text, drop the URL
    text = _BARE_URL.sub("", text)       # drop bare URLs
    return _re.sub(r"[ \t]{2,}", " ", text).strip()


def is_bad_prompt(snippet: str) -> bool:
    """True if a snippet fails the prompt-quality checks."""
    return prompt_rejection_reason(snippet) is not None


def output_rejection_reason(text: str):
    """Reject degeneration, not factual mistakes that the DPO judge must assess."""
    s = text.strip()
    if not s:
        return "empty"
    if not any(ch.isalnum() for ch in s):
        return "no_answer_content"
    return _content_rejection_reason(s) or language_rejection_reason(s)


def is_garbage_output(text: str) -> bool:
    return output_rejection_reason(text) is not None


# Diverse instruction templates (kept modest for a pilot).
PROMPT_TEMPLATES = [
    "Explain, in your own words: {snippet}",
    "Summarize the key point of the following: {snippet}",
    "Define and briefly describe the main concept here: {snippet}",
    "What is the significance of this, and why does it matter? {snippet}",
    "List the main ideas from the following: {snippet}",
    "Explain the following step by step: {snippet}",
]


def _windows(text, window_chars, max_windows=None, repairs=None):
    """Pack sentences instead of cutting arbitrary characters or words.

    Overlong sentences and unfinished tails remain separate for rejection with
    a reason. Paragraph boundaries keep navigation blocks away from good prose.
    """
    if window_chars <= 0:
        raise ValueError("sampling.window_chars must be positive.")
    if not isinstance(text, str) or not text.strip():
        return []
    out = []
    for paragraph in _re.split(r"\n\s*\n", text):
        paragraph = _re.sub(r"[ \t\r\n\f]+", " ", paragraph).strip(" \t\r\n\f")
        pending = ""
        for sentence in _source_sentences(paragraph, repairs):
            if not sentence:
                continue
            complete = _complete_sentence(sentence)
            if pending and (len(pending) + 1 + len(sentence) > window_chars or not complete):
                out.append(pending)
                pending = ""
            if len(sentence) > window_chars or not complete:
                out.append(sentence)
            else:
                pending = f"{pending} {sentence}".strip()
        if pending:
            out.append(pending)
    return out if max_windows is None else out[:max_windows]


def _docs_to_prompts(docs, source, rng, prompts_per_doc, window_chars, limit=None, seen=None):
    prompts = []
    rejected = Counter()
    repairs = Counter()
    seen = set() if seen is None else seen
    if prompts_per_doc <= 0:
        raise ValueError("sampling.prompts_per_doc must be positive.")
    if limit is not None and limit <= 0:
        return prompts
    for d in docs:
        kept_doc = 0
        for raw in _windows(d.get("text", ""), window_chars, repairs=repairs):
            snip = clean_prompt_snippet(raw)
            reason = prompt_rejection_reason(snip, max_chars=window_chars)
            # Inspect controls before whitespace normalization can hide them.
            decoded_raw = unescape(raw)
            raw_prose = _prose(decoded_raw)
            if _CONTROL.search(decoded_raw):
                reason = "encoding_noise"
            elif _HTML_TAG.search(raw_prose) or any(sign.lower() in raw_prose.lower() for sign in _MARKUP_SIGNS):
                reason = "markup"
            if reason:
                rejected[reason] += 1
                continue
            key = snip.casefold()
            if key in seen:
                rejected["duplicate_snippet"] += 1
                continue
            seen.add(key)
            t = rng.choice(PROMPT_TEMPLATES)
            doc_hash = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
            doc_id = d.get("id") or d.get("url") or doc_hash
            prompts.append({
                "prompt": t.format(snippet=snip), "source": source,
                "source_doc_id": f"{d.get('source') or d.get('kind') or source}:{doc_id}",
                "source_doc_hash": doc_hash, "source_text": snip,
                "source_origin": d.get("source"), "source_url": d.get("url"),
                "source_title": d.get("title"), "source_license": d.get("license"),
            })
            kept_doc += 1
            if kept_doc >= prompts_per_doc or (limit is not None and len(prompts) >= limit):
                break
        if limit is not None and len(prompts) >= limit:
            break
    print(f"[sample:{source}] kept={len(prompts)} | rejected windows={dict(rejected)} "
          f"| source boundary repairs={dict(repairs)}")
    return prompts


def build_prompt_set(cfg):
    scfg = cfg["sampling"]
    rng = random.Random(scfg["seed"])
    total = scfg["num_prompts"]
    n_general = int(round(total * scfg.get("general_frac", 0.20)))
    n_domain = total - n_general
    ppd, wc = scfg.get("prompts_per_doc", 3), scfg.get("window_chars", 250)
    seen = set()

    # Older day4.yaml files specify clean_dir but omit train_pool. Still use
    # only the training split, never the unsplit domain/replay source files.
    pool_path = cfg["paths"].get("train_pool") or str(Path(cfg["paths"]["clean_dir"]) / "pool_train.jsonl")
    pool = list(read_jsonl(pool_path))
    if any(d.get("kind") not in {"domain", "replay"} for d in pool):
        raise ValueError("Every training-pool record must have kind='domain' or 'replay'.")
    dom = [d for d in pool if d["kind"] == "domain"]
    rng.shuffle(dom)
    domain_prompts = _docs_to_prompts(dom, "domain", rng, ppd, wc, limit=n_domain, seen=seen)

    gdocs = [d for d in pool if d["kind"] == "replay"]
    rng.shuffle(gdocs)
    general_prompts = _docs_to_prompts(gdocs, "general", rng, ppd, wc, limit=n_general, seen=seen)
    if n_domain and not domain_prompts:
        raise ValueError("No valid domain prompts in the training pool after filtering.")
    if n_general and not general_prompts:
        raise ValueError("General prompts were requested, but no valid training replay remains after filtering.")
    if len(domain_prompts) < n_domain or len(general_prompts) < n_general:
        print(f"[sample] SHORTFALL after filtering: domain={len(domain_prompts)}/{n_domain}, "
              f"general={len(general_prompts)}/{n_general}. Do not assume the requested count was reached.")

    prompts = domain_prompts + general_prompts
    rng.shuffle(prompts)
    print(f"[sample] {sum(p['source']=='domain' for p in prompts)} domain + "
          f"{sum(p['source']=='general' for p in prompts)} general = {len(prompts)}")
    return prompts


def system_for(source, cfg):
    """Domain -> ML-assistant system prompt; general -> NO system prompt."""
    return None if source == "general" else cfg["chat"]["system_prompt"]


def sampling_temperatures(scfg):
    temps = scfg.get("temperatures")
    count = scfg.get("candidates_per_prompt", len(temps) if temps else 1)
    if type(count) is not int or count <= 0:
        raise ValueError("sampling.candidates_per_prompt must be a positive integer.")
    if temps is None:
        temps = [scfg.get("temperature", 0.7)] * count
    if not isinstance(temps, list) or len(temps) != count:
        raise ValueError("sampling.temperatures must have one entry per requested candidate.")
    return temps


def generate_candidates(model, tok, system, prompt, scfg, rejected=None):
    import torch
    # Stable per-prompt sampling seed, including when earlier prompts are skipped on resume.
    torch.manual_seed(scfg["seed"] + int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16))
    temps = sampling_temperatures(scfg)
    dedup = scfg.get("dedup_threshold", 1.0)
    if tok.eos_token_id is None:
        raise ValueError("The SFT tokenizer must define its generation EOS token.")

    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    plen = inputs["input_ids"].shape[1]

    kept = []
    rejected = Counter() if rejected is None else rejected
    # Accept ANY valid end-of-turn token as "finished", so a good answer is never
    # dropped just because the tokenizer's EOS and the model's end token differ.
    vocab = tok.get_vocab()
    stop_ids = sorted({tok.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab})
    for temp in temps:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=scfg["max_new_tokens"],
                do_sample=(temp > 0), temperature=max(temp, 0.01),
                top_p=scfg["top_p"], pad_token_id=tok.pad_token_id,
                eos_token_id=stop_ids,
            )
        generated = out[0][plen:]
        token_ids = generated.tolist()
        ended = bool(token_ids) and token_ids[-1] in stop_ids
        if not ended and len(token_ids) >= scfg["max_new_tokens"]:
            rejected["truncated_at_token_limit"] += 1
            continue
        # Remove the normal terminal EOS, but expose embedded control tokens
        # to the filters instead of hiding them with skip_special_tokens=True.
        ans = tok.decode(token_ids[:-1] if ended else token_ids, skip_special_tokens=False).strip()
        reason = output_rejection_reason(ans)
        ans = strip_urls(ans)
        reason = reason or output_rejection_reason(ans)
        if reason:
            rejected[reason] += 1
            continue
        if any(is_near_duplicate(ans, s, dedup) for s in kept):
            rejected["duplicate_candidate"] += 1
            continue
        kept.append(ans)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day4.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["sampling"]
    sampling_temperatures(scfg)  # Validate before paying for model loading.
    if scfg.get("dedup_threshold", 1.0) < 1.0:
        print("[sample] NOTE: dedup_threshold < 1.0 can discard answers that differ "
              "in a crucial number or negation; use 1.0 for exact-only filtering.")

    from sft_common import load_model_and_tokenizer
    _language_identifier()  # Fail before loading the student if dependency is missing.
    prompts = build_prompt_set(cfg)

    out_rel = cfg["paths"]["candidates"]
    out_path = out_rel if str(out_rel).startswith("/") else str(repo_root() / out_rel)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and Path(out_path).exists():
        with open(out_path, encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                row = _json.loads(line)
                if row.get("filter_version") != FILTER_VERSION:
                    raise ValueError("Existing candidates use older filters. Back up the file and rebuild without --resume.")
                if not row.get("source_doc_id") or not row.get("source_doc_hash"):
                    raise ValueError("Existing candidates lack source provenance. Rebuild without --resume.")
                done.add(row["prompt"])
        print(f"[sample] RESUME: {len(done)} already done")

    model, tok = load_model_and_tokenizer(cfg, for_generation=True)
    f = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    kept, skipped = len(done), 0
    rejected_candidates = Counter()
    try:
        for idx, item in enumerate(prompts, 1):
            if item["prompt"] in done:
                continue
            system = system_for(item["source"], cfg)
            cands = generate_candidates(model, tok, system, item["prompt"], scfg, rejected_candidates)
            if len(cands) < 2:
                skipped += 1
            else:
                f.write(_json.dumps({
                    **item,
                    "filter_version": FILTER_VERSION,
                    "system": system, "candidates": cands,
                    "reference": "",   # judge may add a teacher candidate, not a gold reference
                }, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            if idx % 10 == 0 or idx == len(prompts):
                print(f"[sample] {idx}/{len(prompts)} | kept {kept} | skipped {skipped}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[sample] INTERRUPTED at {idx}; {kept} saved. Re-run with --resume.")
    finally:
        f.close()

    print(f"[sample] DONE: {kept} prompts with >=2 candidates -> {out_path}")
    print(f"[sample] rejected candidates={dict(rejected_candidates)}")
    print("[sample] Next: python data/judge_pairs.py --config configs/day4.yaml")


if __name__ == "__main__":
    main()
