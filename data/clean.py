"""
Step 2 — CLEAN + DEDUPLICATE (aggressive, cross-source).

Reads data/raw/*.jsonl, cleans the text, drops too-short docs, then removes
duplicates in LAYERS:

  1. EXACT   — identical normalized text (fast hash) is dropped outright.
  2. NEAR    — MinHash/LSH finds near-duplicate CANDIDATES across ALL sources
               (so the same content scraped as a doc and described in a paper
               is caught), then a similarity check VERIFIES each candidate.
  3. KEEP BEST — among a near-duplicate group we keep the longest/richest copy
               and record every source it appeared in (`dup_sources`), so no
               provenance is lost.

Text cleaning also strips control characters (PDF junk like form-feed / null)
that can survive extraction — folded in from the old clean_raw_pairs step.

Usage:
    python clean.py --config day2_cpt.yaml
"""

import argparse
import re
from pathlib import Path

from dataio import ensure_dir, load_config, read_jsonl, text_hash, write_jsonl

# Raw files produced by collect.py that we clean here. Replay is cleaned too so
# it gets deduplicated against the domain docs (no general/domain overlap).
RAW_FILES = [
    "domain_wikipedia.jsonl",
    "domain_papers.jsonl",
    "domain_docs.jsonl",
    "replay.jsonl",
]

# Common scraped/page boilerplate lines to drop.
BOILERPLATE = {
    "edit this page", "copied", "join the hugging face community",
    "table of contents", "on this page", "previous", "next",
}

# Control characters to strip: C0 (keep \n and \t) AND C1 controls
# (\x80-\x9f, e.g. the \x96 seen in mangled PDF extractions).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

import unicodedata as _unicodedata


def _fix_unicode(text: str) -> str:
    """Normalize math-styled Unicode back to plain ASCII where possible.

    PDF extraction of papers turns equation letters into 'mathematical
    alphanumeric symbols' (e.g. 'S𝑖𝑗' instead of 'Sij'). NFKC normalization
    maps those to readable ASCII, so the model learns clean text instead of
    Unicode soup. Text without such symbols is unchanged.
    """
    return _unicodedata.normalize("NFKC", text)


def _garbage_ratio(text: str) -> float:
    """Fraction of characters that are NOT normal readable text.

    'Readable' = letters, digits, spaces, common punctuation. A high ratio means
    the chunk is mostly leftover math/layout symbols even after normalization --
    extraction garbage the model shouldn't learn from.
    """
    if not text:
        return 1.0
    readable = sum(
        1 for ch in text
        if ch.isalnum() or ch.isspace() or ch in ".,;:!?()[]{}\"'-+=/*%$#@&_|<>"
    )
    return 1.0 - (readable / len(text))

# Repeated documentation boilerplate that appears across MANY doc pages (license
# headers, frontmatter, HTML comments). Left in, these shared blocks inflate the
# cross-document 13-gram overlap (tripping split.py's leakage check) and add
# low-value tokens. Stripping them improves corpus quality AND removes the false
# overlap. These run on the raw text BEFORE line-level normalization.
_DOC_BOILERPLATE_PATTERNS = [
    # HTML comment blocks -- the Apache license header that opens every HF doc:
    #   <!--Copyright 2023 The HuggingFace Team... under the License. -->
    re.compile(r"<!--.*?-->", re.DOTALL),
    # Markdown frontmatter block at the very start:  --- ... ---
    re.compile(r"\A---\n.*?\n---\n", re.DOTALL),
    # HF's "this file is in Markdown but contains..." rendering-warning note,
    # which repeats verbatim across many pages.
    re.compile(r"⚠️.*?rendered properly in your Markdown viewer\.?", re.DOTALL),
    # d2l / reStructuredText cross-reference directives that add no prose value:
    #   :label:`sec_x`  :numref:`fig_y`  :eqlabel:`eq_z`  :cite:`ref`  :ref:`x`
    re.compile(r":(?:label|numref|eqlabel|ref|cite|width|height):`[^`]*`"),
    # d2l code-cell fences markers like ```{.python .input} -> plain ```
    re.compile(r"```\{[^}]*\}"),
    # RST-style directive lines (.. code::, .. figure::, .. math::)
    re.compile(r"^\.\.\s+\w+::.*$", re.MULTILINE),
]


def strip_doc_boilerplate(text: str) -> str:
    """Remove repeated documentation boilerplate blocks (license headers, etc.)."""
    for pat in _DOC_BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return text


def clean_text(text: str) -> str:
    """Normalize math-styled Unicode to ASCII, then strip control/C1 chars."""
    if not text:
        return ""
    text = _fix_unicode(text)          # 𝑖𝑗 -> ij, etc.
    return _CONTROL_RE.sub("", text)


def normalize(text, source=None):
    """Standardize whitespace and strip boilerplate, WITHOUT destroying code.

    Indentation inside fenced code blocks (``` ... ```) is preserved, because
    Python and other code is meaningless without it. Prose lines are still
    stripped and space-collapsed as before.
    """
    text = clean_text(text)
    text = strip_doc_boilerplate(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    kept_lines = []
    in_code = False
    for line in text.split("\n"):
        # Toggle code-block state on fence lines (```), keeping the fence.
        if line.lstrip().startswith("```"):
            in_code = not in_code
            kept_lines.append(line.rstrip())
            continue
        if in_code:
            # Inside code: keep leading indentation, only trim trailing spaces.
            kept_lines.append(line.rstrip())
            continue
        # Prose line: collapse internal whitespace and strip, as before.
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        if stripped.lower() in BOILERPLATE:
            continue
        kept_lines.append(stripped)

    # Collapse 3+ blank lines (outside code this is fine; code rarely has them).
    joined = "\n".join(kept_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# Near-duplicate detection: MinHash + LSH
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5):
    """Word k-shingles (sets of k consecutive words) used to fingerprint a doc."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingles, num_perm: int):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity, used to VERIFY an LSH candidate pair."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_near(docs, cfg):
    """Layer 2+3: cross-source near-duplicate removal.

    docs: list of already-exact-deduped rows (each has normalized 'text').
    Returns the kept rows, each annotated with `dup_sources` (all sources the
    near-duplicate group appeared in).
    """
    from datasketch import MinHashLSH

    ccfg = cfg["data"]
    threshold = ccfg.get("near_dup_threshold", 0.8)   # Jaccard cutoff
    num_perm = ccfg.get("minhash_perm", 128)
    shingle_k = ccfg.get("shingle_k", 5)

    # Precompute shingles + minhashes.
    shingle_sets, minhashes = [], []
    for d in docs:
        sh = _shingles(d["text"], shingle_k)
        shingle_sets.append(sh)
        minhashes.append(_minhash(sh, num_perm))

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    removed = set()
    merged_sources = {}     # index kept -> set of sources merged into it
    near_dropped = 0

    for i in range(len(docs)):
        if i in removed:
            continue
        # LSH candidates for doc i (approximate near-duplicates).
        candidates = [int(x) for x in lsh.query(minhashes[i])]
        for j in candidates:
            if j == i or j in removed:
                continue
            # VERIFY with exact Jaccard so LSH false-positives don't drop good docs.
            sim = _jaccard(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                # KEEP BEST: whichever text is longer stays; the other is merged.
                keep, drop = (i, j) if len(docs[i]["text"]) >= len(docs[j]["text"]) else (j, i)
                removed.add(drop)
                grp = merged_sources.setdefault(keep, {docs[keep]["source"]})
                grp.add(docs[drop]["source"])
                near_dropped += 1

    kept = []
    for i, d in enumerate(docs):
        if i in removed:
            continue
        # record every source this (near-dup) content appeared in
        sources = merged_sources.get(i, {d["source"]})
        d["dup_sources"] = sorted(sources)
        kept.append(d)

    return kept, near_dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["clean_dir"])
    min_chars = cfg["data"]["min_doc_chars"]
    # Docs with more than this fraction of non-readable chars (after Unicode
    # normalization) are dropped as extraction garbage. 0.30 = keep anything at
    # least ~70% normal text; equations/symbols above that are unrecoverable.
    max_garbage_ratio = cfg["data"].get("max_garbage_ratio", 0.30)

    # ---- Pass 1: normalize + drop short + EXACT dedup (across all files) ----
    seen_hashes = set()
    per_file = {}          # fname -> list of surviving rows (exact-deduped)
    stats = {}

    for fname in RAW_FILES:
        rows = read_jsonl(f"{cfg['paths']['raw_dir']}/{fname}")
        if not rows:
            print(f"[clean] {fname}: not found or empty, skipping")
            # Remove any stale cleaned copy so a missing raw input can't leave
            # old cleaned data behind to confuse later steps.
            stale = Path(cfg["paths"]["clean_dir"]) / fname
            if stale.exists():
                stale.unlink()
                print(f"[clean] {fname}: removed stale cleaned file {stale}")
            continue
        kept, dropped_short, dropped_exact, dropped_garbage = [], 0, 0, 0
        for row in rows:
            text = normalize(row.get("text", ""))
            if len(text) < min_chars:
                dropped_short += 1
                continue
            # Drop docs that are still mostly symbol/math soup after normalization
            # (unrecoverable PDF-extraction garbage). Readable docs pass easily.
            if _garbage_ratio(text) > max_garbage_ratio:
                dropped_garbage += 1
                continue
            h = text_hash(text)
            if h in seen_hashes:
                dropped_exact += 1
                continue
            seen_hashes.add(h)
            row["text"] = text
            row["hash"] = h
            row.setdefault("license", "unknown")   # governance: never blank
            kept.append(row)
        per_file[fname] = kept
        stats[fname] = {"short": dropped_short, "exact": dropped_exact,
                        "garbage": dropped_garbage}

    # ---- Pass 2+3: cross-source NEAR-dup (MinHash/LSH + verify + keep best) --
    all_docs = [row for fname in per_file for row in per_file[fname]]
    print(f"[clean] exact-deduped pool: {len(all_docs)} docs -> running near-dup ...")
    deduped, near_dropped = dedup_near(all_docs, cfg)
    print(f"[clean] near-dup removed: {near_dropped}")

    # ---- Regroup kept docs back into their source files and write ----
    by_file = {fname: [] for fname in per_file}
    # map each kept doc to the file it came from via its hash
    hash_to_file = {}
    for fname, rows in per_file.items():
        for r in rows:
            hash_to_file[r["hash"]] = fname
    for d in deduped:
        by_file[hash_to_file[d["hash"]]].append(d)

    total = 0
    for fname, rows in by_file.items():
        n = write_jsonl(f"{cfg['paths']['clean_dir']}/{fname}", rows)
        s = stats.get(fname, {"short": 0, "exact": 0})
        print(f"[clean] {fname}: kept {n}, dropped_short {s['short']}, "
              f"dropped_exact {s['exact']}, dropped_garbage {s.get('garbage', 0)}")
        total += n

    # license breakdown (governance signal)
    from collections import Counter
    lic = Counter(d.get("license", "unknown") for d in deduped)
    print("\n[clean] license breakdown:")
    for k, v in lic.most_common():
        print(f"    {v:5d}  {k}")

    print(f"\n[clean] total kept: {total}")
    print("Tip: open a few files in data/clean/ and read some docs by hand before continuing.")
    print("Next: python split.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
