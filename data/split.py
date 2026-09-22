"""Preserve useful domain data while keeping evaluation independent.

An existing heldout.jsonl stays locked across reruns. Checks cover train,
validation, domain held-out and, when available, general evaluation documents.
Near-overlap checks use shared word sequences; they do not detect paraphrases.
Frozen evaluation is checked directly and cumulatively against candidate source
documents; a weak/transitive group link never excludes a source document.
For new splits, group exact copies, matching IDs and strong pairwise overlaps.
Only choose evaluation groups whose content is independent of all other retained
source groups. Groups unsuitable for evaluation remain available for training.
Fractions are approximate document counts, not token counts.

Admission to frozen evaluation budgets is deterministic and prefers longer texts.
It is a greedy retention heuristic, not a globally optimal token allocation.
An exclusion report records reasons without changing raw or cleaned source files.

Usage: python split.py --config day2.yaml
"""

import argparse
from array import array
from collections import Counter
import hashlib
from itertools import chain
import math
from pathlib import Path
import random
import os
import tempfile

from dataio import load_config, read_jsonl, text_hash, write_jsonl

DOMAIN_FILES = ["domain_wikipedia.jsonl", "domain_papers.jsonl", "domain_docs.jsonl"]


def shingles(text, k=13):
    """Case-insensitive word sequences; retain a fingerprint for short documents."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def validate_rows(rows, name):
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{name}: document {i} has missing or empty text")


def document_id(row):
    """A stable source ID also identifies revised copies of a held-out article."""
    return (str(row.get("source", "")), str(row["id"])) if row.get("id") is not None else None


def overlap_groups(records):
    """Group retained SOURCES only; return groups and safe evaluation group IDs.

    A pair is linked only by equal hashes, source IDs, or >50% coverage of the
    smaller document. The cumulative eligibility check then measures each group
    against ALL other groups. Low-contribution neighbors are never joined just
    because a document has substantial overlap elsewhere.
    """
    parent = list(range(len(records)))
    sizes = [1] * len(records)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = find(i), find(j)
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]

    postings, fingerprints, identities, hashes = {}, [], {}, {}
    for i, (_, row) in enumerate(records):
        identity = document_id(row)
        if identity is not None:
            if identity in identities:
                join(i, identities[identity])
            else:
                identities[identity] = i
        digest = text_hash(row["text"])
        if digest in hashes:
            join(i, hashes[digest])
        else:
            hashes[digest] = i
        values = sorted({
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
            for s in shingles(row["text"])
        })
        fingerprints.append(array("Q", values))
        for value in values:
            owner = postings.get(value)
            if owner is None:
                postings[value] = i
            elif isinstance(owner, int):
                postings[value] = [owner, i]
            else:
                owner.append(i)
        if (i + 1) % 100 == 0:
            print(f"[split] fingerprinted {i + 1}/{len(records)} documents")

    for i, values in enumerate(fingerprints):
        pair_counts = Counter()
        for value in values:
            owners = postings[value]
            if isinstance(owners, list):
                pair_counts.update(j for j in owners if j > i)
        for j, count in pair_counts.items():
            if count * 2 > min(len(values), len(fingerprints[j])):
                join(i, j)

    # Coverage from several outside documents is still checked. Unlike the old
    # algorithm, a failed check makes this group training-only; it removes nothing.
    roots = [find(i) for i in range(len(records))]
    cross_group = {
        value for value, owners in postings.items()
        if isinstance(owners, list) and any(roots[j] != roots[owners[0]] for j in owners[1:])
    }
    unsafe = {
        roots[i] for i, values in enumerate(fingerprints)
        if sum(value in cross_group for value in values) * 2 > len(values)
    }
    grouped = {}
    for i, record in enumerate(records):
        grouped.setdefault(roots[i], []).append(record)
    groups = list(grouped.values())
    eligible = {i for i, root in enumerate(grouped) if root not in unsafe}
    return groups, eligible


class CoverageGuard:
    """Transactional cumulative coverage budget for one fixed evaluation set."""

    def __init__(self, evaluation, label):
        self.evaluation = evaluation
        self.label = label
        self.hashes, self.identities, self.owners = {}, {}, {}
        self.lengths, self.matches = [], []
        for i, row in enumerate(evaluation):
            self.hashes.setdefault(text_hash(row["text"]), []).append(i)
            identity = document_id(row)
            if identity is not None:
                self.identities.setdefault(identity, []).append(i)
            values = shingles(row["text"])
            self.lengths.append(len(values))
            self.matches.append(set())
            for value in values:
                self.owners.setdefault(value, []).append(i)

    def issue(self, reason, indices, additions=None):
        return {
            "evaluation_set": self.label,
            "reason": reason,
            "evaluation_matches": [
                {
                    "source": self.evaluation[i].get("source"),
                    "id": self.evaluation[i].get("id"),
                    "title": self.evaluation[i].get("title"),
                    **({"coverage_if_admitted": len(self.matches[i] | additions[i]) / self.lengths[i]}
                       if additions is not None else {}),
                } for i in indices
            ],
        }

    def probe(self, row, values):
        digest, identity = text_hash(row["text"]), document_id(row)
        if digest in self.hashes:
            return self.issue("exact_evaluation_copy", self.hashes[digest]), {}
        if identity is not None and identity in self.identities:
            return self.issue("same_evaluation_document_id", self.identities[identity]), {}
        additions = {}
        for value in values:
            for i in self.owners.get(value, ()):
                additions.setdefault(i, set()).add(value)
        failing = [
            i for i, added in additions.items()
            if len(self.matches[i] | added) * 2 > self.lengths[i]
        ]
        if failing:
            return self.issue("cumulative_evaluation_overlap", failing, additions), {}
        return None, additions

    def commit(self, additions):
        for i, added in additions.items():
            self.matches[i].update(added)


def protect_frozen(records, heldout, general):
    """Exclude only candidates that individually break a fixed eval constraint.

    Rejected records never consume a coverage budget. In particular, rejecting a
    cleaned copy of a locked paper cannot implicate its low-overlap neighbors.
    """
    guards = [CoverageGuard(heldout, "domain_heldout"), CoverageGuard(general, "general_eval")]
    # General evaluation must stay independent of fixed domain evaluation too.
    for row in heldout:
        issue, additions = guards[1].probe(row, shingles(row["text"]))
        if issue:
            raise ValueError("Locked domain evaluation overlaps general evaluation; no files were written")
        guards[1].commit(additions)
    retained, excluded = [], []
    ordered = sorted(records, key=lambda item: (
        -len(item[1]["text"]), text_hash(item[1]["text"]),
        item[0], str(document_id(item[1])),
    ))
    for index, (role, row) in enumerate(ordered):
        values = shingles(row["text"]) if any(guard.evaluation for guard in guards) else set()
        pending, issues = [], []
        for guard in guards:
            issue, additions = guard.probe(row, values)
            pending.append((guard, additions))
            if issue:
                issues.append(issue)
        if issues:
            excluded.append({
                "role": role, "source": row.get("source"), "id": row.get("id"),
                "title": row.get("title"), "url": row.get("url"),
                "license": row.get("license"), "characters": len(row["text"]),
                "hash": text_hash(row["text"]), "issues": issues,
            })
        else:
            for guard, additions in pending:
                guard.commit(additions)
            retained.append((role, row))
        if (index + 1) % 100 == 0:
            print(f"[split] checked frozen evaluation against {index + 1}/{len(ordered)} source documents")
    return retained, excluded


def write_outputs(outputs):
    """Stage all JSONL writes before replacing any output; atomic per file."""
    staged = []
    try:
        for target, rows in outputs:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            temporary = Path(name)
            staged.append((temporary, target))
            write_jsonl(temporary, rows)
        for temporary, target in staged:
            temporary.replace(target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def take_groups(groups, target, reserve):
    """Approach a document-count target while reserving groups for other splits."""
    available, selected, count = list(groups), [], 0
    while count < target and len(available) > reserve:
        gap = target - count
        # Groups are already shuffled. Prefer one that fits the remaining gap.
        fitting = next((i for i, group in enumerate(available) if len(group) <= gap), None)
        index = fitting if fitting is not None else min(range(len(available)), key=lambda i: len(available[i]))
        group = available[index]
        if selected and abs(count + len(group) - target) >= abs(count - target):
            break
        selected.extend(available.pop(index))
        count = len(selected)
    if target and not selected:
        raise ValueError(
            "Not enough independent document groups for the requested splits. "
            "Review repeated boilerplate/overlap or collect more independent documents. "
            "No new splits were written."
        )
    return selected, available


def check_leakage(previous, evaluation, label):
    """Reject exact matches or >50% coverage of an eval document's 13-grams.

    Retain only shingles relevant to evaluation, rather than all training text.
    Coverage may come from multiple documents, as in the original checker.
    """
    if not evaluation:
        return
    eval_hashes = {text_hash(row["text"]) for row in evaluation}
    eval_ids = {document_id(row) for row in evaluation} - {None}
    eval_sets = [shingles(row["text"]) for row in evaluation]
    wanted = set().union(*eval_sets)
    matches, exact, same_ids = set(), set(), set()
    for row in previous:
        h = text_hash(row["text"])
        if h in eval_hashes:
            exact.add(h)
        identity = document_id(row)
        if identity in eval_ids:
            same_ids.add(identity)
        matches.update(wanted.intersection(shingles(row["text"])))
    near = sum(bool(s) and len(s & matches) / len(s) > 0.5 for s in eval_sets)
    if exact or near or same_ids:
        raise ValueError(
            f"[split] {label}: leakage detected (exact={len(exact)}, "
            f"same_document_id={len(same_ids)}, overlap_flagged={near}; counts may overlap). "
            "No new splits were written. Review/deduplicate the input documents."
        )


def split_documents(cfg):
    clean_dir = Path(cfg["paths"]["clean_dir"])
    heldout_path = Path(cfg.get("evaluation", {}).get(
        "domain_file", str(Path(cfg["paths"]["eval_heldout_dir"]) / "heldout.jsonl")
    ))
    general_path = Path(cfg.get("evaluation", {}).get(
        "general_file", str(Path(cfg["paths"]["eval_heldout_dir"]) / "general_eval.jsonl")
    ))
    exclusions_path = clean_dir / "split_exclusions.jsonl"
    targets = [clean_dir / "pool_train.jsonl", clean_dir / "pool_val.jsonl", heldout_path, general_path, exclusions_path]
    if len({p.resolve() for p in targets}) != len(targets):
        raise ValueError("Training, validation and evaluation paths must be distinct")
    source_paths = { (clean_dir / name).resolve() for name in DOMAIN_FILES + ["replay.jsonl"] }
    if source_paths & {p.resolve() for p in targets}:
        raise ValueError("Split outputs and evaluation paths must not overwrite cleaned source files")

    data = cfg["data"]
    heldout_frac, val_frac = float(data["eval_heldout_frac"]), float(data["val_frac"])
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (heldout_frac, val_frac)) or heldout_frac + val_frac >= 1:
        raise ValueError("Split fractions must be nonnegative and sum to less than 1")
    rng = random.Random(data["seed"])
    domain = []
    for name in DOMAIN_FILES:
        rows = read_jsonl(clean_dir / name)
        validate_rows(rows, name)
        domain.extend(rows)
        print(f"[split] loaded {len(rows)} from {name}")
    if not domain:
        raise ValueError("No cleaned domain documents; run collection and cleaning first")
    replay = read_jsonl(clean_dir / "replay.jsonl")
    validate_rows(replay, "replay")

    # Exact input copies will stay in one group, rather than being split apart.
    domain.sort(key=lambda row: text_hash(row["text"]))
    replay.sort(key=lambda row: text_hash(row["text"]))
    locked = heldout_path.exists()
    heldout = read_jsonl(heldout_path) if locked else []
    validate_rows(heldout, "locked held-out")
    if locked and heldout_frac > 0 and not heldout:
        raise ValueError("Existing held-out file is empty; supply a valid held-out set")
    general = read_jsonl(general_path)
    validate_rows(general, "general evaluation")
    if not general:
        print("[split] General evaluation is absent/empty; its leakage check was not performed.")

    records = [("domain", row) for row in domain]
    records += [("replay", row) for row in replay]
    print(f"[split] source records: domain={len(domain)} replay={len(replay)}; "
          f"separate evaluation copies: locked_domain={len(heldout)} general={len(general)}")
    retained, excluded = protect_frozen(records, heldout, general)
    groups, eligible = overlap_groups(retained)
    print(f"[split] {len(retained)} retained source records -> {len(groups)} source groups")
    candidates, train_domain, replay_kept = [], [], []
    for group_index, group in enumerate(groups):
        domain_rows = [row for role, row in group if role == "domain"]
        replay_rows = [row for role, row in group if role == "replay"]
        if replay_rows or group_index not in eligible:
            # Replay and groups unsuitable for independent evaluation stay in
            # training. Being unsuitable for evaluation never discards a group.
            train_domain.extend(domain_rows)
            replay_kept.extend(replay_rows)
        elif domain_rows:
            candidates.append(domain_rows)
    rng.shuffle(candidates)
    n_available = sum(role == "domain" for role, _ in retained)
    n_val = max(1, int(n_available * val_frac)) if val_frac else 0
    if locked:
        print(f"[split] preserving {len(heldout)} locked held-out documents")
    else:
        n_heldout = max(1, int(n_available * heldout_frac)) if heldout_frac else 0
        reserve = int(n_val > 0) + int(not train_domain)
        heldout, candidates = take_groups(candidates, n_heldout, reserve)
    val, candidates = take_groups(candidates, n_val, int(not train_domain))
    train_domain.extend(row for group in candidates for row in group)
    if not train_domain:
        raise ValueError("No independent domain training documents remain; no new splits were written")
    train_pool = [dict(row, kind="domain") for row in train_domain]
    train_pool.extend(dict(row, kind="replay") for row in replay_kept)
    if excluded:
        print(f"[split] excluded {len(excluded)} source records that fail a direct or cumulative "
              "frozen-evaluation check; source files are unchanged.")
        reasons = Counter(row["issues"][0]["reason"] for row in excluded)
        print(f"[split] primary exclusion reasons: {dict(reasons)}")
        for row in excluded[:10]:
            print(f"[split] excluded: {row.get('source', '?')} / {row.get('id', '?')} / "
                  f"{row['issues'][0]['reason']} / {row.get('title', '')}")
        if len(excluded) > 10:
            print(f"[split] ... and {len(excluded) - 10} more")
    print("[split] Fractions target retained domain document counts; source groups stay together.")
    check_leakage(train_pool, val, "validation vs training")
    check_leakage(chain(train_pool, val), heldout, "held-out vs training/validation")
    check_leakage(chain(train_pool, val, heldout), general, "general eval vs other splits")

    newly_heldout = 0 if locked else len(heldout)
    accounted = len(train_pool) + len(val) + newly_heldout + len(excluded)
    if accounted != len(records):
        raise RuntimeError("Source accounting mismatch; no new splits were written")
    # All content checks pass before any outputs are changed. The supplied
    # dataio.py is supported without changing its read_jsonl signature.
    outputs = [
        (clean_dir / "pool_train.jsonl", train_pool),
        (clean_dir / "pool_val.jsonl", val),
        (exclusions_path, excluded),
    ]
    if not locked:
        outputs.append((heldout_path, heldout))
    write_outputs(outputs)
    print(f"[split] accounting: {len(records)} source records = {len(train_pool)} training "
          f"+ {len(val)} validation + {newly_heldout} newly held-out + {len(excluded)} excluded")
    print(f"[split] train_domain={len(train_domain)} replay={len(replay_kept)} val={len(val)} heldout={len(heldout)}")
    print(f"[split] exclusion details: {exclusions_path}")
    print("[split] No leakage found by the exact-match and word-overlap checks.")
    print("[split] Training license labels:", dict(Counter(row.get("license") or "unknown" for row in train_pool)))
    return train_pool, val, heldout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2.yaml")
    args = parser.parse_args()
    try:
        split_documents(load_config(args.config))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Next: python tokenize_pack.py --config {args.config}")


if __name__ == "__main__":
    main()
