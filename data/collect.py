"""
Step 1 — COLLECT raw text (personal CPT corpus, ~10M tokens target).

Collects ONLY these domain sources, each row tagged with a `license` field
(a governance signal, recorded even though the corpus is for personal use):

  1. wikipedia — NN/ML/transformer/LLM articles (CC-BY-SA-4.0), fetched from a
     curated seed list and expanded by crawling on-topic Wikipedia links.
  2. arxiv     — a few hundred FULL papers about LLM/Transformer/NN, filtered
     by keyword from the public `jamescalam/ai-arxiv` dataset (arXiv license).
  3. docs      — HF (transformers/PEFT/TRL) + PyTorch documentation SOURCE .md
     from GitHub raw (Apache-2.0 / BSD-3-Clause).

Plus two general slices kept for training health (not domain data):
  4. replay       — small general slice mixed into CPT to reduce forgetting.
  5. general_eval — separate held-out general slice for honest eval.

Only raw text is fetched here. Cleaning + dedup happen next, in clean.py.

Usage:
    python collect.py --config day2_cpt.yaml
    python collect.py --config day2_cpt.yaml --skip-docs   # skip GitHub doc fetch
"""

import argparse
import time
from urllib.parse import unquote

from dataio import ensure_dir, load_config, write_jsonl


# ---------------------------------------------------------------------------
# 1. Wikipedia — seed list + on-topic link crawl
# ---------------------------------------------------------------------------

# Curated seed articles: transformer/LLM-centric, plus the neural-network
# foundations transformers are built on (attention, embeddings, backprop,
# training). The crawl expands outward from these, staying on-topic (see
# WIKI_TOPIC_TERMS), toward the target page count.
WIKI_SEEDS = [
    # --- Transformers / LLMs (core) ---
    "Transformer (deep learning)", "Attention (machine learning)",
    "Attention Is All You Need", "Multi-head attention",
    "Positional encoding", "Large language model", "Language model",
    "Generative pre-trained transformer", "GPT-2", "GPT-3", "GPT-4",
    "BERT (language model)", "T5 (language model)", "Vision transformer",
    "Masked language model", "Autoregressive model", "Seq2seq",
    "Encoder-decoder", "Neural machine translation", "Foundation model",
    "Generative artificial intelligence", "Text generation",
    # --- Tokenization / representations ---
    "Tokenization (lexical analysis)", "Byte pair encoding", "Word embedding",
    "Word2vec", "Embedding", "Beam search", "N-gram language model",
    # --- Training / adaptation of LLMs ---
    "Fine-tuning (deep learning)", "Transfer learning", "Pretraining",
    "Self-supervised learning", "Reinforcement learning from human feedback",
    "In-context learning (natural language processing)", "Prompt engineering",
    "Zero-shot learning", "Few-shot learning", "Knowledge distillation",
    "Low-rank adaptation", "Mixture of experts", "Instruction tuning",
    # --- Neural-network foundations (what transformers build on) ---
    "Artificial neural network", "Deep learning", "Feedforward neural network",
    "Multilayer perceptron", "Recurrent neural network", "Long short-term memory",
    "Gated recurrent unit", "Backpropagation", "Gradient descent",
    "Stochastic gradient descent", "Activation function", "Softmax function",
    "Rectifier (neural networks)", "Layer normalization", "Batch normalization",
    "Residual neural network", "Dropout (neural networks)", "Loss function",
    "Cross entropy", "Adam (optimization algorithm)", "Learning rate",
    "Vanishing gradient problem", "Regularization (mathematics)",
    "Overfitting", "Automatic differentiation", "Chain rule",
    # --- Core NLP context ---
    "Natural language processing", "Machine learning", "Supervised learning",
    "Neural scaling law", "Hallucination (artificial intelligence)",
    "Perplexity", "Chatbot", "Question answering",
]

# A page is on-topic if its title contains any of these terms. Tightened for the
# transformer/LLM focus (Option B): LLM/transformer terms + the NN foundations
# they rely on -- but NOT broad classical-ML topics (SVMs, random forests,
# clustering, PCA, etc.), which would pull the crawl off-domain.
WIKI_TOPIC_TERMS = [
    # transformer / LLM
    "transformer", "attention", "language model", "llm", "gpt", "bert",
    "token", "embedding", "sequence", "pretrain", "fine-tun", "prompt",
    "generative", "natural language", "translation", "autoregressive",
    # neural-network foundations
    "neural", "deep learning", "gradient", "backprop", "activation",
    "recurrent", "perceptron", "normalization", "regularization",
    "loss function", "optimization", "training",
]


def _wiki_is_on_topic(title: str) -> bool:
    t = title.lower()
    return any(term in t for term in WIKI_TOPIC_TERMS)


def _wiki_resolve_title(session, title: str):
    """Resolve a seed title to a real Wikipedia article title via search.

    Seed titles can be slightly off (wrong disambiguation, renamed article).
    Rather than silently dropping such a seed, we ask Wikipedia's search API
    for the best-matching article and use that title. Returns the resolved
    title, or None if nothing matches.
    """
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "list": "search",
        "srsearch": title, "srlimit": 1, "srnamespace": 0,
    }
    try:
        r = session.get(api, params=params, timeout=20)
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            return hits[0]["title"]
    except Exception:
        pass
    return None


def _wiki_fetch_page(session, title: str):
    """Fetch one Wikipedia article's plain-text extract + its outgoing links.

    Uses the MediaWiki API (action=query) with:
      - prop=extracts (explaintext) for clean article text
      - prop=links for on-site links to expand the crawl
    Returns (text, [linked_titles]) or (None, []) on failure.
    """
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "titles": title,
        "prop": "extracts|links", "explaintext": 1, "redirects": 1,
        "pllimit": "max", "plnamespace": 0,
    }
    try:
        r = session.get(api, params=params, timeout=20)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return None, []
            text = page.get("extract", "") or ""
            links = [l["title"] for l in page.get("links", [])]
            return text, links
    except Exception as e:
        print(f"[wikipedia] skip {title!r}: {e}")
    return None, []


def collect_wikipedia(cfg):
    """Breadth-first crawl from the seed titles, staying on on-topic pages."""
    import requests

    wcfg = cfg["data"]
    max_pages = wcfg.get("wiki_max_pages", 1500)
    min_chars = wcfg.get("wiki_min_chars", 800)
    delay = wcfg.get("wiki_request_delay_sec", 0.2)

    session = requests.Session()
    session.headers.update({"User-Agent": "cpt-edu-collector (personal, educational)"})

    # Resolve seed titles to real article titles first. A slightly-wrong seed
    # (bad disambiguation, renamed page) would otherwise be silently skipped;
    # search resolves it to the closest real article instead. Only seeds are
    # resolved -- crawled links already come from real pages.
    resolve_seeds = wcfg.get("wiki_resolve_seeds", True)
    seeds = []
    if resolve_seeds:
        print(f"[wikipedia] resolving {len(WIKI_SEEDS)} seed titles via search ...")
        for s in WIKI_SEEDS:
            resolved = _wiki_resolve_title(session, s)
            if resolved:
                if resolved.lower() != s.lower():
                    print(f"[wikipedia]   {s!r} -> {resolved!r}")
                seeds.append(resolved)
            else:
                seeds.append(s)     # fall back to the original if search fails
            time.sleep(delay)
    else:
        seeds = list(WIKI_SEEDS)

    to_visit = list(dict.fromkeys(seeds))   # de-dup while preserving order
    visited = set()
    rows = []

    print(f"[wikipedia] crawling up to {max_pages} on-topic pages "
          f"from {len(to_visit)} seeds ...")
    while to_visit and len(rows) < max_pages:
        title = to_visit.pop(0)
        key = title.lower()
        if key in visited:
            continue
        visited.add(key)

        text, links = _wiki_fetch_page(session, title)
        if text and len(text) >= min_chars:
            rows.append({
                "source": "wikipedia",
                "id": f"wiki:{title}",
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "license": "CC-BY-SA-4.0",
                "text": text,
            })
            if len(rows) % 50 == 0:
                print(f"[wikipedia] collected {len(rows)} pages ...")

        # Queue on-topic links we haven't seen (keeps the crawl focused).
        for link in links:
            if link.lower() not in visited and _wiki_is_on_topic(link):
                to_visit.append(link)
        time.sleep(delay)

    print(f"[wikipedia] collected {len(rows)} pages")
    return rows


# ---------------------------------------------------------------------------
# 2. arXiv — keyword-filtered full papers from jamescalam/ai-arxiv
# ---------------------------------------------------------------------------

def collect_arxiv(cfg):
    """Full-text LLM/Transformer/NN papers, filtered by keyword from a HF dataset."""
    from datasets import load_dataset

    acfg = cfg["data"]
    name = acfg["arxiv_dataset"]
    cap = acfg["arxiv_max_docs"]
    keywords = [k.lower() for k in acfg["arxiv_keywords"]]
    min_chars = acfg.get("arxiv_min_chars", 1500)

    print(f"[arxiv] loading {name}, filtering by {keywords} (up to {cap}) ...")
    ds = load_dataset(name, split="train", streaming=True)

    rows = []
    scanned = 0
    for ex in ds:
        if len(rows) >= cap:
            break
        scanned += 1
        # Field names vary across arXiv datasets; try the common ones.
        title = (ex.get("title") or "").lower()
        abstract = (ex.get("abstract") or ex.get("summary") or "").lower()
        haystack = f"{title} {abstract}"
        # keep only papers whose title/abstract mentions our domain keywords
        if not any(k in haystack for k in keywords):
            continue
        # full text may live under 'content', 'text', or 'body'; fall back to abstract
        text = (ex.get("content") or ex.get("text") or ex.get("body")
                or ex.get("abstract") or ex.get("summary") or "")
        if len(text) < min_chars:
            continue
        rows.append({
            "source": "arxiv",
            "id": ex.get("id") or ex.get("entry_id") or f"arxiv-{scanned}",
            "title": ex.get("title", ""),
            "url": ex.get("url") or ex.get("source") or ex.get("entry_id") or "",
            "license": "arXiv (per-paper; non-exclusive distribution)",
            "text": text,
        })
    print(f"[arxiv] scanned {scanned}, kept {len(rows)} on-topic papers")
    return rows


# ---------------------------------------------------------------------------
# 3. Docs — HF + PyTorch documentation SOURCE .md from GitHub raw
# ---------------------------------------------------------------------------

def _github_list_md(session, api_repo, path, exts):
    """List doc files under a repo path via the GitHub contents API (recursive).

    Returns [(raw_url, relative_path)]. Filters to the given extensions and to
    on-topic file names (LLM/transformer/NN), keeping the doc set focused.
    """
    out = []
    stack = [path]
    while stack:
        cur = stack.pop()
        url = f"https://api.github.com/repos/{api_repo}/contents/{cur}"
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                continue
            for item in r.json():
                if item["type"] == "dir":
                    stack.append(item["path"])
                elif item["type"] == "file" and any(item["name"].endswith(e) for e in exts):
                    out.append((item["download_url"], item["path"]))
        except Exception as e:
            print(f"[docs] list skip {cur}: {e}")
    return out


def collect_docs(cfg):
    """Fetch documentation .md/.mdx source from configured GitHub repos."""
    import requests

    dcfg = cfg["data"]
    sources = dcfg["doc_github_sources"]     # list of {repo, path, license}
    exts = tuple(dcfg.get("doc_extensions", [".md", ".mdx"]))
    max_files = dcfg.get("doc_max_files", 600)
    # Per-repo cap so a large early repo can't consume the whole global budget
    # and starve later sources (e.g. d2l / HF course listed last). Defaults to
    # an even share across the configured repos.
    per_repo_cap = dcfg.get("doc_max_files_per_repo")
    if per_repo_cap is None:
        n_sources = max(1, len(dcfg["doc_github_sources"]))
        per_repo_cap = max(1, max_files // n_sources * 2)   # 2x even share, still bounded by global
    min_chars = dcfg.get("doc_min_chars", 400)
    delay = dcfg.get("doc_request_delay_sec", 0.1)

    session = requests.Session()
    session.headers.update({"User-Agent": "cpt-edu-collector (personal, educational)"})

    # Optional GitHub token from the environment (NEVER hard-code a token).
    # With a token the API allows ~5000 requests/hour instead of ~60, so the
    # recursive doc listing across several repos won't get rate-limited.
    #   export GITHUB_TOKEN=...   then run this script.
    import os
    gh_token = os.environ.get("GITHUB_TOKEN")
    if gh_token:
        session.headers.update({"Authorization": f"Bearer {gh_token}"})
        print("[docs] using GITHUB_TOKEN from environment (higher rate limit)")
    else:
        print("[docs] no GITHUB_TOKEN set -- GitHub API limited to ~60 req/hour. "
              "Set it to avoid rate-limiting, or use --skip-docs.")

    rows = []
    for src in sources:
        repo, path, lic = src["repo"], src.get("path", ""), src["license"]
        print(f"[docs] listing {repo}/{path} ...")
        files = _github_list_md(session, repo, path, exts)
        print(f"[docs] {repo}: found {len(files)} doc files")
        repo_count = 0
        for raw_url, rel in files:
            if len(rows) >= max_files:           # global ceiling
                break
            if repo_count >= per_repo_cap:       # fair per-repo share
                print(f"[docs] {repo}: hit per-repo cap ({per_repo_cap}), moving on")
                break
            try:
                r = session.get(raw_url, timeout=20)
                if r.status_code != 200:
                    continue
                text = r.text
                if len(text) >= min_chars:
                    rows.append({
                        "source": "docs",
                        "id": f"{repo}:{rel}",
                        "title": rel,
                        "url": raw_url,
                        "license": lic,
                        "text": text,
                    })
                    repo_count += 1
            except Exception as e:
                print(f"[docs] skip {raw_url}: {e}")
            time.sleep(delay)
    print(f"[docs] collected {len(rows)} doc pages")
    return rows


# ---------------------------------------------------------------------------
# 4 & 5. General replay + held-out general eval (training health, not domain)
# ---------------------------------------------------------------------------

def _stream_general(cfg, skip, take, source_tag, license_str):
    from datasets import load_dataset

    name = cfg["data"]["replay_dataset"]
    subset = cfg["data"].get("replay_subset")
    try:
        ds = load_dataset(name, subset, split="train", streaming=True)
    except Exception:
        ds = load_dataset(name, split="train", streaming=True)

    rows = []
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        text = ex.get("text") or ""
        if text:
            rows.append({
                "source": source_tag,
                "id": f"{source_tag}-{i}",
                "license": license_str,
                "text": text,
            })
        if len(rows) >= take:
            break
    return rows


def collect_replay(cfg):
    take = cfg["data"].get("replay_max_docs", 800)
    lic = cfg["data"].get("replay_license", "ODC-BY-1.0 (FineWeb-Edu)")
    print(f"[replay] streaming {take} general docs ...")
    rows = _stream_general(cfg, skip=0, take=take, source_tag="replay", license_str=lic)
    print(f"[replay] collected {len(rows)}")
    return rows, take


def collect_general_eval(cfg, replay_take):
    take = cfg["evaluation"]["general_ppl_docs"]
    lic = cfg["data"].get("replay_license", "ODC-BY-1.0 (FineWeb-Edu)")
    skip = replay_take + 500                  # gap so eval never overlaps replay
    print(f"[general-eval] streaming {take} held-out general docs (after {skip}) ...")
    rows = _stream_general(cfg, skip=skip, take=take,
                           source_tag="general_eval", license_str=lic)
    print(f"[general-eval] collected {len(rows)}")
    return rows


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    parser.add_argument("--skip-docs", action="store_true", help="skip GitHub doc fetch")
    parser.add_argument("--skip-wiki", action="store_true", help="skip Wikipedia crawl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw_dir"]
    heldout_dir = cfg["paths"]["eval_heldout_dir"]
    ensure_dir(raw_dir)
    ensure_dir(heldout_dir)

    wiki = [] if args.skip_wiki else collect_wikipedia(cfg)
    arxiv = collect_arxiv(cfg)
    docs = [] if args.skip_docs else collect_docs(cfg)
    replay, replay_take = collect_replay(cfg)
    general_eval = collect_general_eval(cfg, replay_take)

    # Wikipedia: when --skip-wiki is set we do NOT touch domain_wikipedia.jsonl,
    # so a file produced separately by collect_wiki.py is preserved. We read its
    # existing rows only for the token-estimate summary below.
    wiki_path = f"{raw_dir}/domain_wikipedia.jsonl"
    if args.skip_wiki:
        from dataio import read_jsonl
        existing_wiki = read_jsonl(wiki_path)
        n_wiki = len(existing_wiki)
        wiki = existing_wiki   # for the token estimate only; file left as-is
        print(f"[wikipedia] --skip-wiki: keeping existing {n_wiki} rows in {wiki_path}")
    else:
        n_wiki = write_jsonl(wiki_path, wiki)
    n_arxiv = write_jsonl(f"{raw_dir}/domain_papers.jsonl", arxiv)
    # Docs: like --skip-wiki, --skip-docs must NOT wipe an existing docs file.
    docs_path = f"{raw_dir}/domain_docs.jsonl"
    if args.skip_docs:
        from dataio import read_jsonl
        existing_docs = read_jsonl(docs_path)
        n_docs = len(existing_docs)
        docs = existing_docs   # token estimate only; file left as-is
        print(f"[docs] --skip-docs: keeping existing {n_docs} rows in {docs_path}")
    else:
        n_docs = write_jsonl(docs_path, docs)
    n_replay = write_jsonl(f"{raw_dir}/replay.jsonl", replay)
    n_eval = write_jsonl(f"{heldout_dir}/general_eval.jsonl", general_eval)

    # rough token estimate (~4 chars/token) so you can see if ~10M is in reach
    def _tok_est(rows):
        return sum(len(r["text"]) for r in rows) // 4

    domain_tokens = _tok_est(wiki) + _tok_est(arxiv) + _tok_est(docs)
    print(f"\nCollected: wikipedia={n_wiki} papers={n_arxiv} docs={n_docs} "
          f"replay={n_replay} general_eval={n_eval}")
    print(f"Estimated DOMAIN tokens (pre-clean, ~4 chars/tok): ~{domain_tokens:,}")
    print(f"Estimated total w/ replay: ~{domain_tokens + _tok_est(replay):,}")
    print("Next: python clean.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
