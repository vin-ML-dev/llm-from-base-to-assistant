"""
Step 1 — COLLECT raw text.

Downloads/scrapes text into data/raw/ as JSONL (one {"source", "text", ...} per line):
  1. papers  — LLM/ML papers from a HuggingFace dataset (domain)
  2. docs     — HuggingFace / PyTorch documentation pages (domain, scraped)
  3. replay   — a small slice of a general web corpus (anti-forgetting)
  4. general_eval — a SEPARATE general slice, never trained on (for honest eval)

This only fetches raw text. Cleaning and dedup happen next, in clean.py.

Usage:
    python collect.py --config day2_cpt.yaml
"""

import argparse
import time

from dataio import ensure_dir, load_config, write_jsonl


def collect_papers(cfg):
    """LLM/ML papers with full text, from a HuggingFace dataset."""
    from datasets import load_dataset

    name = cfg["data"]["arxiv_dataset"]
    cap = cfg["data"]["arxiv_max_docs"]
    print(f"[papers] loading {name} (up to {cap}) ...")
    ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= cap:
            break
        text = ex.get("content") or ex.get("text") or ex.get("abstract") or ""
        if text:
            rows.append({"source": "papers", "id": ex.get("id", f"paper-{i}"), "text": text})
    print(f"[papers] collected {len(rows)}")
    return rows


def scrape_docs(cfg):
    """
    Scrape documentation pages, following links within the same websites only.
    A simple breadth-first crawl starting from the seed URLs.
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse

    seed_urls = cfg["data"]["doc_seed_urls"]
    max_pages = cfg["data"]["doc_max_pages"]
    delay = cfg["data"]["doc_request_delay_sec"]

    allowed_sites = {urlparse(u).netloc for u in seed_urls}
    to_visit = list(seed_urls)
    visited = set()
    rows = []
    headers = {"User-Agent": "cpt-edu-scraper (polite)"}

    print(f"[docs] scraping up to {max_pages} pages from {sorted(allowed_sites)} ...")
    while to_visit and len(rows) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # remove non-content page furniture before extracting text
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            if len(text) > 200:
                rows.append({"source": "docs", "url": url, "text": text})
            # queue same-site links we haven't seen
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0]
                if urlparse(link).netloc in allowed_sites and link not in visited:
                    to_visit.append(link)
        except Exception as e:
            print(f"[docs] skip {url}: {e}")
        time.sleep(delay)
    print(f"[docs] collected {len(rows)}")
    return rows


def _stream_general(cfg, skip, take, source_tag):
    """Stream `take` docs from the general corpus, skipping the first `skip`."""
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
            rows.append({"source": source_tag, "id": f"{source_tag}-{i}", "text": text})
        if len(rows) >= take:
            break
    return rows


def collect_replay(cfg):
    """General text mixed into training so the model doesn't forget general skills."""
    #take = max(200, cfg["data"]["arxiv_max_docs"] // 2)
    take = cfg["data"].get("replay_max_docs", 800)
    print(f"[replay] streaming {take} general docs ...")
    rows = _stream_general(cfg, skip=0, take=take, source_tag="replay")
    print(f"[replay] collected {len(rows)}")
    return rows, take


def collect_general_eval(cfg, replay_take):
    """
    A separate general set for measuring forgetting. It comes from the same
    corpus but a LATER slice than replay (plus a gap), so it is never trained on.
    """
    take = cfg["evaluation"]["general_ppl_docs"]
    #take = cfg["eval"]["general_docs"]
    skip = replay_take + 500  # skip past the training-replay slice, leave a gap
    print(f"[general-eval] streaming {take} held-out general docs (after first {skip}) ...")
    rows = _stream_general(cfg, skip=skip, take=take, source_tag="general_eval")
    print(f"[general-eval] collected {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="day2_cpt.yaml")
    parser.add_argument("--skip-docs", action="store_true", help="skip web scraping")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw_dir"]
    heldout_dir = cfg["paths"]["eval_heldout_dir"]
    ensure_dir(raw_dir)
    ensure_dir(heldout_dir)

    papers = collect_papers(cfg)
    docs = [] if args.skip_docs else scrape_docs(cfg)
    replay, replay_take = collect_replay(cfg)
    general_eval = collect_general_eval(cfg, replay_take)

    n_papers = write_jsonl(f"{raw_dir}/domain_papers.jsonl", papers)
    n_docs = write_jsonl(f"{raw_dir}/domain_docs.jsonl", docs)
    n_replay = write_jsonl(f"{raw_dir}/replay.jsonl", replay)
    n_eval = write_jsonl(f"{heldout_dir}/general_eval.jsonl", general_eval)

    print(f"\nCollected: papers={n_papers} docs={n_docs} replay={n_replay} general_eval={n_eval}")
    print("Next: python clean.py --config day2_cpt.yaml")


if __name__ == "__main__":
    main()
