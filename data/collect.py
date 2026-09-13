"""Day 2 · Step 1 — COLLECT raw text.

Pulls three sources into data/raw/ as JSONL ({source, url/id, text}):
  1. domain papers   — jamescalam/ai-arxiv  (LLM/ML papers, full text)
  2. domain docs     — HF/PyTorch docs, scraped directly from the websites
  3. general replay  — a small streamed slice of HuggingFaceFW/fineweb-edu

This only fetches and stores raw text. Cleaning/dedup happens in clean.py.

Usage:
    python data/collect.py --config configs/day2.yaml
"""
from __future__ import annotations

import argparse
import time

from dataio import ensure_dir, load_config, update_manifest, write_jsonl


# ---------------------------------------------------------------- papers
def collect_arxiv(cfg) -> list[dict]:
    from datasets import load_dataset

    name = cfg["data"]["arxiv_dataset"]
    cap = cfg["data"]["arxiv_max_docs"]
    print(f"[papers] loading {name} (cap {cap}) ...")
    ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= cap:
            break
        # ai-arxiv stores full text in a 'content' field (fall back to common keys)
        text = ex.get("content") or ex.get("text") or ex.get("abstract") or ""
        if not text:
            continue
        rows.append({"source": "arxiv", "id": ex.get("id", f"arxiv-{i}"), "text": text})
    print(f"[papers] collected {len(rows)} documents")
    return rows


# ---------------------------------------------------------------- docs (scrape)
def _scrape_site(seed_urls, max_pages, delay, source_tag):
    """Generic same-domain BFS scraper → list of {source, url, text}."""
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse

    allowed = {urlparse(u).netloc for u in seed_urls}
    seen: set[str] = set()
    queue = list(seed_urls)
    rows: list[dict] = []
    headers = {"User-Agent": "llm-from-base-to-assistant/edu (polite scraper)"}

    print(f"[{source_tag}] scraping up to {max_pages} pages from {sorted(allowed)} ...")
    while queue and len(rows) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            if len(text) > 200:
                rows.append({"source": source_tag, "url": url, "text": text})
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0]
                if urlparse(link).netloc in allowed and link not in seen:
                    queue.append(link)
        except Exception as e:  # noqa: BLE001
            print(f"[{source_tag}] skip {url}: {e}")
        time.sleep(delay)
    print(f"[{source_tag}] collected {len(rows)} pages")
    return rows


def collect_docs(cfg) -> list[dict]:
    return _scrape_site(
        cfg["data"]["doc_seed_urls"], cfg["data"]["doc_max_pages"],
        cfg["data"]["doc_request_delay_sec"], "docs")


# ---------------------------------------------------------------- theory (scrape)
def collect_theory_web(cfg) -> list[dict]:
    """HF blog + HF LLM Course — blog-style/beginner explanatory prose."""
    return _scrape_site(
        cfg["data"]["theory_seed_urls"], cfg["data"]["theory_max_pages"],
        cfg["data"]["doc_request_delay_sec"], "theory")


# ---------------------------------------------------------------- d2l (github md)
def collect_d2l(cfg) -> list[dict]:
    """Dive into Deep Learning — pulled as clean Markdown from the GitHub source.
    Each chapter folder has an index.md listing section .md files; we fetch those."""
    import requests

    base = cfg["data"]["d2l_github_raw"]
    chapters = cfg["data"]["d2l_chapters"]
    rows = []
    headers = {"User-Agent": "llm-from-base-to-assistant/edu"}
    print(f"[d2l] fetching {len(chapters)} chapters as Markdown from GitHub ...")
    # GitHub API to list files in each chapter folder
    api = "https://api.github.com/repos/d2l-ai/d2l-en/contents"
    for ch in chapters:
        try:
            listing = requests.get(f"{api}/{ch}?ref=master", headers=headers, timeout=20)
            if listing.status_code != 200:
                print(f"[d2l] skip {ch}: HTTP {listing.status_code}")
                continue
            for item in listing.json():
                name = item.get("name", "")
                if name.endswith(".md"):
                    raw = requests.get(f"{base}/{ch}/{name}", headers=headers, timeout=20)
                    if raw.status_code == 200 and len(raw.text) > 300:
                        rows.append({"source": "d2l", "url": f"{ch}/{name}", "text": raw.text})
            time.sleep(cfg["data"]["doc_request_delay_sec"])
        except Exception as e:  # noqa: BLE001
            print(f"[d2l] skip {ch}: {e}")
    print(f"[d2l] collected {len(rows)} sections")
    return rows


# ---------------------------------------------------------------- arxiv surveys
def collect_arxiv_surveys(cfg) -> list[dict]:
    """arXiv SURVEY papers via the arXiv API (readable overviews). Abstracts +
    titles; full-text PDFs would need extra parsing, so we use the rich abstracts
    which are already good explanatory prose for surveys."""
    import requests
    import xml.etree.ElementTree as ET

    query = cfg["data"]["arxiv_survey_query"].replace(" ", "+")
    n = cfg["data"]["arxiv_survey_max"]
    url = (f"http://export.arxiv.org/api/query?search_query=all:{query}"
           f"&start=0&max_results={n}&sortBy=relevance")
    print(f"[surveys] querying arXiv API for '{cfg['data']['arxiv_survey_query']}' ...")
    rows = []
    try:
        r = requests.get(url, timeout=30)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns).text.strip()
            summary = entry.find("a:summary", ns).text.strip()
            aid = entry.find("a:id", ns).text.strip()
            # keep survey-ish papers
            if "survey" in title.lower() or "overview" in title.lower() or "review" in title.lower():
                rows.append({"source": "survey", "url": aid,
                             "text": f"{title}\n\n{summary}"})
    except Exception as e:  # noqa: BLE001
        print(f"[surveys] error: {e}")
    print(f"[surveys] collected {len(rows)} survey abstracts")
    return rows


# ---------------------------------------------------------------- replay
def collect_replay(cfg) -> list[dict]:
    from datasets import load_dataset

    name = cfg["data"]["replay_dataset"]
    subset = cfg["data"].get("replay_subset")
    # Pull enough general docs to satisfy the replay ratio later; a few hundred is plenty
    cap = max(200, cfg["data"]["arxiv_max_docs"] // 2)
    print(f"[replay] streaming {name}:{subset} (cap {cap}) ...")
    try:
        ds = load_dataset(name, subset, split="train", streaming=True)
    except Exception:
        ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= cap:
            break
        text = ex.get("text") or ""
        if text:
            rows.append({"source": "replay", "id": f"fineweb-{i}", "text": text})
    print(f"[replay] collected {len(rows)} documents")
    return rows


def collect_general_eval(cfg) -> list[dict]:
    """A SEPARATE general held-out set for honest forgetting evaluation.
    Streamed from the SAME general corpus but from a DIFFERENT slice than the
    training replay — so it was never trained on. This is what makes the
    'did it forget general skills?' comparison trustworthy."""
    from datasets import load_dataset

    name = cfg["data"]["replay_dataset"]
    subset = cfg["data"].get("replay_subset")
    train_cap = max(200, cfg["data"]["arxiv_max_docs"] // 2)   # what replay used
    eval_cap = cfg["evaluation"]["general_ppl_docs"]
    print(f"[general-eval] streaming a held-out slice of {name} (skip first {train_cap}) ...")
    try:
        ds = load_dataset(name, subset, split="train", streaming=True)
    except Exception:
        ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i < train_cap + 500:          # skip past the training-replay slice + a gap
            continue
        text = ex.get("text") or ""
        if text:
            rows.append({"source": "general_eval", "id": f"fineweb-eval-{i}", "text": text})
        if len(rows) >= eval_cap:
            break
    print(f"[general-eval] collected {len(rows)} held-out general docs")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/day2.yaml")
    ap.add_argument("--skip-docs", action="store_true", help="skip scraping (papers+replay only)")
    ap.add_argument("--skip-theory", action="store_true", help="skip theory sources (blog/course/d2l/surveys)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["raw_dir"])
    ensure_dir(cfg["paths"]["eval_heldout_dir"])

    papers = collect_arxiv(cfg)
    docs = [] if args.skip_docs else collect_docs(cfg)
    replay = collect_replay(cfg)
    general_eval = collect_general_eval(cfg)   # SEPARATE held-out general set

    # theory sources (beginner -> advanced explanatory prose)
    theory_web, d2l, surveys = [], [], []
    if not args.skip_theory:
        theory_web = collect_theory_web(cfg)   # HF blog + HF LLM Course
        d2l = collect_d2l(cfg)                  # Dive into Deep Learning (markdown)
        surveys = collect_arxiv_surveys(cfg)    # arXiv survey abstracts

    rd = cfg["paths"]["raw_dir"]
    n1 = write_jsonl(f"{rd}/domain_papers.jsonl", papers)
    n2 = write_jsonl(f"{rd}/domain_docs.jsonl", docs)
    n3 = write_jsonl(f"{rd}/replay.jsonl", replay)
    n4 = write_jsonl(f"{rd}/theory_web.jsonl", theory_web)
    n5 = write_jsonl(f"{rd}/theory_d2l.jsonl", d2l)
    n6 = write_jsonl(f"{rd}/theory_surveys.jsonl", surveys)
    n7 = write_jsonl(f"{cfg['paths']['eval_heldout_dir']}/general_eval.jsonl", general_eval)

    update_manifest(cfg["paths"]["manifest"], "collect", {
        "arxiv_docs": n1, "doc_pages": n2, "replay_docs": n3,
        "theory_web_pages": n4, "d2l_sections": n5, "survey_abstracts": n6,
        "general_eval_docs": n7,
        "arxiv_dataset": cfg["data"]["arxiv_dataset"],
        "replay_dataset": cfg["data"]["replay_dataset"],
        "doc_seed_urls": cfg["data"]["doc_seed_urls"],
        "theory_seed_urls": cfg["data"]["theory_seed_urls"],
        "d2l_chapters": cfg["data"]["d2l_chapters"],
    })
    print(f"\nRaw collected → papers={n1}, docs={n2}, replay={n3}, "
          f"theory_web={n4}, d2l={n5}, surveys={n6}, general_eval={n7}")
    print("Next: python data/clean.py --config configs/day2.yaml")


if __name__ == "__main__":
    main()
