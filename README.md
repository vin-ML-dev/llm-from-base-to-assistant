# llm-from-base-to-assistant

Turning an open **base** language model into a helpful domain **chat assistant**, end to end:

```
BASE  →  CPT  →  SFT  →  DPO  →  honestly evaluated  →  served
```

- **Model:** `Qwen/Qwen3-1.7B-Base` (development mule: `Qwen/Qwen3-0.6B-Base`)
- **Domain:** an **LLM/ML tutor** — the assistant learns to explain LLM/ML concepts from basics to advanced.
- **Data:** LLM/ML papers (`jamescalam/ai-arxiv`) + HF/PyTorch docs + theory sources (HF blog, HF LLM Course, [d2l.ai](https://d2l.ai), arXiv surveys), with a [`FineWeb-Edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) slice for general replay.
- **Stack:** PyTorch · Hugging Face Transformers · `datasets` · PEFT · TRL · vLLM

This repo is built day by day alongside a theory guide. **Days 1–2 are complete:**
Day 1 (setup + foundations) gets a pinned environment, confirms the base model runs,
explores the concepts, and locks the eval set; Day 2 (data + continued pretraining)
builds a versioned LLM/ML corpus and produces `cpt-v1`, a domain-adapted base model.

> The point of the project is not "I fine-tuned Qwen." It is being able to **explain, measure honestly, reproduce, and serve** every stage.

---

## Repository layout

```
llm-from-base-to-assistant/
├── configs/        # config-driven experiments (model, data, training)
├── data/           # dataset building + the LOCKED held-out eval set
│   └── eval_heldout/   # never used for CPT / SFT-gen / DPO prompts
├── training/       # CPT / SFT / DPO training entrypoints (added Days 2–4)
├── evaluation/     # eval suite + judge (added Day 5)
├── serving/        # vLLM + Docker (added Day 6)
├── scripts/        # Day 1 exploration + smoke-test scripts
├── tests/          # unit tests (masking test added Day 3; smoke test Day 1)
└── docs/           # decisions log, lineage, reproducibility, results
```

---

## Day 1 — quick start

```bash
# 1. Create the environment (Qwen3 needs transformers >= 4.51, < 5.0)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# On managed images (RunPod) use the Day 2 notebook's install cell instead — it
# reconciles preinstalled packages so the notebook and CLI share one Python env.

# 2. Confirm the base model loads and runs; record measured VRAM + tokens/sec
python scripts/smoke_test.py --config configs/day1.yaml

# 3. Explore the Day 1 concepts hands-on (safe, read-only)
python scripts/explore_tokenizer.py   --config configs/day1.yaml
python scripts/inspect_model.py       --config configs/day1.yaml
python scripts/forward_pass_probs.py  --config configs/day1.yaml --prompt "The capital of France is"
python scripts/base_vs_chat.py        --config configs/day1.yaml
python scripts/generation_settings.py --config configs/day1.yaml --prompt "Explain what a tokenizer does."

# 4. Save your "before" evidence (base model failing to chat)
python scripts/base_vs_chat.py --config configs/day1.yaml --save docs/before_evidence.md

# 5. Lock the held-out eval location (creates the directory + a README guard)
python scripts/lock_eval_set.py
```

Then fill in `docs/decisions.md` with your model + hardware decisions.

### Day 1 checklist (maps to the theory guide)

- [ ] Pinned environment installed
- [ ] `smoke_test.py` prints measured peak VRAM + tokens/sec
- [ ] Tokenizer explored (tokens, IDs, special tokens, two-tokenizer comparison)
- [ ] `print(model)` reviewed; params counted per component
- [ ] Top-10 next-token probabilities inspected
- [ ] Base model behavior observed on the frozen prompts — it tends to *continue* text rather than answer as an assistant (save the actual outputs as before-evidence)
- [ ] Generation settings (temp 0 / 0.7 / 1.2) compared
- [ ] Held-out eval location locked
- [ ] `docs/decisions.md` filled in
- [ ] `pytest tests/test_environment.py` passes

---

## Day 2 — quick start

Full fine-tuning CPT on an A40 (~44 GB). Sources: `jamescalam/ai-arxiv` papers +
scraped HF/PyTorch docs + theory sources (HF blog, HF LLM Course, d2l.ai, arXiv
surveys) + a FineWeb-Edu replay slice. All settings in `configs/day2.yaml`.

```bash
python data/collect.py        --config configs/day2.yaml   # papers + docs + theory + replay
python data/clean.py          --config configs/day2.yaml   # normalize, dedup
python data/split.py          --config configs/day2.yaml   # doc-level split + leakage check
python data/tokenize_pack.py  --config configs/day2.yaml   # 1024-block packing, 85/15 mix
python training/cpt.py        --config configs/day2.yaml   # full fine-tune -> artifacts/cpt-v1
python evaluation/perplexity.py --config configs/day2.yaml # BASE vs CPT, domain vs general
```

Or run `notebooks/day2_runner.ipynb` top-to-bottom on the rented GPU.

> **RunPod-safe install** (baked into the notebook): `pip install -r requirements.txt
> --ignore-installed blinker`, force `transformers>=4.51,<5.0`, and uninstall the
> unused `torchaudio`. All script calls use `{sys.executable}` so they run under the
> notebook kernel's Python.

---

## Day 2 — results

1. **Perplexity (lower is better):** BASE → CPT: domain `7.54 → 6.76`, general `10.62 → 11.13`.
2. **Domain:** `-10.3%` perplexity — evidence of domain specialization.
3. **General:** `+4.8%` perplexity — a measurable forgetting signal.
4. **Qualitative check:** on the same greedy-decoded prompts, CPT gives more domain-relevant explanations for attention and tokenization.
5. **Instruction following:** prompts such as “Give me three tips for fine-tuning an LLM” still show continuation/repetition instead of reliable instruction following.
6. **Expected behavior:** BASE and CPT are completion models, not chat-tuned assistants; repetition and weak stopping behavior at this stage are expected.
7. **Day 2 conclusion:** `cpt-v1` improved domain modeling while preserving a clear reason for Day 3 SFT — instruction following, response structure, and stopping behavior.

---

## License

Code: MIT (see `LICENSE`). The model and datasets carry their own licenses — verify `Qwen/Qwen3-1.7B-Base` (Apache-2.0), `jamescalam/ai-arxiv`, `HuggingFaceFW/fineweb-edu`, d2l.ai, and any scraped docs on their respective pages before use.

### Data usage disclaimer

This is a **non-commercial, educational / research project.** The training data
(papers, documentation, textbook material, and web text) is collected and used
**solely for learning and demonstration** — to study the fine-tuning pipeline — and
**not for any commercial purpose**. Each source retains its own license and terms;
this project does not redistribute the raw data and makes no claim of ownership over
it. Anyone reusing this repository is responsible for verifying and complying with
the license of each individual data source before using it, especially for any
commercial use. **Note in particular that individual arXiv papers carry their own
licenses** (many are not open-license), so a blanket "educational use" statement
does not by itself grant reuse rights for every paper. If you are a rights holder
and have concerns about a source, please open an issue.
