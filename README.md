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
# 1. Create and pin the environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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
- [ ] Base model observed continuing text instead of answering (saved as before-evidence)
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

### Day 2 result (actual run)

> **Continued pretraining reduced domain perplexity by 5.0% (8.40 → 7.98) while
> general perplexity stayed flat (10.50 → 10.28, −2.1%)** — successful domain
> specialization without catastrophic forgetting.

| model | domain ppl | general ppl |
|---|---:|---:|
| BASE | 8.40 | 10.50 |
| CPT  | 7.98 | 10.28 |
| change | **−5.0%** | −2.1% |

Run cost: **~11 minutes** of full fine-tuning, **~21 GB peak VRAM** on the A40
(comfortably within 44 GB). The trained model is preserved on the Hugging Face Hub:
[`vinmlops/cpt-v1`](https://huggingface.co/vinmlops/cpt-v1).

### Checkpoint lineage (`artifacts/cpt-v1/lineage.json`)

Every checkpoint records exactly how it was made — parent model, data, hyperparameters,
runtime, and seed — for reproducibility:

```json
{
  "stage": "cpt-v1",
  "parent_model": "Qwen/Qwen3-1.7B-Base",
  "parent_revision": "main",
  "dataset_manifest": "data/manifest.json",
  "block_size": 1024,
  "token_budget": 3000000,
  "learning_rate": 2e-05,
  "effective_batch": 32,
  "runtime_sec": 713.6,
  "peak_vram_gb": 21.26,
  "seed": 42
}
```

> Want a bigger domain effect? Raise `token_budget` in `configs/day2.yaml` (e.g. to
> 10–30M). On the A40 you're time-limited, not memory-limited, so it mostly costs a
> longer session.

## Roadmap

| Day | Focus | Added to repo |
|----|----|----|
| 1 ✅ | Setup + foundations | env, configs, exploration scripts, eval lock |
| 2 ✅ | Data + continued pretraining | `data/` pipeline, `training/cpt.py`, `evaluation/perplexity.py` |
| 3 | SFT | instruction data, `training/sft.py`, masking test |
| 4 | DPO | preference data, `training/dpo.py` |
| 5 | Honest evaluation | `evaluation/` suite, judge, failure analysis |
| 6 | Serving + portfolio | `serving/` vLLM + Docker, cards, diagram |

## License

Code: MIT (see `LICENSE`). The model and datasets carry their own licenses — verify `Qwen/Qwen3-1.7B-Base` (Apache-2.0), `jamescalam/ai-arxiv`, `HuggingFaceFW/fineweb-edu`, d2l.ai, and any scraped docs on their respective pages before use.
