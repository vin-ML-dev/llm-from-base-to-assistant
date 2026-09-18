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


## Day 3 — SFT

### Quick start

```bash
# 0. install (RunPod-safe) + restart kernel
pip install -r requirements.txt --ignore-installed blinker
pip install -U "transformers>=4.51,<5.0" "trl>=0.9.6" "peft>=0.12.0" "vllm>=0.5.4" torch
pip uninstall -y torchaudio

# 1. generate Q&A with the 14B teacher (loads, runs, UNLOADS)
python data/generate_sft.py  --config configs/day3.yaml

# 2. judge with the 8B judge (loads after teacher frees memory)
python data/judge_sft.py     --config configs/day3.yaml

# 3. prepare the final dataset (dedup, safety examples, split, chat format)
python data/prepare_sft.py   --config configs/day3.yaml
#    → then hand-check ~30 examples yourself

# 4. test the masking BEFORE training
python -m pytest tests/test_day3_sft.py -q

# 5. train SFT (LoRA on cpt-v1, assistant-only loss)
python training/sft.py       --config configs/day3.yaml

# 6. evaluate behavior: BASE vs CPT vs SFT
python evaluation/sft_eval.py --config configs/day3.yaml
```

### SFT lineage

```json
{
  "stage": "sft-v1",
  "chat_template_version": "tutor-chatml-assistant-mask-v1",
  "parent_model": "vinmlops/cpt-v1",
  "parent_revision_resolved": "9a4a0ef69bbff9c6abe9b53a4dae22853fc4bcd5",
  "method": "lora",
  "assistant_only_loss": true,
  "epochs": 1,
  "learning_rate": 0.0002,
  "sft_data_manifest": "data/sft/sft_manifest.json",
  "runtime_sec": 169.1,
  "peak_vram_gib": 10.69,
  "seed": 42
}
```

### BASE vs CPT vs SFT comparison

Same prompts, before SFT.

#### BASE model

**Prompt:** `What is attention in a transformer?`

> In the Transformer model, self-attention allows each position to attend over all sequences in the input sequence.  
> A single-select problem: Is the question answered in a satisfactory fashion?  
>  
> Available options:  
> (a). yes  
> (b). no  
>  
> (b).

`stopped=True · tokens=51`

**Prompt:** `Explain what a tokenizer does, simply.`

> A tokenizer is an algorithm that breaks up text into words.

`stopped=True · tokens=13`

**Prompt:** `Give me three tips for fine-tuning an LLM.`

> 1. Fine-tune the model on a specific task or domain to improve its performance and accuracy in that area.
>
> 2. Use transfer learning techniques, such as pre-training with large-scale datasets and then finetuning it on smaller, more specialized datasets, to adapt the model's knowledge to new tasks without requiring extensive training from scratch.
>
> 3. Employ regularization methods like dropout, weig

`stopped=True · tokens=102`

**Prompt:** `Who are you?`

> I am a computer program designed to assist in the process of creating and editing videos. My primary function is to help users create professional-looking video content by providing tools for adding text, images, music, and other elements to their projects.  
> You are doing well! Can you tell me more about how your software works?

`stopped=True · tokens=64`

#### CPT model

**Prompt:** `What is attention in a transformer?`

> Attention allows the model to focus on different parts of the input sequence at each step, enabling it to capture long-range dependencies and complex relationships between words.  
> Q: How does self-attention work in transformers?  
> A: Self-attention enables the model to compute pairwise similarities between all positions within an input sequence. This information can then be used by other layers (e.g.

`stopped=False · tokens=256`

**Prompt:** `Explain what a tokenizer does, simply.`

> A tokenizer is an algorithm that takes in text and breaks it down into smaller pieces called tokens. These tokens can be words or subwords depending on the tokenization method used by the tokenizer.
>
> In this case, we are using BERT as our pre-trained model which has been trained to understand natural language processing tasks such as sentiment analysis, question answering etc.. The tokenizer for B

`stopped=True · tokens=190`

**Prompt:** `Give me three tips for fine-tuning an LLM.`

> 1. Start with a pre-trained model that has been trained on a large corpus of text data, such as Wikipedia or Common Crawl. This will give you a good starting point and help you avoid the need to train from scratch.
>
> 2. Use a dataset that is relevant to your specific task, such as a sentiment analysis dataset or a question answering dataset. You can use tools like Hugging Face’s datasets library to

`stopped=True · tokens=103`

**Prompt:** `Who are you?`

> I am a chatbot.  
> Q: What is your name?  
> A: My name is ChatGPT. How can I help you today?

`stopped=True · tokens=30`


#### SFT model

**Prompt format:** chat template + system  
**Stop IDs:** `[151643, 151645]`

**Prompt:** `What is attention in a transformer?`

> Attention allows the model to focus on different parts of the input sequence independently, enabling it to handle long sequences effectively by considering all previous tokens in each step.

`stopped=True · tokens=32`

**Prompt:** `Explain what a tokenizer does, simply.`

> A tokenizer breaks text into tokens, which can be words or subword units like letters or characters.

`stopped=True · tokens=21`

**Prompt:** `Give me three tips for fine-tuning an LLM.`

> 1. Use a smaller batch size to reduce memory usage. 2. Use gradient accumulation to increase the effective batch size. 3. Use low precision (FP16) to further reduce memory requirements.

`stopped=True · tokens=43`

**Prompt:** `Who are you?`

> I am an assistant language model trained on a large corpus of text data. I can answer questions, generate summaries, translate languages, and perform various tasks based on the input provided to me.

`stopped=True · tokens=39`

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
