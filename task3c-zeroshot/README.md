# Task 3(c) — two scripts, two different purposes

The brief for 3(c) asks for **at least two open-source decoder-only LLMs**,
evaluated with **zero-shot and few-shot prompting**, with the prompts
documented and justified.

| Script | What it does | Satisfies 3(c)? |
|---|---|---|
| `classify_decoder_llm.py` | Prompts Qwen2.5-1.5B-Instruct and Phi-3-mini-4k-instruct (decoder-only LLMs) with zero-shot and few-shot prompts | **Yes — this is the required deliverable.** See `PROMPT_DESIGN.md`. |
| `classify_zeroshot.py` | Runs BART-large-mnli and DeBERTa-v3-zeroshot through HuggingFace's NLI-based `zero-shot-classification` pipeline | Supplementary baseline only. Neither model is decoder-only, and there's no prompting or few-shot variant — kept as an extra comparison point in Task 3(d), not as the answer to 3(c). |

## Running order

1. `classify_zeroshot.py` (already run — results in `results/bart-large-mnli/`, `results/deberta-v3-zeroshot/`)
2. `classify_decoder_llm.py` (run this on Colab with a GPU runtime; needs `transformers` with chat-template support)

Both scripts write into the same `results/` and `plots/` directories in
this folder, namespaced by model (and strategy, for the decoder-LLM
script) so nothing overwrites the other.
