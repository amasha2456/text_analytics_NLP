# Task 3(d) — Model Comparison Discussion

## Summary table

| Model | Category | Accuracy | F1-Weighted | F1-Macro | Precision-Macro | Recall-Macro |
|---|---|---|---|---|---|---|
| BERT (fine-tuned) | Fine-tuned Transformer | 0.88 | 0.88 | 0.87 | 0.87 | 0.87 |
| RoBERTa (fine-tuned) | Fine-tuned Transformer | 0.87 | 0.87 | 0.86 | 0.86 | 0.86 |
| DeBERTa-v3 (NLI zero-shot) | Zero-Shot NLI Classifier | 0.36 | 0.33 | 0.38 | 0.41 | 0.45 |
| Phi-3-mini (few-shot prompt) | Decoder LLM (Few-Shot) | 0.35 | 0.34 | 0.36 | 0.39 | 0.41 |
| Phi-3-mini (zero-shot prompt) | Decoder LLM (Zero-Shot) | 0.34 | 0.33 | 0.35 | 0.39 | 0.40 |
| Qwen2.5-1.5B (few-shot prompt) | Decoder LLM (Few-Shot) | 0.27 | 0.26 | 0.27 | 0.43 | 0.32 |
| BART-MNLI (NLI zero-shot) | Zero-Shot NLI Classifier | 0.23 | 0.20 | 0.23 | 0.43 | 0.25 |
| Qwen2.5-1.5B (zero-shot prompt) | Decoder LLM (Zero-Shot) | 0.17 | 0.17 | 0.19 | 0.49 | 0.22 |

*(source: `results/summary_comparison.csv`, `results/comparison_report.txt`. Note: the two NLI classifiers and BERT/RoBERTa were evaluated on the full 9,446-post test set; the four decoder-LLM prompting runs were evaluated on a 2,000-post stratified subsample of that same test set, per the compute-cost tradeoff documented in `../task3c-zeroshot/PROMPT_DESIGN.md`.)*

## Fine-tuned Transformers (BERT, RoBERTa)

**Strengths**
- Both models cleared 0.86+ on every aggregate metric (BERT: 0.88 accuracy / 0.87 F1-macro; RoBERTa: 0.87 / 0.86) — a ~2.4x jump over the best of the six zero/few-shot approaches (DeBERTa-v3 NLI at 0.36 accuracy).
- Consistent across classes: per-class F1 stayed in a tight 0.80–0.94 band for both models (`results/per_class_f1_comparison.csv`), with no category collapsing the way every zero/few-shot approach did on "Sri Lanka Identity & Culture" (F1 0.00–0.27 across all six).
- BERT edges out RoBERTa on every metric here, though the gap is small (≤0.02) and within the noise of a single fine-tuning run — not strong evidence of one architecture being fundamentally better for this corpus.

**Limitations**
- Both are only as good as the silver-standard labels from Task 3a's clustering — errors or ambiguity in the cluster-derived categories cap the ceiling on true accuracy, since "accuracy" here really measures agreement with an unsupervised label, not ground truth.
- Fixed 10-way output head: adding or splitting a category requires re-fine-tuning, unlike every other approach tested here, which takes new label names/descriptions at inference time with no retraining.

**Data efficiency** — required ~37,783 labeled training examples (80% of the 47,229-post corpus) to reach these scores; this labeling was itself synthetic (Task 3a clusters), so real-world deployment on a genuinely unlabeled domain would need either the same clustering bootstrap or manual annotation at similar scale.

**Interpretability** — outputs a softmax probability per class, but the decision itself is opaque without extra tooling (e.g. attention visualization, SHAP/LIME on top of the fine-tuned head); harder to audit *why* a post was routed to a category than the NLI entailment scores or the LLMs' generated text below.

**Ease of deployment** — each fine-tuned checkpoint is ~440MB; inference is a single forward pass per post (cheap relative to the NLI classifiers' per-label passes or the decoder LLMs' autoregressive generation), but a GPU is effectively required for the fine-tuning step, and the taxonomy is baked into the weights, so any change to categories means retraining and redeploying a new checkpoint.

## Zero-Shot NLI Classifiers (BART-MNLI, DeBERTa-v3-zeroshot)

These are *not* decoder-only LLMs — they're encoder(-decoder) models run through HuggingFace's NLI-based `zero-shot-classification` pipeline, which scores each candidate label as an entailment hypothesis rather than generating text. Kept here as a useful baseline; see the next section for the decoder-only LLMs Task 3(c) actually requires.

**Strengths**
- No labeled training data required — both models classified directly off the natural-language label strings (`"This Reddit post is about {}."`) with zero fine-tuning.
- DeBERTa-v3-zeroshot in particular did reasonably on high-signal categories with distinctive vocabulary: Education (F1 0.59), Employment & Career (0.52), Travel & Tourism (0.47) — categories where the label text itself is close to the words that actually appear in-post.
- New categories can be added or renamed at inference time by editing the candidate-label list, with no retraining pipeline.

**Limitations**
- Large aggregate gap vs. fine-tuned models: DeBERTa-v3 reached only 0.36 accuracy / 0.33 F1-weighted, BART-MNLI just 0.23 / 0.20.
- Catastrophic failure on broad, vocabulary-overlapping categories: "Sri Lanka Identity & Culture" (the largest test class) scored F1 0.12 (BART) and 0.09 (DeBERTa) — the label name is too generic and semantically overlaps with almost every other category, so the entailment model can't isolate it.
- High macro-precision but low macro-recall for BART-MNLI (0.43 precision vs. 0.25 recall) indicates it defaults heavily to a small subset of "safe" labels rather than spreading predictions across all 10 categories.

**Data efficiency** — zero training examples needed; quality depends entirely on how well the candidate label names and hypothesis template match the actual vocabulary of the domain.

**Interpretability** — entailment scores per label give a natural, ranked confidence output, but that transparency doesn't translate into accuracy here.

**Ease of deployment** — no training pipeline needed, but each prediction costs one forward pass *per candidate label* (10x the compute of a fine-tuned classifier per post).

## Decoder-Only LLM Prompting (Qwen2.5-1.5B-Instruct, Phi-3-mini-4k-instruct)

This is the approach Task 3(c) actually specifies: two open-source decoder-only LLMs, prompted (not fine-tuned) under both zero-shot and few-shot strategies. Full prompt templates and design rationale are in `../task3c-zeroshot/PROMPT_DESIGN.md`.

**Strengths**
- Few-shot consistently beat zero-shot for both models: Qwen2.5-1.5B jumped from 0.17 to 0.27 accuracy (+0.10), Phi-3-mini from 0.34 to 0.35 (+0.01) — one worked example per category is enough to meaningfully calibrate the smaller model, though the effect shrinks as the base model gets stronger.
- Phi-3-mini (3.8B params) clearly outperformed Qwen2.5-1.5B at both strategies (0.34–0.35 vs. 0.17–0.27 accuracy), and did so while producing far fewer unparseable outputs (0.15%/1.30% vs. 1.10%/0.75% unparsed rate) — model scale bought both accuracy and instruction-following reliability.
- Phi-3-mini few-shot (0.35 accuracy) essentially matches the best NLI classifier (DeBERTa-v3, 0.36) despite using a completely different mechanism (generation + string-matching vs. entailment scoring), and without any task-specific training.
- The `UNPARSED` rate is itself a useful diagnostic unavailable from the other two approaches — it directly measures how reliably a model follows the "respond with only the category name" instruction, separate from whether the *content* of its answer is correct.

**Limitations**
- Still a large gap vs. fine-tuned models: even the best run (Phi-3-mini few-shot, 0.35 accuracy) is roughly a quarter of BERT's 0.88.
- The same "Sri Lanka Identity & Culture" failure mode seen in the NLI baseline persists here, worse for the smaller model: Qwen2.5-1.5B scored F1 0.00 (zero-shot) and 0.02 (few-shot) on this class — despite an explicit gloss describing what the category covers, the model seems to almost never predict it, likely defaulting to more specific-sounding categories instead. Phi-3-mini did meaningfully better here (F1 0.24–0.27) but still trailed its own performance on other classes.
- Qwen2.5-1.5B's "Help & General Q&A" F1 (0.03 zero-shot, 0.16 few-shot) suggests the smaller model struggles specifically with the catch-all/ambiguous categories that require judging what a post *isn't* about, not just pattern-matching on vocabulary.
- Results here come from a 2,000-post stratified subsample of the test set, not the full 9,446 used for the other six models — a deliberate cost tradeoff (documented in `PROMPT_DESIGN.md`) given that generation is far more expensive per example than a classification-head forward pass or an NLI entailment pass. Numbers are internally comparable across the four decoder-LLM runs, but not on the exact same posts as the other rows in this table.

**Data efficiency** — zero fine-tuning examples needed for zero-shot; few-shot needs only 10 worked examples total (1 per category) reused across every inference call, dramatically cheaper than the ~37,783 examples fine-tuning required.

**Interpretability** — the model's raw generated text is directly human-readable (unlike a softmax vector), and the `UNPARSED` metric gives a second, orthogonal signal about instruction-following reliability. However, unlike the NLI classifiers' entailment scores, a single greedy-decoded label has no natural confidence/ranking signal unless one deliberately extracts token probabilities.

**Ease of deployment** — no training pipeline, and categories can be redefined by editing the prompt with no retraining. But autoregressive generation is markedly slower and more resource-intensive per post than either alternative — Phi-3-mini (3.8B, fp16) needs meaningfully more GPU memory and time than a 110M-parameter fine-tuned BERT head plus a single forward pass, and few-shot prompts multiply the per-request cost further by re-processing the worked examples on every single call.

## Overall recommendation

For this specific problem — categorizing r/srilanka posts into the 10 silver-standard categories from Task 3a — **fine-tuned transformers (BERT/RoBERTa) remain the clear choice**, now confirmed against six different zero/few-shot baselines rather than two: nothing without task-specific training got within 2.4x of BERT's accuracy, and every non-fine-tuned approach shared the same core weakness on broad, vocabulary-overlapping categories like "Sri Lanka Identity & Culture." Since Task 3a already produces a usable silver-standard label set at no marginal labeling cost, there's no real data-scarcity constraint here to justify accepting the zero/few-shot accuracy penalty.

Where the zero/few-shot approaches *do* earn their keep is as a fast, no-training baseline or for scenarios with a genuinely fluid taxonomy (e.g. before Task 3a's clustering existed at all). Within that family, the ranking is: **decoder-LLM few-shot prompting with a sufficiently capable model (Phi-3-mini) ≈ the best NLI classifier (DeBERTa-v3) > decoder-LLM zero-shot ≈ weaker decoder-LLM few-shot (Qwen2.5-1.5B) > the weaker NLI classifier (BART-MNLI)**. The clearest single lesson from adding the decoder-LLM runs is that *model scale and prompting strategy both matter more than the zero-shot-vs-few-shot distinction alone* — Phi-3-mini zero-shot (0.34) already beat Qwen2.5-1.5B few-shot (0.27), meaning a bigger base model with no examples outperformed a smaller model with calibration examples.
