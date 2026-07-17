# Task 3(d) — Model Comparison Discussion

## Summary table

| Model | Category | Accuracy | F1-Weighted | F1-Macro | Precision-Macro | Recall-Macro |
|---|---|---|---|---|---|---|
| BERT (fine-tuned) | Fine-tuned Transformer | 0.88 | 0.88 | 0.87 | 0.87 | 0.87 |
| RoBERTa (fine-tuned) | Fine-tuned Transformer | 0.87 | 0.87 | 0.86 | 0.86 | 0.86 |
| DeBERTa-v3 (zero-shot) | Zero-Shot LLM | 0.36 | 0.33 | 0.38 | 0.41 | 0.45 |
| BART-MNLI (zero-shot) | Zero-Shot LLM | 0.23 | 0.20 | 0.23 | 0.43 | 0.25 |

*(source: `results/summary_comparison.csv`, `results/comparison_report.txt`)*

## Fine-tuned Transformers (BERT, RoBERTa)

**Strengths**
- Both models cleared 0.86+ on every aggregate metric (BERT: 0.88 accuracy / 0.87 F1-macro; RoBERTa: 0.87 / 0.86), a ~2.3–2.5x jump over the better zero-shot model.
- Consistent across classes: per-class F1 stayed in a tight 0.80–0.94 band for both models (`results/per_class_f1_comparison.csv`), with no category collapsing the way zero-shot did on "Sri Lanka Identity & Culture" (F1 0.09–0.12).
- BERT edges out RoBERTa on every metric here, though the gap is small (≤0.02) and within the noise of a single fine-tuning run — not strong evidence of one architecture being fundamentally better for this corpus.

**Limitations**
- Both are only as good as the silver-standard labels from Task 3a's clustering — errors or ambiguity in the cluster-derived categories cap the ceiling on true accuracy, since "accuracy" here really measures agreement with an unsupervised label, not ground truth.
- Fixed 10-way output head: adding or splitting a category requires re-fine-tuning, unlike the zero-shot models which take new label names at inference time with no retraining.

**Data efficiency** — required ~37,783 labeled training examples (80% of the 47,229-post corpus) to reach these scores; this labeling was itself synthetic (Task 3a clusters), so real-world deployment on a genuinely unlabeled domain would need either the same clustering bootstrap or manual annotation at similar scale.

**Interpretability** — outputs a softmax probability per class, but the decision itself is opaque without extra tooling (e.g. attention visualization, SHAP/LIME on top of the fine-tuned head); harder to audit *why* a post was routed to a category than the zero-shot entailment scores below.

**Ease of deployment** — each fine-tuned checkpoint is ~440MB (`results/bert-base-uncased/model`, `results/roberta-base/model`); inference is a single forward pass per post (cheap relative to zero-shot's per-label passes), but a GPU is effectively required for the fine-tuning step, and the taxonomy is baked into the weights, so any change to categories means retraining and redeploying a new checkpoint.

## Zero-Shot LLMs (BART-MNLI, DeBERTa-v3-zeroshot)

**Strengths**
- No labeled training data required — both models classified directly off the natural-language label strings (`"This Reddit post is about {}."`) with zero fine-tuning.
- DeBERTa-v3-zeroshot in particular did reasonably on high-signal categories with distinctive vocabulary: Education (F1 0.59), Employment & Career (0.52), Travel & Tourism (0.47) — categories where the label text itself is close to the words that actually appear in-post.
- New categories can be added or renamed at inference time by editing the candidate-label list, with no retraining pipeline.

**Limitations**
- Large aggregate gap vs. fine-tuned models: DeBERTa-v3 reached only 0.36 accuracy / 0.33 F1-weighted, BART-MNLI just 0.23 / 0.20 — roughly a 2.5x drop from BERT/RoBERTa.
- Catastrophic failure on broad, vocabulary-overlapping categories: "Sri Lanka Identity & Culture" (the largest test class at 1,909 support) scored F1 0.12 (BART) and 0.09 (DeBERTa) — the label name is too generic and semantically overlaps with almost every other category, so the entailment model can't isolate it. "Finance, Banking & Telecoms" was similarly weak (F1 0.04 / 0.31) despite being a distinctive topic, suggesting the label phrasing itself, not just topic difficulty, is a major factor.
- High macro-precision but low macro-recall for BART-MNLI (0.43 precision vs. 0.25 recall) indicates it defaults heavily to a small subset of "safe" labels rather than spreading predictions across all 10 categories — visible in the confusion matrices (`plots/cm_bart_large_mnli.png`).

**Data efficiency** — zero training examples needed; quality depends entirely on how well the candidate label names and hypothesis template match the actual vocabulary of the domain, which the results above show is a real bottleneck, not a theoretical one.

**Interpretability** — entailment scores per label give a natural, ranked confidence output (arguably more interpretable out-of-the-box than a fine-tuned softmax head, since each score maps to an explicit "does this post entail this label" judgment), but that transparency doesn't translate into accuracy here.

**Ease of deployment** — no training pipeline needed, but each prediction costs one forward pass *per candidate label* (10x the compute of a fine-tuned classifier per post), which matters for latency/cost at the ~47k-post scale of this corpus and would compound further with more categories.

## Overall recommendation

For this specific problem — categorizing r/srilanka posts into the 10 silver-standard categories from Task 3a — **fine-tuned transformers (BERT/RoBERTa) are the clear choice** given the ~2.5x gap in every aggregate metric and their far more even per-class performance. The zero-shot approach is only justified when labeled data genuinely doesn't exist and the taxonomy needs to stay fluid (e.g. an exploratory first pass before Task 3a's clustering existed at all), or as a cheap baseline/sanity-check against the fine-tuned models. Given that Task 3a already produces a usable silver-standard label set at no marginal cost, there's no real data-scarcity constraint here to justify accepting zero-shot's accuracy penalty — the fine-tuned models' downsides (fixed taxonomy, retraining cost, GPU requirement) are a reasonable trade for the accuracy gain, especially since the corpus size (47k+ posts) comfortably supports fine-tuning without overfitting risk.
