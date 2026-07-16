# Task 3(d) — Model Comparison Discussion

*Fill in after running `compare_models.py` with all four classification reports in place.*

## Summary table

*(paste `results/summary_comparison.csv` here as a markdown table once generated)*

## Fine-tuned Transformers (BERT, RoBERTa)

**Strengths**
- TODO: cite actual accuracy/F1 once available (task3b results)
-

**Limitations**
-

**Data efficiency** — required ~38k labeled training examples; performance vs. label volume trade-off.

**Interpretability** — TODO: attention/probability outputs, but not inherently explainable without extra tooling (e.g. SHAP, attention visualization).

**Ease of deployment** — requires hosting a fine-tuned checkpoint (~440MB+ per model), GPU recommended for low-latency inference, versioning/retraining needed if label taxonomy changes.

## Zero-Shot LLMs (BART-MNLI, DeBERTa-v3-zeroshot)

**Strengths**
- No labeled training data required — works directly off natural-language label names.
-

**Limitations**
- TODO: cite actual accuracy/F1 gap vs. fine-tuned models (task3c results)
-

**Data efficiency** — zero training examples needed; quality depends entirely on how well candidate label names/hypothesis template match the domain.

**Interpretability** — entailment scores per label give a natural confidence ranking, arguably more interpretable out-of-the-box than a fine-tuned softmax head.

**Ease of deployment** — no training pipeline needed, but each prediction costs one forward pass *per candidate label* (10x more compute per example than a fine-tuned classifier), which matters for latency/cost at scale.

## Overall recommendation

TODO once numbers are in — which approach fits this classification problem (10-category r/srilanka posts) best, and under what constraints (labeled data availability, latency budget, need to explain predictions).
