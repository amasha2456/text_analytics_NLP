# Task 3(c) — Prompt Design for Decoder-Only LLM Classification

Applies to `classify_decoder_llm.py`, which prompts two open-source
decoder-only instruction-tuned models — **Qwen2.5-1.5B-Instruct** and
**microsoft/Phi-3-mini-4k-instruct** — to classify r/srilanka posts into
the 10 silver-standard categories from Task 3(a), under both zero-shot and
few-shot strategies.

## Why these two models

Both are open-weight, ungated on HuggingFace (no license click-through or
access token needed, unlike Llama or Gemma), instruction-tuned out of the
box, and small enough (1.5B / 3.8B params) to run in fp16 on a single
Colab T4 GPU alongside the rest of the pipeline. They also come from
different training lineages (Alibaba vs. Microsoft), giving a genuine
"two different LLMs" comparison rather than two checkpoints of the same
base model.

## Zero-shot prompt

**System message:**
```
You are a text classification assistant for posts from the r/srilanka
subreddit. Classify the post the user gives you into exactly ONE of the
following categories:

1. Education — schools, universities, degrees, exams, studying, scholarships
2. Employment & Career — jobs, salaries, interviews, resumes, workplace issues, work visas
3. Finance, Banking & Telecoms — banking, credit cards, payments, mobile/internet providers, bills, money transfers
4. Help & General Q&A — general questions or requests for advice not specific to any topic below
5. Politics & National Affairs — government, elections, policy, national news, protests
6. Relationships & Social Life — family, friendships, dating, personal or social situations
7. Shopping & Local Recommendations — where to buy things, product or service recommendations, local businesses
8. Sri Lanka Identity & Culture — national identity, culture, traditions, general discussion about Sri Lanka as a country or society
9. Transport & Vehicles — buses, trains, vehicles, driving, traffic, vehicle imports
10. Travel & Tourism — trips, destinations, travel tips, tourism within or outside Sri Lanka

Respond with ONLY the exact category name from the list above, with no
extra words, quotes, explanation, or punctuation.
```

**User message:**
```
Post: {post text, truncated to 120 words}
Category:
```

## Few-shot prompt

Identical system message, with a `EXAMPLES:` block appended containing
**one worked example per category** (10 total), each truncated to 40
words, sampled from the training split (same train/test split used in
Task 3b, so no test-set leakage):

```
Post: {example post 1}
Category: Education

Post: {example post 2}
Category: Employment & Career

...
```

The user message for the target post is unchanged.

## Rationale behind the design choices

- **Category glosses, not bare label names.** The NLI-based baseline in
  `classify_zeroshot.py` scored F1 0.09–0.12 on "Sri Lanka Identity &
  Culture" — the single largest test class — because the label text alone
  is too generic and overlaps semantically with almost every other
  category. Pairing every label with a one-line description of what it
  actually covers directly targets that failure mode.
- **"Respond with ONLY the category name."** Forces a single-token-ish
  answer instead of free-form explanation, which keeps `max_new_tokens`
  small (cheaper/faster generation) and lets output parsing stay a simple
  string match rather than requiring a constrained-decoding library or a
  JSON schema. The unparsed rate is tracked explicitly per run
  (`unparsed_rate` in each `classification_report.txt`) as a measure of
  how well each model actually follows the instruction — this is itself a
  useful comparison point between the two models, and between zero-shot
  and few-shot for the same model.
- **One example per class for few-shot, not more.** With 10 categories,
  even one example each means every single inference call re-processes a
  10-example block. More examples per class would improve calibration
  further but multiply prompt length (and therefore latency/cost) for
  every one of the 2,000 evaluated posts; one per class is the minimum
  needed to show the model the expected input/output *format* and give it
  a concrete anchor per category, which is the main gap zero-shot prompts
  have.
- **Greedy decoding (`do_sample=False`).** There is no benefit to sampling
  diversity in a single-label classification task — greedy decoding
  is deterministic and reproducible, and removes temperature/top-p as
  confounding variables when comparing models and strategies.
- **2,000-post stratified eval subsample**, not the full 9,446-post test
  set. Autoregressive generation is far more expensive per example than a
  classification-head forward pass (as used by the fine-tuned transformers
  in 3b, or the NLI pipeline in the zero-shot baseline). Running 2 models ×
  2 strategies over the full test set was impractical on a single Colab
  session; a stratified subsample preserves class proportions from the
  full test set while keeping total runtime to roughly 20–30 minutes.
  This is disclosed as a limitation in `DISCUSSION.md` once results are
  in — the subsample means these numbers are directly comparable to each
  other, but not on exactly the same posts as the 3b/NLI-baseline numbers.
