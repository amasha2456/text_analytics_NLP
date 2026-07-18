# Task 2(c) — Tokenization Scheme Recommendation

This closes the reference left in `task2b-tokenization/tokenization_report.txt`
("the recommended scheme for subsequent LLM tasks is discussed in
task2c-evaluation"), using the comparison already computed there.

## Test-set methodology

The 47,229-post cleaned corpus was split 90% train (42,506) / 10% test
(4,723), **stratified by document length** rather than a plain random
split — length is the single biggest driver of token-count and
perplexity variance in this corpus (10–500 words per Task 2(a)'s
thresholds), so a length-stratified test set avoids accidentally
evaluating perplexity on a skewed (e.g. unusually short or long) subset
and keeps the comparison fair across schemes. All four schemes were
evaluated on the identical test split.

## Comparison (from `task2b-tokenization/tokenization_report.txt`)

| Scheme | Total tokens | Unique tokens | Vocab size | Avg tokens/doc | Perplexity (bigram, test set) | Compression ratio |
|---|---|---|---|---|---|---|
| Word + Lemma (traditional) | 1,813,024 | 39,933 | 39,933 (open) | 38.4 | **6,151.6** | 1.985 |
| BPE | 4,183,854 | 15,606 | 16,000 | 88.6 | 536.0 | 0.860 |
| **WordPiece** | 4,212,667 | 15,238 | 16,000 | 89.2 | **519.9** | **0.854** |
| SentencePiece (Unigram, bonus) | 4,236,869 | 16,759 | 16,000 | 89.7 | 555.3 | 0.850 |

(Compression ratio = tokens produced ÷ whitespace-word count; lower means
fewer tokens per document, i.e. more efficient. Perplexity is a bigram
language model fit on the train split and evaluated on the held-out test
split, in each scheme's own token space — lower is better and indicates
a more predictable, less sparse token distribution.)

## Recommendation: **WordPiece**

**WordPiece is the recommended scheme for all subsequent tasks in this
assignment.**

- **Lowest perplexity of all four schemes (519.9)**, beating BPE (536.0)
  and SentencePiece (555.3), and roughly **12x lower than the
  traditional Word+Lemma scheme (6,151.6)**. The traditional scheme's
  perplexity is inflated by its open, 39,933-token vocabulary — every
  rare surface form (misspellings, code-mixed Sinhala/Singlish, typos)
  becomes its own sparse token with almost no training signal. A fixed
  16,000-token subword vocabulary avoids this by decomposing anything
  unseen into familiar sub-word pieces instead of treating it as
  effectively unseen.
- **Best compression ratio among the fixed-vocab subword schemes
  (0.854)** — SentencePiece is marginally better (0.850) but at the cost
  of higher perplexity; WordPiece gives the better perplexity/compression
  trade-off of the three.
- **Directly reusable downstream.** WordPiece is BERT's native
  tokenization scheme — Task 3(b)'s fine-tuned `bert-base-uncased`
  classifier already tokenizes this exact corpus with WordPiece via
  `AutoTokenizer`, so this recommendation is consistent with (and
  validates) what Task 3 onward actually used in practice, rather than
  being a purely theoretical choice made in isolation.

## Where this recommendation does — and doesn't — apply

This recommendation governs the **corpus-level tokenization choice**
(Task 2(b)/(c)'s own deliverable). It does not mean every later model in
this assignment literally re-uses these WordPiece token IDs: pretrained
models loaded via HuggingFace (RoBERTa, Qwen2.5, Phi-3-mini, DeBERTa-v3,
BART) each carry their own bundled tokenizer, which is standard practice
and not something this project's tokenizer choice can or should override.
The perplexity comparison above is what justifies WordPiece as the
project's own answer to "which scheme is most suitable for this corpus,"
independent of which specific pretrained tokenizer any individual
downstream model happens to ship with.
