"""
Task 2(b) — Tokenization Scheme Comparison
============================================
Input  : ../task2a-clean/cleaned_corpus.csv
Output : tokenized/          — tokenized outputs (.csv each)
         models/             — saved sub-word tokenizer models
         plots/              — visualisations
         tokenization_report.txt

Schemes implemented
-------------------
  1. Word + Stopword Removal + Lemmatization  (NLTK)
     Traditional regex-based word split, NLTK English stopwords removed,
     tokens then lemmatized using WordNetLemmatizer.

  2. Byte Pair Encoding (BPE)  — HuggingFace tokenizers
     Trained from scratch on the corpus. Merges the most frequent character
     pairs iteratively until vocab_size is reached. Handles OOV naturally
     via byte-level fallback.

  3. WordPiece  — HuggingFace tokenizers
     Trained from scratch. Splits unknown words into sub-word units using
     a greedy longest-match-first strategy (## prefix for continuations).
     Same algorithm as BERT tokenizers.

  4. SentencePiece (Unigram LM)  — sentencepiece  [BONUS]
     Language-model-based sub-word segmentation; does not require pre-
     tokenization so it handles scripts without whitespace naturally.
"""

import re
import ssl
import string
from pathlib import Path
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence as NormSeq, Strip

import sentencepiece as spm

ssl._create_default_https_context = ssl._create_unverified_context
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CORPUS_CSV = ROOT / "task2a-clean" / "cleaned_corpus.csv"
OUT_DIR    = Path(__file__).parent
TOK_DIR    = OUT_DIR / "tokenized"
MOD_DIR    = OUT_DIR / "models"
PLOTS_DIR  = OUT_DIR / "plots"
REPORT     = OUT_DIR / "tokenization_report.txt"

for d in (TOK_DIR, MOD_DIR, PLOTS_DIR):
    d.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#E05252"

# ── Config ─────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 16_000   # shared target vocab size for sub-word schemes
SP_VOCAB     = 16_000
TEST_FRAC    = 0.10     # 10 % held-out for perplexity
RANDOM_SEED  = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
def hline(c="─", w=72):
    return c * w

def save_plot(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → plots/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# Load corpus
# ══════════════════════════════════════════════════════════════════════════════
print("Loading cleaned corpus …")
df = pd.read_csv(CORPUS_CSV)
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Train / test split (stratified on word-count buckets for representativeness)
df["wc_bucket"] = pd.qcut(df["clean_wc"], q=10, labels=False, duplicates="drop")
test_df  = df.groupby("wc_bucket", group_keys=False).apply(
    lambda g: g.sample(frac=TEST_FRAC, random_state=RANDOM_SEED)
)
train_df = df.drop(test_df.index)

print(f"  Total  : {len(df):,} entries")
print(f"  Train  : {len(train_df):,} entries")
print(f"  Test   : {len(test_df):,} entries")

texts_all   = df["clean_text"].tolist()
texts_train = train_df["clean_text"].tolist()
texts_test  = test_df["clean_text"].tolist()

# ══════════════════════════════════════════════════════════════════════════════
# Scheme 1 — Word tokenization + Stopword Removal + Lemmatization
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Word + Stopwords + Lemmatization (NLTK) …")

STOP_EN   = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def tokenize_word(text: str) -> list[str]:
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]   # keep alphabetic only
    tokens = [t for t in tokens if t not in STOP_EN]      # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]    # lemmatize
    return tokens

word_tokens_all = []
for i, t in enumerate(texts_all, 1):
    word_tokens_all.append(tokenize_word(t))
    if i % 10000 == 0:
        print(f"    {i:,}/{len(texts_all):,} …")

word_tokens_train = [word_tokens_all[i] for i in train_df.index - df.index[0]]
word_tokens_test  = [word_tokens_all[i] for i in test_df.index  - df.index[0]]

# Re-index safely
idx_map = {orig: new for new, orig in enumerate(df.index)}
word_tokens_train = [word_tokens_all[idx_map[i]] for i in train_df.index]
word_tokens_test  = [word_tokens_all[idx_map[i]] for i in test_df.index]

# Stats
all_word_flat   = [t for doc in word_tokens_all   for t in doc]
train_word_flat = [t for doc in word_tokens_train for t in doc]
test_word_flat  = [t for doc in word_tokens_test  for t in doc]

word_stats = {
    "total_tokens"  : len(all_word_flat),
    "unique_tokens" : len(set(all_word_flat)),
    "vocab"         : len(set(all_word_flat)),
    "avg_per_doc"   : np.mean([len(d) for d in word_tokens_all]),
    "train_tokens"  : len(train_word_flat),
    "test_tokens"   : len(test_word_flat),
}
print(f"    Total tokens  : {word_stats['total_tokens']:,}")
print(f"    Unique tokens : {word_stats['unique_tokens']:,}")

# Save
df_w = df[["entry_id", "clean_text"]].copy()
df_w["tokens"]     = [" ".join(t) for t in word_tokens_all]
df_w["token_count"] = [len(t) for t in word_tokens_all]
df_w.to_csv(TOK_DIR / "tokenized_word_lemma.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Scheme 2 — BPE (Byte Pair Encoding)  — HuggingFace tokenizers
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] BPE tokenizer (HuggingFace, train from scratch) …")

bpe_model_path = MOD_DIR / "bpe_tokenizer.json"

bpe_tok = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tok.normalizer  = NormSeq([Strip(), Lowercase()])
bpe_tok.pre_tokenizer = Whitespace()

bpe_trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2,
    show_progress=False,
)
bpe_tok.train_from_iterator(texts_train, trainer=bpe_trainer)
bpe_tok.save(str(bpe_model_path))
print(f"    BPE model saved → models/bpe_tokenizer.json")

bpe_encoded_all = bpe_tok.encode_batch(texts_all, is_pretokenized=False)
bpe_tokens_all  = [e.tokens for e in bpe_encoded_all]

bpe_tokens_train = [bpe_tokens_all[idx_map[i]] for i in train_df.index]
bpe_tokens_test  = [bpe_tokens_all[idx_map[i]] for i in test_df.index]

all_bpe_flat = [t for doc in bpe_tokens_all for t in doc]
bpe_stats = {
    "total_tokens"  : len(all_bpe_flat),
    "unique_tokens" : len(set(all_bpe_flat)),
    "vocab"         : bpe_tok.get_vocab_size(),
    "avg_per_doc"   : np.mean([len(d) for d in bpe_tokens_all]),
    "train_tokens"  : len([t for doc in bpe_tokens_train for t in doc]),
    "test_tokens"   : len([t for doc in bpe_tokens_test  for t in doc]),
}
print(f"    Vocab size    : {bpe_stats['vocab']:,}")
print(f"    Total tokens  : {bpe_stats['total_tokens']:,}")

df_bpe = df[["entry_id", "clean_text"]].copy()
df_bpe["tokens"]      = [" ".join(t) for t in bpe_tokens_all]
df_bpe["token_count"] = [len(t) for t in bpe_tokens_all]
df_bpe.to_csv(TOK_DIR / "tokenized_bpe.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Scheme 3 — WordPiece  — HuggingFace tokenizers
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] WordPiece tokenizer (HuggingFace, train from scratch) …")

wp_model_path = MOD_DIR / "wordpiece_tokenizer.json"

wp_tok = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_tok.normalizer    = NormSeq([Strip(), Lowercase()])
wp_tok.pre_tokenizer = Whitespace()

wp_trainer = WordPieceTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2,
    show_progress=False,
)
wp_tok.train_from_iterator(texts_train, trainer=wp_trainer)
wp_tok.save(str(wp_model_path))
print(f"    WordPiece model saved → models/wordpiece_tokenizer.json")

wp_encoded_all = wp_tok.encode_batch(texts_all, is_pretokenized=False)
wp_tokens_all  = [e.tokens for e in wp_encoded_all]

wp_tokens_train = [wp_tokens_all[idx_map[i]] for i in train_df.index]
wp_tokens_test  = [wp_tokens_all[idx_map[i]] for i in test_df.index]

all_wp_flat = [t for doc in wp_tokens_all for t in doc]
wp_stats = {
    "total_tokens"  : len(all_wp_flat),
    "unique_tokens" : len(set(all_wp_flat)),
    "vocab"         : wp_tok.get_vocab_size(),
    "avg_per_doc"   : np.mean([len(d) for d in wp_tokens_all]),
    "train_tokens"  : len([t for doc in wp_tokens_train for t in doc]),
    "test_tokens"   : len([t for doc in wp_tokens_test  for t in doc]),
}
print(f"    Vocab size    : {wp_stats['vocab']:,}")
print(f"    Total tokens  : {wp_stats['total_tokens']:,}")

df_wp = df[["entry_id", "clean_text"]].copy()
df_wp["tokens"]      = [" ".join(t) for t in wp_tokens_all]
df_wp["token_count"] = [len(t) for t in wp_tokens_all]
df_wp.to_csv(TOK_DIR / "tokenized_wordpiece.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Scheme 4 — SentencePiece (Unigram LM)  [BONUS]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] SentencePiece / Unigram LM [BONUS] …")

sp_train_txt = OUT_DIR / "sp_train_input.txt"
sp_model_prefix = str(MOD_DIR / "spm_unigram")

# Write training text
with open(sp_train_txt, "w", encoding="utf-8") as f:
    for t in texts_train:
        f.write(t.replace("\n", " ") + "\n")

spm.SentencePieceTrainer.train(
    input=str(sp_train_txt),
    model_prefix=sp_model_prefix,
    vocab_size=SP_VOCAB,
    model_type="unigram",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    character_coverage=0.9995,
    input_sentence_size=len(texts_train),
    shuffle_input_sentence=True,
)
sp_train_txt.unlink()   # clean temp file

sp = spm.SentencePieceProcessor()
sp.load(sp_model_prefix + ".model")
print(f"    SentencePiece model saved → models/spm_unigram.model")

sp_tokens_all = [sp.encode(t, out_type=str) for t in texts_all]
sp_tokens_train = [sp_tokens_all[idx_map[i]] for i in train_df.index]
sp_tokens_test  = [sp_tokens_all[idx_map[i]] for i in test_df.index]

all_sp_flat = [t for doc in sp_tokens_all for t in doc]
sp_stats = {
    "total_tokens"  : len(all_sp_flat),
    "unique_tokens" : len(set(all_sp_flat)),
    "vocab"         : sp.get_piece_size(),
    "avg_per_doc"   : np.mean([len(d) for d in sp_tokens_all]),
    "train_tokens"  : len([t for doc in sp_tokens_train for t in doc]),
    "test_tokens"   : len([t for doc in sp_tokens_test  for t in doc]),
}
print(f"    Vocab size    : {sp_stats['vocab']:,}")
print(f"    Total tokens  : {sp_stats['total_tokens']:,}")

df_sp = df[["entry_id", "clean_text"]].copy()
df_sp["tokens"]      = [" ".join(t) for t in sp_tokens_all]
df_sp["token_count"] = [len(t) for t in sp_tokens_all]
df_sp.to_csv(TOK_DIR / "tokenized_sentencepiece.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Perplexity — bigram language model on each scheme
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing perplexity (bigram LM) on held-out test set …")

def bigram_perplexity(train_docs: list[list[str]],
                      test_docs:  list[list[str]],
                      smoothing_k: float = 0.5) -> float:
    """
    Add-k smoothed bigram perplexity.
    PP = exp(- (1/N) * sum_i log P(w_i | w_{i-1}))
    """
    # Build unigram and bigram counts from training
    unigram: Counter = Counter()
    bigram:  Counter = Counter()
    for doc in train_docs:
        tokens = ["<s>"] + doc + ["</s>"]
        for w in tokens:
            unigram[w] += 1
        for a, b in zip(tokens, tokens[1:]):
            bigram[(a, b)] += 1

    vocab_size = len(unigram)

    log_prob_sum = 0.0
    N = 0
    for doc in test_docs:
        tokens = ["<s>"] + doc + ["</s>"]
        for a, b in zip(tokens, tokens[1:]):
            count_ab = bigram.get((a, b), 0)
            count_a  = unigram.get(a, 0)
            # add-k smoothing
            prob = (count_ab + smoothing_k) / (count_a + smoothing_k * vocab_size)
            log_prob_sum += np.log(prob)
            N += 1

    return float(np.exp(-log_prob_sum / N)) if N > 0 else float("inf")

pp_word = bigram_perplexity(word_tokens_train, word_tokens_test)
pp_bpe  = bigram_perplexity(bpe_tokens_train,  bpe_tokens_test)
pp_wp   = bigram_perplexity(wp_tokens_train,   wp_tokens_test)
pp_sp   = bigram_perplexity(sp_tokens_train,   sp_tokens_test)

print(f"  Perplexity — Word+Lemma     : {pp_word:,.1f}")
print(f"  Perplexity — BPE            : {pp_bpe:,.1f}")
print(f"  Perplexity — WordPiece      : {pp_wp:,.1f}")
print(f"  Perplexity — SentencePiece  : {pp_sp:,.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# Compression ratio  (raw words / scheme tokens per document)
# ══════════════════════════════════════════════════════════════════════════════
raw_wc_mean = df["clean_wc"].mean()
comp = {
    "Word+Lemma"    : raw_wc_mean / word_stats["avg_per_doc"],
    "BPE"           : raw_wc_mean / bpe_stats["avg_per_doc"],
    "WordPiece"     : raw_wc_mean / wp_stats["avg_per_doc"],
    "SentencePiece" : raw_wc_mean / sp_stats["avg_per_doc"],
}

# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════
def fmt(stats):
    return (
        f"    Total tokens     : {stats['total_tokens']:>12,}\n"
        f"    Unique tokens    : {stats['unique_tokens']:>12,}\n"
        f"    Vocabulary size  : {stats['vocab']:>12,}\n"
        f"    Avg tokens/doc   : {stats['avg_per_doc']:>12.1f}\n"
        f"    Train tokens     : {stats['train_tokens']:>12,}\n"
        f"    Test tokens      : {stats['test_tokens']:>12,}"
    )

report_lines = [
    "Task 2(b) — Tokenization Scheme Comparison Report",
    hline("="),
    "",
    "CORPUS SPLIT",
    hline(),
    f"  Total entries  : {len(df):,}",
    f"  Train entries  : {len(train_df):,}  (90 %)",
    f"  Test entries   : {len(test_df):,}  (10 %, stratified by length)",
    f"  Target vocab   : {VOCAB_SIZE:,}  (sub-word schemes)",
    "",
    "══════════════════════════════════════════════════════════════════════════",
    "SCHEME 1 — Word Tokenization + Stopword Removal + Lemmatization (NLTK)",
    hline(),
    "  Method: NLTK word_tokenize → remove non-alpha → remove English",
    "  stopwords → WordNetLemmatizer. No fixed vocabulary; any surface",
    "  form that survives becomes a token.",
    fmt(word_stats),
    f"    Perplexity (bigram) : {pp_word:,.1f}",
    f"    Compression ratio   : {comp['Word+Lemma']:.3f}",
    "",
    "══════════════════════════════════════════════════════════════════════════",
    "SCHEME 2 — Byte Pair Encoding (BPE) — HuggingFace tokenizers",
    hline(),
    "  Method: Start with characters; iteratively merge the most frequent",
    "  adjacent pair until vocab_size is reached. Trained on training split.",
    f"  Vocab size target: {VOCAB_SIZE:,}.",
    fmt(bpe_stats),
    f"    Perplexity (bigram) : {pp_bpe:,.1f}",
    f"    Compression ratio   : {comp['BPE']:.3f}",
    "",
    "══════════════════════════════════════════════════════════════════════════",
    "SCHEME 3 — WordPiece — HuggingFace tokenizers",
    hline(),
    "  Method: Like BPE but selects merges that maximise likelihood of",
    "  training data rather than raw frequency. Continuation pieces are",
    f"  prefixed with '##'. Trained on training split. Vocab: {VOCAB_SIZE:,}.",
    fmt(wp_stats),
    f"    Perplexity (bigram) : {pp_wp:,.1f}",
    f"    Compression ratio   : {comp['WordPiece']:.3f}",
    "",
    "══════════════════════════════════════════════════════════════════════════",
    "SCHEME 4 — SentencePiece (Unigram LM) [BONUS]",
    hline(),
    "  Method: Language-model-based sub-word segmentation. Does NOT rely",
    "  on pre-tokenization (no whitespace assumption), making it robust",
    "  to scripts without natural word boundaries and to transliterated",
    f"  text (Singlish). Trained on training split. Vocab: {SP_VOCAB:,}.",
    fmt(sp_stats),
    f"    Perplexity (bigram) : {pp_sp:,.1f}",
    f"    Compression ratio   : {comp['SentencePiece']:.3f}",
    "",
    "══════════════════════════════════════════════════════════════════════════",
    "SUMMARY COMPARISON",
    hline(),
    f"  {'Scheme':<25} {'Total tokens':>14} {'Unique':>10} {'Vocab':>8} "
    f"{'Avg/doc':>9} {'Perplexity':>12} {'Compress':>10}",
    hline(),
    f"  {'Word+Lemma':<25} {word_stats['total_tokens']:>14,} "
    f"{word_stats['unique_tokens']:>10,} {word_stats['vocab']:>8,} "
    f"{word_stats['avg_per_doc']:>9.1f} {pp_word:>12,.1f} "
    f"{comp['Word+Lemma']:>10.3f}",
    f"  {'BPE':<25} {bpe_stats['total_tokens']:>14,} "
    f"{bpe_stats['unique_tokens']:>10,} {bpe_stats['vocab']:>8,} "
    f"{bpe_stats['avg_per_doc']:>9.1f} {pp_bpe:>12,.1f} "
    f"{comp['BPE']:>10.3f}",
    f"  {'WordPiece':<25} {wp_stats['total_tokens']:>14,} "
    f"{wp_stats['unique_tokens']:>10,} {wp_stats['vocab']:>8,} "
    f"{wp_stats['avg_per_doc']:>9.1f} {pp_wp:>12,.1f} "
    f"{comp['WordPiece']:>10.3f}",
    f"  {'SentencePiece [BONUS]':<25} {sp_stats['total_tokens']:>14,} "
    f"{sp_stats['unique_tokens']:>10,} {sp_stats['vocab']:>8,} "
    f"{sp_stats['avg_per_doc']:>9.1f} {pp_sp:>12,.1f} "
    f"{comp['SentencePiece']:>10.3f}",
    "",
    "RECOMMENDATION (for Task 2c)",
    hline(),
    "  Based on perplexity and compression ratio, the recommended scheme",
    "  for subsequent LLM tasks is discussed in task2c-evaluation.",
]

with open(REPORT, "w") as f:
    f.write("\n".join(report_lines))

for line in report_lines:
    print(line)

# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots …")

schemes     = ["Word+Lemma", "BPE", "WordPiece", "SentencePiece"]
all_stats   = [word_stats, bpe_stats, wp_stats, sp_stats]
perplexities = [pp_word, pp_bpe, pp_wp, pp_sp]
palette      = ["#5B8DB8", "#E05252", "#7B5EA7", "#4CAF82"]

# (A) Total tokens comparison
fig, ax = plt.subplots(figsize=(9, 5))
vals = [s["total_tokens"] for s in all_stats]
bars = ax.bar(schemes, vals, color=palette, edgecolor="white", lw=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Total tokens")
ax.set_title("Total Token Count by Scheme")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1e6):.1f}M"))
save_plot(fig, "01_total_tokens")

# (B) Unique tokens / vocabulary size
fig, ax = plt.subplots(figsize=(9, 5))
unique_vals = [s["unique_tokens"] for s in all_stats]
vocab_vals  = [s["vocab"]         for s in all_stats]
x = np.arange(len(schemes))
w = 0.35
bars1 = ax.bar(x - w/2, unique_vals, w, label="Unique tokens seen", color=palette, alpha=0.8)
bars2 = ax.bar(x + w/2, vocab_vals,  w, label="Vocabulary size",    color=palette, alpha=0.45,
               hatch="//", edgecolor="gray")
ax.set_xticks(x)
ax.set_xticklabels(schemes)
ax.set_ylabel("Count")
ax.set_title("Unique Tokens vs Vocabulary Size")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_plot(fig, "02_unique_vs_vocab")

# (C) Average tokens per document
fig, ax = plt.subplots(figsize=(9, 5))
avg_vals = [s["avg_per_doc"] for s in all_stats]
bars = ax.bar(schemes, avg_vals, color=palette, edgecolor="white", lw=0.5)
ax.axhline(raw_wc_mean, color="gray", ls="--", lw=1.4,
           label=f"Raw word count mean ({raw_wc_mean:.1f})")
for bar, val in zip(bars, avg_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Avg tokens per document")
ax.set_title("Average Tokens per Document by Scheme")
ax.legend()
save_plot(fig, "03_avg_tokens_per_doc")

# (D) Compression ratio
fig, ax = plt.subplots(figsize=(9, 5))
comp_vals = [comp[s] for s in schemes]
bars = ax.bar(schemes, comp_vals, color=palette, edgecolor="white", lw=0.5)
ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="Ratio = 1 (no change)")
for bar, val in zip(bars, comp_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}×", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Raw words / scheme tokens (avg per doc)")
ax.set_title("Compression Ratio (higher = fewer tokens than raw words)")
ax.legend()
save_plot(fig, "04_compression_ratio")

# (E) Perplexity comparison
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(schemes, perplexities, color=palette, edgecolor="white", lw=0.5)
for bar, val in zip(bars, perplexities):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val:,.1f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Bigram perplexity (lower = better)")
ax.set_title("Bigram Language Model Perplexity on Test Set")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_plot(fig, "05_perplexity")

# (F) Token length distribution — all four schemes overlapping
tok_len_data = {
    "Word+Lemma"    : [len(t) for doc in word_tokens_all for t in doc],
    "BPE"           : [len(t) for doc in bpe_tokens_all  for t in doc],
    "WordPiece"     : [len(t) for doc in wp_tokens_all   for t in doc],
    "SentencePiece" : [len(t) for doc in sp_tokens_all   for t in doc],
}
fig, ax = plt.subplots(figsize=(11, 5))
for (scheme, lengths), color in zip(tok_len_data.items(), palette):
    cap = int(np.percentile(lengths, 98))
    ax.hist(np.clip(lengths, 0, cap), bins=40, alpha=0.5,
            color=color, label=scheme, edgecolor="white", lw=0.2)
ax.set_xlabel("Token length (characters, capped at P98)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Token Lengths by Scheme")
ax.legend()
save_plot(fig, "06_token_length_dist")

# (G) Top-20 tokens per scheme — 2×2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
scheme_data = [
    ("Word+Lemma",    word_tokens_all),
    ("BPE",           bpe_tokens_all),
    ("WordPiece",     wp_tokens_all),
    ("SentencePiece", sp_tokens_all),
]
for ax, (scheme, tok_docs), color in zip(axes.flat, scheme_data, palette):
    flat = [t for doc in tok_docs for t in doc]
    top = Counter(flat).most_common(20)
    words, counts = zip(*top)
    ax.barh(list(words)[::-1], list(counts)[::-1], color=color)
    ax.set_title(f"Top-20 Tokens — {scheme}", fontsize=10)
    ax.set_xlabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="y", labelsize=8)
fig.suptitle("Top-20 Most Frequent Tokens per Scheme", fontsize=13, y=1.01)
fig.tight_layout()
save_plot(fig, "07_top20_tokens_grid")

# (H) Doc-level token count distributions — violin
fig, ax = plt.subplots(figsize=(10, 6))
doc_token_counts = [
    [len(d) for d in word_tokens_all],
    [len(d) for d in bpe_tokens_all],
    [len(d) for d in wp_tokens_all],
    [len(d) for d in sp_tokens_all],
]
parts = ax.violinplot(doc_token_counts, showmedians=True, showextrema=False)
for i, (pc, color) in enumerate(zip(parts["bodies"], palette)):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
parts["cmedians"].set_color("white")
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(schemes)
ax.set_ylabel("Tokens per document")
ax.set_ylim(0, 350)
ax.set_title("Token Count Distribution per Document (Violin Plot)")
save_plot(fig, "08_token_count_violin")

# (I) Radar / spider chart — summary comparison
categories   = ["Total\nTokens", "Unique\nTokens", "Vocab\nSize",
                 "Avg/Doc",  "1/Perplexity\n(×10³)"]
raw_matrices = np.array([
    [word_stats["total_tokens"], word_stats["unique_tokens"],
     word_stats["vocab"],        word_stats["avg_per_doc"],   1000/pp_word],
    [bpe_stats["total_tokens"],  bpe_stats["unique_tokens"],
     bpe_stats["vocab"],         bpe_stats["avg_per_doc"],    1000/pp_bpe],
    [wp_stats["total_tokens"],   wp_stats["unique_tokens"],
     wp_stats["vocab"],          wp_stats["avg_per_doc"],     1000/pp_wp],
    [sp_stats["total_tokens"],   sp_stats["unique_tokens"],
     sp_stats["vocab"],          sp_stats["avg_per_doc"],     1000/pp_sp],
], dtype=float)
# Normalise each column to [0,1] for radar
col_min = raw_matrices.min(axis=0)
col_max = raw_matrices.max(axis=0)
normed  = (raw_matrices - col_min) / (col_max - col_min + 1e-9)

N    = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
for row, scheme, color in zip(normed, schemes, palette):
    vals = row.tolist() + [row[0]]
    ax.plot(angles, vals, color=color, lw=2, label=scheme)
    ax.fill(angles, vals, color=color, alpha=0.10)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_yticklabels([])
ax.set_title("Scheme Comparison Radar\n(normalised per metric)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
save_plot(fig, "09_radar_comparison")

print(f"\n{hline('═')}")
print(f"  Tokenized outputs : task2b-tokenization/tokenized/  (4 CSVs)")
print(f"  Saved models      : task2b-tokenization/models/")
print(f"  Plots             : task2b-tokenization/plots/  (9 files)")
print(f"  Report            : task2b-tokenization/tokenization_report.txt")
print(hline("═"))
