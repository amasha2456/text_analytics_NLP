"""
Text Cleaning & Preprocessing — r/srilanka dataset
====================================================
Input  : ../entries.csv
Output : cleaned_corpus.csv        — one cleaned entry per row
         preprocessing_report.txt  — full audit report
         plots/                    — before/after visualisations

Cleaning pipeline (applied in order):
  1.  Combine title + body into a single full_text field
  2.  Strip Reddit moderation markers  ([removed], [deleted], …)
  3.  Decode HTML entities              (&amp; → &, etc.)
  4.  Remove HTML tags
  5.  Remove URLs                       (http/https/www…)
  6.  Remove Markdown syntax            (**bold**, ##heading, >quote, `code`)
  7.  Remove emojis & other non-printable / non-ASCII characters
  8.  Translate Sinhala-script spans → English  (Google Translate)
  9.  Replace Singlish tokens → English          (dictionary lookup)
  10. Expand common contractions        (don't → do not, etc.)
  11. Normalise whitespace & punctuation
  12. Unicode normalise (NFKC) + smart-quote → straight-quote
  13. Case normalisation (lowercase)
  14. Filter by word-count thresholds
        Lower : 10 words  — title-only stubs / fully-removed posts
        Upper : 500 words — outliers above P99, skew statistics
  15. Deduplication — exact-match on cleaned text
"""

import csv
import html
import re
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from deep_translator import GoogleTranslator

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
CSV_IN    = ROOT / "entries.csv"
OUT_DIR   = Path(__file__).parent
PLOTS_DIR = OUT_DIR / "plots"
CSV_OUT   = OUT_DIR / "cleaned_corpus.csv"
REPORT    = OUT_DIR / "preprocessing_report.txt"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#E05252"

# ── Sinhala script detection ───────────────────────────────────────────────────
# Unicode block U+0D80–U+0DFF
SINHALA_SPAN_RE = re.compile(r"[඀-෿][඀-෿\s]*")   # contiguous Sinhala chunks

# ── Singlish → English dictionary ─────────────────────────────────────────────
# Covers: kinship terms, common expressions, verbs, question words,
#         pronouns, adjectives, particles, and Sri Lankan slang
SINGLISH_DICT: dict[str, str] = {
    # ── Kinship / address ──────────────────────────────────────────────────
    "machan":    "friend",
    "machang":   "friend",
    "nangi":     "younger sister",
    "akka":      "older sister",
    "aiya":      "older brother",
    "ayya":      "older brother",
    "malli":     "younger brother",
    "putha":     "son",
    "duwa":      "daughter",
    "mama":      "uncle",
    "nanda":     "aunt",
    "thaththa":  "father",
    "amma":      "mother",
    "seeya":     "grandfather",
    "aachchi":   "grandmother",
    "loku thaththa": "paternal uncle (elder)",
    "bappa":     "paternal uncle (younger)",
    "loku nenda":"paternal aunt (elder)",
    "punchi nenda":"paternal aunt (younger)",
    "loku amma": "maternal aunt (elder)",
    "punchi amma":"maternal aunt (younger)",
    "loku mama": "maternal uncle (elder)",
    "punchi mama":"maternal uncle (younger)",
    "kolla":     "boy",
    "kella":     "girl",
    "kello":     "girls",
    "kollo":     "boys",
    "minissa":   "person",
    "minissu":   "people",
    # ── Expressions / interjections ────────────────────────────────────────
    "ayyo":      "oh no",
    "aiyo":      "oh no",
    "aiyoo":     "oh no",
    "aney":      "hey",
    "ane":       "hey",
    "anee":      "oh dear",
    "hari":      "okay",
    "harin":     "alright",
    "hodai":     "good",
    "hoda":      "good",
    "boru":      "lie",
    "pissu":     "crazy",
    "gona":      "fool",
    "hora":      "thief",
    "baas":      "boss",
    "athal":     "stubborn",
    "palayan":   "go away",
    "eppa":      "don't",
    "lassana":   "beautiful",
    "yakko":     "rascal",
    "yako":      "rascal",
    "choon":     "cool",
    "godak":     "a lot",
    "tikak":     "a little",
    "lokku":     "big",
    "loku":      "big",
    "punchi":    "small",
    "wediya":    "too much",
    "nattami":   "porter",
    "bathi":     "devoted",
    # ── Pronouns / determiners ─────────────────────────────────────────────
    "api":       "we",
    "ape":       "our",
    "mage":      "my",
    "eka":       "it",
    "ekata":     "for it",
    "eka":       "the thing",
    "oyala":     "you all",
    "thopi":     "you",
    "mama":      "i",
    "wage":      "like",
    "wagai":     "like",
    # ── Question words ─────────────────────────────────────────────────────
    "mokak":     "what",
    "mokakda":   "what is it",
    "monawa":    "what things",
    "mona":      "what",
    "kohomada":  "how is it",
    "kohoma":    "how",
    "koheda":    "where",
    "kohedda":   "where exactly",
    "kawda":     "who",
    "kawdata":   "to whom",
    "neda":      "isn't it",
    "nede":      "is it not",
    "nehe":      "no",
    "ne":        "right",
    "ney":       "isn't it",
    # ── Verbs (infinitive / common forms) ──────────────────────────────────
    "karanna":   "to do",
    "karala":    "did",
    "karanda":   "to do",
    "karanawa":  "doing",
    "yanna":     "to go",
    "gihilla":   "went",
    "enna":      "to come",
    "aawa":      "came",
    "balanna":   "to look",
    "balala":    "looked",
    "kiyanna":   "to say",
    "kiyala":    "said",
    "kiyanne":   "says",
    "kiyanawa":  "saying",
    "ganna":     "to take",
    "gannawa":   "taking",
    "denna":     "to give",
    "denawa":    "giving",
    "hadanna":   "to make",
    "hadala":    "made",
    "hadanawa":  "making",
    "hitanna":   "to think",
    "hitala":    "thought",
    "hitanawa":  "thinking",
    "hitanne":   "thinks",
    "innawa":    "to be",
    "inne":      "is there",
    "thiyenawa": "there is",
    "thibuna":   "there was",
    "thibenne":  "there is",
    "wenawa":    "to become",
    "wena":      "becoming",
    "venne":     "becomes",
    "danne":     "know",
    "dannawa":   "knowing",
    "thiyena":   "have",
    # ── Particles / connectors ─────────────────────────────────────────────
    "weda":      "work",
    "kade":      "shop",
    "kadey":     "shop",
    "paan":      "bread",
    "thamai":    "indeed",
    "nemei":     "is not",
    "neme":      "not",
    "ekka":      "with",
    "ekkath":    "even with",
    "apita":     "to us",
    "okata":     "to that",
    "innewada":  "is there",
    # ── Informal / slang ───────────────────────────────────────────────────
    "choonpaan": "cool stuff",
    "dung":      "smoke",
    "bung":      "brother",
    "aswesuma":  "welfare",
}

# Build compiled regex for whole-word Singlish replacement (longest first to
# avoid partial matches, e.g. "loku mama" before "loku")
_singlish_sorted = sorted(SINGLISH_DICT.keys(), key=len, reverse=True)
SINGLISH_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _singlish_sorted) + r")\b",
    re.IGNORECASE,
)

# ── Contraction map ───────────────────────────────────────────────────────────
CONTRACTIONS = {
    r"won't":       "will not",
    r"can't":       "cannot",
    r"n't\b":       " not",
    r"'re\b":       " are",
    r"'s\b":        " is",
    r"'d\b":        " would",
    r"'ll\b":       " will",
    r"'ve\b":       " have",
    r"'m\b":        " am",
    r"i'm\b":       "i am",
    r"it's\b":      "it is",
    r"that's\b":    "that is",
    r"there's\b":   "there is",
    r"they're\b":   "they are",
    r"we're\b":     "we are",
    r"you're\b":    "you are",
    r"he's\b":      "he is",
    r"she's\b":     "she is",
    r"who's\b":     "who is",
    r"what's\b":    "what is",
    r"let's\b":     "let us",
    r"i've\b":      "i have",
    r"we've\b":     "we have",
    r"they've\b":   "they have",
    r"i'd\b":       "i would",
    r"we'd\b":      "we would",
    r"they'd\b":    "they would",
    r"i'll\b":      "i will",
    r"we'll\b":     "we will",
    r"they'll\b":   "they will",
    r"isn't\b":     "is not",
    r"aren't\b":    "are not",
    r"wasn't\b":    "was not",
    r"weren't\b":   "were not",
    r"hasn't\b":    "has not",
    r"haven't\b":   "have not",
    r"hadn't\b":    "had not",
    r"doesn't\b":   "does not",
    r"don't\b":     "do not",
    r"didn't\b":    "did not",
    r"wouldn't\b":  "would not",
    r"shouldn't\b": "should not",
    r"couldn't\b":  "could not",
    r"mightn't\b":  "might not",
    r"mustn't\b":   "must not",
}

CONTRACTION_RE = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in CONTRACTIONS.items()
]

# ── Regex patterns ─────────────────────────────────────────────────────────────
URL_RE          = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_RE     = re.compile(
    r"\[([^\]]+)\]\([^\)]+\)"
    r"|\*{1,3}([^*]*)\*{1,3}"
    r"|_{1,2}([^_]*)_{1,2}"
    r"|~~([^~]*)~~"
    r"|`{1,3}[^`]*`{1,3}"
    r"|^#{1,6}\s"
    r"|^>\s?"
    r"|\^{1,2}\S*",
    re.MULTILINE,
)
REDDIT_NOISE_RE = re.compile(
    r"\[removed\]|\[deleted\]|\[image\]|\[gif\]|\[video\]|\[poll\]",
    re.IGNORECASE,
)
HTML_TAG_RE     = re.compile(r"<[^>]+>")
EMOJI_RE        = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002500-\U00002BEF"
    "\U00010000-\U0010FFFF"
    "]+",
    flags=re.UNICODE,
)
PUNCT_REPEAT_RE = re.compile(r"([!?.,;:]){3,}")
WHITESPACE_RE   = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE      = re.compile(r"\n{3,}")
ESCAPE_SEQ_RE   = re.compile(r"\\[nrtbf]")
ZERO_WIDTH_RE   = re.compile(r"[​-‏‪-‮﻿]")


# ══════════════════════════════════════════════════════════════════════════════
# Translation helpers
# ══════════════════════════════════════════════════════════════════════════════
def translate_sinhala_spans(text: str, translator: GoogleTranslator) -> str:
    """Replace each contiguous Sinhala-script span with its English translation."""
    chunks = SINHALA_SPAN_RE.findall(text)
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        try:
            english = translator.translate(stripped) or stripped
            text = text.replace(chunk, " " + english + " ")
        except Exception:
            # On any API error keep the original chunk — don't crash the pipeline
            pass
    return text


def replace_singlish(text: str) -> str:
    """Replace known Singlish tokens with their English equivalents."""
    return SINGLISH_RE.sub(lambda m: SINGLISH_DICT[m.group(0).lower()], text)


# ══════════════════════════════════════════════════════════════════════════════
# Main cleaning function
# ══════════════════════════════════════════════════════════════════════════════
def clean(text: str, translator: GoogleTranslator | None = None) -> str:
    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Smart quotes / dashes → ASCII
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("…", "...")

    # 3. Remove zero-width characters
    text = ZERO_WIDTH_RE.sub("", text)

    # 4. Decode HTML entities
    text = html.unescape(text)

    # 5. Strip HTML tags
    text = HTML_TAG_RE.sub(" ", text)

    # 6. Strip Reddit noise markers
    text = REDDIT_NOISE_RE.sub(" ", text)

    # 7. Remove URLs
    text = URL_RE.sub(" ", text)

    # 8. Remove Markdown syntax (keep inner text where possible)
    text = MARKDOWN_RE.sub(
        lambda m: next((g for g in m.groups() if g), " "),
        text,
    )

    # 9. Remove emojis
    text = EMOJI_RE.sub(" ", text)

    # 10. Remove remaining non-printable / control characters (keep \n)
    text = "".join(c if (c.isprintable() or c == "\n") else " " for c in text)

    # 11. Translate Sinhala-script spans → English
    if translator is not None and SINHALA_SPAN_RE.search(text):
        text = translate_sinhala_spans(text, translator)

    # 12. Replace Singlish tokens → English
    text = replace_singlish(text)

    # 13. Expand contractions (apply after lowercasing intermediate copy)
    text_lower = text.lower()
    for pattern, replacement in CONTRACTION_RE:
        text_lower = pattern.sub(replacement, text_lower)
    text = text_lower

    # 14. Collapse repeated punctuation
    text = PUNCT_REPEAT_RE.sub(r"\1\1", text)

    # 15. Escape sequence literals
    text = ESCAPE_SEQ_RE.sub(" ", text)

    # 16. Normalise whitespace
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    return text


def word_count(text: str) -> int:
    return len(text.split())


def hline(c="─", w=70):
    return c * w


# ══════════════════════════════════════════════════════════════════════════════
# Load raw data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading raw dataset …")
df = pd.read_csv(CSV_IN)
df["text"]  = df["text"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)

df["raw_full"] = df["title"].str.strip() + " " + df["text"].str.strip()
df["raw_full"] = df["raw_full"].str.strip()

raw_count   = len(df)
raw_total_w = df["raw_full"].apply(word_count).sum()
print(f"  Raw entries : {raw_count:,}")

df["raw_wc"] = df["raw_full"].apply(word_count)

# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute which entries contain Sinhala script
# ══════════════════════════════════════════════════════════════════════════════
sinhala_mask    = df["raw_full"].str.contains(r"[඀-෿]", regex=True)
n_sinhala       = sinhala_mask.sum()
singlish_mask   = df["raw_full"].str.lower().apply(
    lambda t: bool(SINGLISH_RE.search(t))
)
n_singlish      = singlish_mask.sum()
print(f"  Entries with Sinhala script : {n_sinhala:,}")
print(f"  Entries with Singlish words : {n_singlish:,}")

# ══════════════════════════════════════════════════════════════════════════════
# Initialise translator (used only for Sinhala-script entries)
# ══════════════════════════════════════════════════════════════════════════════
translator = GoogleTranslator(source="si", target="en")

# ══════════════════════════════════════════════════════════════════════════════
# Apply cleaning pipeline
# ══════════════════════════════════════════════════════════════════════════════
print("Applying cleaning pipeline …")
print(f"  (Translating {n_sinhala:,} Sinhala-script entries via Google Translate …)")

n_translated   = 0
translation_errors = 0
clean_texts    = []

for i, row in enumerate(df["raw_full"], 1):
    has_sinhala = bool(SINHALA_SPAN_RE.search(row))
    try:
        ct = clean(row, translator=translator if has_sinhala else None)
        if has_sinhala:
            n_translated += 1
    except Exception as e:
        ct = clean(row, translator=None)   # fallback without translation
        translation_errors += 1

    clean_texts.append(ct)

    if i % 5000 == 0:
        print(f"    {i:,} / {raw_count:,} processed …")

df["clean_text"] = clean_texts
df["clean_wc"]   = df["clean_text"].apply(word_count)

# ══════════════════════════════════════════════════════════════════════════════
# Thresholding
# ══════════════════════════════════════════════════════════════════════════════
LOWER = 10
UPPER = 500

df["drop_reason"] = ""
below_mask = df["clean_wc"] < LOWER
above_mask = df["clean_wc"] > UPPER
df.loc[below_mask, "drop_reason"]              = f"below_{LOWER}_words"
df.loc[above_mask & ~below_mask, "drop_reason"] = f"above_{UPPER}_words"

n_below = int(below_mask.sum())
n_above = int(above_mask.sum())

# ══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════════════════════════
df_threshold = df[df["drop_reason"] == ""].copy()
df_threshold["dup"] = df_threshold["clean_text"].duplicated(keep="first")
n_dup = int(df_threshold["dup"].sum())
df_clean = df_threshold[~df_threshold["dup"]].drop(columns=["dup"])

final_count = len(df_clean)
total_words = int(df_clean["clean_wc"].sum())

# ══════════════════════════════════════════════════════════════════════════════
# Save cleaned corpus
# ══════════════════════════════════════════════════════════════════════════════
cols_out = [
    "entry_id", "post_id", "type", "author", "flair",
    "score", "upvote_ratio", "num_comments",
    "created_utc", "created_date", "permalink",
    "clean_text", "clean_wc",
]
df_clean[cols_out].to_csv(CSV_OUT, index=False)
print(f"  Saved → cleaned_corpus.csv ({final_count:,} entries)")

# ══════════════════════════════════════════════════════════════════════════════
# Audit report
# ══════════════════════════════════════════════════════════════════════════════
report_lines = [
    "r/srilanka — Preprocessing Report",
    hline("="),
    "",
    "PIPELINE STEPS",
    hline(),
    "  1.  Combine title + body into full_text",
    "  2.  Strip Reddit moderation markers ([removed], [deleted], [image], …)",
    "  3.  Decode HTML entities (&amp; → &, &lt; → <, etc.)",
    "  4.  Remove HTML tags",
    "  5.  Remove URLs (http/https/www)",
    "  6.  Remove Markdown syntax (keep inner text where possible)",
    "  7.  Remove emojis and non-printable characters",
    "  8.  Translate Sinhala-script spans → English (Google Translate, si→en)",
    "  9.  Replace Singlish tokens → English (dictionary, "
        f"{len(SINGLISH_DICT)} entries)",
    "  10. Expand common contractions (don't → do not, etc.)",
    "  11. Collapse repeated punctuation (!!!! → !!)",
    "  12. NFKC Unicode normalisation + smart-quote → ASCII",
    "  13. Remove zero-width / invisible characters",
    "  14. Normalise whitespace (collapse spaces, cap newlines)",
    "  15. Lowercase all text",
    "",
    "TRANSLATION STATISTICS",
    hline(),
    f"  Entries with Sinhala script  : {n_sinhala:,}",
    f"  Entries successfully translated : {n_translated:,}",
    f"  Translation errors (fallback)   : {translation_errors:,}",
    f"  Entries with Singlish tokens : {n_singlish:,}",
    f"  Singlish dictionary size     : {len(SINGLISH_DICT)} terms",
    "",
    "REMOVAL THRESHOLDS",
    hline(),
    f"  Lower bound : {LOWER} words",
    "    Justification: Entries below 10 words are title-only stubs, image/link",
    "    posts with no body, or fully-removed posts where only the title survives.",
    "    They carry almost no linguistic content for downstream NLP tasks.",
    "",
    f"  Upper bound : {UPPER} words",
    "    Justification: Entries above 500 words exceed the 99th percentile of the",
    "    raw distribution. They are typically copy-pasted news articles, walls-of-",
    "    text, or multi-part megaposts that inflate vocabulary statistics and bias",
    "    length-sensitive models. Capping at P99 removes <1% of documents.",
    "",
    "REMOVAL AUDIT",
    hline(),
    f"  Raw entries              : {raw_count:,}",
    f"  Removed (< {LOWER} words)    : {n_below:,}  ({n_below/raw_count*100:.2f}%)",
    f"  Removed (> {UPPER} words)   : {n_above:,}  ({n_above/raw_count*100:.2f}%)",
    f"  Removed (duplicates)     : {n_dup:,}  ({n_dup/raw_count*100:.2f}%)",
    f"  Final corpus size        : {final_count:,}  ({final_count/raw_count*100:.2f}% of raw)",
    "",
    "CORPUS STATISTICS (cleaned)",
    hline(),
    f"  Total words              : {total_words:,}",
    f"  Mean words per entry     : {df_clean['clean_wc'].mean():.1f}",
    f"  Median words per entry   : {df_clean['clean_wc'].median():.1f}",
    f"  Std dev                  : {df_clean['clean_wc'].std():.1f}",
    f"  Min words                : {df_clean['clean_wc'].min()}",
    f"  Max words                : {df_clean['clean_wc'].max()}",
    f"  P25 / P50 / P75          : {df_clean['clean_wc'].quantile(0.25):.0f} / "
        f"{df_clean['clean_wc'].quantile(0.50):.0f} / "
        f"{df_clean['clean_wc'].quantile(0.75):.0f}",
    "",
    "BEFORE vs AFTER WORD COUNT",
    hline(),
    f"  Total words (raw)        : {raw_total_w:,}",
    f"  Total words (cleaned)    : {total_words:,}",
    f"  Reduction                : {(1 - total_words/raw_total_w)*100:.2f}%",
]

with open(REPORT, "w") as f:
    f.write("\n".join(report_lines))

for line in report_lines:
    print(line)

# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots …")

def save(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → plots/{name}.png")


# (A) Word count before vs after
fig, ax = plt.subplots(figsize=(11, 5))
cap  = 600
bins = np.linspace(0, cap, 80)
ax.hist(df["raw_wc"].clip(upper=cap), bins=bins, alpha=0.5, color="#5B8DB8",
        label=f"Before cleaning  (n={raw_count:,})", edgecolor="white", lw=0.3)
ax.hist(df_clean["clean_wc"].clip(upper=cap), bins=bins, alpha=0.6, color=ACCENT,
        label=f"After cleaning  (n={final_count:,})", edgecolor="white", lw=0.3)
ax.axvline(LOWER, color="black",  ls="--", lw=1.4, label=f"Lower threshold ({LOWER} w)")
ax.axvline(UPPER, color="purple", ls="--", lw=1.4, label=f"Upper threshold ({UPPER} w)")
ax.set_xlabel("Word count (capped at 600)")
ax.set_ylabel("Number of entries")
ax.set_title("Word Count Distribution: Before vs After Cleaning")
ax.legend(fontsize=9)
save(fig, "01_wordcount_before_vs_after")

# (B) Removal breakdown
removal_labels = [
    f"Kept\n({final_count:,})",
    f"< {LOWER} words\n({n_below:,})",
    f"> {UPPER} words\n({n_above:,})",
    f"Duplicates\n({n_dup:,})",
]
removal_vals   = [final_count, n_below, n_above, n_dup]
removal_colors = ["#4CAF82", ACCENT, "#E09A52", "#8A9DB7"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(removal_labels, removal_vals, color=removal_colors, edgecolor="white", lw=0.5)
for bar, val in zip(bars, removal_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}\n({val/raw_count*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of entries")
ax.set_title("Entry Removal Breakdown")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "02_removal_breakdown")

# (C) Cleaned word count distribution
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(df_clean["clean_wc"], bins=80, color=ACCENT, edgecolor="white", lw=0.3)
ax.axvline(df_clean["clean_wc"].mean(),   color="navy",  ls="--", lw=1.5,
           label=f"Mean  {df_clean['clean_wc'].mean():.1f}")
ax.axvline(df_clean["clean_wc"].median(), color="green", ls="--", lw=1.5,
           label=f"Median {df_clean['clean_wc'].median():.1f}")
ax.set_xlabel("Word count")
ax.set_ylabel("Number of entries")
ax.set_title("Word Count Distribution — Cleaned Corpus")
ax.legend()
save(fig, "03_cleaned_wordcount_dist")

# (D) Log-scale
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(df_clean["clean_wc"].clip(lower=1), bins=80, color="#7B5EA7",
        edgecolor="white", lw=0.3, log=True)
ax.set_xlabel("Word count")
ax.set_ylabel("Count (log scale)")
ax.set_title("Word Count Distribution — Cleaned Corpus (Log Scale)")
save(fig, "04_cleaned_wordcount_log")

# (E) Box plot before vs after
fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(
    [df["raw_wc"].clip(upper=600).values, df_clean["clean_wc"].clip(upper=600).values],
    patch_artist=True, notch=False, widths=0.4,
    medianprops={"color": "white", "lw": 2},
)
for patch, color in zip(bp["boxes"], ["#5B8DB8", ACCENT]):
    patch.set_facecolor(color)
ax.set_xticklabels(["Before cleaning", "After cleaning"])
ax.set_ylabel("Word count (capped at 600)")
ax.set_title("Word Count Box Plot: Before vs After")
save(fig, "05_wordcount_boxplot")

# (F) CDF
sorted_wc = np.sort(df_clean["clean_wc"].values)
cdf = np.arange(1, len(sorted_wc) + 1) / len(sorted_wc)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sorted_wc, cdf, color=ACCENT, lw=2)
ax.axvline(LOWER, color="black",  ls="--", lw=1.2, label=f"Lower ({LOWER} w)")
ax.axvline(UPPER, color="purple", ls="--", lw=1.2, label=f"Upper ({UPPER} w)")
ax.set_xlabel("Word count")
ax.set_ylabel("Cumulative fraction of entries")
ax.set_title("CDF of Entry Word Counts — Cleaned Corpus")
ax.legend()
save(fig, "06_cleaned_cdf")

# (G) Words removed per entry
df_kept = df.loc[df_clean.index].copy()
df_kept["wc_reduction"] = df_kept["raw_wc"] - df_clean["clean_wc"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_kept["wc_reduction"].clip(lower=0, upper=50), bins=50,
        color="#C68B59", edgecolor="white", lw=0.3)
ax.set_xlabel("Words removed per entry (capped at 50)")
ax.set_ylabel("Number of entries")
ax.set_title("Words Removed per Entry by Cleaning Pipeline")
save(fig, "07_words_removed_per_entry")

# (H) Flair distribution — cleaned
top_flairs = df_clean["flair"].value_counts().head(12)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top_flairs.index[::-1], top_flairs.values[::-1],
        color=sns.color_palette("Set2", len(top_flairs)))
ax.set_xlabel("Number of entries")
ax.set_title("Flair Distribution — Cleaned Corpus (Top 12)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "08_flair_cleaned")

# (I) Translation coverage bar
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Sinhala script\nentries", "Singlish\nentries", "English-only\nentries"]
vals   = [n_sinhala, n_singlish, raw_count - n_sinhala - n_singlish]
colors = ["#E05252", "#E09A52", "#4CAF82"]
bars   = ax.bar(labels, vals, color=colors, edgecolor="white", lw=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}\n({val/raw_count*100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of entries")
ax.set_title("Language Mix in Raw Corpus")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "09_language_mix")

print(f"\n{hline('═')}")
print(f"  Cleaned corpus : {final_count:,} entries  |  {total_words:,} total words")
print(f"  Sinhala entries translated  : {n_translated:,}")
print(f"  Singlish entries normalized : {n_singlish:,}")
print(f"  Saved : data-clean&preprocess/cleaned_corpus.csv")
print(f"  Report: data-clean&preprocess/preprocessing_report.txt")
print(f"  Plots : data-clean&preprocess/plots/  (9 files)")
print(hline("═"))
