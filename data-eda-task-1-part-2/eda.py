"""
Exploratory Data Analysis — r/srilanka dataset (entries.csv)
Outputs:
  - Console summary report
  - plots/ directory with all visualizations
  - eda_report.txt with full text statistics
"""

import ssl
import os
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

import nltk

# Fix macOS SSL for NLTK downloads
try:
    _ctx = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _ctx

for pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "entries.csv"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = Path(__file__).parent / "eda_report.txt"

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT = "#E05252"   # reddit-ish red for highlights

# ── Helpers ───────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
REDDIT_NOISE = {
    "https", "http", "www", "reddit", "com", "r", "u", "amp",
    "removed", "deleted", "edit", "gt", "lt", "nbsp",
}
STOP_WORDS |= REDDIT_NOISE


def word_count(text: str) -> int:
    return len(text.split())


def char_count(text: str) -> int:
    return len(text)


def sentence_count(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+", text.strip())))


def clean_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = word_tokenize(text)
    return [
        t for t in tokens
        if t.isalpha() and t not in STOP_WORDS and len(t) > 2
    ]


def save(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → plots/{name}.png")


def hline(char="─", width=70):
    return char * width


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & basic cleaning
# ══════════════════════════════════════════════════════════════════════════════
print(hline("═"))
print("  r/srilanka EDA")
print(hline("═"))
print("Loading dataset …")

df = pd.read_csv(CSV_PATH)

# Coerce types
df["score"]        = pd.to_numeric(df["score"],        errors="coerce")
df["upvote_ratio"] = pd.to_numeric(df["upvote_ratio"], errors="coerce")
df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
df["created_utc"]  = pd.to_numeric(df["created_utc"],  errors="coerce")
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", utc=True)

df["text"]  = df["text"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)

# Combined text field (title + body)
df["full_text"] = df["title"].str.strip() + " " + df["text"].str.strip()
df["full_text"] = df["full_text"].str.strip()

# Flag removed / deleted entries
df["is_removed"] = df["text"].str.strip().isin(["[removed]", "[deleted]", ""])

# ── Temporal features
df["year"]  = df["created_date"].dt.year
df["month"] = df["created_date"].dt.month
df["dow"]   = df["created_date"].dt.day_of_week   # 0 = Monday
df["hour"]  = df["created_date"].dt.hour
df["ym"]    = df["created_date"].dt.tz_localize(None).dt.to_period("M")

# ── Length features  (on full_text)
df["word_count"]     = df["full_text"].apply(word_count)
df["char_count"]     = df["full_text"].apply(char_count)
df["sentence_count"] = df["full_text"].apply(sentence_count)
df["avg_word_len"]   = df["char_count"] / df["word_count"].replace(0, np.nan)

print(f"  Rows loaded : {len(df):,}")
print(f"  Columns     : {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Dimension summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + hline())
print("DATASET DIMENSIONS")
print(hline())

dim_lines = [
    f"Total entries       : {len(df):,}",
    f"Unique post IDs     : {df['post_id'].nunique():,}",
    f"Unique authors      : {df['author'].nunique():,}",
    f"Removed entries     : {df['is_removed'].sum():,}  ({df['is_removed'].mean()*100:.1f}%)",
    f"Date range          : {df['created_date'].min().date()} → {df['created_date'].max().date()}",
    f"Distinct flairs     : {df['flair'].nunique()}",
]
for l in dim_lines:
    print("  " + l)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Length statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + hline())
print("ENTRY LENGTH STATISTICS (words)")
print(hline())

wc = df["word_count"]
length_stats = {
    "Min"     : wc.min(),
    "Max"     : wc.max(),
    "Mean"    : wc.mean(),
    "Median"  : wc.median(),
    "Std Dev" : wc.std(),
    "P25"     : wc.quantile(0.25),
    "P75"     : wc.quantile(0.75),
    "P90"     : wc.quantile(0.90),
    "P99"     : wc.quantile(0.99),
}
for k, v in length_stats.items():
    print(f"  {k:<10}: {v:>8.1f}")

print("\nENTRY LENGTH STATISTICS (characters)")
cc = df["char_count"]
for k, fn in [("Min", cc.min), ("Max", cc.max), ("Mean", cc.mean),
              ("Median", cc.median), ("Std Dev", cc.std)]:
    print(f"  {k:<10}: {fn():>10.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Engagement statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + hline())
print("ENGAGEMENT STATISTICS")
print(hline())

for col in ["score", "upvote_ratio", "num_comments"]:
    s = df[col]
    print(f"  {col:<15}: mean={s.mean():.2f}  median={s.median():.2f}  max={s.max():.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Top authors & flairs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + hline())
print("TOP 10 AUTHORS (by post count)")
print(hline())
top_authors = df["author"].value_counts().head(10)
for author, cnt in top_authors.items():
    print(f"  {author:<30} {cnt:>5}")

print("\nTOP 15 FLAIRS")
print(hline())
top_flairs = df["flair"].value_counts().head(15)
for flair, cnt in top_flairs.items():
    print(f"  {flair:<35} {cnt:>6}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Word frequency
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding word frequency (this may take a moment) …")
all_tokens: list[str] = []
for text in df["full_text"]:
    all_tokens.extend(clean_tokens(text))

freq = Counter(all_tokens)
print(f"  Unique clean tokens : {len(freq):,}")
print(f"  Total clean tokens  : {sum(freq.values()):,}")

print("\nTOP 30 WORDS")
print(hline())
for word, cnt in freq.most_common(30):
    bar = "█" * (cnt * 40 // freq.most_common(1)[0][1])
    print(f"  {word:<20} {cnt:>7,}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Write text report
# ══════════════════════════════════════════════════════════════════════════════
with open(REPORT_PATH, "w") as f:
    f.write("r/srilanka EDA Report\n")
    f.write(hline("=") + "\n\n")
    f.write("DATASET DIMENSIONS\n" + hline() + "\n")
    for l in dim_lines:
        f.write(l + "\n")

    f.write("\nLENGTH STATISTICS (words)\n" + hline() + "\n")
    for k, v in length_stats.items():
        f.write(f"  {k:<10}: {v:>8.1f}\n")

    f.write("\nTOP 15 FLAIRS\n" + hline() + "\n")
    for flair, cnt in top_flairs.items():
        f.write(f"  {flair:<35} {cnt:>6}\n")

    f.write("\nTOP 30 WORDS\n" + hline() + "\n")
    for word, cnt in freq.most_common(30):
        f.write(f"  {word:<20} {cnt:>7,}\n")

print(f"\n  Report → eda_report.txt")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots …")

# ── (A) Word-count histogram ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cap = int(np.percentile(df["word_count"], 98))
ax.hist(df["word_count"].clip(upper=cap), bins=80, color=ACCENT, edgecolor="white", linewidth=0.4)
ax.axvline(df["word_count"].mean(),   color="navy",  ls="--", lw=1.5, label=f"Mean  {df['word_count'].mean():.1f}")
ax.axvline(df["word_count"].median(), color="green", ls="--", lw=1.5, label=f"Median {df['word_count'].median():.1f}")
ax.set_xlabel("Word count (capped at 98th percentile)")
ax.set_ylabel("Number of entries")
ax.set_title("Distribution of Entry Lengths (Word Count)")
ax.legend()
save(fig, "01_word_count_histogram")

# ── (B) Character-count histogram ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cap_c = int(np.percentile(df["char_count"], 98))
ax.hist(df["char_count"].clip(upper=cap_c), bins=80, color="#5B8DB8", edgecolor="white", linewidth=0.4)
ax.axvline(df["char_count"].mean(),   color="navy",  ls="--", lw=1.5, label=f"Mean  {df['char_count'].mean():.0f}")
ax.axvline(df["char_count"].median(), color="green", ls="--", lw=1.5, label=f"Median {df['char_count'].median():.0f}")
ax.set_xlabel("Character count (capped at 98th percentile)")
ax.set_ylabel("Number of entries")
ax.set_title("Distribution of Entry Lengths (Character Count)")
ax.legend()
save(fig, "02_char_count_histogram")

# ── (C) Log-scale word-count distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["word_count"].clip(lower=1), bins=100, color="#7B5EA7", edgecolor="white", linewidth=0.3, log=True)
ax.set_xlabel("Word count")
ax.set_ylabel("Number of entries (log scale)")
ax.set_title("Word Count Distribution (Log Scale)")
save(fig, "03_word_count_log")

# ── (D) Top-20 word frequencies ──────────────────────────────────────────────
top20 = pd.DataFrame(freq.most_common(20), columns=["word", "count"])
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(top20["word"][::-1], top20["count"][::-1], color=sns.color_palette("muted", 20))
ax.set_xlabel("Frequency")
ax.set_title("Top 20 Most Frequent Words (stop-words removed)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, top20["count"][::-1]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
save(fig, "04_top20_words")

# ── (E) Top-50 word frequencies ──────────────────────────────────────────────
top50 = pd.DataFrame(freq.most_common(50), columns=["word", "count"])
fig, ax = plt.subplots(figsize=(12, 10))
colors = sns.color_palette("tab20", 50)
ax.barh(top50["word"][::-1], top50["count"][::-1], color=colors)
ax.set_xlabel("Frequency")
ax.set_title("Top 50 Most Frequent Words")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.tick_params(axis="y", labelsize=8)
save(fig, "05_top50_words")

# ── (F) Word cloud ────────────────────────────────────────────────────────────
wc_gen = WordCloud(
    width=1400, height=700, background_color="white",
    colormap="RdYlBu", max_words=200,
    collocations=False,
).generate_from_frequencies(freq)
fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wc_gen, interpolation="bilinear")
ax.axis("off")
ax.set_title("Word Cloud — r/srilanka (stop-words removed)", fontsize=16)
save(fig, "06_wordcloud")

# ── (G) Flair distribution ───────────────────────────────────────────────────
top_n_flairs = df["flair"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(11, 6))
colors = sns.color_palette("Set2", len(top_n_flairs))
ax.barh(top_n_flairs.index[::-1], top_n_flairs.values[::-1], color=colors)
ax.set_xlabel("Number of posts")
ax.set_title("Top 15 Post Flairs")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for i, (val, name) in enumerate(zip(top_n_flairs.values[::-1], top_n_flairs.index[::-1])):
    ax.text(val + 50, i, f"{val:,}", va="center", fontsize=9)
save(fig, "07_flair_distribution")

# ── (H) Flair pie chart ───────────────────────────────────────────────────────
others = df["flair"].value_counts().iloc[10:].sum()
pie_data = df["flair"].value_counts().head(10)
pie_labels = list(pie_data.index) + ["Other"]
pie_vals   = list(pie_data.values) + [others]
fig, ax = plt.subplots(figsize=(9, 9))
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_labels, autopct="%1.1f%%",
    colors=sns.color_palette("pastel", len(pie_vals)),
    startangle=140, pctdistance=0.82,
)
for t in autotexts:
    t.set_fontsize(8)
ax.set_title("Flair Distribution (top 10 + Other)", fontsize=13)
save(fig, "08_flair_pie")

# ── (I) Posts over time (monthly) ────────────────────────────────────────────
monthly = df.groupby("ym").size().reset_index(name="count")
monthly["ym_dt"] = monthly["ym"].dt.to_timestamp()
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(monthly["ym_dt"], monthly["count"], alpha=0.3, color="#E05252")
ax.plot(monthly["ym_dt"], monthly["count"], color="#E05252", lw=2)
ax.set_xlabel("Month")
ax.set_ylabel("Number of posts")
ax.set_title("Posts Per Month Over Time")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.autofmt_xdate()
save(fig, "09_posts_per_month")

# ── (J) Posts by day of week ─────────────────────────────────────────────────
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_counts = df["dow"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(dow_labels, dow_counts.values, color=sns.color_palette("Set3", 7))
ax.set_xlabel("Day of Week")
ax.set_ylabel("Number of posts")
ax.set_title("Post Volume by Day of Week")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "10_posts_by_dow")

# ── (K) Posts by hour of day ─────────────────────────────────────────────────
hour_counts = df["hour"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(hour_counts.index, hour_counts.values, color="#5B8DB8")
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("Number of posts")
ax.set_title("Post Volume by Hour of Day (UTC)")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "11_posts_by_hour")

# ── (L) Score distribution (log) ─────────────────────────────────────────────
positive_scores = df[df["score"] > 0]["score"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(positive_scores.clip(upper=positive_scores.quantile(0.99)), bins=80,
        color="#7B5EA7", edgecolor="white", linewidth=0.3, log=True)
ax.set_xlabel("Score (capped at 99th percentile)")
ax.set_ylabel("Count (log scale)")
ax.set_title("Distribution of Post Scores (log scale, positive only)")
save(fig, "12_score_distribution")

# ── (M) Upvote ratio distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["upvote_ratio"].dropna(), bins=50, color="#4CAF82", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Upvote Ratio")
ax.set_ylabel("Number of posts")
ax.set_title("Distribution of Upvote Ratios")
save(fig, "13_upvote_ratio")

# ── (N) Comments distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
nc = df["num_comments"].clip(upper=df["num_comments"].quantile(0.99))
ax.hist(nc, bins=60, color="#E09A52", edgecolor="white", linewidth=0.3, log=True)
ax.set_xlabel("Number of comments (capped at 99th pct)")
ax.set_ylabel("Count (log scale)")
ax.set_title("Distribution of Comment Counts per Post")
save(fig, "14_comment_count_distribution")

# ── (O) Score vs word count scatter ──────────────────────────────────────────
sample = df[(df["score"] > 0) & (df["word_count"] > 0)].sample(
    min(5000, len(df)), random_state=42
)
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(np.log1p(sample["word_count"]), np.log1p(sample["score"]),
           alpha=0.25, s=10, color=ACCENT)
ax.set_xlabel("log(1 + word count)")
ax.set_ylabel("log(1 + score)")
ax.set_title("Score vs. Entry Length (log-log, 5k sample)")
save(fig, "15_score_vs_length")

# ── (P) Box plots — word count by flair (top 8) ───────────────────────────────
top8_flairs = df["flair"].value_counts().head(8).index
flair_df = df[df["flair"].isin(top8_flairs)]
fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(
    data=flair_df, x="flair", y="word_count",
    order=top8_flairs,
    hue="flair", palette="Set2", legend=False,
    showfliers=False, ax=ax,
)
cap_box = int(np.percentile(flair_df["word_count"], 95))
ax.set_ylim(0, cap_box)
ax.set_xlabel("Flair")
ax.set_ylabel("Word count")
ax.set_title("Entry Length Distribution by Top-8 Flairs (outliers hidden)")
ax.tick_params(axis="x", rotation=30)
save(fig, "16_wordcount_by_flair")

# ── (Q) Removed / valid ratio ────────────────────────────────────────────────
removed_counts = df["is_removed"].value_counts()
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Valid", "Removed/Empty"],
       [removed_counts.get(False, 0), removed_counts.get(True, 0)],
       color=["#4CAF82", ACCENT])
ax.set_ylabel("Number of entries")
ax.set_title("Valid vs Removed/Empty Entries")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "17_removed_vs_valid")

# ── (R) Heatmap: posts by day-of-week × hour ─────────────────────────────────
pivot = df.groupby(["dow", "hour"]).size().unstack(fill_value=0)
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Post count"})
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("Day of Week")
ax.set_title("Activity Heatmap: Day-of-Week × Hour (UTC)")
save(fig, "18_activity_heatmap")

# ── (S) Sentence count histogram ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cap_s = int(np.percentile(df["sentence_count"], 97))
ax.hist(df["sentence_count"].clip(upper=cap_s), bins=60,
        color="#8A9DB7", edgecolor="white", linewidth=0.4)
ax.axvline(df["sentence_count"].mean(), color="navy", ls="--", lw=1.5,
           label=f"Mean {df['sentence_count'].mean():.1f}")
ax.set_xlabel("Sentence count (capped at 97th percentile)")
ax.set_ylabel("Number of entries")
ax.set_title("Distribution of Sentence Counts per Entry")
ax.legend()
save(fig, "19_sentence_count")

# ── (T) Average word length distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
awl = df["avg_word_len"].dropna()
awl_cap = awl.clip(upper=awl.quantile(0.99))
ax.hist(awl_cap, bins=60, color="#C68B59", edgecolor="white", linewidth=0.4)
ax.axvline(awl.mean(), color="navy", ls="--", lw=1.5, label=f"Mean {awl.mean():.2f}")
ax.set_xlabel("Average word length (characters)")
ax.set_ylabel("Number of entries")
ax.set_title("Distribution of Average Word Length per Entry")
ax.legend()
save(fig, "20_avg_word_length")

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + hline("═"))
print(f"  All plots saved in  → data-eda/plots/  ({len(list(OUT_DIR.glob('*.png')))} files)")
print(f"  Text report saved   → data-eda/eda_report.txt")
print(hline("═"))
