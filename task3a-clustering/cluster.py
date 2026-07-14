# Must be first — forces 'fork' start method so gensim's internal
# multiprocessing (CoherenceModel, LdaModel workers) works on macOS Python 3.13
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

"""
Task 3(a) — Post Categorisation via Clustering & Topic Modelling
=================================================================
Input  : ../data-clean&preprocess-task-2-part-1/cleaned_corpus.csv
Output : labeled_corpus.csv      — corpus with silver-standard cluster labels
         clustering_report.txt   — full findings report
         plots/                  — all visualisations

Two complementary unsupervised approaches are used:

  A. K-Means on TF-IDF + SVD
     - TF-IDF vectorises the text (50k features, sublinear TF)
     - TruncatedSVD reduces to 100 latent dimensions (LSA)
     - K-Means run for k = 5..20; best k chosen by silhouette score
     - This is the primary method that produces the final silver labels

  B. LDA (Latent Dirichlet Allocation) — Topic Modelling
     - Run for n_topics = 5..20; best chosen by coherence (c_v)
     - Used as cross-validation of the theme structure found by K-Means

The cluster with the most coherent top-terms is assigned a human-readable
topic label. These become the 'silver standard' classes used in Tasks 3b/3c.
"""

import ssl
import warnings
warnings.filterwarnings("ignore")

import ssl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

import nltk
ssl._create_default_https_context = ssl._create_unverified_context
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet",
            "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import Normalizer

import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
from gensim.utils import simple_preprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CORPUS_CSV = ROOT / "data-clean&preprocess-task-2-part-1" / "cleaned_corpus.csv"
OUT_DIR    = Path(__file__).parent
PLOTS_DIR  = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT  = "#E05252"
PALETTE = ["#E05252","#5B8DB8","#7B5EA7","#4CAF82","#E09A52",
           "#C68B59","#8A9DB7","#E8C45A","#6DBFBF","#D4748C",
           "#A0C878","#B07DBF"]

RANDOM_SEED   = 42
N_SVD_DIMS    = 100       # LSA latent dims for K-Means
K_RANGE       = range(5, 21)   # k values to evaluate
LDA_RANGE     = range(5, 21)   # n_topics to evaluate
STOP_EN       = set(stopwords.words("english"))
lemmatizer    = WordNetLemmatizer()

def hline(c="─", w=72):
    return c * w

def save_plot(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → plots/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load corpus
# ══════════════════════════════════════════════════════════════════════════════
print("Loading corpus …")
df = pd.read_csv(CORPUS_CSV)
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["flair"]      = df["flair"].fillna("Unknown").astype(str)
texts = df["clean_text"].tolist()
print(f"  {len(df):,} entries loaded")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Pre-process for LDA (tokenise + remove stopwords)
# ══════════════════════════════════════════════════════════════════════════════
print("Tokenising for LDA …")
def lda_preprocess(text: str) -> list[str]:
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    tokens = [t for t in tokens if t not in STOP_EN]
    return [lemmatizer.lemmatize(t) for t in tokens]

lda_tokens = [lda_preprocess(t) for t in texts]

# Gensim dictionary + corpus
id2word  = corpora.Dictionary(lda_tokens)
id2word.filter_extremes(no_below=5, no_above=0.6)   # drop very rare/common
bow_corpus = [id2word.doc2bow(doc) for doc in lda_tokens]

# ══════════════════════════════════════════════════════════════════════════════
# 3. TF-IDF + SVD (for K-Means)
# ══════════════════════════════════════════════════════════════════════════════
print("Building TF-IDF + SVD (LSA) representation …")

tfidf = TfidfVectorizer(
    max_features=50_000,
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
    ngram_range=(1, 2),
    stop_words="english",
)
X_tfidf = tfidf.fit_transform(texts)
print(f"  TF-IDF matrix : {X_tfidf.shape}")

svd = TruncatedSVD(n_components=N_SVD_DIMS, random_state=RANDOM_SEED)
X_svd = svd.fit_transform(X_tfidf)
X_norm = Normalizer(copy=False).fit_transform(X_svd)
print(f"  SVD explained variance : {svd.explained_variance_ratio_.sum():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 4A. K-Means — sweep k, collect silhouette + inertia
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nK-Means sweep k={K_RANGE.start}..{K_RANGE.stop-1} …")

inertias    = []
silhouettes = []
db_scores   = []

for k in K_RANGE:
    km = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_SEED, batch_size=4096,
        n_init=10, max_iter=300,
    )
    labels = km.fit_predict(X_norm)
    # Sample 8k docs for silhouette (full is too slow)
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(
        len(X_norm), min(8000, len(X_norm)), replace=False
    )
    sil = silhouette_score(X_norm[sample_idx], labels[sample_idx], metric="cosine")
    db  = davies_bouldin_score(X_norm[sample_idx], labels[sample_idx])
    inertias.append(km.inertia_)
    silhouettes.append(sil)
    db_scores.append(db)
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}  DB={db:.4f}")

best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
print(f"\n  → Best k by silhouette : {best_k}  (score={max(silhouettes):.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 4B. LDA — sweep n_topics, collect coherence
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLDA sweep n_topics={LDA_RANGE.start}..{LDA_RANGE.stop-1} …")

coherences = []
for n in LDA_RANGE:
    lda_m = LdaModel(
        corpus=bow_corpus, id2word=id2word,
        num_topics=n,
        passes=5, random_state=RANDOM_SEED,
    )
    cm = CoherenceModel(
        model=lda_m, texts=lda_tokens,
        dictionary=id2word, coherence="c_v",
    )
    coh = cm.get_coherence()
    coherences.append(coh)
    print(f"  n={n:2d}  coherence(c_v)={coh:.4f}")

best_n_lda = list(LDA_RANGE)[int(np.argmax(coherences))]
print(f"\n  → Best n_topics by coherence : {best_n_lda}  (c_v={max(coherences):.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Final K-Means with best_k
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFitting final K-Means with k={best_k} …")
km_final = MiniBatchKMeans(
    n_clusters=best_k, random_state=RANDOM_SEED,
    batch_size=4096, n_init=15, max_iter=500,
)
km_labels = km_final.fit_predict(X_norm)
df["kmeans_cluster"] = km_labels

# ══════════════════════════════════════════════════════════════════════════════
# 6. Final LDA with best_n_lda
# ══════════════════════════════════════════════════════════════════════════════
print(f"Fitting final LDA with n_topics={best_n_lda} …")
lda_final = LdaModel(
    corpus=bow_corpus, id2word=id2word,
    num_topics=best_n_lda,
    passes=10, random_state=RANDOM_SEED,
)
# Assign dominant topic per document
lda_topic_per_doc = []
for bow in bow_corpus:
    topic_dist = lda_final.get_document_topics(bow, minimum_probability=0)
    dominant   = max(topic_dist, key=lambda x: x[1])[0]
    lda_topic_per_doc.append(dominant)
df["lda_topic"] = lda_topic_per_doc

# ══════════════════════════════════════════════════════════════════════════════
# 7. Name clusters using top TF-IDF terms per K-Means cluster
# ══════════════════════════════════════════════════════════════════════════════
feature_names = tfidf.get_feature_names_out()

def top_terms_for_cluster(cluster_id, n=15):
    """Mean TF-IDF of all docs in the cluster → top n terms."""
    mask    = km_labels == cluster_id
    mean_v  = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
    top_idx = mean_v.argsort()[::-1][:n]
    return [feature_names[i] for i in top_idx]

cluster_top_terms = {k: top_terms_for_cluster(k) for k in range(best_k)}

# ── Explicit per-cluster labels (based on top-term inspection, k=17) ─────────
# Derived by reading the actual top TF-IDF terms per cluster rather than
# relying on a keyword heuristic that repeatedly mis-assigned "Health & Medical".
EXPLICIT_CLUSTER_LABELS = {
    0:  "Shopping & Colombo Recommendations",  # buy, place buy, colombo
    1:  "Relationships & Social Life",          # like, feel, people, did
    2:  "Employment & Tech Jobs",               # advice, job, work, software, company
    3:  "General Q&A",                          # does know, know, mean, place (generic)
    4:  "Transport & Vehicles",                 # new, car, vehicle, license, driving
    5:  "Title-Only / Generic Posts",           # title says, basically title (no body)
    6:  "Local Recommendations",                # good, recommend, place, good place
    7:  "General Sri Lanka Discussion",         # lanka, sri lanka, sri, know, like
    8:  "Help Requests",                        # help, need, need help, sinhala, guys
    9:  "Politics & National Affairs",          # lankans, country, election, think
    10: "Miscellaneous Discussion",             # sl, like, know, best, does
    11: "Sri Lanka Identity & Culture",         # lankan, sri lankan, sri, lanka
    12: "Travel & Tourism (Colombo area)",      # colombo, looking, place, places, good
    13: "Sri Lanka Expat & Identity",           # srilanka, srilankan, know, best
    14: "Education & University",               # degree, university, science, doing, uni
    15: "Finance, Banking & Telecoms",          # dialog, slt, bank, visa, card
    16: "Travel & Tourism (Upcountry)",         # kandy, ella, looking, travel, places
}

# ── 10 final clean categories: merge overlapping raw clusters ────────────────
# Rationale:
#  - Clusters 3, 5, 8, 10  → all generic/help content with no strong theme
#  - Clusters 7, 11, 13    → all broad "Sri Lanka" identity/culture discussion
#  - Clusters 0, 6         → both about recommending or finding things locally
#  - Clusters 12, 16       → both about travel destinations inside Sri Lanka
CLUSTER_TO_FINAL = {
    0:  "Shopping & Local Recommendations",
    1:  "Relationships & Social Life",
    2:  "Employment & Career",
    3:  "Help & General Q&A",
    4:  "Transport & Vehicles",
    5:  "Help & General Q&A",
    6:  "Shopping & Local Recommendations",
    7:  "Sri Lanka Identity & Culture",
    8:  "Help & General Q&A",
    9:  "Politics & National Affairs",
    10: "Help & General Q&A",
    11: "Sri Lanka Identity & Culture",
    12: "Travel & Tourism",
    13: "Sri Lanka Identity & Culture",
    14: "Education",
    15: "Finance, Banking & Telecoms",
    16: "Travel & Tourism",
}

FINAL_10_CATEGORIES = sorted(set(CLUSTER_TO_FINAL.values()))

cluster_labels = EXPLICIT_CLUSTER_LABELS.copy()

df["topic_label"]    = df["kmeans_cluster"].map(cluster_labels)
df["final_category"] = df["kmeans_cluster"].map(CLUSTER_TO_FINAL)

print("\nCluster → Topic Label mapping:")
for k in sorted(cluster_labels):
    n    = (df["kmeans_cluster"] == k).sum()
    top5 = ", ".join(cluster_top_terms[k][:5])
    print(f"  Cluster {k:2d} → {cluster_labels[k]:<30} n={n:>5,}  top: {top5}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Save labeled corpus
# ══════════════════════════════════════════════════════════════════════════════
cols_out = [
    "entry_id","post_id","author","flair","score","upvote_ratio",
    "num_comments","created_date","permalink",
    "clean_text","clean_wc","kmeans_cluster","lda_topic",
    "topic_label","final_category",
]
df[cols_out].to_csv(OUT_DIR / "labeled_corpus.csv", index=False)
print(f"\n  Saved → labeled_corpus.csv  ({len(df):,} entries, {best_k} clusters, 10 final categories)")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Report
# ══════════════════════════════════════════════════════════════════════════════
cluster_sizes  = df["topic_label"].value_counts()
final_sizes    = df["final_category"].value_counts()
lda_top_words  = {i: [w for w, _ in lda_final.show_topic(i, topn=10)]
                  for i in range(best_n_lda)}

report_lines = [
    "Task 3(a) — Clustering & Topic Modelling Report",
    hline("="),
    "",
    "METHOD A — K-Means on TF-IDF + LSA (primary, produces silver labels)",
    hline(),
    f"  Vectoriser   : TF-IDF  (max_features=50,000  ngram=(1,2)  sublinear_tf)",
    f"  Dimensionality reduction : TruncatedSVD ({N_SVD_DIMS} dims, LSA)",
    f"  Explained variance after SVD : {svd.explained_variance_ratio_.sum():.1%}",
    f"  k range evaluated : {K_RANGE.start} – {K_RANGE.stop-1}",
    f"  Metric used       : Silhouette score (cosine, 8k sample)",
    f"  Best k            : {best_k}  (silhouette={max(silhouettes):.4f})",
    "",
    "METHOD B — LDA Topic Modelling (cross-validation)",
    hline(),
    f"  Library         : Gensim LdaModel",
    f"  n_topics range  : {LDA_RANGE.start} – {LDA_RANGE.stop-1}",
    f"  Metric used     : Coherence score (c_v)",
    f"  Best n_topics   : {best_n_lda}  (coherence={max(coherences):.4f})",
    "",
    "RAW CLUSTER ASSIGNMENTS  (K-Means, k=" + str(best_k) + ")",
    hline(),
    f"  {'Cluster':<5} {'Raw Label':<38} {'Final Category':<35} {'Size':>7}",
    hline(),
]
for k in sorted(cluster_labels):
    n = int((df["kmeans_cluster"] == k).sum())
    report_lines.append(
        f"  {k:<5} {cluster_labels[k]:<38} {CLUSTER_TO_FINAL[k]:<35} {n:>7,}"
    )

report_lines += [
    "",
    "10 FINAL SILVER-STANDARD CATEGORIES",
    hline(),
    f"  {'Category':<40} {'Posts':>8}  {'% of corpus':>12}",
    hline(),
]
for cat, cnt in final_sizes.items():
    report_lines.append(
        f"  {cat:<40} {cnt:>8,}  {cnt/len(df)*100:>11.1f}%"
    )

report_lines += [
    "",
    "LDA TOP WORDS PER TOPIC  (n=" + str(best_n_lda) + ")",
    hline(),
]
for i, words in lda_top_words.items():
    report_lines.append(f"  Topic {i:2d} : {', '.join(words)}")

report_lines += [
    "",
    "FINAL CATEGORY SIZE DISTRIBUTION",
    hline(),
    f"  Largest  : {final_sizes.max():,}  ({final_sizes.idxmax()})",
    f"  Smallest : {final_sizes.min():,}  ({final_sizes.idxmin()})",
    f"  Mean     : {final_sizes.mean():,.0f}",
    f"  Std dev  : {final_sizes.std():,.0f}",
    "",
    "QUALITY CHECK — Final Category vs Reddit Flairs",
    hline(),
    "  Flairs are user-assigned and noisy; comparison is indicative only.",
]
for cat in FINAL_10_CATEGORIES:
    mask      = df["final_category"] == cat
    top_flair = df.loc[mask, "flair"].value_counts().head(3)
    flair_str = "  |  ".join(f"{f}({n})" for f, n in top_flair.items())
    report_lines.append(f"  {cat[:35]:<35} → {flair_str}")

with open(OUT_DIR / "clustering_report.txt", "w") as f:
    f.write("\n".join(report_lines))

for line in report_lines:
    print(line)

# ══════════════════════════════════════════════════════════════════════════════
# 10. Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots …")

k_list  = list(K_RANGE)
n_list  = list(LDA_RANGE)

# (A) Silhouette scores
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(k_list, silhouettes, "o-", color=ACCENT, lw=2)
ax.axvline(best_k, color="navy", ls="--", lw=1.4, label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title("K-Means: Silhouette Score vs Number of Clusters")
ax.set_xticks(k_list)
ax.legend()
save_plot(fig, "01_silhouette_scores")

# (B) Elbow (inertia)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(k_list, inertias, "s-", color="#5B8DB8", lw=2)
ax.axvline(best_k, color="navy", ls="--", lw=1.4, label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia (within-cluster sum of squares)")
ax.set_title("K-Means Elbow Plot")
ax.set_xticks(k_list)
ax.legend()
save_plot(fig, "02_elbow_inertia")

# (C) Davies-Bouldin scores
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(k_list, db_scores, "^-", color="#7B5EA7", lw=2)
ax.axvline(best_k, color="navy", ls="--", lw=1.4, label=f"Best k={best_k}")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Davies-Bouldin score (lower = better)")
ax.set_title("K-Means: Davies-Bouldin Score vs Number of Clusters")
ax.set_xticks(k_list)
ax.legend()
save_plot(fig, "03_davies_bouldin")

# (D) LDA coherence
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(n_list, coherences, "D-", color="#4CAF82", lw=2)
ax.axvline(best_n_lda, color="navy", ls="--", lw=1.4, label=f"Best n={best_n_lda}")
ax.set_xlabel("Number of topics")
ax.set_ylabel("Coherence score (c_v, higher = better)")
ax.set_title("LDA: Topic Coherence vs Number of Topics")
ax.set_xticks(n_list)
ax.legend()
save_plot(fig, "04_lda_coherence")

# (E1) Raw cluster sizes (17)
sorted_labels = df["topic_label"].value_counts()
colors_bar = (PALETTE * 4)[:len(sorted_labels)]
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(sorted_labels.index[::-1], sorted_labels.values[::-1],
               color=colors_bar[::-1], edgecolor="white", lw=0.4)
for bar, val in zip(bars, sorted_labels.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8)
ax.set_xlabel("Number of posts")
ax.set_title(f"Raw K-Means Cluster Sizes (k={best_k})")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_plot(fig, "05_raw_cluster_sizes")

# (E2) Final 10-category sizes
final_sorted = df["final_category"].value_counts()
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(final_sorted.index[::-1], final_sorted.values[::-1],
               color=PALETTE[:len(final_sorted)][::-1], edgecolor="white", lw=0.4)
for bar, val in zip(bars, final_sorted.values[::-1]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
            f"{val:,}  ({val/len(df)*100:.1f}%)", va="center", fontsize=9)
ax.set_xlabel("Number of posts")
ax.set_title("10 Final Silver-Standard Categories")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_plot(fig, "05b_final_10_categories")

# (F) Top-10 terms per cluster — grid of bar charts
n_cols = 3
n_rows = int(np.ceil(best_k / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
axes_flat = axes.flat if best_k > 1 else [axes]
for k, ax in zip(range(best_k), axes_flat):
    terms  = cluster_top_terms[k][:10]
    mask   = km_labels == k
    scores = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
    top_idx = np.argsort(scores)[::-1][:10]
    vals    = [scores[i] for i in top_idx]
    ax.barh(terms[::-1], vals[::-1], color=PALETTE[k % len(PALETTE)])
    ax.set_title(f"Cluster {k}: {cluster_labels[k][:25]}", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
for ax in list(axes_flat)[best_k:]:
    ax.set_visible(False)
fig.suptitle("Top-10 TF-IDF Terms per K-Means Cluster", fontsize=13, y=1.01)
fig.tight_layout()
save_plot(fig, "06_cluster_top_terms")

# (G) LDA top terms — grid
n_rows_lda = int(np.ceil(best_n_lda / n_cols))
fig, axes = plt.subplots(n_rows_lda, n_cols,
                          figsize=(15, n_rows_lda * 3.2))
axes_flat = axes.flat if best_n_lda > 1 else [axes]
for i, ax in zip(range(best_n_lda), axes_flat):
    words_scores = lda_final.show_topic(i, topn=10)
    words  = [w for w, _ in words_scores]
    scores = [s for _, s in words_scores]
    ax.barh(words[::-1], scores[::-1], color=PALETTE[i % len(PALETTE)])
    ax.set_title(f"LDA Topic {i}", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
for ax in list(axes_flat)[best_n_lda:]:
    ax.set_visible(False)
fig.suptitle(f"LDA Topic Word Distributions (n_topics={best_n_lda})", fontsize=13, y=1.01)
fig.tight_layout()
save_plot(fig, "07_lda_top_terms")

# (H) t-SNE coloured by the 10 final categories
print("  Computing t-SNE (this takes ~2 min) …")
from sklearn.manifold import TSNE
sample_n   = min(5000, len(df))
rng        = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(len(df), sample_n, replace=False)
X_sample   = X_norm[sample_idx]
final_sample = df["final_category"].iloc[sample_idx].values

tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=40,
            max_iter=1000, metric="cosine", init="pca")
X_2d = tsne.fit_transform(X_sample)

fig, ax = plt.subplots(figsize=(13, 10))
for cat, color in zip(FINAL_10_CATEGORIES, PALETTE):
    mask_c = final_sample == cat
    ax.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1],
               s=6, alpha=0.5, color=color, label=cat)
ax.legend(loc="upper right", fontsize=8, markerscale=3,
          bbox_to_anchor=(1.42, 1.0))
ax.set_title(f"t-SNE — 10 Final Categories ({sample_n:,} sample)", fontsize=12)
ax.axis("off")
save_plot(fig, "08_tsne_final_categories")

# (I) Final category × flair heatmap
top_flairs = df["flair"].value_counts().head(12).index.tolist()
hm_data = pd.crosstab(df["final_category"], df["flair"])[top_flairs]
hm_norm = hm_data.div(hm_data.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(hm_norm, cmap="YlOrRd", ax=ax, fmt=".2f", annot=True,
            annot_kws={"size": 8}, linewidths=0.3)
ax.set_title("Final Category × Reddit Flair (row-normalised proportion)")
ax.set_xlabel("Reddit Flair")
ax.set_ylabel("Final Category")
ax.tick_params(axis="x", rotation=35, labelsize=8)
ax.tick_params(axis="y", labelsize=9)
save_plot(fig, "09_final_category_flair_heatmap")

# (J) Final 10-category pie
fig, ax = plt.subplots(figsize=(10, 10))
sizes_pie  = final_sorted.values
labels_pie = [f"{l}\n({v:,})" for l, v in zip(final_sorted.index, sizes_pie)]
ax.pie(sizes_pie, labels=labels_pie,
       colors=PALETTE[:len(sizes_pie)],
       autopct="%1.1f%%", startangle=140, pctdistance=0.82,
       textprops={"fontsize": 8})
ax.set_title("10 Final Category Distribution", fontsize=13)
save_plot(fig, "10_final_category_pie")

print(f"\n{'═'*72}")
print(f"  K-Means raw clusters : {best_k}  (silhouette={max(silhouettes):.4f})")
print(f"  LDA best n_topics    : {best_n_lda}  (coherence={max(coherences):.4f})")
print(f"  Final categories     : 10  →  labeled_corpus.csv  (column: final_category)")
print(f"  10 categories:")
for cat, cnt in final_sorted.items():
    print(f"    {cat:<40} {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
print(f"  Plots  : task3a-clustering/plots/  (11 files)")
print(f"  Report : task3a-clustering/clustering_report.txt")
print(f"{'═'*72}")
