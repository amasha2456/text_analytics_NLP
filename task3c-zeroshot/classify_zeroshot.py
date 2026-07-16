import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20     # must match task3b so both scripts evaluate on the same held-out posts
BATCH_SIZE  = 16
HYPOTHESIS_TEMPLATE = "This Reddit post is about {}."

MODELS = {
    "bart-large-mnli":     "facebook/bart-large-mnli",
    "deberta-v3-zeroshot": "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
}

BASE       = Path(__file__).parent
CORPUS_CSV = BASE.parent / "task3a-clustering" / "labeled_corpus.csv"
OUT_DIR    = BASE / "results"
PLOT_DIR   = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, label_names, model_key):
    cm      = confusion_matrix(y_true, y_pred, labels=label_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix — {model_key}", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / f"cm_{model_key.replace('-', '_')}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_comparison(all_reports):
    keys    = list(all_reports.keys())
    metrics = ["accuracy", "f1_weighted", "f1_macro"]
    labels  = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]

    x     = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4C72B0", "#DD8452"]
    for i, mk in enumerate(keys):
        vals = [all_reports[mk][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=mk, color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Zero-Shot Model Comparison — r/srilanka")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "01_model_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_per_class_f1(all_reports, label_names):
    keys  = list(all_reports.keys())
    x     = np.arange(len(label_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#4C72B0", "#DD8452"]
    for i, mk in enumerate(keys):
        f1s = [all_reports[mk]["report"][ln]["f1-score"] for ln in label_names]
        ax.bar(x + i * width, f1s, width, label=mk, color=colors[i % len(colors)])

    ax.set_xlabel("Category")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Category F1 Score by Zero-Shot Model")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "02_per_class_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    # 1. Load corpus
    print("\nLoading labeled corpus …")
    df = pd.read_csv(CORPUS_CSV)
    df = df.dropna(subset=["clean_text", "final_category"])
    label_names = sorted(df["final_category"].unique())
    print(f"  {len(df):,} posts  |  {len(label_names)} categories")
    print(f"  Labels: {label_names}")

    # 2. Reproduce the same train/test split used in task3b so results are comparable
    _, X_test, _, y_test = train_test_split(
        df["clean_text"].values, df["final_category"].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df["final_category"].values,
    )
    print(f"  Test set : {len(X_test):,} posts (same split as task3b)")

    # 3. Zero-shot classify with each model
    all_reports: dict = {}

    for model_key, model_name in MODELS.items():
        print(f"\n{'═'*62}")
        print(f"  Zero-shot : {model_name}")
        print(f"{'═'*62}")

        model_out = OUT_DIR / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        zsc = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            batch_size=BATCH_SIZE,
        )

        print("  Classifying test set … (this can take a while, no GPU shortcuts here)")
        outputs = zsc(
            list(X_test),
            candidate_labels=label_names,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
        )
        preds = [o["labels"][0] for o in outputs]

        pd.DataFrame({
            "text":       X_test,
            "true_label": y_test,
            "pred_label": preds,
        }).to_csv(model_out / "predictions.csv", index=False)

        rep_dict = classification_report(
            y_test, preds, labels=label_names, target_names=label_names, output_dict=True, zero_division=0)
        rep_str = classification_report(
            y_test, preds, labels=label_names, target_names=label_names, zero_division=0)

        all_reports[model_key] = {
            "report":      rep_dict,
            "report_str":  rep_str,
            "accuracy":    rep_dict["accuracy"],
            "f1_weighted": rep_dict["weighted avg"]["f1-score"],
            "f1_macro":    rep_dict["macro avg"]["f1-score"],
        }

        print(f"\n  Accuracy     : {rep_dict['accuracy']:.4f}")
        print(f"  F1 (Weighted): {rep_dict['weighted avg']['f1-score']:.4f}")
        print(f"  F1 (Macro)   : {rep_dict['macro avg']['f1-score']:.4f}")
        print(rep_str)

        with open(model_out / "classification_report.txt", "w") as f:
            f.write(f"Model : {model_name}\n{'='*60}\n\n")
            f.write(f"Accuracy     : {rep_dict['accuracy']:.4f}\n")
            f.write(f"F1 Weighted  : {rep_dict['weighted avg']['f1-score']:.4f}\n")
            f.write(f"F1 Macro     : {rep_dict['macro avg']['f1-score']:.4f}\n\n")
            f.write(rep_str)

        plot_confusion_matrix(y_test, preds, label_names, model_key)

    # 4. Summary comparison
    print(f"\n{'═'*62}")
    print("  FINAL COMPARISON")
    print(f"{'═'*62}")
    print(f"  {'Model':<28} {'Accuracy':>9} {'F1-Wt':>8} {'F1-Mac':>8}")
    print(f"  {'-'*53}")
    for mk, res in all_reports.items():
        print(f"  {mk:<28} {res['accuracy']:>9.4f} {res['f1_weighted']:>8.4f} {res['f1_macro']:>8.4f}")

    with open(OUT_DIR / "comparison_report.txt", "w") as f:
        f.write("Task 3(c) — Zero-Shot Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("SETTINGS\n")
        f.write(f"  Models              : {', '.join(MODELS.keys())}\n")
        f.write(f"  Hypothesis template : \"{HYPOTHESIS_TEMPLATE}\"\n")
        f.write(f"  Test size           : {len(X_test):,} (same split as task3b)\n")
        f.write(f"  Device              : {device}\n\n")
        f.write("MODEL COMPARISON\n")
        f.write(f"  {'Model':<28} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}\n")
        f.write(f"  {'-'*60}\n")
        for mk, res in all_reports.items():
            f.write(f"  {mk:<28} {res['accuracy']:>10.4f} {res['f1_weighted']:>12.4f} {res['f1_macro']:>10.4f}\n")
        f.write("\n\n")
        for mk, res in all_reports.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"  Model : {mk}\n")
            f.write(f"{'='*60}\n\n")
            f.write(res["report_str"])

    print("\nGenerating comparison plots …")
    plot_comparison(all_reports)
    plot_per_class_f1(all_reports, label_names)

    print(f"\n{'═'*62}")
    print(f"  Results → {OUT_DIR}")
    print(f"  Plots   → {PLOT_DIR}")
    print("  Task 3(c) complete ✓")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
