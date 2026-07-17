import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
# Each entry: display name -> (category, path to classification_report.txt)
BASE = Path(__file__).parent

MODELS = {
    "BERT (fine-tuned)":    ("Fine-tuned Transformer", BASE.parent / "task3b-transformers" / "results" / "bert-base-uncased" / "classification_report.txt"),
    "RoBERTa (fine-tuned)": ("Fine-tuned Transformer", BASE.parent / "task3b-transformers" / "results" / "roberta-base" / "classification_report.txt"),
    "BART-MNLI (NLI zero-shot)":    ("Zero-Shot NLI Classifier", BASE.parent / "task3c-zeroshot" / "results" / "bart-large-mnli" / "classification_report.txt"),
    "DeBERTa-v3 (NLI zero-shot)":   ("Zero-Shot NLI Classifier", BASE.parent / "task3c-zeroshot" / "results" / "deberta-v3-zeroshot" / "classification_report.txt"),
    "Qwen2.5-1.5B (zero-shot prompt)":  ("Decoder LLM (Zero-Shot)", BASE.parent / "task3c-zeroshot" / "results" / "qwen2.5-1.5b-instruct-zero-shot" / "classification_report.txt"),
    "Qwen2.5-1.5B (few-shot prompt)":   ("Decoder LLM (Few-Shot)",  BASE.parent / "task3c-zeroshot" / "results" / "qwen2.5-1.5b-instruct-few-shot" / "classification_report.txt"),
    "Phi-3-mini (zero-shot prompt)":    ("Decoder LLM (Zero-Shot)", BASE.parent / "task3c-zeroshot" / "results" / "phi-3-mini-instruct-zero-shot" / "classification_report.txt"),
    "Phi-3-mini (few-shot prompt)":     ("Decoder LLM (Few-Shot)",  BASE.parent / "task3c-zeroshot" / "results" / "phi-3-mini-instruct-few-shot" / "classification_report.txt"),
}

OUT_DIR  = BASE / "results"
PLOT_DIR = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ROW_RE = re.compile(
    r"^\s*(?P<label>.+?)\s{2,}"
    r"(?P<precision>[\d.]+)\s+"
    r"(?P<recall>[\d.]+)\s+"
    r"(?P<f1>[\d.]+)\s+"
    r"(?P<support>\d+)\s*$"
)
ACCURACY_RE = re.compile(
    r"^\s*accuracy\s+(?P<f1>[\d.]+)\s+(?P<support>\d+)\s*$"
)


def parse_classification_report(path: Path) -> dict:
    text = path.read_text()
    per_class, overall = {}, {}
    for line in text.splitlines():
        m = ACCURACY_RE.match(line)
        if m:
            overall["accuracy"] = float(m["f1"])
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        label = m["label"].strip()
        row = {
            "precision": float(m["precision"]),
            "recall":    float(m["recall"]),
            "f1-score":  float(m["f1"]),
            "support":   int(m["support"]),
        }
        if label in ("macro avg", "weighted avg"):
            overall[label] = row
        else:
            per_class[label] = row
    return {"per_class": per_class, "overall": overall}


def main():
    parsed = {}
    missing = []
    for name, (category, path) in MODELS.items():
        if not path.exists():
            missing.append((name, path))
            continue
        parsed[name] = {"category": category, **parse_classification_report(path)}

    if missing:
        print("Missing report files — cannot build the full comparison yet:")
        for name, path in missing:
            print(f"  ✗ {name:<28} expected at {path}")
        print("\nDrop the classification_report.txt files in place (from the task3b zip "
              "and the task3c Colab run) and re-run this script.")
        return

    # ── Summary table (accuracy / macro / weighted F1 per model) ──────────────
    summary_rows = []
    for name, res in parsed.items():
        ov = res["overall"]
        summary_rows.append({
            "model":        name,
            "category":     res["category"],
            "accuracy":     ov["accuracy"],
            "f1_weighted":  ov["weighted avg"]["f1-score"],
            "f1_macro":     ov["macro avg"]["f1-score"],
            "precision_macro": ov["macro avg"]["precision"],
            "recall_macro":    ov["macro avg"]["recall"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("f1_weighted", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary_comparison.csv", index=False)

    print(f"\n{'═'*70}")
    print("  ALL-MODEL COMPARISON — Task 3(d)")
    print(f"{'═'*70}")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── Per-class F1 table across all models ───────────────────────────────────
    label_names = sorted(next(iter(parsed.values()))["per_class"].keys())
    per_class_rows = []
    for name, res in parsed.items():
        row = {"model": name}
        for ln in label_names:
            row[ln] = res["per_class"].get(ln, {}).get("f1-score", np.nan)
        per_class_rows.append(row)
    per_class_df = pd.DataFrame(per_class_rows).set_index("model")
    per_class_df.to_csv(OUT_DIR / "per_class_f1_comparison.csv")

    # ── Write full text report ─────────────────────────────────────────────────
    with open(OUT_DIR / "comparison_report.txt", "w") as f:
        f.write("Task 3(d) — Comprehensive Model Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write("MODELS COMPARED\n")
        for name, (category, _) in MODELS.items():
            f.write(f"  {name:<28} [{category}]\n")
        f.write("\nSUMMARY\n")
        f.write(f"  {'Model':<26} {'Category':<24} {'Accuracy':>9} {'F1-Wt':>8} {'F1-Mac':>8}\n")
        f.write(f"  {'-'*78}\n")
        for _, r in summary_df.iterrows():
            f.write(f"  {r['model']:<26} {r['category']:<24} {r['accuracy']:>9.4f} "
                     f"{r['f1_weighted']:>8.4f} {r['f1_macro']:>8.4f}\n")
        f.write("\nPER-CLASS F1\n")
        f.write(per_class_df.round(3).to_string())
        f.write("\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    # 1. Overall metric comparison (grouped bars)
    metrics = ["accuracy", "f1_weighted", "f1_macro"]
    metric_labels = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]
    models_order = summary_df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(models_order)
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("deep", len(models_order))
    for i, name in enumerate(models_order):
        row = summary_df[summary_df["model"] == name].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=palette[i])
    ax.set_xticks(x + width * (len(models_order) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("All-Model Comparison — Fine-tuned Transformers vs Zero-Shot LLMs")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "01_all_model_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")

    # 2. Per-class F1 heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(per_class_df[label_names], annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"label": "F1-score"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Per-Category F1 Score — All Models")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / "02_per_class_f1_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")

    print(f"\nResults → {OUT_DIR}")
    print(f"Plots   → {PLOT_DIR}")
    print("Task 3(d) comparison table complete ✓")
    print("\nNOTE: the written discussion (strengths/limitations — data efficiency,")
    print("interpretability, ease of deployment) is a separate qualitative writeup,")
    print("not generated by this script. See DISCUSSION.md.")


if __name__ == "__main__":
    main()
