import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
MAX_LEN     = 128       # tokens – good balance of coverage vs speed
BATCH_SIZE  = 16
NUM_EPOCHS  = 3
LR          = 2e-5
TEST_SIZE   = 0.20

MODELS = {
    "bert-base-uncased": "google-bert/bert-base-uncased",
    "roberta-base":      "FacebookAI/roberta-base",
}

BASE       = Path(__file__).parent
CORPUS_CSV = BASE.parent / "task3a-clustering" / "labeled_corpus.csv"
OUT_DIR    = BASE / "results"
PLOT_DIR   = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── Dataset ───────────────────────────────────────────────────────────────────
class PostDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
        )
        enc["labels"] = self.labels[idx]
        return enc

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_macro":    float(f1_score(labels, preds, average="macro")),
    }

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, label_names, model_key):
    cm      = confusion_matrix(y_true, y_pred)
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
    ax.set_title("Encoder-Only Transformer Comparison — r/srilanka")
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
    ax.set_title("Per-Category F1 Score by Model")
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


def plot_training_history(logs, model_key):
    train_loss = [(e["epoch"], e["loss"])
                  for e in logs if "loss" in e and "eval_loss" not in e]
    eval_rows  = [(e["epoch"], e["eval_loss"], e.get("eval_f1_weighted", None))
                  for e in logs if "eval_loss" in e]
    if not eval_rows:
        return

    ep_t, loss_t = zip(*train_loss) if train_loss else ([], [])
    ep_e, loss_e, f1_e = zip(*eval_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if loss_t:
        ax1.plot(ep_t, loss_t, label="Train loss")
    ax1.plot(ep_e, loss_e, label="Eval loss", marker="o")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {model_key}"); ax1.legend(); ax1.grid(alpha=0.3)

    f1_e_clean = [v for v in f1_e if v is not None]
    if f1_e_clean:
        ax2.plot(ep_e[:len(f1_e_clean)], f1_e_clean, marker="o", color="#DD8452")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Weighted")
        ax2.set_title(f"Validation F1 — {model_key}")
        ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / f"03_history_{model_key.replace('-', '_')}.png"
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
    print(f"  {len(df):,} posts  |  {df['final_category'].nunique()} categories")

    # 2. Encode labels
    le         = LabelEncoder()
    df["label_id"] = le.fit_transform(df["final_category"])
    label_names    = list(le.classes_)
    label_map      = {i: n for i, n in enumerate(label_names)}
    print(f"  Labels: {label_names}")

    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # 3. Train / test split  (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values, df["label_id"].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df["label_id"].values,
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    # 4. Fine-tune each model
    all_reports: dict = {}
    all_preds:   dict = {}

    for model_key, model_name in MODELS.items():
        print(f"\n{'═'*62}")
        print(f"  Fine-tuning : {model_name}")
        print(f"{'═'*62}")

        model_out = OUT_DIR / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        collator  = DataCollatorWithPadding(tokenizer)

        train_ds = PostDataset(X_train, y_train, tokenizer)
        test_ds  = PostDataset(X_test,  y_test,  tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels = len(label_names),
            id2label   = label_map,
            label2id   = {v: k for k, v in label_map.items()},
            ignore_mismatched_sizes = True,
        )

        training_args = TrainingArguments(
            output_dir                  = str(model_out / "checkpoints"),
            num_train_epochs            = NUM_EPOCHS,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = BATCH_SIZE * 2,
            learning_rate               = LR,
            weight_decay                = 0.01,
            warmup_ratio                = 0.06,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1_weighted",
            greater_is_better           = True,
            logging_steps               = 200,
            seed                        = RANDOM_SEED,
            report_to                   = "none",
            fp16                        = (device == "cuda"),
            bf16                        = False,
        )

        trainer = Trainer(
            model            = model,
            args             = training_args,
            train_dataset    = train_ds,
            eval_dataset     = test_ds,
            processing_class = tokenizer,
            data_collator    = collator,
            compute_metrics = compute_metrics,
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("  Training …")
        train_result = trainer.train()

        print("  Evaluating …")
        pred_out = trainer.predict(test_ds)
        preds    = np.argmax(pred_out.predictions, axis=-1)
        all_preds[model_key] = preds

        rep_dict = classification_report(
            y_test, preds, target_names=label_names, output_dict=True, zero_division=0)
        rep_str  = classification_report(
            y_test, preds, target_names=label_names, zero_division=0)

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

        trainer.save_model(str(model_out / "model"))
        tokenizer.save_pretrained(str(model_out / "model"))

        plot_confusion_matrix(y_test, preds, label_names, model_key)
        plot_training_history(trainer.state.log_history, model_key)

    # 5. Summary comparison
    print(f"\n{'═'*62}")
    print("  FINAL COMPARISON")
    print(f"{'═'*62}")
    print(f"  {'Model':<28} {'Accuracy':>9} {'F1-Wt':>8} {'F1-Mac':>8}")
    print(f"  {'-'*53}")
    for mk, res in all_reports.items():
        print(f"  {mk:<28} {res['accuracy']:>9.4f} {res['f1_weighted']:>8.4f} {res['f1_macro']:>8.4f}")

    # Write combined report
    with open(OUT_DIR / "comparison_report.txt", "w") as f:
        f.write("Task 3(b) — Encoder-Only Transformer Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("SETTINGS\n")
        f.write(f"  Models              : {', '.join(MODELS.keys())}\n")
        f.write(f"  Max sequence length : {MAX_LEN} tokens\n")
        f.write(f"  Batch size          : {BATCH_SIZE}\n")
        f.write(f"  Epochs              : {NUM_EPOCHS}\n")
        f.write(f"  Learning rate       : {LR}\n")
        f.write(f"  Train / Test split  : {1-TEST_SIZE:.0%} / {TEST_SIZE:.0%}\n")
        f.write(f"  Train size          : {len(X_train):,}\n")
        f.write(f"  Test size           : {len(X_test):,}\n")
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
    print("  Task 3(b) complete ✓")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
