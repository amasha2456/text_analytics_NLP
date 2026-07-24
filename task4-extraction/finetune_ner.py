"""
Task 4(b)(ii) — fine-tune a BERT-based token-classification (NER) model on
the weakly-labeled data from build_ner_data.py, and compare it against
the LLM prompting approach in extract_llm.py on accuracy and efficiency.

Entity-level (span + type exact match) precision/recall/F1 is computed
with a small self-contained scorer rather than the seqeval package, to
avoid adding a dependency that might not be preinstalled in a fresh
Colab runtime.
"""
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
MODEL_NAME  = "google-bert/bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 8
NUM_EPOCHS  = 15     # small dataset (~400 train posts) needs more epochs than task3b's 37k-example run
LR          = 3e-5
VAL_FRACTION = 0.10  # carved out of train.jsonl for early stopping; test.jsonl stays fully held out

BASE     = Path(__file__).parent
DATA_DIR = BASE / "ner_data"
OUT_DIR  = BASE / "results" / "ner-bert-base-uncased"
PLOT_DIR = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── Dataset ───────────────────────────────────────────────────────────────────
def tokenize_and_align(tokens, tags, tokenizer, tag2id, max_length):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    label_ids, prev_wid = [], None
    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)
        elif wid != prev_wid:
            label_ids.append(tag2id[tags[wid]])
        else:
            label_ids.append(-100)  # only the first subword of each word carries the label
        prev_wid = wid
    enc["labels"] = label_ids
    return enc


class NERDataset(Dataset):
    def __init__(self, records, tokenizer, tag2id, max_length):
        self.records, self.tokenizer, self.tag2id, self.max_length = records, tokenizer, tag2id, max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return tokenize_and_align(r["tokens"], r["tags"], self.tokenizer, self.tag2id, self.max_length)


# ── Entity-level scoring ────────────────────────────────────────────────────────
def get_entities(tags):
    """Decode a BIO tag sequence into a set of (type, start, end) spans."""
    entities, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((etype, start, i))
            start, etype = i, tag[2:]
        elif tag.startswith("I-") and etype == tag[2:]:
            continue
        else:
            if start is not None:
                entities.append((etype, start, i))
            start, etype = None, None
    return entities


def entity_level_report(all_gold_tags, all_pred_tags, entity_types):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for gold_tags, pred_tags in zip(all_gold_tags, all_pred_tags):
        gold_ents, pred_ents = set(get_entities(gold_tags)), set(get_entities(pred_tags))
        for e in pred_ents:
            (tp if e in gold_ents else fp)[e[0]] += 1
        for e in gold_ents:
            if e not in pred_ents:
                fn[e[0]] += 1

    report = {}
    total_tp = total_fp = total_fn = 0
    for etype in entity_types:
        p = tp[etype] / (tp[etype] + fp[etype]) if (tp[etype] + fp[etype]) else 0.0
        r = tp[etype] / (tp[etype] + fn[etype]) if (tp[etype] + fn[etype]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        support = tp[etype] + fn[etype]
        report[etype] = {"precision": p, "recall": r, "f1-score": f1, "support": support}
        total_tp += tp[etype]; total_fp += fp[etype]; total_fn += fn[etype]

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_p = float(np.mean([report[e]["precision"] for e in entity_types])) if entity_types else 0.0
    macro_r = float(np.mean([report[e]["recall"] for e in entity_types])) if entity_types else 0.0
    macro_f1 = float(np.mean([report[e]["f1-score"] for e in entity_types])) if entity_types else 0.0
    report["micro avg"] = {"precision": micro_p, "recall": micro_r, "f1-score": micro_f1, "support": total_tp + total_fn}
    report["macro avg"] = {"precision": macro_p, "recall": macro_r, "f1-score": macro_f1, "support": total_tp + total_fn}
    return report


def format_report(report, entity_types):
    lines = [f"{'':<20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"]
    for etype in entity_types + ["micro avg", "macro avg"]:
        r = report[etype]
        lines.append(f"{etype:<20}{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1-score']:>10.2f}{r['support']:>10}")
    return "\n".join(lines)


def make_compute_metrics(id2label, entity_types):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        all_gold, all_pred = [], []
        for p_row, l_row in zip(preds, labels):
            gold_tags, pred_tags = [], []
            for p_id, l_id in zip(p_row, l_row):
                if l_id == -100:
                    continue
                gold_tags.append(id2label[int(l_id)])
                pred_tags.append(id2label[int(p_id)])
            all_gold.append(gold_tags)
            all_pred.append(pred_tags)
        rep = entity_level_report(all_gold, all_pred, entity_types)
        return {
            "f1_micro": rep["micro avg"]["f1-score"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "precision_micro": rep["micro avg"]["precision"],
            "recall_micro": rep["micro avg"]["recall"],
        }
    return compute_metrics


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_entity_f1(report, entity_types):
    vals = [report[e]["f1-score"] for e in entity_types]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(entity_types, vals, color="#4C72B0")
    ax.set_xlabel("F1 (entity-level, span + type exact match)")
    ax.set_title("Task 4(b)(ii) — Fine-tuned NER: Per-Entity-Type F1 (test set)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / "04_ner_per_entity_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_training_history(logs):
    train_loss = [(e["epoch"], e["loss"]) for e in logs if "loss" in e and "eval_loss" not in e]
    eval_rows = [(e["epoch"], e["eval_loss"], e.get("eval_f1_micro")) for e in logs if "eval_loss" in e]
    if not eval_rows:
        return
    ep_t, loss_t = zip(*train_loss) if train_loss else ([], [])
    ep_e, loss_e, f1_e = zip(*eval_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if loss_t:
        ax1.plot(ep_t, loss_t, label="Train loss")
    ax1.plot(ep_e, loss_e, label="Eval loss", marker="o")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss — NER fine-tuning"); ax1.legend(); ax1.grid(alpha=0.3)

    f1_clean = [v for v in f1_e if v is not None]
    if f1_clean:
        ax2.plot(ep_e[:len(f1_clean)], f1_clean, marker="o", color="#DD8452")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Micro (entity-level)")
        ax2.set_title("Validation F1 — NER fine-tuning")
        ax2.set_ylim(0, 1); ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / "05_ner_training_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    if not (DATA_DIR / "train.jsonl").exists():
        print(f"Missing {DATA_DIR/'train.jsonl'} — run build_ner_data.py first "
              f"(it needs results/extractions.csv from extract_llm.py).")
        return

    tag_vocab = json.load(open(DATA_DIR / "tag_vocab.json"))
    id2label = {v: k for k, v in tag_vocab.items()}
    entity_types = sorted({t[2:] for t in tag_vocab if t.startswith("B-")})
    print(f"Tag vocab: {len(tag_vocab)} tags, {len(entity_types)} entity types: {entity_types}")

    all_train = load_jsonl(DATA_DIR / "train.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(all_train))
    n_val = max(1, int(len(all_train) * VAL_FRACTION))
    val_idx, train_idx = set(idx[:n_val].tolist()), set(idx[n_val:].tolist())
    train_records = [all_train[i] for i in sorted(train_idx)]
    val_records = [all_train[i] for i in sorted(val_idx)]
    print(f"Train: {len(train_records)}  |  Val: {len(val_records)}  |  Test: {len(test_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorForTokenClassification(tokenizer)

    train_ds = NERDataset(train_records, tokenizer, tag_vocab, MAX_LEN)
    val_ds = NERDataset(val_records, tokenizer, tag_vocab, MAX_LEN)
    test_ds = NERDataset(test_records, tokenizer, tag_vocab, MAX_LEN)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(tag_vocab), id2label=id2label, label2id=tag_vocab,
    )

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep only the best checkpoint on disk — with up to 15 epochs
                              # and no cap, this was silently accumulating ~1-1.3GB per epoch
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=20,
        seed=RANDOM_SEED,
        report_to="none",
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label, entity_types),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nTraining …")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    print("\nEvaluating on held-out test set …")
    t1 = time.time()
    pred_out = trainer.predict(test_ds)
    inference_time = time.time() - t1
    preds = np.argmax(pred_out.predictions, axis=-1)

    all_gold, all_pred = [], []
    for p_row, l_row in zip(preds, pred_out.label_ids):
        gold_tags, pred_tags = [], []
        for p_id, l_id in zip(p_row, l_row):
            if l_id == -100:
                continue
            gold_tags.append(id2label[int(l_id)])
            pred_tags.append(id2label[int(p_id)])
        all_gold.append(gold_tags)
        all_pred.append(pred_tags)

    report = entity_level_report(all_gold, all_pred, entity_types)
    report_str = format_report(report, entity_types)
    print(report_str)

    trainer.save_model(str(OUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUT_DIR / "model"))

    model_size_mb = sum(f.stat().st_size for f in (OUT_DIR / "model").rglob("*") if f.is_file()) / 1e6

    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Model : {MODEL_NAME} (token classification, entity-level scoring)\n{'='*60}\n\n")
        f.write(f"F1 Micro   : {report['micro avg']['f1-score']:.4f}\n")
        f.write(f"F1 Macro   : {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Precision  : {report['micro avg']['precision']:.4f}\n")
        f.write(f"Recall     : {report['micro avg']['recall']:.4f}\n\n")
        f.write(report_str)
        f.write("\n\nEFFICIENCY (comparison point vs. extract_llm.py's timing in results/extraction_report.txt)\n")
        f.write(f"  Train examples      : {len(train_records)}\n")
        f.write(f"  Training time       : {train_time:.1f}s ({train_time/max(len(train_records),1):.3f}s/example)\n")
        f.write(f"  Test examples       : {len(test_records)}\n")
        f.write(f"  Inference time      : {inference_time:.1f}s ({inference_time/max(len(test_records),1):.3f}s/example)\n")
        f.write(f"  Saved model size    : {model_size_mb:.1f} MB\n")

    print(f"\nModel size: {model_size_mb:.1f} MB")
    print("\nGenerating plots …")
    plot_entity_f1(report, entity_types)
    plot_training_history(trainer.state.log_history)

    print(f"\nResults → {OUT_DIR}")
    print(f"Plots   → {PLOT_DIR}")
    print("Task 4(b)(ii) NER fine-tuning complete ✓")


if __name__ == "__main__":
    main()
