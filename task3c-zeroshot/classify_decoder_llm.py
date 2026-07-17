"""
Task 3(c) — decoder-only LLM classification via prompting (zero-shot + few-shot).

This is the script that fulfils the brief's actual Task 3(c) requirement:
"Apply at least two different open-source decoder-only LLMs to the same
classification task. Evaluate their performance using both zero-shot and
few-shot prompting strategies."

`classify_zeroshot.py` in this same folder uses BART-MNLI / DeBERTa-v3 via
HuggingFace's NLI-based `zero-shot-classification` pipeline — that is a
useful baseline, but those are not decoder-only LLMs and involve no
prompting, so it does not satisfy 3(c) on its own. See README.md for how
the two scripts relate.

PROMPT DESIGN (see PROMPT_DESIGN.md for the full writeup):
  - Category names alone are ambiguous (this is exactly why the NLI
    baseline collapsed on "Sri Lanka Identity & Culture", F1 0.09-0.12 —
    the label text overlaps semantically with almost every other category).
    The system prompt therefore pairs every category with a one-line gloss
    describing what it covers.
  - The model is instructed to answer with the category name ONLY, so
    output parsing can stay a simple string match instead of needing a
    constrained-decoding library.
  - Few-shot uses exactly one worked example per category (10 total),
    sampled from the training split — enough to calibrate the output
    format and disambiguate boundary cases without bloating the prompt
    (and therefore latency/cost) for every single inference call.
  - Greedy decoding (do_sample=False) is used throughout for
    reproducibility — there's no benefit to sampling diversity in a
    single-label classification task.
"""
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TEST_SIZE    = 0.20     # must match task3b/task3c-NLI so all three are comparable
EVAL_SIZE    = 2000     # stratified subsample of the 9,446-post test set (generation is
                         # far more expensive per-example than the NLI classification head;
                         # 2,000 stratified posts keeps class proportions intact while
                         # keeping 2 models x 2 strategies runnable on a single Colab T4 session)
BATCH_SIZE   = 8
MAX_NEW_TOKENS = 16
N_FEWSHOT_PER_CLASS = 1
EXAMPLE_TRUNCATE_WORDS = 40
POST_TRUNCATE_WORDS    = 120

MODELS = {
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi-3-mini-instruct":   "microsoft/Phi-3-mini-4k-instruct",
}
STRATEGIES = ["zero-shot", "few-shot"]

CATEGORY_DESCRIPTIONS = {
    "Education": "schools, universities, degrees, exams, studying, scholarships",
    "Employment & Career": "jobs, salaries, interviews, resumes, workplace issues, work visas",
    "Finance, Banking & Telecoms": "banking, credit cards, payments, mobile/internet providers, bills, money transfers",
    "Help & General Q&A": "general questions or requests for advice not specific to any topic below",
    "Politics & National Affairs": "government, elections, policy, national news, protests",
    "Relationships & Social Life": "family, friendships, dating, personal or social situations",
    "Shopping & Local Recommendations": "where to buy things, product or service recommendations, local businesses",
    "Sri Lanka Identity & Culture": "national identity, culture, traditions, general discussion about Sri Lanka as a country or society",
    "Transport & Vehicles": "buses, trains, vehicles, driving, traffic, vehicle imports",
    "Travel & Tourism": "trips, destinations, travel tips, tourism within or outside Sri Lanka",
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


def truncate_words(text: str, n: int) -> str:
    words = str(text).split()
    return " ".join(words[:n])


# ── Prompt construction ────────────────────────────────────────────────────────
def build_system_prompt(label_names, fewshot_block=None):
    cat_lines = "\n".join(
        f"{i+1}. {name} — {CATEGORY_DESCRIPTIONS[name]}"
        for i, name in enumerate(label_names)
    )
    prompt = (
        "You are a text classification assistant for posts from the r/srilanka "
        "subreddit. Classify the post the user gives you into exactly ONE of the "
        "following categories:\n\n"
        f"{cat_lines}\n\n"
        "Respond with ONLY the exact category name from the list above, "
        "with no extra words, quotes, explanation, or punctuation."
    )
    if fewshot_block:
        prompt += "\n\nEXAMPLES:\n" + fewshot_block
    return prompt


def build_fewshot_block(X_train, y_train, label_names, rng):
    lines = []
    for label in label_names:
        idx = np.where(y_train == label)[0]
        chosen = rng.choice(idx)
        text = truncate_words(X_train[chosen], EXAMPLE_TRUNCATE_WORDS)
        lines.append(f"Post: {text}\nCategory: {label}")
    return "\n\n".join(lines)


def build_messages(text, system_prompt):
    user = f"Post: {truncate_words(text, POST_TRUNCATE_WORDS)}\nCategory:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# ── Output parsing ──────────────────────────────────────────────────────────────
def extract_response_text(item):
    entry = item[0] if isinstance(item, list) else item
    convo = entry["generated_text"]
    if isinstance(convo, list):
        return str(convo[-1]["content"]).strip()
    return str(convo).strip()


def parse_label(response, label_names):
    resp = response.strip().strip('"').strip("'").strip(".")
    resp_lower = resp.lower()
    for ln in label_names:
        if resp_lower == ln.lower():
            return ln
    candidates = [ln for ln in label_names if ln.lower() in resp_lower]
    if candidates:
        return max(candidates, key=len)
    return "UNPARSED"


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, label_names, run_key):
    labels_with_unparsed = label_names + (["UNPARSED"] if "UNPARSED" in y_pred else [])
    cm = confusion_matrix(y_true, y_pred, labels=labels_with_unparsed)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels_with_unparsed, yticklabels=labels_with_unparsed, ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Normalised Confusion Matrix — {run_key}", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / f"cm_{run_key.replace('-', '_').replace(' ', '_')}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_comparison(all_reports):
    keys    = list(all_reports.keys())
    metrics = ["accuracy", "f1_weighted", "f1_macro"]
    labels  = ["Accuracy", "F1 (Weighted)", "F1 (Macro)"]

    x     = np.arange(len(metrics))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("deep", len(keys))
    for i, mk in enumerate(keys):
        vals = [all_reports[mk][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=mk, color=palette[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Decoder-LLM Prompting Comparison — r/srilanka (Task 3c)")
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "03_decoder_llm_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_per_class_f1(all_reports, label_names):
    keys  = list(all_reports.keys())
    x     = np.arange(len(label_names))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = sns.color_palette("deep", len(keys))
    for i, mk in enumerate(keys):
        f1s = [all_reports[mk]["report"].get(ln, {}).get("f1-score", 0.0) for ln in label_names]
        ax.bar(x + i * width, f1s, width, label=mk, color=palette[i])

    ax.set_xlabel("Category")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Category F1 — Decoder-LLM Prompting Strategies (Task 3c)")
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "04_decoder_llm_per_class_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")
    rng = np.random.default_rng(RANDOM_SEED)

    print("\nLoading labeled corpus …")
    df = pd.read_csv(CORPUS_CSV)
    df = df.dropna(subset=["clean_text", "final_category"])
    label_names = sorted(df["final_category"].unique())
    print(f"  {len(df):,} posts  |  {len(label_names)} categories")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].values, df["final_category"].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df["final_category"].values,
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,} (same split as task3b/3c-NLI)")

    X_eval, _, y_eval, _ = train_test_split(
        X_test, y_test, train_size=EVAL_SIZE, random_state=RANDOM_SEED, stratify=y_test,
    )
    print(f"  Eval subsample : {len(X_eval):,} posts (stratified, for decoder-LLM prompting cost)")

    fewshot_block = build_fewshot_block(X_train, y_train, label_names, rng)
    zero_shot_system = build_system_prompt(label_names)
    few_shot_system  = build_system_prompt(label_names, fewshot_block)

    all_reports: dict = {}

    for model_key, model_name in MODELS.items():
        print(f"\n{'═'*62}")
        print(f"  Loading : {model_name}")
        print(f"{'═'*62}")

        dtype = torch.float16 if device != -1 else torch.float32
        pipe = pipeline(
            "text-generation", model=model_name,
            torch_dtype=dtype, device=device, batch_size=BATCH_SIZE,
        )
        if pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

        for strategy in STRATEGIES:
            run_key = f"{model_key} ({strategy})"
            print(f"\n  ── {run_key} " + "─" * (50 - len(run_key)))

            system_prompt = few_shot_system if strategy == "few-shot" else zero_shot_system
            messages = [build_messages(t, system_prompt) for t in X_eval]

            print("  Generating …")
            raw_outputs = pipe(
                messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=pipe.tokenizer.pad_token_id,
            )
            responses = [extract_response_text(o) for o in raw_outputs]
            preds     = [parse_label(r, label_names) for r in responses]

            unparsed_rate = sum(p == "UNPARSED" for p in preds) / len(preds)
            print(f"  Unparsed rate: {unparsed_rate:.2%}")

            run_out = OUT_DIR / f"{model_key}-{strategy}"
            run_out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "text": X_eval, "true_label": y_eval,
                "raw_response": responses, "pred_label": preds,
            }).to_csv(run_out / "predictions.csv", index=False)

            rep_dict = classification_report(
                y_eval, preds, labels=label_names, target_names=label_names,
                output_dict=True, zero_division=0,
            )
            rep_str = classification_report(
                y_eval, preds, labels=label_names, target_names=label_names, zero_division=0,
            )

            all_reports[run_key] = {
                "report": rep_dict,
                "report_str": rep_str,
                "accuracy": rep_dict["accuracy"],
                "f1_weighted": rep_dict["weighted avg"]["f1-score"],
                "f1_macro": rep_dict["macro avg"]["f1-score"],
                "unparsed_rate": unparsed_rate,
            }

            print(f"  Accuracy     : {rep_dict['accuracy']:.4f}")
            print(f"  F1 (Weighted): {rep_dict['weighted avg']['f1-score']:.4f}")
            print(f"  F1 (Macro)   : {rep_dict['macro avg']['f1-score']:.4f}")
            print(rep_str)

            with open(run_out / "classification_report.txt", "w") as f:
                f.write(f"Model    : {model_name}\nStrategy : {strategy}\n{'='*60}\n\n")
                f.write(f"Accuracy       : {rep_dict['accuracy']:.4f}\n")
                f.write(f"F1 Weighted    : {rep_dict['weighted avg']['f1-score']:.4f}\n")
                f.write(f"F1 Macro       : {rep_dict['macro avg']['f1-score']:.4f}\n")
                f.write(f"Unparsed rate  : {unparsed_rate:.2%}\n\n")
                f.write(rep_str)

            plot_confusion_matrix(y_eval, preds, label_names, run_key)

        del pipe
        gc.collect()
        if device != -1 and device != "mps":
            torch.cuda.empty_cache()

    print(f"\n{'═'*62}")
    print("  FINAL COMPARISON — Decoder-LLM Prompting (Task 3c)")
    print(f"{'═'*62}")
    print(f"  {'Run':<38} {'Accuracy':>9} {'F1-Wt':>8} {'F1-Mac':>8} {'Unparsed':>9}")
    print(f"  {'-'*76}")
    for rk, res in all_reports.items():
        print(f"  {rk:<38} {res['accuracy']:>9.4f} {res['f1_weighted']:>8.4f} "
              f"{res['f1_macro']:>8.4f} {res['unparsed_rate']:>9.2%}")

    with open(OUT_DIR / "comparison_report_decoder_llm.txt", "w") as f:
        f.write("Task 3(c) — Decoder-Only LLM Prompting Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("SETTINGS\n")
        f.write(f"  Models             : {', '.join(MODELS.values())}\n")
        f.write(f"  Strategies         : {', '.join(STRATEGIES)}\n")
        f.write(f"  Few-shot exemplars : {N_FEWSHOT_PER_CLASS} per class ({len(label_names)} total)\n")
        f.write(f"  Eval subsample     : {len(X_eval):,} stratified posts "
                f"(of {len(X_test):,}-post test set, same split as task3b)\n")
        f.write(f"  Decoding           : greedy (do_sample=False), max_new_tokens={MAX_NEW_TOKENS}\n")
        f.write(f"  Device             : {device}\n\n")
        f.write("COMPARISON\n")
        f.write(f"  {'Run':<38} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10} {'Unparsed':>10}\n")
        f.write(f"  {'-'*82}\n")
        for rk, res in all_reports.items():
            f.write(f"  {rk:<38} {res['accuracy']:>10.4f} {res['f1_weighted']:>12.4f} "
                     f"{res['f1_macro']:>10.4f} {res['unparsed_rate']:>10.2%}\n")
        f.write("\n\n")
        for rk, res in all_reports.items():
            f.write(f"\n{'='*60}\n  Run : {rk}\n{'='*60}\n\n")
            f.write(res["report_str"])

    print("\nGenerating comparison plots …")
    plot_comparison(all_reports)
    plot_per_class_f1(all_reports, label_names)

    print(f"\n{'═'*62}")
    print(f"  Results → {OUT_DIR}")
    print(f"  Plots   → {PLOT_DIR}")
    print("  Task 3(c) decoder-LLM prompting complete ✓")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
