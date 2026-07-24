"""
Task 4(b)(i) — instruction-tuned LLM information extraction via advanced prompting.

Extracts the entity/relationship schema defined in SCHEMA.md from a sample
of Employment & Career posts, using microsoft/Phi-3-mini-4k-instruct (the
strongest instruction-follower of the two decoder LLMs evaluated in
Task 3(c) — lowest UNPARSED rate across all four of that task's runs).

ADVANCED PROMPT ENGINEERING TECHNIQUES USED (see also SCHEMA.md):
  1. Explicit JSON schema in the system prompt, every field documented
     with its type and allowed values.
  2. Grounding instruction — the model is told to only fill a field when
     the value is explicitly present in the post, and to prefer null /
     empty list over guessing. This is checked post-hoc (see
     `check_groundedness`) by verifying every extracted string literally
     occurs in the source post, giving a measurable hallucination proxy
     rather than a purely qualitative one.
  3. Few-shot exemplars deliberately span different completeness levels
     (a fully-populated job-offer post, a career-advice post with almost
     everything null, a salary-inquiry post) — this teaches the model
     that leaving fields null is the *expected*, not the failure, case.
  4. Constrained output vocabularies — `post_intent` and relation
     `predicate` are both drawn from fixed sets; anything else the model
     produces is normalized away rather than trusted (see `normalize`).
  5. Self-repair retry — posts whose first-pass output fails to parse as
     JSON get a second generation pass that shows the model its own
     invalid output and asks it to correct it, before finally falling
     back to a null-extraction record.
"""
import gc
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
SAMPLE_SIZE  = 500      # "small annotated sample" per the brief; also feeds task4b(ii)'s
                         # NER weak-labels (400 train / 100 test split, see build_ner_data.py)
MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
BATCH_SIZE   = 4
MAX_NEW_TOKENS = 300
POST_TRUNCATE_WORDS = 150

BASE       = Path(__file__).parent
CORPUS_CSV = BASE.parent / "task3a-clustering" / "labeled_corpus.csv"
TARGET_CATEGORY = "Employment & Career"
OUT_DIR    = BASE / "results"
PLOT_DIR   = BASE / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STRING_FIELDS = ["job_title", "company", "location", "salary", "employment_type", "experience_required"]
ALLOWED_INTENTS = {"seeking_job", "job_posting", "salary_inquiry", "career_advice", "complaint", "other"}
ALLOWED_PREDICATES = {"requires_skill", "offered_by", "located_in", "has_salary", "requires_experience"}


def get_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def truncate_words(text: str, n: int) -> str:
    return " ".join(str(text).split()[:n])


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an information extraction assistant for r/srilanka job/career posts.

Given a post, extract the following fields as a single JSON object:

- job_title: string or null — the job role/position the post is about
- company: string or null — employer/company name, only if explicitly named
- location: string or null — city/country associated with the job
- salary: string or null — a compensation figure or range, only if a concrete amount is mentioned
- employment_type: string or null — e.g. "internship", "full-time", "freelance"
- experience_required: string or null — years/level of experience mentioned
- skills: list of strings — specific skills, tools, or qualifications named (empty list if none)
- visa_sponsorship_mentioned: true or false — whether visa/work-permit sponsorship is discussed
- post_intent: one of "seeking_job", "job_posting", "salary_inquiry", "career_advice", "complaint", "other"
- relations: list of {"subject": ..., "predicate": ..., "object": ...} triples, where predicate is
  one of "requires_skill", "offered_by", "located_in", "has_salary", "requires_experience"

RULES:
- Only fill a field if its value is explicitly stated in the post. If it is not mentioned, use null
  (or an empty list for skills/relations). Do NOT guess or infer values that aren't in the text.
- Respond with ONLY the JSON object, no explanation, no markdown code fences.

EXAMPLES:

Post: ux/ui and frontend developer looking for an internship in colombo. i have basic figma and react skills, no prior experience yet.
JSON: {"job_title": "ux/ui and frontend developer", "company": null, "location": "colombo", "salary": null, "employment_type": "internship", "experience_required": null, "skills": ["figma", "react"], "visa_sponsorship_mentioned": false, "post_intent": "seeking_job", "relations": [{"subject": "ux/ui and frontend developer", "predicate": "located_in", "object": "colombo"}, {"subject": "ux/ui and frontend developer", "predicate": "requires_skill", "object": "figma"}, {"subject": "ux/ui and frontend developer", "predicate": "requires_skill", "object": "react"}]}

Post: feeling lost in my career path, should i consider an internship again at 29? i have a bsc degree and 3 years of experience in marketing but want to switch fields.
JSON: {"job_title": null, "company": null, "location": null, "salary": null, "employment_type": "internship", "experience_required": "3 years", "skills": [], "visa_sponsorship_mentioned": false, "post_intent": "career_advice", "relations": []}

Post: software engineer at wso2 with 2 years experience, offered 180000 lkr per month in colombo. is this a fair salary for the role?
JSON: {"job_title": "software engineer", "company": "wso2", "location": "colombo", "salary": "180000 lkr per month", "employment_type": null, "experience_required": "2 years", "skills": [], "visa_sponsorship_mentioned": false, "post_intent": "salary_inquiry", "relations": [{"subject": "software engineer", "predicate": "offered_by", "object": "wso2"}, {"subject": "software engineer", "predicate": "located_in", "object": "colombo"}, {"subject": "software engineer", "predicate": "has_salary", "object": "180000 lkr per month"}, {"subject": "software engineer", "predicate": "requires_experience", "object": "2 years"}]}"""


def build_messages(post_text, system_prompt=SYSTEM_PROMPT):
    user = f"Post: {truncate_words(post_text, POST_TRUNCATE_WORDS)}\nJSON:"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def build_repair_messages(post_text, bad_response):
    user = (
        f"Post: {truncate_words(post_text, POST_TRUNCATE_WORDS)}\n\n"
        f"Your previous response could not be parsed as valid JSON:\n{bad_response}\n\n"
        "Return ONLY the corrected, valid JSON object matching the schema. No explanation."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ── Generation ──────────────────────────────────────────────────────────────────
def extract_response_text(item):
    entry = item[0] if isinstance(item, list) else item
    convo = entry["generated_text"]
    if isinstance(convo, list):
        return str(convo[-1]["content"]).strip()
    return str(convo).strip()


def run_batched_generation(pipe, all_messages, batch_size, max_new_tokens, tag=""):
    responses = []
    n = len(all_messages)
    for start in range(0, n, batch_size):
        chunk = all_messages[start:start + batch_size]
        outputs = pipe(
            chunk, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        responses.extend(extract_response_text(o) for o in outputs)
        batch_no = start // batch_size
        if batch_no % 10 == 0:
            print(f"  {tag}{min(start + batch_size, n)}/{n}")
    return responses


# ── JSON parsing / validation ────────────────────────────────────────────────────
def extract_json_block(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def try_parse_json(text):
    block = extract_json_block(text)
    if block is None:
        return None
    for candidate in (block, block.replace("'", '"')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def normalize(parsed):
    if not isinstance(parsed, dict):
        return None
    out = {}
    for f in STRING_FIELDS:
        v = parsed.get(f)
        out[f] = v.strip() if isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none") else None
    skills = parsed.get("skills")
    out["skills"] = [s.strip() for s in skills if isinstance(s, str) and s.strip()] if isinstance(skills, list) else []
    out["visa_sponsorship_mentioned"] = parsed.get("visa_sponsorship_mentioned") is True
    intent = parsed.get("post_intent")
    out["post_intent"] = intent if intent in ALLOWED_INTENTS else "other"
    relations = parsed.get("relations")
    clean_relations = []
    if isinstance(relations, list):
        for r in relations:
            if (isinstance(r, dict) and r.get("predicate") in ALLOWED_PREDICATES
                    and isinstance(r.get("subject"), str) and isinstance(r.get("object"), str)
                    and r["subject"].strip() and r["object"].strip()):
                clean_relations.append({
                    "subject": r["subject"].strip(), "predicate": r["predicate"], "object": r["object"].strip(),
                })
    out["relations"] = clean_relations
    return out


def check_groundedness(post_text, normalized):
    text_lower = post_text.lower()
    checks = {}
    for f in STRING_FIELDS:
        v = normalized.get(f)
        if v:
            checks[f] = v.lower() in text_lower
    for i, s in enumerate(normalized.get("skills", [])):
        checks[f"skill_{i}"] = s.lower() in text_lower
    return checks


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_field_fill_rates(fill_rates):
    fig_labels = list(fill_rates.keys())
    vals = list(fill_rates.values())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fig_labels, vals, color="#4C72B0")
    ax.set_xlabel("Fill rate (fraction of posts with a non-null value)")
    ax.set_title("Task 4(b)(i) — Field Fill Rates (Employment & Career, n=%d)" % SAMPLE_SIZE)
    ax.set_xlim(0, 1)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / "01_field_fill_rates.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_groundedness(grounded_rate_by_field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_labels = list(grounded_rate_by_field.keys())
    vals = list(grounded_rate_by_field.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#55A868" if v >= 0.9 else "#DD8452" if v >= 0.7 else "#C44E52" for v in vals]
    ax.barh(fig_labels, vals, color=colors)
    ax.set_xlabel("Groundedness rate (extracted value literally present in source post)")
    ax.set_title("Task 4(c) — Extraction Groundedness by Field (hallucination proxy)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    p = PLOT_DIR / "02_groundedness_by_field.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_intent_distribution(intent_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(intent_counts.index, intent_counts.values, color="#4C72B0")
    ax.set_ylabel("Count")
    ax.set_title("Extracted post_intent Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = PLOT_DIR / "03_post_intent_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    print("\nLoading corpus …")
    df = pd.read_csv(CORPUS_CSV)
    sub = df[df["final_category"] == TARGET_CATEGORY].dropna(subset=["clean_text"])
    sample = sub.sample(n=min(SAMPLE_SIZE, len(sub)), random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  {TARGET_CATEGORY}: {len(sub):,} posts total, sampling {len(sample):,}")

    dtype = torch.float16 if device != -1 else torch.float32
    print(f"\nLoading {MODEL_NAME} …")
    pipe = pipeline("text-generation", model=MODEL_NAME, torch_dtype=dtype, device=device, batch_size=BATCH_SIZE)
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    print("\nPass 1: extracting …")
    t0 = time.time()
    messages = [build_messages(t) for t in sample["clean_text"]]
    responses = run_batched_generation(pipe, messages, BATCH_SIZE, MAX_NEW_TOKENS, tag="  ")
    pass1_time = time.time() - t0

    parsed = [try_parse_json(r) for r in responses]
    fail_idx = [i for i, p in enumerate(parsed) if p is None]
    print(f"  Pass 1 parse failures: {len(fail_idx)}/{len(parsed)} ({len(fail_idx)/len(parsed):.1%})")

    repair_time = 0.0
    if fail_idx:
        print(f"\nPass 2: repairing {len(fail_idx)} failed extractions …")
        t1 = time.time()
        repair_messages = [build_repair_messages(sample["clean_text"].iloc[i], responses[i]) for i in fail_idx]
        repair_responses = run_batched_generation(pipe, repair_messages, BATCH_SIZE, MAX_NEW_TOKENS, tag="  repair ")
        repair_time = time.time() - t1
        for i, r in zip(fail_idx, repair_responses):
            p = try_parse_json(r)
            if p is not None:
                parsed[i] = p
                responses[i] = r

    total_time = pass1_time + repair_time
    still_failed = sum(1 for p in parsed if p is None)
    print(f"\nFinal parse failures after repair: {still_failed}/{len(parsed)} ({still_failed/len(parsed):.1%})")
    print(f"Total generation time: {total_time:.1f}s ({total_time/len(sample):.2f}s/post)")

    del pipe
    gc.collect()
    if device not in (-1, "mps"):
        torch.cuda.empty_cache()

    # ── Normalize + groundedness ──────────────────────────────────────────────
    records = []
    all_grounded_checks = []
    for i, row in sample.iterrows():
        norm = normalize(parsed[i]) if parsed[i] is not None else None
        if norm is None:
            norm = {f: None for f in STRING_FIELDS}
            norm.update({"skills": [], "visa_sponsorship_mentioned": False, "post_intent": "other", "relations": []})
            parse_success = False
        else:
            parse_success = True
        grounded = check_groundedness(row["clean_text"], norm) if parse_success else {}
        all_grounded_checks.append(grounded)
        records.append({
            "entry_id": row.get("entry_id", i),
            "clean_text": row["clean_text"],
            "raw_response": responses[i],
            "parse_success": parse_success,
            **{f: norm[f] for f in STRING_FIELDS},
            "skills": json.dumps(norm["skills"]),
            "visa_sponsorship_mentioned": norm["visa_sponsorship_mentioned"],
            "post_intent": norm["post_intent"],
            "relations": json.dumps(norm["relations"]),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_DIR / "extractions.csv", index=False)

    # ── Stats ──────────────────────────────────────────────────────────────────
    parse_success_rate = out_df["parse_success"].mean()
    fill_rates = {f: out_df[f].notna().mean() for f in STRING_FIELDS}
    fill_rates["skills (>=1)"] = (out_df["skills"].apply(lambda s: len(json.loads(s)) > 0)).mean()
    fill_rates["relations (>=1)"] = (out_df["relations"].apply(lambda s: len(json.loads(s)) > 0)).mean()

    grounded_flat = {}
    for checks in all_grounded_checks:
        for k, v in checks.items():
            field = "skill" if k.startswith("skill_") else k
            grounded_flat.setdefault(field, []).append(v)
    grounded_rate_by_field = {f: float(np.mean(v)) for f, v in grounded_flat.items() if v}
    overall_grounded = float(np.mean([v for vals in grounded_flat.values() for v in vals])) if grounded_flat else float("nan")

    intent_counts = out_df["post_intent"].value_counts()

    with open(OUT_DIR / "extraction_report.txt", "w") as f:
        f.write("Task 4(b)(i) — LLM Extraction Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model              : {MODEL_NAME}\n")
        f.write(f"Target category    : {TARGET_CATEGORY}\n")
        f.write(f"Sample size        : {len(sample):,}\n")
        f.write(f"Device             : {device}\n\n")
        f.write("PARSING\n")
        f.write(f"  Pass-1 parse success rate : {1 - len(fail_idx)/len(parsed):.2%}\n")
        f.write(f"  Final parse success rate  : {parse_success_rate:.2%} (after self-repair retry)\n")
        f.write(f"  Unrecoverable failures    : {still_failed}\n\n")
        f.write("TIMING (efficiency comparison point for task4b-ii)\n")
        f.write(f"  Pass-1 generation time : {pass1_time:.1f}s\n")
        f.write(f"  Repair pass time       : {repair_time:.1f}s\n")
        f.write(f"  Total time             : {total_time:.1f}s ({total_time/len(sample):.3f}s/post)\n\n")
        f.write("FIELD FILL RATES (fraction of posts where the field was extracted as non-null)\n")
        for k, v in fill_rates.items():
            f.write(f"  {k:<20} {v:.2%}\n")
        f.write(f"\nGROUNDEDNESS (extracted value literally found in source post — hallucination proxy)\n")
        f.write(f"  Overall groundedness rate : {overall_grounded:.2%}\n")
        for k, v in grounded_rate_by_field.items():
            f.write(f"  {k:<20} {v:.2%}\n")
        f.write(f"\nPOST_INTENT DISTRIBUTION\n")
        for k, v in intent_counts.items():
            f.write(f"  {k:<20} {v}\n")

    print(f"\nParse success rate (final): {parse_success_rate:.2%}")
    print(f"Overall groundedness rate : {overall_grounded:.2%}")
    print("\nField fill rates:")
    for k, v in fill_rates.items():
        print(f"  {k:<20} {v:.2%}")

    print("\nGenerating plots …")
    plot_field_fill_rates(fill_rates)
    plot_groundedness(grounded_rate_by_field)
    plot_intent_distribution(intent_counts)

    print(f"\nResults → {OUT_DIR}")
    print(f"Plots   → {PLOT_DIR}")
    print("Task 4(b)(i) extraction complete ✓")


if __name__ == "__main__":
    main()
