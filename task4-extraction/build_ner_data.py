"""
Task 4(b)(ii), data prep — converts extract_llm.py's LLM extractions into
BIO-tagged token sequences to weakly-supervise a fine-tuned NER model.

There is no gold NER annotation for this corpus (out of scope per the
brief — "manually annotating such a large dataset is out of scope", same
justification as Task 3(a)'s silver-standard clustering). Instead, each
LLM-extracted entity string is located as a literal token span inside its
source post; entities that can't be found verbatim (the model paraphrased
or hallucinated them) are silently dropped from the NER labels rather than
forced in — this keeps the weak labels at least self-consistent, at the
cost of the NER model only ever learning to find spans that look like
things the LLM already found. That tradeoff — and what it means for
comparing the two approaches "fairly" — is discussed in EVALUATION.md.

Run locally (no GPU/model download needed): python build_ner_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
TEST_FRACTION = 0.20

BASE = Path(__file__).parent
EXTRACTIONS_CSV = BASE / "results" / "extractions.csv"
OUT_DIR = BASE / "ner_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# field name in extractions.csv -> BIO entity type
FIELD_TO_TYPE = {
    "job_title": "JOB_TITLE",
    "company": "COMPANY",
    "location": "LOCATION",
    "salary": "SALARY",
    "employment_type": "EMPLOYMENT_TYPE",
    "experience_required": "EXPERIENCE",
}
ENTITY_TYPES = list(FIELD_TO_TYPE.values()) + ["SKILL"]

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(str(text))


def find_span(tokens_lower, entity_tokens_lower, claimed):
    """First non-overlapping match of entity_tokens_lower inside tokens_lower."""
    n, m = len(tokens_lower), len(entity_tokens_lower)
    if m == 0:
        return None
    for start in range(n - m + 1):
        if any(claimed[start:start + m]):
            continue
        if tokens_lower[start:start + m] == entity_tokens_lower:
            return start, start + m
    return None


def tag_post(text, field_values):
    """field_values: list of (entity_type, string_value), longest-first for greedy matching."""
    tokens = tokenize(text)
    tokens_lower = [t.lower() for t in tokens]
    tags = ["O"] * len(tokens)
    claimed = [False] * len(tokens)
    matched, unmatched = 0, 0

    ordered = sorted(field_values, key=lambda kv: -len(kv[1]))
    for etype, value in ordered:
        ent_tokens = [t.lower() for t in tokenize(value)]
        span = find_span(tokens_lower, ent_tokens, claimed)
        if span is None:
            unmatched += 1
            continue
        start, end = span
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
        for i in range(start, end):
            claimed[i] = True
        matched += 1

    return tokens, tags, matched, unmatched


def build_tag_vocab():
    tags = ["O"]
    for t in ENTITY_TYPES:
        tags.append(f"B-{t}")
        tags.append(f"I-{t}")
    return {tag: i for i, tag in enumerate(tags)}


def main():
    if not EXTRACTIONS_CSV.exists():
        print(f"Missing {EXTRACTIONS_CSV} — run extract_llm.py first (on Colab, then bring")
        print("results/extractions.csv back into this folder).")
        return

    df = pd.read_csv(EXTRACTIONS_CSV)
    df = df[df["parse_success"] == True].reset_index(drop=True)  # noqa: E712
    print(f"Using {len(df):,} successfully-parsed extractions as weak-label source")

    records = []
    total_matched, total_unmatched = 0, 0
    for _, row in df.iterrows():
        field_values = []
        for field, etype in FIELD_TO_TYPE.items():
            v = row.get(field)
            if isinstance(v, str) and v.strip():
                field_values.append((etype, v.strip()))
        skills = json.loads(row["skills"]) if isinstance(row.get("skills"), str) else []
        for s in skills:
            if isinstance(s, str) and s.strip():
                field_values.append(("SKILL", s.strip()))

        tokens, tags, matched, unmatched = tag_post(row["clean_text"], field_values)
        total_matched += matched
        total_unmatched += unmatched
        records.append({
            "entry_id": row.get("entry_id"),
            "tokens": tokens,
            "tags": tags,
            "n_entities": matched,
        })

    match_rate = total_matched / (total_matched + total_unmatched) if (total_matched + total_unmatched) else float("nan")
    print(f"Entity span-matching rate: {match_rate:.2%} ({total_matched} matched, {total_unmatched} unmatched)")

    n_with_entities = sum(1 for r in records if r["n_entities"] > 0)
    print(f"Posts with >=1 tagged entity: {n_with_entities}/{len(records)} ({n_with_entities/len(records):.1%})")

    # post-level train/test split (shuffled, fixed seed)
    import random
    rng = random.Random(RANDOM_SEED)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * TEST_FRACTION))
    test_idx, train_idx = set(idx[:n_test]), set(idx[n_test:])

    train_records = [records[i] for i in sorted(train_idx)]
    test_records = [records[i] for i in sorted(test_idx)]

    with open(OUT_DIR / "train.jsonl", "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "test.jsonl", "w") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    tag_vocab = build_tag_vocab()
    with open(OUT_DIR / "tag_vocab.json", "w") as f:
        json.dump(tag_vocab, f, indent=2)

    with open(OUT_DIR / "build_report.txt", "w") as f:
        f.write("Task 4(b)(ii) — NER Weak-Label Build Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source extractions (parse_success=True) : {len(df):,}\n")
        f.write(f"Entity span-matching rate                : {match_rate:.2%}\n")
        f.write(f"  Matched                                : {total_matched}\n")
        f.write(f"  Unmatched (dropped)                    : {total_unmatched}\n")
        f.write(f"Posts with >=1 tagged entity              : {n_with_entities}/{len(records)} ({n_with_entities/len(records):.1%})\n\n")
        f.write(f"Train posts : {len(train_records)}\n")
        f.write(f"Test posts  : {len(test_records)}\n")
        f.write(f"Tag vocab   : {len(tag_vocab)} tags ({len(ENTITY_TYPES)} entity types x B-/I- + O)\n")

    print(f"\nTrain: {len(train_records)} posts → {OUT_DIR/'train.jsonl'}")
    print(f"Test  : {len(test_records)} posts → {OUT_DIR/'test.jsonl'}")
    print(f"Tag vocab → {OUT_DIR/'tag_vocab.json'}")
    print("Task 4(b)(ii) NER data build complete ✓")


if __name__ == "__main__":
    main()
