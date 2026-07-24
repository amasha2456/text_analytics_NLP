"""
Task 5(b) — quality, factual-consistency, and bias assessment methodology
for the Task 5(a) final summary.

Two independent checks:

1. FACTUAL CONSISTENCY (deterministic, no LLM) — every number/percentage
   in final_summary.md is extracted and checked against
   results/aggregate_stats.json, the ONLY source of numbers the
   summarization prompt was allowed to cite. A number that doesn't trace
   back to the stats file is either a rounding deviation or a fabrication
   by construction, not a subjective judgment call.

2. LLM-AS-JUDGE qualitative scoring — Qwen2.5-1.5B-Instruct (a
   DIFFERENT model from the Phi-3-mini that wrote the summary, so it
   isn't grading its own homework) scores the summary 1-5 on coherence,
   relevance, conciseness, and absence-of-hallucination, with a short
   justification per axis.

A small-scale manual review (the third leg the brief asks for) is done
separately in BIAS_ASSESSMENT.md by reading final_summary.md against a
handful of actual source posts — that part is qualitative and isn't
automated here.
"""
import json
import re
import warnings
from pathlib import Path

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # deliberately different from the Phi-3-mini summarizer
MAX_NEW_TOKENS = 400

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
SUMMARY_PATH = RESULTS_DIR / "final_summary.md"
STATS_PATH = RESULTS_DIR / "aggregate_stats.json"


def get_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


# ── 1. Factual consistency ──────────────────────────────────────────────────────
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
COUNT_RE = re.compile(r"\b(\d+)\s+posts?\b")


def collect_allowed_numbers(stats):
    allowed_pct, allowed_counts = set(), set()
    for v in stats["post_intent_pct"].values():
        allowed_pct.add(round(float(v), 1))
    allowed_pct.add(round(float(stats["salary_mentioned_pct"]), 1))
    allowed_pct.add(round(float(stats["company_mentioned_pct"]), 1))
    allowed_pct.add(round(float(stats["job_posting_pct"]), 1))
    allowed_pct.add(round(float(stats["seeking_job_pct"]), 1))

    allowed_counts.add(stats["n_posts"])
    for v in stats["top_employment_types"].values():
        allowed_counts.add(int(v))
    allowed_counts.add(int(stats["other_employment_type_count"]))
    for _, c in stats["top_skills"]:
        allowed_counts.add(int(c))
    for _, c in stats["top_locations"]:
        allowed_counts.add(int(c))
    return allowed_pct, allowed_counts


def check_factual_consistency(summary_text, stats):
    allowed_pct, allowed_counts = collect_allowed_numbers(stats)

    found_pct = [float(m) for m in PCT_RE.findall(summary_text)]
    found_counts = [int(m) for m in COUNT_RE.findall(summary_text)]

    pct_results = [{"value": v, "allowed": round(v, 1) in allowed_pct} for v in found_pct]
    count_results = [{"value": v, "allowed": v in allowed_counts} for v in found_counts]

    n_checked = len(pct_results) + len(count_results)
    n_flagged = sum(1 for r in pct_results + count_results if not r["allowed"])

    return {
        "percentages_found": pct_results,
        "counts_found": count_results,
        "n_numbers_checked": n_checked,
        "n_flagged_unsupported": n_flagged,
        "consistency_rate": round(1 - n_flagged / n_checked, 3) if n_checked else None,
    }


# ── 2. LLM-as-judge ───────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are evaluating a summary report written by another AI system. The report analyzes Reddit posts about jobs and careers in Sri Lanka and gives advice to job seekers, employers, and policymakers.

Score the report on these 4 criteria, each from 1 (poor) to 5 (excellent):
- coherence: is it well-organized and easy to follow?
- relevance: does it stay focused on jobs/careers in Sri Lanka without going off-topic?
- conciseness: is it appropriately brief without unnecessary repetition?
- absence_of_hallucination: does it avoid making confident claims that sound too specific/invented rather than appropriately general?

Respond with ONLY a JSON object in this exact format, no other text:
{"coherence": {"score": <1-5>, "reason": "<one sentence>"}, "relevance": {"score": <1-5>, "reason": "<one sentence>"}, "conciseness": {"score": <1-5>, "reason": "<one sentence>"}, "absence_of_hallucination": {"score": <1-5>, "reason": "<one sentence>"}}"""


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
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        try:
            return json.loads(block.replace("'", '"'))
        except json.JSONDecodeError:
            return None


def extract_response_text(item):
    entry = item[0] if isinstance(item, list) else item
    convo = entry["generated_text"]
    return str(convo[-1]["content"]).strip() if isinstance(convo, list) else str(convo).strip()


def run_judge(summary_text, device):
    dtype = torch.float16 if device != -1 else torch.float32
    pipe = pipeline("text-generation", model=JUDGE_MODEL, torch_dtype=dtype, device=device, batch_size=1)
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": summary_text}]
    out = pipe([messages], max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=pipe.tokenizer.pad_token_id)
    response = extract_response_text(out[0])
    parsed = try_parse_json(response)

    if parsed is None:
        # one repair attempt
        repair_messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": summary_text},
            {"role": "assistant", "content": response},
            {"role": "user", "content": "That was not valid JSON. Return ONLY the corrected JSON object."},
        ]
        out2 = pipe([repair_messages], max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                     pad_token_id=pipe.tokenizer.pad_token_id)
        response2 = extract_response_text(out2[0])
        parsed = try_parse_json(response2)
        response = response2 if parsed is not None else response

    return parsed, response


def main():
    device = get_device()
    print(f"Device : {device}")
    print(f"PyTorch: {torch.__version__}")

    if not SUMMARY_PATH.exists() or not STATS_PATH.exists():
        print(f"Missing {SUMMARY_PATH} or {STATS_PATH} — run summarize.py first.")
        return

    summary_text = SUMMARY_PATH.read_text()
    # strip the appended "verified statistics" block before both checks — that
    # block is our own grounding text, not part of what the LLM wrote
    summary_body = summary_text.split("## Verified statistics used for grounding")[0]
    stats = json.loads(STATS_PATH.read_text())

    print("\n[1/2] Factual consistency check …")
    consistency = check_factual_consistency(summary_body, stats)
    print(f"  Numbers found in summary : {consistency['n_numbers_checked']}")
    print(f"  Flagged as unsupported   : {consistency['n_flagged_unsupported']}")
    print(f"  Consistency rate         : {consistency['consistency_rate']}")
    for r in consistency["percentages_found"] + consistency["counts_found"]:
        flag = "OK" if r["allowed"] else "** NOT IN STATS **"
        print(f"    {r['value']} -> {flag}")

    print(f"\n[2/2] LLM-as-judge scoring ({JUDGE_MODEL}) …")
    judge_scores, judge_raw = run_judge(summary_body, device)
    if judge_scores:
        for axis, d in judge_scores.items():
            print(f"  {axis:<28} {d['score']}/5 — {d['reason']}")
    else:
        print("  Judge output could not be parsed as JSON. Raw response saved for manual review.")

    report = {
        "factual_consistency": consistency,
        "judge_model": JUDGE_MODEL,
        "judge_scores": judge_scores,
        "judge_raw_response": judge_raw,
    }
    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(RESULTS_DIR / "evaluation_report.txt", "w") as f:
        f.write("Task 5(b) — Summary Evaluation Report\n" + "=" * 60 + "\n\n")
        f.write("FACTUAL CONSISTENCY\n")
        f.write(f"  Numbers found in summary : {consistency['n_numbers_checked']}\n")
        f.write(f"  Flagged as unsupported   : {consistency['n_flagged_unsupported']}\n")
        f.write(f"  Consistency rate         : {consistency['consistency_rate']}\n\n")
        f.write("LLM-AS-JUDGE SCORES (%s)\n" % JUDGE_MODEL)
        if judge_scores:
            for axis, d in judge_scores.items():
                f.write(f"  {axis:<28} {d['score']}/5 — {d['reason']}\n")
        else:
            f.write("  (judge output failed to parse — see evaluation_report.json for raw response)\n")

    print(f"\nResults → {RESULTS_DIR}")
    print("Task 5(b) automated evaluation complete ✓")


if __name__ == "__main__":
    main()
