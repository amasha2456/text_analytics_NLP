# Task 4(c) — Extraction Accuracy, Consistency, and Failure Modes

## Quantitative summary

**Task 4(b)(i) — Phi-3-mini-4k-instruct, 500 Employment & Career posts**
(`results/extraction_report.txt`)

| Metric | Value |
|---|---|
| Final JSON parse success rate (after 1 self-repair retry) | 97.80% (489/500) |
| Pass-1 parse success (before repair) | 97.20% |
| Overall groundedness rate (extracted value verified present in source post) | 87.53% |
| Total extraction time | 1,948.4s (3.90s/post) |

Groundedness by field ranged from 99.02% (`location`) down to 61.86%
(`employment_type`) — see "Failure mode 2" below for why that one field
is the outlier, and why the raw number overstates the actual problem.

**Task 4(b)(ii) — fine-tuned BERT NER, entity-level (span+type exact
match) scoring, 97 held-out test posts** (`results/ner-bert-base-uncased/classification_report.txt`)

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SALARY | 0.00 | 0.00 | 0.00 | 3 |
| JOB_TITLE | 0.16 | 0.14 | 0.15 | 22 |
| EMPLOYMENT_TYPE | 0.36 | 0.38 | 0.37 | 13 |
| LOCATION | 0.43 | 0.43 | 0.43 | 21 |
| COMPANY | 0.45 | 0.36 | 0.40 | 14 |
| SKILL | 0.53 | 0.39 | 0.45 | 46 |
| EXPERIENCE | 0.71 | 0.33 | 0.45 | 15 |
| **Micro avg** | **0.42** | **0.34** | **0.38** | 134 |

**Efficiency comparison** (the brief explicitly asks for this):

| | LLM prompting (Phi-3-mini) | Fine-tuned NER (BERT) |
|---|---|---|
| Per-example latency | 3.90s/post (generation) | 0.005s/post (single forward pass) — **~780x faster** |
| Setup cost | Zero training, just prompt engineering | Needed the LLM's own output as weak-label training data, plus a 227s fine-tuning run |
| Output richness | Full schema: 7 entities + `post_intent` + typed `relations` | Entities only — no intent classification, no relations (a token-classification head has no mechanism to output relational triples) |
| Deployment cost | Requires hosting/calling a 3.8B-parameter generative model | 436MB checkpoint, CPU-feasible for low-throughput use |

## Successful extraction — example

> *"expected salary for a tech lead (c#) at 99x or similar companies? i am
> considering a tech lead role with a strong focus on backend using c#
> and related technologies. i have around 5 years of experience and have
> handled small teams..."*

Extracted: `job_title="tech lead (c#)"`, `company="99x or similar companies"`,
`location="sri lanka"`, `skills=["c#", "backend", "team management", "architecture"]`,
plus four correctly-typed relation triples (`requires_experience`,
3x `requires_skill`). Every field here is grounded, `salary` was correctly
left `null` (the post asks *about* salary, none is stated), and the
relations correctly link each skill back to the job title rather than to
each other — exactly the intended behavior from the schema design.

## Failure mode 1 — output truncation on entity-dense posts

10 of the 11 unrecoverable parse failures (after repair) share the same
signature: the raw response is valid-looking JSON that simply stops
mid-string, never reaching a closing `}`. E.g. one response ends
`..."skills": [...7 items...], "visa_sponsorship_mentioned": false, "post_intent"` —
cut off exactly at the 300-token `max_new_tokens` budget. These are
posts with unusually many extractable facts (long skill lists, multiple
relations), where the schema's own richness works against it: the more
a post genuinely contains, the more likely the fixed token budget clips
the output before it closes. The self-repair pass (asking the model to
resend corrected JSON) didn't fix these, since the *content* was fine —
only the length budget was the problem — and a repair prompt doesn't
raise `max_new_tokens` for the retry. A more direct fix would be to
detect truncation specifically (response doesn't end in `}`) and retry
with a larger token budget rather than a generic repair prompt.

## Failure mode 2 — an "unstated default" bias on `employment_type`

`employment_type` groundedness (61.86%) is far below every other field
(87–99%). Manually reviewing the ungrounded cases splits them into two
distinct categories:

**(a) Benign paraphrase, not hallucination — a measurement artifact.**
Several "ungrounded" extractions are actually correct, just not a literal
substring match: *"monthly payment $35 (2 hours per day - part
time/flexible)"* → extracted `"part-time"` (source says "part time",
no hyphen); *"hired as intern mobile development engineer"* →
`"internship"` (source says "intern", not "internship"). The
groundedness check in `extract_llm.py` is a strict case-insensitive
substring match, so it can't credit morphological variants — the *true*
hallucination rate for this field is lower than 61.86% suggests.

**(b) Genuine hallucination — the model defaults to "full-time" when
nothing is stated.** Other cases are real errors: *"looking for a
reliable lady cook ... to work on weekends"* → extracted `"full-time"`
(weekend-only work is the opposite of full-time); *"i have recently
joined as the manager of a fuel station"* → extracted `"full-time"` with
no employment-type language anywhere in the post at all. This is a
consistent, directional bias — not random noise — suggesting the model
treats "full-time" as an implicit default for any post that reads like a
real job rather than correctly leaving the field `null` per the prompt's
explicit grounding instruction. This is a useful, concrete illustration
of why the grounding instruction alone isn't sufficient — the model
still applies a prior about what's "normally" true rather than reporting
only what's stated.

## Failure mode 3 — weak-label errors compound into the NER model

`build_ner_data.py`'s span-matching step (`ner_data/build_report.txt`)
could only locate 615 of 730 LLM-extracted entity mentions (84.25%) as
literal token spans in the source post — the other 115 (mostly the same
kind of paraphrase/hallucination cases as Failure mode 2) were silently
dropped rather than mislabeled. This means the fine-tuned NER model's
0.38 micro-F1 should be read as **"how well BERT learned to reproduce
Phi-3-mini's own (imperfect, 87.5%-grounded) extractions,"** not as
accuracy against independent ground truth — there is no ground truth
here, only the LLM's silver labels one step removed. The NER model's
ceiling is bounded by the quality of its teacher's output, and every
systematic bias the LLM has (like the `employment_type` default) is a
bias the NER model has no way to unlearn from this data alone.

This also plausibly explains why `JOB_TITLE` — conceptually the most
central field in the schema — scored the *worst* NER F1 (0.15) despite
having reasonable support (22 test instances): job titles are highly
variable multi-token free text ("tech lead (c#)", "ux/ui and frontend
developer", "associate software engineer") with inconsistent boundaries,
and with only 353 actual training examples (392 posts minus the 10%
validation carve-out) and highly variable phrasing, there's too little
signal for BERT to learn a general job-title span pattern — versus
`LOCATION` (0.43 F1), which draws from a comparatively small, more
consistent vocabulary of Sri Lankan city/country names that repeat
across posts.

## Challenges from unstructured Reddit text

- **Rhetorical / advice-seeking posts read differently from postings.**
  Many Employment & Career posts (e.g. "is this normal in IT sector in
  Sri Lanka?", "should I do another internship at 29?") aren't reporting
  a job at all — they're asking for validation or advice. The
  `post_intent` field exists specifically to let the schema distinguish
  these (45 `career_advice` + 44 `complaint` + 119 `other` vs. 267
  `seeking_job`, per the intent distribution in `extraction_report.txt`),
  but it also means a large fraction of "extractable" posts legitimately
  have most entity fields `null` — the extraction task here is not
  "find the 7 fields" so much as "correctly recognize how much of the
  schema even applies."
- **Title/body duplication from the cleaning pipeline.** Task 2's
  preprocessing concatenates the post title and body, so nearly every
  post repeats its own opening sentence verbatim (visible in every
  example above). This doesn't break extraction, but it does mean
  token counts and "post length" are systematically inflated relative to
  the actual unique content.
- **Currency/number formatting is inconsistent.** Salary mentions appear
  as "150,000 LKR", "$35", "10,000 MYR (around 630k LKR)", or bare
  numbers with implied currency — a real deployment would need a
  normalization layer downstream of extraction (not attempted here) to
  make salary figures comparable across posts, which matters directly
  for Task 5's stakeholder-facing summarization of this category.

## What this means for choosing an approach

Neither approach is strictly better — they trade off in the way Task
3(d) already showed for classification. The LLM is more accurate, richer
(relations + intent, which a token-classification head structurally
cannot produce), and requires no training data of its own, but costs
~780x more per inference and inherits no ground-truth check on its own
outputs beyond the grounding heuristic. The fine-tuned NER model is fast
and cheap enough for real-time or high-volume use, but is only ever as
good as the LLM-derived weak labels it learned from, and — as the
`JOB_TITLE` result shows — needs meaningfully more than ~350 training
examples to learn free-text-heavy entity types reliably. A practical
deployment would likely use the LLM to (re-)generate and periodically
refresh weak labels at moderate volume, with the NER model doing the
actual high-throughput inference in production.
