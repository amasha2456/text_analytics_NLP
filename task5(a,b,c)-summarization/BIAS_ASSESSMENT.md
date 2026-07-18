# Task 5(b)/(c) — Quality, Factual Consistency, and Bias Assessment

Assessment of `results/final_summary.md`, the Task 5(a) multi-stakeholder
summary generated from 489 successfully-extracted Employment & Career
posts (same sample as Task 4).

## Methodology (three independent checks)

1. **Automated factual consistency** (`evaluate_summary.py`) — every
   number/percentage in the summary is traced against
   `results/aggregate_stats.json`, computed directly with pandas from
   Task 4's extraction data. A number not in that file is unsupported by
   construction.
2. **LLM-as-judge** — Qwen2.5-1.5B-Instruct (deliberately a different
   model from the Phi-3-mini that wrote the summary) scores it 1–5 on
   coherence, relevance, conciseness, and absence-of-hallucination.
3. **Small-scale manual review** — reading the final summary against the
   actual source posts and intermediate `batch_summaries.json` /
   `cluster_summaries.json`, tracing specific claims back to where they
   entered the pipeline.

## Results

| Check | Result |
|---|---|
| Numeric factual consistency | **4/4 (100%)** — every number cited traces to the grounding stats |
| LLM-judge: coherence | 4/5 |
| LLM-judge: relevance | 4/5 |
| LLM-judge: conciseness | 4/5 |
| LLM-judge: absence of hallucination | 4/5 — *"avoids making confident claims that sound too specific or invented"* |

Taken alone, these two automated checks would suggest the summary is
essentially clean. **Manual review found this to be false.**

## What manual review found that both automated checks missed

**Fabrication: "custom jewelry."** The Overview claims themes span
"tech to custom jewelry" and Policymakers are told to "consider... custom
jewelry" entrepreneurship. Searching all 500 source posts for "jewelry"
returns **zero matches**. Tracing the pipeline: the phrase does not
appear in any of the 25 batch summaries either — it first appears in
`cluster_summaries.json`'s cluster 0, meaning the reduce-1 stage
*invented* it while consolidating five batch summaries, despite an
explicit prompt instruction not to add information beyond what the
inputs contain. This is a pure hallucination introduced during
synthesis, not a distortion of anything real.

**Amplification: "navigating the job market as a transgender
individual."** This one *is* grounded — exactly one of the 500 posts
("seeking advice: what can i expect when trying to find employment as a
trans woman in sri lanka...") is the source. It's correctly summarized
in batch 4 ("the challenges of finding employment as a transgender
individual"), where it accurately reflects that batch's 20 posts. But as
it propagates unchanged through cluster 0's reduce and into the final
Overview, all frequency information is lost — a detail from 1 of 500
posts (0.2%) reads, in the final report, as if it were a recurring
community theme on par with the 54.6% who are job-seeking.

**Why this matters more than the raw numbers suggest:** the judge model
explicitly praised the summary for avoiding "confident claims that sound
too specific or invented" — the opposite of what manual review found.
An LLM judge sharing the same basic failure mode as the summarizer (no
mechanism to verify claims against source data, only to assess surface
plausibility) is not an independent check on hallucination, only on
style. **The brief's "if possible, a small-scale manual review" is not
optional in practice — for this pipeline, it was the only method of the
three that caught either issue.**

## Ethical implications — fairness and representativeness

- **A single vulnerable individual's disclosure became "market
  intelligence."** The trans-woman job-seeker's post is public, but
  writing an aggregate report that hands "navigating the job market as a
  transgender individual" to *employers and policymakers* as a
  stakeholder-relevant theme — drawn from n=1 — is a real harm distinct
  from ordinary hallucination. It over-states how representative that
  experience is of "the community," and it repackages one person's
  personal, potentially sensitive disclosure into a decision-relevant
  input for exactly the audiences (employers, government) whose
  decisions could most affect that person, without their knowledge or
  consent for that specific use.
- **The "For Employers & Recruiters" section is built on almost no
  employer data.** Only 7 of 489 posts (1.4%) are actual job postings —
  the rest are job-seekers, advice-seekers, and complainants. Advice like
  "offer competitive salaries" and "promote company culture" is
  extrapolated almost entirely from what job *seekers* say they want, not
  from anything employers themselves posted. The summary presents this
  with the same confident, bulleted authority as the job-seeker section,
  giving no signal that it rests on roughly 20x less underlying data.
- **r/srilanka is not Sri Lanka's workforce.** The top skills mentioned
  (`python`, `english`, `software engineering`, `c#`, `html`/`css`) and
  top locations (`sri lanka`, `colombo`) reflect a subreddit skewing
  toward younger, urban, English-literate, internet-connected, likely
  IT/tech-sector users. A "For Policymakers" section built from this
  sample and phrased as if describing general "youth employment" risks
  policymakers over-weighting a demographically narrow, self-selected
  online population — informal-sector workers, non-English speakers, and
  rural job-seekers are structurally near-invisible to this pipeline, not
  because they lack concerns, but because they aren't on this subreddit
  in the first place.

## Task 5(c) — mitigation strategies for real-world deployment

1. **Carry frequency/provenance through the pipeline, not just text.**
   The map stage should output `(theme, supporting_batch_count)` pairs
   rather than free text alone, and the final-reduce prompt should be
   instructed to only include a theme if it clears an explicit minimum
   count (e.g., appears in ≥3 of 25 batches) — this directly prevents the
   "transgender individual" failure, where a single-batch detail reached
   the final report with no signal of its rarity.
2. **Audit non-numeric claims, not just numbers.** The current
   `evaluate_summary.py` only checks numbers against the stats file. The
   "custom jewelry" fabrication passed because there is currently no
   check on named entities/topics — extending the checker to verify that
   every specific noun phrase in the summary has *some* keyword-level
   support in the source corpus (the same manual check performed above,
   `df['clean_text'].str.contains(...)`) would have caught it
   automatically.
3. **Do not treat LLM-judge scores as a hallucination check.** Use a
   judge for style/coherence, but pair it with a grounding-verification
   method (like #2) that actually looks at source data — this
   assessment's own judge run demonstrates a same-family model can rate
   a hallucination-containing summary highly on "absence of
   hallucination."
4. **Gate sensitive/identity-related content before it can be
   summarized into stakeholder-facing output.** A pre-filtering pass that
   flags posts disclosing protected characteristics (health, gender
   identity, immigration status, disability, etc.) and either excludes
   them from theme-extraction or requires human sign-off before any
   related phrase reaches a report shown to employers/policymakers would
   directly address the ethical concern above, independent of whether the
   detail is statistically rare enough to be filtered by strategy #1.
5. **State the sample's demographic limits in the output itself**, not
   just in a separate methodology doc — e.g. a standing disclaimer that
   this reflects public r/srilanka posts and likely skews toward younger,
   urban, English-speaking, tech-sector users, so a policymaker reading it
   doesn't mistake subreddit demographics for national representativeness.
6. **Flag confidence by underlying sample size per section.** The
   Employers section (7 posts) and the Job Seekers section (267 posts)
   are not equally well-supported; a real deployment should visibly
   differentiate "high-confidence, broadly-supported" from "low-confidence,
   thin-data" sections rather than presenting both with identical bulleted
   authority.
7. **Keep a human in the loop before publication**, especially for any
   report that will inform actual hiring, policy, or resource-allocation
   decisions — this assessment itself demonstrates that a ~15-minute
   manual read caught two significant issues that a 100%-passing
   automated numeric check and a second LLM judge both missed.
