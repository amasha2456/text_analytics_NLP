# Task 4(a) — Information Extraction Schema

## Target category

**Employment & Career** (2,583 posts in the Task 3(a) silver-standard labels)
— chosen because it has consistently structured underlying information
(what job, where, for how much, what's required) even though the surface
text is informal, and because it feeds directly into Task 5's
stakeholder-facing summarization (job seekers, employers, and
policy/labour-market stakeholders all have a clear use for this data).

## Entity types

| Entity | Description | Example (from real posts) |
|---|---|---|
| `JOB_TITLE` | The job role/position the post is about | "program manager", "ux/ui & frontend developer" |
| `COMPANY` | Employer or company name, if named | "Dialog", "a BPO company" |
| `LOCATION` | City/country associated with the job | "Dubai/UAE", "Colombo", "remote" |
| `SALARY` | Compensation figure or range mentioned | "150,000 LKR", "2.5 year experience" *(see note)* |
| `EMPLOYMENT_TYPE` | Nature of the role | "internship", "full-time", "freelance" |
| `EXPERIENCE` | Experience level/duration required or held | "2.5 years", "entry level" |
| `SKILL` | Specific skill, tool, or qualification named | "Figma", "CIMA", "driving license" |

These seven map to BIO tag types for the Task 4(b)(ii) token-classification
model (`O` + `B-`/`I-` per type = 15 tags total).

## Post-level fields (not token-span entities)

| Field | Description | Allowed values |
|---|---|---|
| `visa_sponsorship_mentioned` | Whether visa/work-permit sponsorship is discussed | `true` / `false` |
| `post_intent` | What the post is doing | `seeking_job`, `job_posting`, `salary_inquiry`, `career_advice`, `complaint`, `other` |

## Relationships

Extracted as `(subject, predicate, object)` triples grounded in the
entities above, using a fixed predicate vocabulary (deliberately
constrained, not open — this keeps output parseable and comparable across
posts):

- `requires_skill` — (JOB_TITLE, SKILL)
- `offered_by` — (JOB_TITLE, COMPANY)
- `located_in` — (JOB_TITLE, LOCATION)
- `has_salary` — (JOB_TITLE, SALARY)
- `requires_experience` — (JOB_TITLE, EXPERIENCE)

## Output JSON schema (per post)

```json
{
  "job_title": "string or null",
  "company": "string or null",
  "location": "string or null",
  "salary": "string or null",
  "employment_type": "string or null",
  "experience_required": "string or null",
  "skills": ["list", "of", "strings"],
  "visa_sponsorship_mentioned": true,
  "post_intent": "seeking_job",
  "relations": [
    {"subject": "string", "predicate": "requires_skill", "object": "string"}
  ]
}
```

## Design rationale

- **Every field nullable, `skills`/`relations` default to empty lists.**
  Most Reddit posts don't mention every field (e.g., a salary-advice post
  rarely names a company) — forcing a value would push the model toward
  hallucination, which is exactly the failure mode Task 4(c) needs to
  measure and discuss.
- **Fixed predicate vocabulary for relations**, rather than letting the
  model invent relation names freely. Open relation extraction is a much
  harder and noisier problem; constraining to five predicates keeps
  output parseable with simple validation and keeps the comparison in
  Task 4(b) fair between the LLM and the fine-tuned NER model (the latter
  can only ever produce entity spans, not free-form relations, so relations
  in this project are always derived from entity co-occurrence + the fixed
  predicate list — see `extract_llm.py`).
- **`SALARY` is a loose bucket, not just currency amounts.** Real posts
  often discuss compensation adequacy without a hard figure (e.g., "is my
  salary ok" threads) — treated as a `SALARY`-relevant span when a
  concrete figure or range is quoted, separate from `EXPERIENCE`.
- **Grounding requirement.** The extraction prompt explicitly instructs
  the model to only fill a field when the value is stated in the post,
  and (see `extract_llm.py`) every extracted span is checked for literal
  presence in the source text after generation — this "groundedness rate"
  is used in Task 4(c) as a concrete, measurable proxy for hallucination,
  rather than relying only on qualitative impressions.
