# Project structure log

## Step 1 — created the requested structure

What we did: created `paper`, `Report`, `code`, `checkpoint`, and `data`.

Why: these are the only project folders requested.

What we found: there were no existing project files to move into them.

## Step 2 — added the starting file

What we did: added `main.py` in the project root.

Why: it is the single starting point requested.

What we found: running it prints a clear terminal confirmation.

## Step 3 — built the factual dataset

What we did: saved a local Wikidata source snapshot and generated 100 capital-city concepts with five fixed realization families each.

Why: this holds C constant while changing S in a complete factorial layout.

What we found: the dataset has 500 rows. Rejected source records were saved with their reasons instead of being silently discarded.

## Step 4 — validated and audited the dataset

What we did: ran the validator and inspected a fixed sample of 50 concept groups.

Why: this checks factor consistency, split separation, full S coverage, and meaning across realizations.

What we found: validation passed with zero leaks, duplicates, missing values, or inconsistent C fields. The semantic audit passed 50 out of 50 groups.

## Step 5 — designed reusable semantic structure

What we did: defined a future fact as domain, relation, subject, and value.

Why: repeated relations and overlapping subjects give later analysis reusable structure instead of isolated capital facts.

What we found: the design uses four domains and repeated relations, but does not create any new dataset rows.

## Step 6 — audited relation feasibility

What we did: checked every proposed relation for source property, value cardinality, reuse target, overlap, and ambiguity risk.

Why: ambiguous facts would make C poorly defined before the larger dataset is built.

What we found: clean geography and chemical-element relations can proceed after source filtering; multi-valued people and object relations are deferred.

## Step 7 — ran source-backed relation coverage

What we did: queried Wikidata and saved factual tables, a source snapshot, rejection reasons, and value-reuse statistics.

Why: each domain needs at least 50 subjects shared across all three relations before text construction.

What we found: geography has 174 shared subjects. Replacing the unusable group relation with Royal Society of Chemistry `period_of` gives science 118 shared subjects; both coverage checks pass.

## Step 8 — built the factual C-matrix

What we did: created one row per domain, relation, subject, and value proposition from the approved common-subject sets.

Why: this fixes semantic facts and subject-level splits before adding any surface text.

What we found: the matrix has 876 rows. Every retained subject has exactly three facts, and validation found no subject leakage or text/S fields.

## Step 9 — tested surface templates

What we did: rendered three factual examples per relation across five surface families and checked the candidate texts.

Why: templates must preserve each proposition before full text generation is allowed.

What we found: 216 candidates contain their fixed subject and value. Each sampled fact has 3 declarative, 3 question, 3 paraphrase, 2 formal, and 1 structured variant. No full surface dataset was generated.

## Step 10 — generated strict V1 surface rows

What we did: generated all approved variants from the strict source-validated factual matrix.

Why: V1 rejects unresolved labels instead of introducing a second label source.

What we found: 148 geography subjects and 118 science subjects produce 798 facts and 9,576 rows. All validations and the 50-fact audit pass.
