# Full surface dataset audit — fixed 50-fact sample

## What we inspected

The fixed audit sample uses seed `20260825` and contains 50 fact groups × 12 rows: 600 generated texts. All six relations are represented.

## Results

| Question | Result |
|---|---|
| Do all 12 rows preserve the same stored C factors? | PASS — 50/50 groups |
| Are surface realizations genuinely different within and across families? | PASS — 50/50 groups |
| Does any realization introduce or omit a fact that changes the proposition? | PASS — 50/50 groups |

The automated checks confirm each group has 3 declarative, 3 question, 3 paraphrase, 2 formal, and 1 structured row. The manual template audit confirms the relation-specific wording only expresses the stored relation, subject, and value. No retained subject or value label is an unresolved Wikidata ID.

## Decision

The strict source-validated V1 surface dataset passes its fixed-sample audit. The earlier 10,512-row output is preserved separately under `data/INVALID_source_label_unresolved/` and is marked do-not-train.
