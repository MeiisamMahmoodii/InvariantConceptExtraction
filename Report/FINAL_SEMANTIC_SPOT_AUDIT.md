# Final semantic spot audit — strict V1

## Sample

The audit sample is fixed by seed `20260825` and stratified across the six relations. It contains 10 facts per relation × 12 variants: 60 fact groups and 720 rows.

| Relation | Fact groups | Rows |
|---|---:|---:|
| capital_of | 10 | 120 |
| continent_of | 10 | 120 |
| currency_of | 10 | 120 |
| atomic_number_of | 10 | 120 |
| period_of | 10 | 120 |
| chemical_symbol_of | 10 | 120 |

## Audit questions

| Question | Result |
|---|---|
| Do all 12 variants preserve the same proposition `(relation, subject, value)`? | PASS — 60/60 groups |
| Are the variants genuinely different in wording or format? | PASS — 60/60 groups |
| Does any variant introduce, omit, or contradict a fact that changes the task? | PASS — 60/60 groups |

## What was checked

For every sampled group, stored C fields are identical across its 12 rows; all five families are present with the approved 3/3/3/2/1 variant counts; and every text contains its selected subject and value. The semantic review confirms that templates express only the relevant capital, continent, currency, atomic number, period, or chemical-symbol proposition. The question family includes its answer, so it does not omit the selected value.

## Result

The final semantic spot audit passes. Strict V1 is ready for the next representation-extraction stage; this audit does not perform that stage.
