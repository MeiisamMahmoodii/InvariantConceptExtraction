# Factual C-matrix report

## What we did

Built one factual row per approved proposition `C = [domain, relation, subject, value]` from the saved common-subject universes. No text, S-family, or contrastive-pair field exists in this matrix.

## Why

All three relations for one subject must remain together before surface realization is introduced.

## What we found

| Domain | Subjects in all three relations | Facts | Relations |
|---|---:|---:|---|
| Geography | 148 | 444 | capital, continent, currency |
| Science | 118 | 354 | atomic number, period, chemical symbol |

Total: 798 factual rows. The subject-level split assigns every subject once, then applies that split to all of its facts. Validation found zero subject leakage, zero missing values, zero common-universe mismatches, and no text or S columns.

Value reuse is saved in `factual_matrix_validation.json`. The strongest reused values are continents (7 values, mean 21.143 subjects/value) and periods (7 values, mean 16.857 subjects/value).
