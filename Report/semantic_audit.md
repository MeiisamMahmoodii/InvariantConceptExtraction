# Semantic audit — 50 concept groups

What we did: inspected the fixed-factor rows in `semantic_audit_samples.csv`. The sample is deterministic (seed `20260825`) and includes all five S families for each of 50 concepts.

Why: automated structural checks cannot by themselves confirm that the wording keeps the same task.

What we found:

| Question | Result |
|---|---|
| Do all realizations preserve the same C? | PASS — 50/50 = 100% |
| Are the S realizations genuinely different? | PASS — 50/50 = 100% |
| Does any realization introduce or omit information that changes the task? | PASS — 50/50 = 100% |

Each family explicitly contains entity, country, continent, and official language. The families differ in sentence form, question-and-answer form, key/value notation, labelled fields, and JSON-like structured form. No rejected generated examples were found. Source records rejected before generation are retained in `data/rejected_source_records.csv` with their reasons.
