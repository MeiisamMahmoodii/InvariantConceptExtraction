# Surface-template design — preflight only

## Rule

For every candidate, the underlying proposition `(relation, subject, value)` must stay fixed. Domain and canonical relation names are metadata; templates must not include `C_domain`, `C_relation`, IDs, split names, or source fields. A relation word may occur where ordinary language requires it, but is varied across the natural-language families.

## Families

`declarative` states the proposition. `question` asks and immediately answers it so no factor is omitted. `paraphrase` changes the wording without adding facts. `formal` uses ordinary human-readable field labels. `structured` uses natural JSON keys, never canonical metadata labels such as `capital_of`.

## Relation-specific preservation rules

| Relation | Preserve | Do not add or change |
|---|---|---|
| capital_of | country and its selected capital city | government seat, population, geography |
| continent_of | country and its selected continent | regions, subcontinents, borders |
| currency_of | country and its selected currency | exchange rate, legal-tender qualifiers |
| atomic_number_of | element and its atomic number | mass, isotope, discovery facts |
| period_of | element and its periodic-table period | group, block, electron configuration |
| chemical_symbol_of | element and its symbol | formula, atomic number, pronunciation |

## Candidate audit scope

`surface_template_candidates.csv` has three source-backed facts for every relation and every proposed family: 90 rows in total. This is the only text produced in this stage. Full generation remains unapproved.
