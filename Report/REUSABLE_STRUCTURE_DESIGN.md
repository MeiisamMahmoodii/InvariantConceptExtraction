# Reusable fact structure — design only

## Goal

The next dataset will not treat one capital-city record as one isolated concept. One underlying fact will instead be:

`C = [domain, relation, subject, value]`

- `domain`: broad topic shared by many facts.
- `relation`: predicate shared by many subject/value pairs.
- `subject`: entity the fact is about.
- `value`: object of that relation.

Example: `[geography, capital_of, France, Paris]` and `[geography, capital_of, Japan, Tokyo]` share both the domain and relation. `[geography, currency_of, Japan, Yen]` shares the domain and subject with the Japan capital fact, but changes relation and value.

## Proposed domains and relations

| Domain | Relation | Subject type | Value type | Reuse target |
|---|---|---|---|---:|
| Geography | capital_of | country | city | 50+ countries |
| Geography | continent_of | country | continent | 50+ countries |
| Geography | currency_of | country | currency | 50+ countries |
| Geography | official_language_of | country | language | 50+ countries |
| People | occupation_of | person | occupation | 50+ people |
| People | birth_country_of | person | country | 50+ people |
| People | field_of_work_of | person | field | 50+ people |
| People | citizenship_of | person | country | 50+ people |
| Science | atomic_number_of | chemical element | integer | 50+ elements |
| Science | period_of | chemical element | periodic-table period | 50+ elements |
| Science | chemical_symbol_of | chemical element | symbol | 50+ elements |
| Science | discoverer_of | chemical element | person | 50+ elements where recorded |
| Objects | material_of | manufactured object | material | 50+ objects |
| Objects | color_of | manufactured object | color | 50+ objects |
| Objects | category_of | manufactured object | object category | 50+ objects |

## Reusable hierarchy

Each row stores all four C factors. Dataset analysis can therefore group rows at several levels:

1. Same fact: identical `[domain, relation, subject, value]`.
2. Same relation: same `[domain, relation]`, different subjects and values.
3. Same subject: same `[domain, subject]`, different relations and values.
4. Same domain: different relations, subjects, and values inside one topic.
5. Different domain: unrelated factual structure.

This gives the later SAE tests repeated relation components, repeated subject components, and domain-level components instead of only one-off capital facts.

## Surface realization remains separate

For each fact `C_i`, create one row for every fixed surface family `S_j`:

- declarative
- question-and-answer
- paraphrase
- formal labelled fields
- structured JSON-like form

All variants must express the same relation, subject, and value naturally. `C_domain` and canonical `C_relation` are metadata labels and do not have to occur literally in text. The `S_family` changes wording and format only. The current S split remains: declarative/question/paraphrase for `S_train`; formal/structured for `S_test`.

## Future row schema

Keep the existing identifiers, text, C split, S split, reproducibility fields, and factual provenance. Replace the old capital-specific C fields with:

`C_domain`, `C_relation`, `C_subject`, `C_value`

Optional type fields make audits clearer: `C_subject_type`, `C_value_type`. Store canonical `C_subject_id` and `C_value_id` for global entity-overlap reports. The S schema is `S_family`, `S_variant`; the present one-variant design is Sandbox 0 only.

## Construction rules for the later dataset

1. Choose a factual structured source before selecting facts; save its local snapshot and source IDs.
2. Select only relations that pass the relation feasibility audit; use one canonical value only when the selected subject set is genuinely single-valued.
3. Select overlapping subjects within a domain where possible. For example, each retained country should contribute capital, continent, currency, and official-language facts.
4. Split by subject/entity groups, not individual facts, so a held-out country or person cannot leak through another relation. Report both subject overlap and global entity overlap.
5. Keep complete S families for every fact and hold out whole S families exactly as before.
6. Validate repeated relation coverage, subject overlap, global entity overlap, factual consistency, C-split leakage, and S-split leakage before any model work. Do not set a contrastive-pair policy at this stage.

## What this design deliberately does not do

It does not generate a larger dataset, select factual instances, or train a model. Those need a separate approved construction step after the factual source and exact relation coverage are chosen.
