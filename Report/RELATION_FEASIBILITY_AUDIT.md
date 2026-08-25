# Relation feasibility audit — design only

## Decision rule

Version 1 should use a relation only when the selected subject set has one canonical value for it, after an explicit source-based filter. A relation that is naturally multi-valued is deferred; the dataset will not silently choose one value. The structured source for this audit is Wikidata. Each property ID below links a fact to its recorded value.

`50+ usable subjects` is the minimum construction target after filtering, not a claim about an unverified global count.

| Proposed domain | Relation | Wikidata property | Value policy | Expected usable subjects | Subject overlap | Risk | Decision |
|---|---|---|---|---:|---|---|---|
| Geography | capital_of | [P36](https://www.wikidata.org/wiki/Property:P36) | exactly one current capital per retained country | 50+ | same countries as other geography relations | multiple/historical capitals | retain with filter |
| Geography | continent_of | [P30](https://www.wikidata.org/wiki/Property:P30) | reject unless exactly one direct best-rank continent claim | 50+ | same countries | transcontinental countries | retain with filter |
| Geography | currency_of | [P38](https://www.wikidata.org/wiki/Property:P38) | exactly one currency per retained country | 50+ | same countries | multi-currency countries | retain with filter |
| Geography | official_language_of | [P37](https://www.wikidata.org/wiki/Property:P37) | set-valued | not used in V1 | same countries | several official languages | defer |
| People | occupation_of | [P106](https://www.wikidata.org/wiki/Property:P106) | set-valued | not used in V1 | people could overlap | multiple occupations and granularity | defer |
| People | birth_country_of | [P19](https://www.wikidata.org/wiki/Property:P19) plus place hierarchy | not directly stored as a country | not used in V1 | people/countries | historical borders and place resolution | defer |
| People | field_of_work_of | [P101](https://www.wikidata.org/wiki/Property:P101) | set-valued | not used in V1 | people | multiple fields and taxonomy noise | defer |
| People | citizenship_of | [P27](https://www.wikidata.org/wiki/Property:P27) | set-valued | not used in V1 | people/countries | multiple and changing citizenships | defer |
| Science | atomic_number_of | [P1086](https://www.wikidata.org/wiki/Property:P1086) | one integer per selected element | 50+ | same elements as other science relations | none material | retain |
| Science | period_of | Royal Society of Chemistry element fact box | exactly one explicit period value | 118 | same elements | source lookup must resolve | retain |
| Science | chemical_symbol_of | [P246](https://www.wikidata.org/wiki/Property:P246) | one symbol per selected element | 50+ | same elements | none material | retain |
| Science | discovered_by | [P61](https://www.wikidata.org/wiki/Property:P61) | set-valued | not used in V1 | elements/people | contested and multiple discoverers | defer |
| Objects | material_of | [P186](https://www.wikidata.org/wiki/Property:P186) | set-valued | not used in V1 | objects/materials | composite materials | defer |
| Objects | color_of | [P462](https://www.wikidata.org/wiki/Property:P462) | set-valued | not used in V1 | objects/colors | multiple, subjective, time-dependent | defer |
| Objects | category_of | [P31](https://www.wikidata.org/wiki/Property:P31) | often set-valued | not used in V1 | objects/categories | multiple classification levels | defer |

## Recommended V1 relation set

The six-relation design is feasible with Wikidata for geography and atomic-number/symbol facts, plus the Royal Society of Chemistry for period facts:

1. geography: `capital_of`, `continent_of`, `currency_of`;
2. science: `atomic_number_of`, `period_of`, `chemical_symbol_of`.

Do not add a weak relation merely to reach a target number. Any future relation must pass the same source and cardinality checks.

## Required source queries before construction

For each retained relation, save a query result that includes subject ID, value ID, labels, and qualifiers. Then reject and save every subject having zero, multiple, deprecated, historical, or unit-incompatible candidate values. The audit report must count retained subjects, distinct values, and shared subjects across relations.

## Split and entity policy

Store canonical global IDs for every subject and value, not just labels. Report both:

1. **subject overlap**: test subjects versus train subjects;
2. **global entity overlap**: every test entity ID (subject or value) versus every train entity ID.

Subject-disjoint splits are acceptable for the first sandbox. Globally unseen-entity splits are a separate, stronger evaluation and must be reported rather than assumed.

## Surface policy

The metadata is `C = [C_domain, C_relation, C_subject, C_value]`. A text must express the same proposition `(relation, subject, value)` naturally. `C_domain` and the canonical relation label are metadata and must not be mechanically exposed in every text.

The schema supports `S = [S_family, S_variant]`. Sandbox 0 may use one variant per family; a later dataset should use two to four independent variants in natural-language families. No variants are generated by this document.

## Deliberately postponed

This audit does not build facts, select instances, decide contrastive positives or negatives, or train a model. The hierarchy is retained only as metadata for future analysis.
