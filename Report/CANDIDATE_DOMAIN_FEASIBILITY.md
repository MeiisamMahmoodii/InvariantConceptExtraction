# Candidate-domain relation feasibility

No text was generated and no model was trained. Values come from Wikidata direct best-rank claims. Retained source tables require exactly one value for all three relations; no multi-valued record was silently collapsed.

## Candidate status

| Domain | Status | Shared subjects |
|---|---|---:|
| Films | provisional: exact-one source table passes; explicit cardinality audit timed out | 1981 |
| Books | eligible: exact-one table and explicit cardinality rejection log pass | 1114 |
| Taxa | provisional: exact-one source table passes; explicit cardinality audit returned gateway error | 1970 |

## Films

| Relation | Retained subjects | Distinct values | Mean subjects/value | Singleton-value fraction |
|---|---:|---:|---:|---:|
| director_of | 1981 | 1472 | 1.35 | 0.798 |
| country_of_origin | 1981 | 61 | 32.48 | 0.295 |
| original_language | 1981 | 53 | 37.38 | 0.302 |

## Books

| Relation | Retained subjects | Distinct values | Mean subjects/value | Singleton-value fraction |
|---|---:|---:|---:|---:|
| author_of | 1114 | 545 | 2.04 | 0.774 |
| country_of_origin | 1114 | 66 | 16.88 | 0.409 |
| original_language | 1114 | 40 | 27.85 | 0.325 |

## Taxa

| Relation | Retained subjects | Distinct values | Mean subjects/value | Singleton-value fraction |
|---|---:|---:|---:|---:|
| taxon_rank | 1970 | 2 | 985.00 | 0.000 |
| parent_taxon | 1970 | 1228 | 1.60 | 0.757 |
| conservation_status | 1970 | 2 | 985.00 | 0.500 |

## Decision

Books is eligible for a later construction stage. Films and Taxa have strong exact-one coverage but are not approved yet because the explicit cardinality-rejection query failed at the source service; rerun that source audit before construction. People and Objects remain deferred.
