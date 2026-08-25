# Relation coverage report

## What we did

Ran a Wikidata direct best-rank-claim query for the six proposed relations. Saved one factual table per relation, a raw source snapshot, and a rejection log.

## Why

The subject intersection across all three relations in a domain matters more than each relation's separate size. No text dataset was created.

## What we found

| Domain | Relation | Retained subjects | Distinct values | Mean subjects/value | Singleton-value fraction |
|---|---|---:|---:|---:|---:|
| Geography | capital_of | 193 | 192 | 1.005 | 0.995 |
| Geography | continent_of | 193 | 7 | 27.571 | 0.000 |
| Geography | currency_of | 161 | 138 | 1.167 | 0.949 |
| Science | atomic_number_of | 174 | 174 | 1.000 | 1.000 |
| Science | period_of | 118 | 7 | 16.857 | 0.000 |
| Science | chemical_symbol_of | 174 | 174 | 1.000 | 1.000 |

The strict geography common-subject set is 148 countries, and the science common-subject set is 118 elements. Both pass the 50-subject requirement. Countries with unresolved Wikidata labels are rejected rather than repaired. `period_of` comes from the Royal Society of Chemistry fact box and is joined to Wikidata elements through the recorded atomic-number identity; period is never derived from atomic number.

31 multi-valued records were rejected and saved with their reasons. No unresolved multi-valued record was retained. The full machine-readable result is `relation_coverage_report.json`.

## Decision

The relation-coverage acceptance criteria pass. This report does not itself authorize factual-matrix or text construction.
