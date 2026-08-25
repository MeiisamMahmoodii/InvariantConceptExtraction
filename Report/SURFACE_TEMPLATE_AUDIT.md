# Surface-template candidate audit

## What we inspected

216 candidate texts: 3 factual examples for each of 6 relations, rendered in all 5 proposed S-families with 12 controlled variants per fact. The candidate table is `surface_template_candidates.csv`.

## Result

| Check | Result |
|---|---|
| Subject and value are present in every candidate | PASS — 216/216 |
| Every sampled fact has all five S-families | PASS — 18/18 |
| Natural-language family variants are distinct | PASS — 3 each for declarative, question, and paraphrase |
| Cross-family variants are distinct | PASS — 12/12 per fact group |
| Candidate wording keeps the same relation, subject, and value | PASS — 18/18 fact groups |
| Candidate wording adds a separate factual proposition | PASS — none found |
| Canonical metadata labels or IDs appear in candidate text | PASS — none found |

## Manual template review

The geography templates state only capital city, continent, or currency. The science templates state only atomic number, periodic-table period, or chemical symbol. The atomic-number paraphrase uses “number of protons,” which is the defining quantity of atomic number; no isotope or mass claim is added. The period paraphrase uses “row” only in the explicit periodic-table context.

The `formal` and `structured` families use readable field names such as `capital_city` and `atomic_number`, never the canonical metadata labels (`capital_of`, `atomic_number_of`) or domain/split/source data. Natural-language subjects begin with normal sentence capitalization; the original source labels remain unchanged in metadata.

## Decision

The candidate-template audit passes. This document approves the templates for a later generation decision; it does not itself generate the full surface dataset.
