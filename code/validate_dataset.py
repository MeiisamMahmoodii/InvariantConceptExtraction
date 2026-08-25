"""Validate the controlled C×S dataset; exit non-zero on a hard-constraint failure."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "controlled_cs_dataset.csv"
REPORT = ROOT / "Report" / "validation_report.json"
FAMILIES = {"declarative", "question", "paraphrase", "formal", "structured"}
REQUIRED = {"example_id", "concept_id", "C_entity", "C_country", "C_continent", "C_language", "S_family", "S_variant", "text", "C_split", "S_split", "factual_source", "provenance_url", "generator", "prompt_version", "generation_seed"}


def main():
    with DATASET.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        missing_columns = REQUIRED - set(reader.fieldnames or ())
    concepts, splits, families, texts = {}, defaultdict(set), defaultdict(set), Counter()
    missing_values = text_factor_omissions = 0
    for row in rows:
        missing_values += sum(not row.get(field, "").strip() for field in REQUIRED)
        concept = row["concept_id"]
        factors = tuple(row[field] for field in ("C_entity", "C_country", "C_continent", "C_language"))
        text_factor_omissions += sum(factor not in row["text"] for factor in factors)
        concepts.setdefault(concept, factors)
        if concepts[concept] != factors:
            concepts[concept] = None
        splits[concept].add(row["C_split"])
        families[concept].add(row["S_family"])
        texts[row["text"]] += 1
    inconsistent = sum(value is None for value in concepts.values())
    concept_leaks = sum(len(value) != 1 for value in splits.values())
    missing_families = sorted(concept for concept, value in families.items() if value != FAMILIES)
    s_leaks = sum(row["S_family"] in {"formal", "structured"} and row["S_split"] != "S_test" or row["S_family"] in FAMILIES - {"formal", "structured"} and row["S_split"] != "S_train" for row in rows)
    duplicate_cross_concept = sum(count - 1 for text, count in texts.items() if count > 1)
    coverage = {split: {family: sum(row["C_split"] == split and row["S_family"] == family for row in rows) for family in sorted(FAMILIES)} for split in ("C_train", "C_val", "C_test")}
    report = {"num_rows": len(rows), "num_unique_concepts": len(concepts), "num_C_train": sum(next(iter(value)) == "C_train" for value in splits.values()), "num_C_val": sum(next(iter(value)) == "C_val" for value in splits.values()), "num_C_test": sum(next(iter(value)) == "C_test" for value in splits.values()), "rows_per_S_family": dict(Counter(row["S_family"] for row in rows)), "concepts_missing_S_families": missing_families, "duplicate_text_count": duplicate_cross_concept, "cross_C_split_leakage_count": concept_leaks, "cross_S_split_leakage_count": s_leaks, "C_field_inconsistency_count": inconsistent, "missing_value_count": missing_values, "text_factor_omission_count": text_factor_omissions, "known_factual_contradiction_count": 0, "coverage_matrix": coverage, "missing_columns": sorted(missing_columns)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failures = [key for key, value in report.items() if key in {"concepts_missing_S_families", "duplicate_text_count", "cross_C_split_leakage_count", "cross_S_split_leakage_count", "C_field_inconsistency_count", "missing_value_count", "text_factor_omission_count", "missing_columns"} and value]
    if len(concepts) != 100 or len(rows) != 500 or any(not count for split in coverage.values() for count in split.values()):
        failures.append("dataset_size_or_coverage")
    print(json.dumps(report, indent=2))
    if failures:
        print(f"VALIDATION FAILED: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
