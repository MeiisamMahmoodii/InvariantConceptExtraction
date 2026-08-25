"""Fail if full controlled surface generation violates the approved constraints."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "controlled_surface_dataset.csv"
REPORT = ROOT / "Report" / "controlled_surface_validation.json"
FAMILY_COUNTS = {"declarative": 3, "question": 3, "paraphrase": 3, "formal": 2, "structured": 1}
COLUMNS = ("C_domain", "C_relation", "C_subject_id", "C_subject_label", "C_value_id", "C_value_label", "C_subject_type", "C_value_type", "source_name", "source_record_id", "source_provenance", "C_split")


def main():
    with DATASET.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_fact, splits, texts = defaultdict(list), defaultdict(set), defaultdict(set)
    missing = 0
    for row in rows:
        by_fact[row["fact_id"]].append(row)
        splits[(row["C_domain"], row["C_subject_id"])].add(row["C_split"])
        texts[row["text"]].add(row["fact_id"])
        missing += sum(not row.get(field, "") for field in (*COLUMNS, "example_id", "S_family", "S_variant", "S_split", "text", "generator", "template_version", "generation_seed"))
    family_failures = sum(Counter(row["S_family"] for row in group) != FAMILY_COUNTS for group in by_fact.values())
    c_failures = sum(len({tuple(row[column] for column in COLUMNS) for row in group}) != 1 for group in by_fact.values())
    s_failures = sum((row["S_family"] in {"declarative", "question", "paraphrase"}) != (row["S_split"] == "S_train") for row in rows)
    report = {"num_rows": len(rows), "num_facts": len(by_fact), "rows_per_fact_failure_count": sum(len(group) != 12 for group in by_fact.values()), "family_count_failure_count": family_failures, "C_field_inconsistency_count": c_failures, "cross_subject_split_leakage_count": sum(len(value) != 1 for value in splits.values()), "S_split_failure_count": s_failures, "duplicate_text_across_fact_count": sum(1 for facts in texts.values() if len(facts) > 1), "missing_value_count": missing, "rows_per_S_family": dict(Counter(row["S_family"] for row in rows)), "provenance_missing_count": sum(not row["source_record_id"] or not row["source_provenance"] for row in rows)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failures = [key for key, value in report.items() if key.endswith("_count") and key not in {"num_rows", "num_facts"} and value]
    if report["num_rows"] != 10512 or report["num_facts"] != 876:
        failures.append("approved_size")
    print(json.dumps(report, indent=2))
    if failures:
        print(f"VALIDATION FAILED: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
