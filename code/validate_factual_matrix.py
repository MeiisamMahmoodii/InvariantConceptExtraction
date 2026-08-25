"""Fail if the factual C-matrix violates its approved structure."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "factual_matrix"
MATRIX = DATA / "factual_c_matrix.csv"
UNIVERSE = DATA / "domain_common_subjects.csv"
REPORT = ROOT / "Report" / "factual_matrix_validation.json"
REQUIRED = {"fact_id", "C_domain", "C_relation", "C_subject_id", "C_subject_label", "C_value_id", "C_value_label", "C_subject_type", "C_value_type", "source_name", "source_record_id", "source_provenance", "C_split"}
EXPECTED = {"geography": (174, {"capital_of", "continent_of", "currency_of"}), "science": (118, {"atomic_number_of", "period_of", "chemical_symbol_of"})}


def load(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader), set(reader.fieldnames or ())


def main():
    rows, columns = load(MATRIX)
    universe, _ = load(UNIVERSE)
    subject_rows, subject_splits, relation_sets = defaultdict(list), defaultdict(set), defaultdict(set)
    missing = 0
    for row in rows:
        missing += sum(not row.get(field, "") for field in REQUIRED)
        key = (row["C_domain"], row["C_subject_id"])
        subject_rows[key].append(row)
        subject_splits[key].add(row["C_split"])
        relation_sets[key].add(row["C_relation"])
    domain_subjects = {domain: {subject for (row_domain, subject) in subject_rows if row_domain == domain} for domain in EXPECTED}
    value_counts = {relation: Counter(row["C_value_id"] for row in rows if row["C_relation"] == relation) for relation in {row["C_relation"] for row in rows}}
    universe_keys = {(row["C_domain"], row["C_subject_id"]) for row in universe}
    report = {"num_facts": len(rows), "subject_count_by_domain": {domain: len(subjects) for domain, subjects in domain_subjects.items()}, "facts_per_subject_failure_count": sum(len(value) != 3 for value in subject_rows.values()), "relation_set_failure_count": sum(relation_sets[(domain, subject)] != expected_relations for domain, (_, expected_relations) in EXPECTED.items() for subject in domain_subjects[domain]), "subject_split_leakage_count": sum(len(value) != 1 for value in subject_splits.values()), "common_universe_mismatch_count": len(set(subject_rows) ^ universe_keys), "missing_value_count": missing, "missing_required_columns": sorted(REQUIRED - columns), "universe_subject_count": len(universe), "unexpected_text_or_S_columns": sorted(column for column in columns if column == "text" or column.startswith("S_")), "value_reuse": {relation: {"distinct_values": len(counts), "mean_subjects_per_value": sum(counts.values()) / len(counts), "singleton_value_fraction": sum(value == 1 for value in counts.values()) / len(counts)} for relation, counts in value_counts.items()}}
    failures = [key for key in ("facts_per_subject_failure_count", "relation_set_failure_count", "subject_split_leakage_count", "common_universe_mismatch_count", "missing_value_count", "missing_required_columns", "unexpected_text_or_S_columns") if report[key]]
    if len(rows) != 876 or report["subject_count_by_domain"] != {"geography": 174, "science": 118} or len(universe) != 292:
        failures.append("approved_matrix_size")
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        print(f"VALIDATION FAILED: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
