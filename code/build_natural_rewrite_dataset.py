"""Replace only paraphrase text with controlled natural rewrites; validate it."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from render_surface_candidates import render

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "three_domain_diversity"
OUT = ROOT / "data" / "three_domain_natural_rewrite"
REPORT = ROOT / "Report" / "three_domain_natural_rewrite_validation.json"


def main():
    with (SOURCE / "controlled_surface_dataset.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    changed = []
    for row in rows:
        if row["S_family"] != "paraphrase":
            continue
        number = int(row["S_variant"].removeprefix("v"))
        row["text"] = render(row["C_relation"], row["C_subject_label"], row["C_value_label"], "paraphrase")[number - 1]
        row["generator"] = "code/build_natural_rewrite_dataset.py"
        row["template_version"] = "three-domain-natural-rewrite-v1"
        changed.append(row)
    duplicates = defaultdict(set)
    for row in rows:
        duplicates[row["text"]].add(row["fact_id"])
    rejected_subjects = {
        (row["C_domain"], row["C_subject_id"])
        for row in rows
        if row["C_domain"] == "books" and len(duplicates[row["text"]]) > 1
    }
    rejection_rows = [{"C_domain": domain, "C_subject_id": subject, "reason": "cross_fact_exact_text_collision"} for domain, subject in sorted(rejected_subjects)]
    rows = [row for row in rows if (row["C_domain"], row["C_subject_id"]) not in rejected_subjects]
    by_fact = defaultdict(list)
    for row in rows:
        by_fact[row["fact_id"]].append(row)
    duplicates = defaultdict(set)
    for row in rows:
        duplicates[row["text"]].add(row["fact_id"])
    report = {
        "facts": len(by_fact),
        "surface_rows": len(rows),
        "natural_rewrite_rows": sum(row["S_family"] == "paraphrase" for row in rows),
        "changed_family": "paraphrase",
        "held_out_natural_family": "indirect",
        "subject_or_value_missing": sum(row["C_subject_label"].casefold() not in row["text"].casefold() or row["C_value_label"].casefold() not in row["text"].casefold() for row in changed),
        "rows_per_fact_failures": sum(len(group) != 13 for group in by_fact.values()),
        "cross_fact_exact_text_duplicates": sum(len(facts) > 1 for facts in duplicates.values()),
        "rejected_subjects": len(rejected_subjects),
        "training_performed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / "factual_c_matrix.csv").open(newline="", encoding="utf-8") as file:
        facts = [row for row in csv.DictReader(file) if (row["C_domain"], row["C_subject_id"]) not in rejected_subjects]
    with (OUT / "factual_c_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(facts[0])); writer.writeheader(); writer.writerows(facts)
    with (OUT / "controlled_surface_dataset.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / "rejection_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("C_domain", "C_subject_id", "reason")); writer.writeheader(); writer.writerows(rejection_rows)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(report[key] for key in ("subject_or_value_missing", "rows_per_fact_failures", "cross_fact_exact_text_duplicates")):
        raise SystemExit("natural rewrite validation failed")


if __name__ == "__main__":
    main()
