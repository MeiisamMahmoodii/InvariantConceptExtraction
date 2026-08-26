"""Create the fixed-template control on the natural-rewrite retained subjects."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "three_domain_diversity"
NATURAL = ROOT / "data" / "three_domain_natural_rewrite"
OUT = ROOT / "data" / "three_domain_template_matched"
REPORT = ROOT / "Report" / "three_domain_template_matched_validation.json"


def main():
    with (NATURAL / "rejection_log.csv").open(newline="", encoding="utf-8") as file:
        rejected = {(row["C_domain"], row["C_subject_id"]) for row in csv.DictReader(file)}
    with (SOURCE / "controlled_surface_dataset.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if (row["C_domain"], row["C_subject_id"]) not in rejected]
    with (SOURCE / "factual_c_matrix.csv").open(newline="", encoding="utf-8") as file:
        facts = [row for row in csv.DictReader(file) if (row["C_domain"], row["C_subject_id"]) not in rejected]
    by_fact = defaultdict(list)
    for row in rows: by_fact[row["fact_id"]].append(row)
    report = {"facts": len(facts), "surface_rows": len(rows), "rejected_subjects": len(rejected), "rows_per_fact_failures": sum(len(group) != 13 for group in by_fact.values()), "same_subject_population_as_natural_rewrite": True, "training_performed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, values in (("factual_c_matrix.csv", facts), ("controlled_surface_dataset.csv", rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    shutil.copyfile(NATURAL / "rejection_log.csv", OUT / "rejection_log.csv")
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["rows_per_fact_failures"]: raise SystemExit("matched control validation failed")


if __name__ == "__main__": main()
