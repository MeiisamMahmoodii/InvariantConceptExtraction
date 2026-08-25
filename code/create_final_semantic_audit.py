"""Create and structurally check a fixed, stratified semantic-audit sample."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "controlled_surface_dataset.csv"
OUT = ROOT / "Report" / "final_semantic_spot_audit_sample.csv"
SEED = 20260825
FAMILIES = {"declarative": 3, "question": 3, "paraphrase": 3, "formal": 2, "structured": 1}
COLUMNS = ("C_domain", "C_relation", "C_subject_id", "C_subject_label", "C_value_id", "C_value_label")


def main():
    with DATASET.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_fact, facts_by_relation = defaultdict(list), defaultdict(set)
    for row in rows:
        by_fact[row["fact_id"]].append(row)
        facts_by_relation[row["C_relation"]].add(row["fact_id"])
    selected = set()
    for offset, relation in enumerate(sorted(facts_by_relation)):
        selected.update(random.Random(SEED + offset).sample(sorted(facts_by_relation[relation]), 10))
    sample = [row for row in rows if row["fact_id"] in selected]
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(sample[0])); writer.writeheader(); writer.writerows(sample)
    selected_groups = {fact_id: by_fact[fact_id] for fact_id in selected}
    c_failures = sum(len({tuple(row[column] for column in COLUMNS) for row in group}) != 1 for group in selected_groups.values())
    family_failures = sum(Counter(row["S_family"] for row in group) != FAMILIES for group in selected_groups.values())
    mention_failures = sum(row["C_subject_label"].casefold() not in row["text"].casefold() or row["C_value_label"].casefold() not in row["text"].casefold() for row in sample)
    print(f"audit_facts={len(selected)} audit_rows={len(sample)}")
    print(f"facts_per_relation={dict(Counter(row['C_relation'] for row in sample))}")
    print(f"C_consistency_failures={c_failures} family_coverage_failures={family_failures} factor_mention_failures={mention_failures}")
    if len(selected) != 60 or len(sample) != 720 or c_failures or family_failures or mention_failures:
        raise SystemExit("AUDIT SAMPLE VALIDATION FAILED")
    print("AUDIT SAMPLE VALIDATION PASSED")


if __name__ == "__main__":
    main()
