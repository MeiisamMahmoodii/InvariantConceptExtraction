"""Generate the full controlled surface dataset from the approved factual matrix."""

import csv
import random
from pathlib import Path

from render_surface_candidates import FAMILIES, render


ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "data" / "factual_matrix" / "factual_c_matrix.csv"
OUT = ROOT / "data" / "controlled_surface_dataset.csv"
REJECTIONS = ROOT / "data" / "surface_generation_rejections.csv"
AUDIT = ROOT / "Report" / "full_surface_audit_sample.csv"
SEED = 20260825


def main():
    with FACTS.open(newline="", encoding="utf-8") as file:
        facts = list(csv.DictReader(file))
    rows, rejected = [], []
    for fact in facts:
        for family in FAMILIES:
            for number, text in enumerate(render(fact["C_relation"], fact["C_subject_label"], fact["C_value_label"], family), 1):
                if not text or fact["C_subject_label"].casefold() not in text.casefold() or fact["C_value_label"].casefold() not in text.casefold():
                    rejected.append({"fact_id": fact["fact_id"], "S_family": family, "S_variant": number, "reason": "rendered text does not contain its subject and value"})
                    continue
                rows.append({**fact, "example_id": f"{fact['fact_id']}-{family}-v{number}", "S_family": family, "S_variant": f"v{number}", "S_split": "S_train" if family in {"declarative", "question", "paraphrase"} else "S_test", "text": text, "generator": "code/generate_controlled_surface_dataset.py", "template_version": "surface-v2", "generation_seed": SEED})
    fields = list(rows[0])
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with REJECTIONS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("fact_id", "S_family", "S_variant", "reason")); writer.writeheader(); writer.writerows(rejected)
    sample_ids = set(random.Random(SEED).sample([fact["fact_id"] for fact in facts], 50))
    with AUDIT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(row for row in rows if row["fact_id"] in sample_ids)
    print(f"Created {OUT}: {len(rows)} rows from {len(facts)} facts.")
    print(f"Saved {len(rejected)} rejected rows to {REJECTIONS}.")
    print(f"Saved fixed 50-fact audit sample to {AUDIT}.")


if __name__ == "__main__":
    main()
