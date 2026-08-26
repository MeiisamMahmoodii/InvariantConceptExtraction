"""Append one held-out indirect natural-language family without altering V1."""

import csv
from pathlib import Path

from render_surface_candidates import render

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "controlled_surface_dataset.csv"
FACTS = ROOT / "data" / "factual_matrix" / "factual_c_matrix.csv"
OUT = ROOT / "data" / "controlled_surface_dataset_with_indirect.csv"
AUDIT = ROOT / "Report" / "indirect_family_candidates.csv"


def main():
    with BASE.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    with FACTS.open(newline="", encoding="utf-8") as file: facts = list(csv.DictReader(file))
    indirect = []
    for fact in facts:
        text = render(fact["C_relation"], fact["C_subject_label"], fact["C_value_label"], "indirect")[0]
        assert fact["C_subject_label"].casefold() in text.casefold() and fact["C_value_label"].casefold() in text.casefold()
        indirect.append({**fact, "example_id": f"{fact['fact_id']}-indirect-v1", "S_family": "indirect", "S_variant": "v1", "S_split": "S_test", "text": text, "generator": "code/generate_indirect_family_expansion.py", "template_version": "indirect-v1", "generation_seed": "20260825"})
    fields = list(rows[0])
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows + indirect)
    sample = []
    for relation in sorted({f["C_relation"] for f in facts}): sample.extend([r for r in indirect if r["C_relation"] == relation][:3])
    with AUDIT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(sample)
    print(f"Created {OUT}: {len(rows) + len(indirect)} rows; appended={len(indirect)} indirect rows.")
    print(f"Saved {len(sample)} fixed indirect candidates to {AUDIT}.")


if __name__ == "__main__": main()
