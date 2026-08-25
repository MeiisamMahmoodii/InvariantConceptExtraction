"""Render a 90-row surface-template preflight; not the final text dataset."""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "data" / "factual_matrix" / "factual_c_matrix.csv"
OUT = ROOT / "Report" / "surface_template_candidates.csv"
FAMILIES = ("declarative", "question", "paraphrase", "formal", "structured")


def render(relation, subject, value, family):
    templates = {
        "capital_of": (f"{value} is the capital city of {subject}.", f"Which city is the capital of {subject}? {value}.", f"{subject}'s national capital is {value}.", f"Country: {subject}. Capital city: {value}.", {"country": subject, "capital_city": value}),
        "continent_of": (f"{subject} is in {value}.", f"Which continent is {subject} in? {value}.", f"{value} is the continent that includes {subject}.", f"Country: {subject}. Continent: {value}.", {"country": subject, "continent": value}),
        "currency_of": (f"{subject} uses the {value}.", f"Which currency does {subject} use? {value}.", f"The currency used in {subject} is the {value}.", f"Country: {subject}. Currency: {value}.", {"country": subject, "currency": value}),
        "atomic_number_of": (f"{subject} has atomic number {value}.", f"What is the atomic number of {subject}? {value}.", f"For {subject}, the number of protons is {value}.", f"Element: {subject}. Atomic number: {value}.", {"element": subject, "atomic_number": value}),
        "period_of": (f"{subject} is in period {value} of the periodic table.", f"Which period contains {subject} on the periodic table? {value}.", f"On the periodic table, {subject} appears in row {value}.", f"Element: {subject}. Period: {value}.", {"element": subject, "period": value}),
        "chemical_symbol_of": (f"The chemical symbol for {subject} is {value}.", f"What is the chemical symbol for {subject}? {value}.", f"{subject} is written as {value} on the periodic table.", f"Element: {subject}. Symbol: {value}.", {"element": subject, "symbol": value}),
    }
    result = templates[relation][FAMILIES.index(family)]
    return json.dumps(result, ensure_ascii=False, sort_keys=True) if isinstance(result, dict) else result


def main():
    with FACTS.open(newline="", encoding="utf-8") as file:
        facts = list(csv.DictReader(file))
    by_relation = defaultdict(list)
    for fact in facts:
        by_relation[fact["C_relation"]].append(fact)
    rows = []
    for relation in sorted(by_relation):
        for fact in sorted(by_relation[relation], key=lambda row: row["fact_id"])[:3]:
            for family in FAMILIES:
                rows.append({"candidate_id": f"{fact['fact_id']}-{family}", "fact_id": fact["fact_id"], "C_relation": relation, "C_subject_label": fact["C_subject_label"], "C_value_label": fact["C_value_label"], "S_family": family, "S_variant": "v1_candidate", "candidate_text": render(relation, fact["C_subject_label"], fact["C_value_label"], family)})
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Created {OUT}: {len(rows)} candidate realizations (3 facts × 6 relations × 5 families).")


if __name__ == "__main__":
    main()
