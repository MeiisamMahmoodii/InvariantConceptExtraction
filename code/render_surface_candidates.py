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
    subject_sentence = subject[:1].upper() + subject[1:]
    templates = {
        "capital_of": {"declarative": [f"{value} is the capital city of {subject_sentence}.", f"{subject_sentence} has {value} as its capital.", f"The capital of {subject_sentence} is {value}."], "question": [f"Which city is the capital of {subject_sentence}? {value}.", f"What is the capital of {subject_sentence}? {value}.", f"{subject_sentence} has which city as its capital? {value}."], "paraphrase": [f"In {subject_sentence}, the capital city is {value}.", f"{value} holds the role of capital for {subject_sentence}.", f"For {subject_sentence}, the capital is {value}."], "formal": [f"Country: {subject_sentence}. Capital city: {value}.", f"Capital city — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "capital_city": value}]},
        "continent_of": {"declarative": [f"{subject_sentence} is in {value}.", f"{subject_sentence} lies in {value}.", f"{value} includes {subject_sentence}."], "question": [f"Which continent is {subject_sentence} in? {value}.", f"{subject_sentence} belongs to which continent? {value}.", f"What continent contains {subject_sentence}? {value}."], "paraphrase": [f"The continent containing {subject_sentence} is {value}.", f"{subject_sentence} is part of {value}.", f"For {subject_sentence}, the relevant continent is {value}.",], "formal": [f"Country: {subject_sentence}. Continent: {value}.", f"Continent — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "continent": value}]},
        "currency_of": {"declarative": [f"{subject_sentence} uses the {value}.", f"The {value} is used in {subject_sentence}.", f"{subject_sentence}'s currency is the {value}."], "question": [f"Which currency does {subject_sentence} use? {value}.", f"What currency is used in {subject_sentence}? {value}.", f"{subject_sentence} uses which currency? {value}."], "paraphrase": [f"For {subject_sentence}, the currency is the {value}.", f"The currency associated with {subject_sentence} is the {value}.", f"In {subject_sentence}, payment uses the {value}.",], "formal": [f"Country: {subject_sentence}. Currency: {value}.", f"Currency — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "currency": value}]},
        "atomic_number_of": {"declarative": [f"{subject_sentence} has atomic number {value}.", f"{subject_sentence} is assigned atomic number {value}.", f"The atomic number associated with {subject_sentence} is {value}."], "question": [f"What is the atomic number of {subject_sentence}? {value}.", f"Which atomic number belongs to {subject_sentence}? {value}.", f"{subject_sentence} has what atomic number? {value}."], "paraphrase": [f"An atom of {subject_sentence} contains {value} protons.", f"{value} is the atomic number assigned to {subject_sentence}.", f"{subject_sentence} occupies atomic-number position {value}.",], "formal": [f"Element: {subject_sentence}. Atomic number: {value}.", f"Atomic number — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "atomic_number": value}]},
        "period_of": {"declarative": [f"{subject_sentence} is in period {value} of the periodic table.", f"{subject_sentence} belongs to period {value} on the periodic table.", f"Period {value} of the periodic table contains {subject_sentence}."], "question": [f"Which period contains {subject_sentence} on the periodic table? {value}.", f"What periodic-table period is {subject_sentence} in? {value}.", f"{subject_sentence} appears in which period? {value}.",], "paraphrase": [f"On the periodic table, {subject_sentence} appears in row {value}.", f"The periodic-table row for {subject_sentence} is {value}.", f"{subject_sentence} is placed on row {value} of the periodic table.",], "formal": [f"Element: {subject_sentence}. Period: {value}.", f"Period — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "period": value}]},
        "chemical_symbol_of": {"declarative": [f"The chemical symbol for {subject_sentence} is {value}.", f"{subject_sentence}'s chemical symbol is {value}.", f"{value} is the chemical symbol of {subject_sentence}."], "question": [f"What is the chemical symbol for {subject_sentence}? {value}.", f"Which symbol represents {subject_sentence}? {value}.", f"{subject_sentence} is represented by which symbol? {value}.",], "paraphrase": [f"{subject_sentence} is written as {value} on the periodic table.", f"On the periodic table, {value} denotes {subject_sentence}.", f"The notation {value} stands for {subject_sentence}.",], "formal": [f"Element: {subject_sentence}. Symbol: {value}.", f"Symbol — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "symbol": value}]},
    }
    return [json.dumps(result, ensure_ascii=False, sort_keys=True) if isinstance(result, dict) else result for result in templates[relation][family]]


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
                for number, text in enumerate(render(relation, fact["C_subject_label"], fact["C_value_label"], family), 1):
                    rows.append({"candidate_id": f"{fact['fact_id']}-{family}-v{number}", "fact_id": fact["fact_id"], "C_relation": relation, "C_subject_label": fact["C_subject_label"], "C_value_label": fact["C_value_label"], "S_family": family, "S_variant": f"v{number}_candidate", "candidate_text": text})
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Created {OUT}: {len(rows)} candidate realizations (3 facts × 6 relations × 12 variants).")


if __name__ == "__main__":
    main()
