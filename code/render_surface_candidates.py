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
    # Independently authored conversational rewrites.  Each still states only
    # the source-backed proposition and preserves both canonical labels.
    natural_paraphrases = {
        "capital_of": [f"Asked for {subject_sentence}'s capital, the answer is {value}.", f"If you are naming the capital of {subject_sentence}, say {value}.", f"The city people mean when they mention {subject_sentence}'s capital is {value}."],
        "continent_of": [f"On a world map, you would place {subject_sentence} in {value}.", f"If you are locating {subject_sentence} by continent, it belongs in {value}.", f"The continent to associate with {subject_sentence} is {value}."],
        "currency_of": [f"When paying in {subject_sentence}, the currency is the {value}.", f"A visit to {subject_sentence} means using the {value} for money.", f"For everyday payments, {subject_sentence} uses the {value}."],
        "atomic_number_of": [f"For {subject_sentence}, the atomic-number value to remember is {value}.", f"In a chemistry lookup, {subject_sentence} is listed under atomic number {value}.", f"The number attached to {subject_sentence} on the periodic table is {value}."],
        "period_of": [f"Reading the periodic table by rows puts {subject_sentence} in period {value}.", f"On the periodic table, look for {subject_sentence} in period {value}.", f"The row assigned to {subject_sentence} is period {value}."],
        "chemical_symbol_of": [f"Chemists write {subject_sentence} as {value}.", f"In chemical notation, {subject_sentence} appears as {value}.", f"The shorthand used for {subject_sentence} is {value}."],
        "author_of": [f"The person who wrote {subject_sentence} is {value}.", f"For {subject_sentence}, the writer's name is {value}.", f"If you are crediting {subject_sentence}, credit {value} as its author."],
        "country_of_origin": [f"{subject_sentence} comes from {value}.", f"To place {subject_sentence} by origin, use {value}.", f"The country associated with {subject_sentence}'s origin is {value}."],
        "original_language": [f"{subject_sentence} first appeared in {value}.", f"The language {subject_sentence} was originally written in is {value}.", f"For the original version of {subject_sentence}, the language is {value}."],
    }
    if family == "paraphrase":
        return natural_paraphrases[relation]
    book_templates = {
        "author_of": {"declarative": [f"{value} wrote {subject_sentence}.", f"The author of {subject_sentence} is {value}.", f"{subject_sentence} was written by {value}."], "question": [f"Who wrote {subject_sentence}? {value}.", f"Who is the author of {subject_sentence}? {value}.", f"{subject_sentence} was written by whom? {value}."], "paraphrase": [f"{value} is credited as the author of {subject_sentence}.", f"The writer behind {subject_sentence} is {value}.", f"For {subject_sentence}, the named author is {value}.",], "formal": [f"Book: {subject_sentence}. Author: {value}.", f"Author — {value}; book — {subject_sentence}."], "structured": [{"book": subject_sentence, "author": value}], "indirect": [f"To identify who wrote {subject_sentence}, use {value}."]},
        "country_of_origin": {"declarative": [f"{subject_sentence} originates from {value}.", f"The country of origin of {subject_sentence} is {value}.", f"{value} is the country of origin for {subject_sentence}."], "question": [f"Which country is {subject_sentence} from? {value}.", f"What is the country of origin of {subject_sentence}? {value}.", f"{subject_sentence} originates in which country? {value}."], "paraphrase": [f"{subject_sentence} is a work from {value}.", f"For {subject_sentence}, the origin country is {value}.", f"The work {subject_sentence} comes from {value}.",], "formal": [f"Book: {subject_sentence}. Country of origin: {value}.", f"Country of origin — {value}; book — {subject_sentence}."], "structured": [{"book": subject_sentence, "country_of_origin": value}], "indirect": [f"To place {subject_sentence} by country of origin, use {value}."]},
        "original_language": {"declarative": [f"{subject_sentence} was originally written in {value}.", f"The original language of {subject_sentence} is {value}.", f"{value} is the original language of {subject_sentence}."], "question": [f"What language was {subject_sentence} originally written in? {value}.", f"Which is the original language of {subject_sentence}? {value}.", f"{subject_sentence} was first written in which language? {value}."], "paraphrase": [f"{subject_sentence} first appeared in {value}.", f"For {subject_sentence}, the source language is {value}.", f"The language of the original version of {subject_sentence} is {value}.",], "formal": [f"Book: {subject_sentence}. Original language: {value}.", f"Original language — {value}; book — {subject_sentence}."], "structured": [{"book": subject_sentence, "original_language": value}], "indirect": [f"To identify the original language of {subject_sentence}, use {value}."]},
    }
    if relation in book_templates:
        return [json.dumps(result, ensure_ascii=False, sort_keys=True) if isinstance(result, dict) else result for result in book_templates[relation][family]]
    templates = {
        "capital_of": {"declarative": [f"{value} is the capital city of {subject_sentence}.", f"{subject_sentence} has {value} as its capital.", f"The capital of {subject_sentence} is {value}."], "question": [f"Which city is the capital of {subject_sentence}? {value}.", f"What is the capital of {subject_sentence}? {value}.", f"{subject_sentence} has which city as its capital? {value}."], "paraphrase": [f"In {subject_sentence}, the capital city is {value}.", f"{value} holds the role of capital for {subject_sentence}.", f"For {subject_sentence}, the capital is {value}."], "formal": [f"Country: {subject_sentence}. Capital city: {value}.", f"Capital city — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "capital_city": value}], "indirect": [f"For {subject_sentence}, the city to name as its capital is {value}."]},
        "continent_of": {"declarative": [f"{subject_sentence} is in {value}.", f"{subject_sentence} lies in {value}.", f"{value} includes {subject_sentence}."], "question": [f"Which continent is {subject_sentence} in? {value}.", f"{subject_sentence} belongs to which continent? {value}.", f"What continent contains {subject_sentence}? {value}."], "paraphrase": [f"The continent containing {subject_sentence} is {value}.", f"{subject_sentence} is part of {value}.", f"For {subject_sentence}, the relevant continent is {value}.",], "formal": [f"Country: {subject_sentence}. Continent: {value}.", f"Continent — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "continent": value}], "indirect": [f"To place {subject_sentence} by continent, use {value}."]},
        "currency_of": {"declarative": [f"{subject_sentence} uses the {value}.", f"The {value} is used in {subject_sentence}.", f"{subject_sentence}'s currency is the {value}."], "question": [f"Which currency does {subject_sentence} use? {value}.", f"What currency is used in {subject_sentence}? {value}.", f"{subject_sentence} uses which currency? {value}."], "paraphrase": [f"For {subject_sentence}, the currency is the {value}.", f"The currency associated with {subject_sentence} is the {value}.", f"In {subject_sentence}, payment uses the {value}.",], "formal": [f"Country: {subject_sentence}. Currency: {value}.", f"Currency — {value}; country — {subject_sentence}."], "structured": [{"country": subject_sentence, "currency": value}], "indirect": [f"For transactions in {subject_sentence}, the currency to use is the {value}."]},
        "atomic_number_of": {"declarative": [f"{subject_sentence} has atomic number {value}.", f"{subject_sentence} is assigned atomic number {value}.", f"The atomic number associated with {subject_sentence} is {value}."], "question": [f"What is the atomic number of {subject_sentence}? {value}.", f"Which atomic number belongs to {subject_sentence}? {value}.", f"{subject_sentence} has what atomic number? {value}."], "paraphrase": [f"An atom of {subject_sentence} contains {value} protons.", f"{value} is the atomic number assigned to {subject_sentence}.", f"{subject_sentence} occupies atomic-number position {value}.",], "formal": [f"Element: {subject_sentence}. Atomic number: {value}.", f"Atomic number — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "atomic_number": value}], "indirect": [f"To identify {subject_sentence} by atomic number, use {value}."]},
        "period_of": {"declarative": [f"{subject_sentence} is in period {value} of the periodic table.", f"{subject_sentence} belongs to period {value} on the periodic table.", f"Period {value} of the periodic table contains {subject_sentence}."], "question": [f"Which period contains {subject_sentence} on the periodic table? {value}.", f"What periodic-table period is {subject_sentence} in? {value}.", f"{subject_sentence} appears in which period? {value}.",], "paraphrase": [f"On the periodic table, {subject_sentence} appears in row {value}.", f"The periodic-table row for {subject_sentence} is {value}.", f"{subject_sentence} is placed on row {value} of the periodic table.",], "formal": [f"Element: {subject_sentence}. Period: {value}.", f"Period — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "period": value}], "indirect": [f"To place {subject_sentence} on the periodic table, use period {value}."]},
        "chemical_symbol_of": {"declarative": [f"The chemical symbol for {subject_sentence} is {value}.", f"{subject_sentence}'s chemical symbol is {value}.", f"{value} is the chemical symbol of {subject_sentence}."], "question": [f"What is the chemical symbol for {subject_sentence}? {value}.", f"Which symbol represents {subject_sentence}? {value}.", f"{subject_sentence} is represented by which symbol? {value}.",], "paraphrase": [f"{subject_sentence} is written as {value} on the periodic table.", f"On the periodic table, {value} denotes {subject_sentence}.", f"The notation {value} stands for {subject_sentence}.",], "formal": [f"Element: {subject_sentence}. Symbol: {value}.", f"Symbol — {value}; element — {subject_sentence}."], "structured": [{"element": subject_sentence, "symbol": value}], "indirect": [f"When writing {subject_sentence} in chemical notation, use {value}."]},
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
