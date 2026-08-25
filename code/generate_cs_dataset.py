"""Build the controlled C×S sandbox from a local Wikidata snapshot."""

import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "Report"
SOURCE = DATA / "wikidata_capitals.json"
OUTPUT = DATA / "controlled_cs_dataset.csv"
REJECTED = DATA / "rejected_source_records.csv"
AUDIT = REPORT / "semantic_audit_samples.csv"
SAMPLE = REPORT / "human_readable_sample.csv"
SEED = 20260825
FAMILIES = ("declarative", "question", "paraphrase", "formal", "structured")


def value(row, name):
    return row[name]["value"]


def text_for(concept, family):
    e, country, continent, language = (concept[k] for k in ("C_entity", "C_country", "C_continent", "C_language"))
    if family == "declarative":
        return f"{e} is in {country}, a country in {continent} where {language} is an official language."
    if family == "question":
        return f"Which country, continent, and official language correspond to {e}? Answer: {country}; {continent}; {language}."
    if family == "paraphrase":
        return f"For {e}: country={country}; continent={continent}; official language={language}."
    if family == "formal":
        return f"Entity: {e}. Country: {country}. Continent: {continent}. Official language: {language}."
    return json.dumps({"entity": e, "country": country, "continent": continent, "language": language}, ensure_ascii=False, sort_keys=True)


def main():
    bindings = json.loads(SOURCE.read_text(encoding="utf-8"))["results"]["bindings"]
    chosen, rejected, seen_countries, seen_entities = [], [], set(), set()
    for row in bindings:
        try:
            entity, country, continent, language = (value(row, f"{key}Label") for key in ("entity", "country", "continent", "language"))
            country_id, entity_id = value(row, "country"), value(row, "entity")
            if any(label.startswith("Q") for label in (entity, country, continent, language)):
                raise ValueError("missing English label")
            if country_id in seen_countries or entity_id in seen_entities:
                continue
            seen_countries.add(country_id)
            seen_entities.add(entity_id)
            chosen.append({"C_entity": entity, "C_country": country, "C_continent": continent, "C_language": language,
                           "source_entity_id": entity_id, "source_country_id": country_id,
                           "provenance_url": f"https://www.wikidata.org/wiki/{country_id}",
                           "factual_source": "Wikidata SPARQL snapshot (2026-08-25)"})
            if len(chosen) == 100:
                break
        except (KeyError, ValueError) as error:
            rejected.append({"reason": str(error), "raw_record": json.dumps(row, ensure_ascii=False)})
    if len(chosen) != 100:
        raise RuntimeError(f"Need 100 usable concepts; found {len(chosen)}")

    rows = []
    for index, concept in enumerate(chosen, 1):
        c_split = "C_train" if index <= 70 else "C_val" if index <= 85 else "C_test"
        concept_id = f"capital-{index:03d}"
        for family in FAMILIES:
            rows.append({"example_id": f"{concept_id}-{family}", "concept_id": concept_id, **concept,
                         "S_family": family, "S_variant": "v1", "text": text_for(concept, family),
                         "C_split": c_split, "S_split": "S_train" if family in FAMILIES[:3] else "S_test",
                         "generator": "code/generate_cs_dataset.py", "prompt_version": "template-v1", "generation_seed": SEED})
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with REJECTED.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("reason", "raw_record"))
        writer.writeheader()
        writer.writerows(rejected)
    sample_ids = set(random.Random(SEED).sample([f"capital-{i:03d}" for i in range(1, 101)], 50))
    with AUDIT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row["concept_id"] in sample_ids)
    with SAMPLE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["concept_id"].rsplit("-", 1)[1]) <= 10)
    print(f"Created {OUTPUT}: {len(rows)} rows, {len(chosen)} concepts, {len(FAMILIES)} S-families.")
    print(f"Saved rejected source records to {REJECTED}: {len(rejected)} records.")
    print(f"Saved 50-group audit sample to {AUDIT}.")
    print(f"Saved 10-group readable table to {SAMPLE}.")


if __name__ == "__main__":
    main()
