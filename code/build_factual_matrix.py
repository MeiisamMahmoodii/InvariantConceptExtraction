"""Build the approved factual C-matrix only; no text or S realizations are created."""

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "relation_feasibility"
OUT = ROOT / "data" / "factual_matrix"
REPORT = ROOT / "Report"
DOMAINS = {
    "geography": {"relations": ("capital_of", "continent_of", "currency_of"), "subject_type": "country", "value_types": {"capital_of": "city", "continent_of": "continent", "currency_of": "currency"}},
    "science": {"relations": ("atomic_number_of", "period_of", "chemical_symbol_of"), "subject_type": "chemical_element", "value_types": {"atomic_number_of": "integer", "period_of": "period", "chemical_symbol_of": "chemical_symbol"}},
}


def read_relation(relation):
    with (SOURCE / f"{relation}.csv").open(newline="", encoding="utf-8") as file:
        return {row["subject_id"]: row for row in csv.DictReader(file)}


def split(subjects):
    total = len(subjects)
    train, val = round(total * .70), round(total * .15)
    return {subject: "C_train" if index < train else "C_val" if index < train + val else "C_test" for index, subject in enumerate(sorted(subjects))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, universes = [], []
    for domain, spec in DOMAINS.items():
        relations = {relation: read_relation(relation) for relation in spec["relations"]}
        subjects = set.intersection(*(set(records) for records in relations.values()))
        assignments = split(subjects)
        for subject_id in sorted(subjects):
            universes.append({"C_domain": domain, "C_subject_id": subject_id, "C_subject_label": relations[spec["relations"][0]][subject_id]["subject_label"], "relations": ";".join(spec["relations"]), "C_split": assignments[subject_id]})
            for relation in spec["relations"]:
                source = relations[relation][subject_id]
                if relation == "period_of":
                    provenance = source["qualifiers_status"].split("source=", 1)[1]
                    source_name, record_id = "Royal Society of Chemistry", f"rsc:{subject_id}:period"
                else:
                    property_id = {"capital_of": "P36", "continent_of": "P30", "currency_of": "P38", "atomic_number_of": "P1086", "chemical_symbol_of": "P246"}[relation]
                    provenance = f"https://www.wikidata.org/wiki/{subject_id}#{property_id}"
                    source_name, record_id = "Wikidata", f"wd:{subject_id}:{property_id}"
                rows.append({"fact_id": f"{domain}-{subject_id}-{relation}", "C_domain": domain, "C_relation": relation, "C_subject_id": subject_id, "C_subject_label": source["subject_label"], "C_value_id": source["value_id"], "C_value_label": source["value_label"], "C_subject_type": spec["subject_type"], "C_value_type": spec["value_types"][relation], "source_name": source_name, "source_record_id": record_id, "source_provenance": provenance, "C_split": assignments[subject_id]})
    fields = list(rows[0])
    with (OUT / "factual_c_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (OUT / "domain_common_subjects.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("C_domain", "C_subject_id", "C_subject_label", "relations", "C_split")); writer.writeheader(); writer.writerows(universes)
    report = {"num_facts": len(rows), "domain_subject_counts": dict(Counter(row["C_domain"] for row in universes)), "domain_fact_counts": dict(Counter(row["C_domain"] for row in rows)), "split_fact_counts": dict(Counter(row["C_split"] for row in rows))}
    (REPORT / "factual_matrix_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Created {OUT / 'factual_c_matrix.csv'} with no text or S fields.")


if __name__ == "__main__":
    main()
