"""Build and validate the Geography+Science+Books factual/surface expansion."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from render_surface_candidates import FAMILIES, render

ROOT = Path(__file__).resolve().parents[1]
BASE_FACTS = ROOT / "data" / "factual_matrix" / "factual_c_matrix.csv"
BASE_SURFACE = ROOT / "data" / "controlled_surface_dataset_with_indirect.csv"
BOOKS = ROOT / "data" / "candidate_domain_feasibility"
OUT = ROOT / "data" / "three_domain_diversity"
REPORT = ROOT / "Report" / "three_domain_dataset_validation.json"
RELATIONS = ("author_of", "country_of_origin", "original_language")


def split(subject):
    bucket = int(hashlib.sha256(subject.encode()).hexdigest(), 16) % 100
    return "C_train" if bucket < 70 else "C_val" if bucket < 85 else "C_test"


def main():
    with BASE_FACTS.open(newline="", encoding="utf-8") as file: facts = list(csv.DictReader(file))
    with BASE_SURFACE.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    tables = {}
    for relation in RELATIONS:
        with (BOOKS / f"books_{relation}.csv").open(newline="", encoding="utf-8") as file: tables[relation] = {r["subject_id"]: r for r in csv.DictReader(file)}
    subjects = set.intersection(*(set(tables[r]) for r in RELATIONS)); assert len(subjects) == 1114
    book_facts = []
    for subject in sorted(subjects):
        for relation in RELATIONS:
            row = tables[relation][subject]; property_id = {"author_of": "P50", "country_of_origin": "P495", "original_language": "P407"}[relation]
            book_facts.append({"fact_id": f"books-{subject}-{relation}", "C_domain": "books", "C_relation": relation, "C_subject_id": subject, "C_subject_label": row["subject_label"], "C_value_id": row["value_id"], "C_value_label": row["value_label"], "C_subject_type": "book", "C_value_type": {"author_of": "author", "country_of_origin": "country", "original_language": "language"}[relation], "source_name": "Wikidata", "source_record_id": f"wd:{subject}:{property_id}", "source_provenance": f"https://www.wikidata.org/wiki/{subject}#{property_id}", "C_split": split(subject)})
    facts += book_facts
    OUT.mkdir(parents=True, exist_ok=True); fields = list(facts[0])
    with (OUT / "factual_c_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(facts)
    for fact in book_facts:
        for family in (*FAMILIES, "indirect"):
            for number, text in enumerate(render(fact["C_relation"], fact["C_subject_label"], fact["C_value_label"], family), 1):
                assert fact["C_subject_label"].casefold() in text.casefold() and fact["C_value_label"].casefold() in text.casefold()
                rows.append({**fact, "example_id": f"{fact['fact_id']}-{family}-v{number}", "S_family": family, "S_variant": f"v{number}", "S_split": "S_test" if family == "indirect" else "S_train", "text": text, "generator": "code/build_three_domain_dataset.py", "template_version": "three-domain-v1", "generation_seed": "20260825"})
    with (OUT / "controlled_surface_dataset.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_subject, by_fact = defaultdict(list), defaultdict(list)
    for fact in facts: by_subject[(fact["C_domain"], fact["C_subject_id"])].append(fact)
    for row in rows: by_fact[row["fact_id"]].append(row)
    report = {"facts": len(facts), "surface_rows": len(rows), "domain_subjects": dict(Counter((f["C_domain"] for f in facts))), "book_subjects": len(subjects), "book_shared_intersection": len(subjects), "facts_per_subject_failures": sum(len(group) != 3 for group in by_subject.values()), "surface_rows_per_fact_failures": sum(len(group) != 13 for group in by_fact.values()), "C_inconsistency_failures": sum(len({tuple(r[k] for k in fields) for r in group}) != 1 for group in by_fact.values()), "unresolved_label_count": sum(f["C_value_label"].startswith("Q") and f["C_value_label"][1:].isdigit() for f in facts), "provenance_missing_count": sum(not f["source_record_id"] or not f["source_provenance"] for f in facts), "training_performed": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))
    if any(report[k] for k in ("facts_per_subject_failures", "surface_rows_per_fact_failures", "C_inconsistency_failures", "unresolved_label_count", "provenance_missing_count")): raise SystemExit("validation failed")


if __name__ == "__main__": main()
