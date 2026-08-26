"""Summarize candidate-domain source tables without constructing text data."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "candidate_domain_feasibility"
OUT = ROOT / "Report" / "CANDIDATE_DOMAIN_FEASIBILITY.md"
JSON_OUT = ROOT / "Report" / "candidate_domain_feasibility_final.json"
SPECS = {
    "films": ("director_of", "country_of_origin", "original_language"),
    "books": ("author_of", "country_of_origin", "original_language"),
    "taxa": ("taxon_rank", "parent_taxon", "conservation_status"),
}
STATUS = {"films": "provisional: exact-one source table passes; explicit cardinality audit timed out", "books": "eligible: exact-one table and explicit cardinality rejection log pass", "taxa": "provisional: exact-one source table passes; explicit cardinality audit returned gateway error"}


def main():
    result = {"policy": "one direct Wikidata value per relation; no multi-valued value is chosen", "domains": {}}
    for domain, relations in SPECS.items():
        tables = {}
        for relation in relations:
            with (DATA / f"{domain}_{relation}.csv").open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
            counts = Counter(row["value_id"] for row in rows)
            tables[relation] = {"retained_subjects": len(rows), "distinct_values": len(counts), "mean_subjects_per_value": len(rows) / len(counts), "singleton_value_fraction": sum(n == 1 for n in counts.values()) / len(counts), "subjects": {row["subject_id"] for row in rows}}
        shared = len(set.intersection(*(item["subjects"] for item in tables.values())))
        result["domains"][domain] = {"status": STATUS[domain], "relations": {key: {k: v for k, v in value.items() if k != "subjects"} for key, value in tables.items()}, "shared_subject_intersection": shared}
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Candidate-domain relation feasibility", "", "No text was generated and no model was trained. Values come from Wikidata direct best-rank claims. Retained source tables require exactly one value for all three relations; no multi-valued record was silently collapsed.", "", "## Candidate status", "", "| Domain | Status | Shared subjects |", "|---|---|---:|"]
    for domain, item in result["domains"].items(): lines.append(f"| {domain.title()} | {item['status']} | {item['shared_subject_intersection']} |")
    for domain, item in result["domains"].items():
        lines += ["", f"## {domain.title()}", "", "| Relation | Retained subjects | Distinct values | Mean subjects/value | Singleton-value fraction |", "|---|---:|---:|---:|---:|"]
        for relation, stats in item["relations"].items(): lines.append(f"| {relation} | {stats['retained_subjects']} | {stats['distinct_values']} | {stats['mean_subjects_per_value']:.2f} | {stats['singleton_value_fraction']:.3f} |")
    lines += ["", "## Decision", "", "Books is eligible for a later construction stage. Films and Taxa have strong exact-one coverage but are not approved yet because the explicit cardinality-rejection query failed at the source service; rerun that source audit before construction. People and Objects remain deferred."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
