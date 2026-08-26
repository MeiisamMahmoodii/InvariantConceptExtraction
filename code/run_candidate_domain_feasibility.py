"""Source-backed feasibility audit for candidate multi-domain expansion; no text."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "candidate_domain_feasibility"
REPORT = ROOT / "Report" / "candidate_domain_feasibility_report.json"
ENDPOINT = "https://query.wikidata.org/sparql"
CANDIDATES = {
    "films": {"class": "Q11424", "relations": {"director_of": "P57", "country_of_origin": "P495", "original_language": "P364"}},
    "books": {"class": "Q571", "relations": {"author_of": "P50", "country_of_origin": "P495", "original_language": "P407"}},
    "taxa": {"class": "Q16521", "relations": {"taxon_rank": "P105", "parent_taxon": "P171", "conservation_status": "P141"}},
}


def query(domain, spec):
    relations = list(spec["relations"].items())
    select = " ".join(f"(SAMPLE(?v{i}) AS ?value{i})" for i in range(3))
    where = " ".join(f"?subject wdt:{pid} ?v{i}." for i, (_, pid) in enumerate(relations))
    having = " && ".join(f"COUNT(DISTINCT ?v{i}) = 1" for i in range(3))
    sparql = f'''SELECT ?subject ?subjectLabel ?value0 ?value0Label ?value1 ?value1Label ?value2 ?value2Label WHERE {{
      {{ SELECT ?subject {select} WHERE {{ ?subject wdt:P31 wd:{spec['class']}. {where} }} GROUP BY ?subject HAVING ({having}) LIMIT 2000 }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}'''
    request = Request(f"{ENDPOINT}?{urlencode({'format': 'json', 'query': sparql})}", headers={"User-Agent": "InvariantConceptExtraction/1.0"})
    with urlopen(request, timeout=180) as response: bindings = json.load(response)["results"]["bindings"]
    return sparql, bindings


def query_cardinality(spec):
    relations = list(spec["relations"].items())
    counts = " ".join(f"(COUNT(DISTINCT ?v{i}) AS ?n{i})" for i in range(3))
    optional = " ".join(f"OPTIONAL {{ ?subject wdt:{pid} ?v{i}. }}" for i, (_, pid) in enumerate(relations))
    sparql = f'''SELECT ?subject ?subjectLabel ?n0 ?n1 ?n2 WHERE {{
      {{ SELECT ?subject {counts} WHERE {{ ?subject wdt:P31 wd:{spec['class']}. {optional} }} GROUP BY ?subject LIMIT 2000 }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}'''
    request = Request(f"{ENDPOINT}?{urlencode({'format': 'json', 'query': sparql})}", headers={"User-Agent": "InvariantConceptExtraction/1.0"})
    with urlopen(request, timeout=180) as response: return sparql, json.load(response)["results"]["bindings"]


def value(row, name): return row[name]["value"]
def entity_id(raw): return raw.rsplit("/", 1)[1] if raw.startswith("http://www.wikidata.org/entity/") else f"literal:{raw}"
def missing_label(label): return label.startswith("Q") and label[1:].isdigit()


def main():
    OUT.mkdir(parents=True, exist_ok=True); report = {"source": "Wikidata SPARQL direct best-rank values", "policy": "retain only subjects with exactly one direct value per chosen relation and English source labels", "domains": {}, "accepted_domains": [], "rejected_domains": []}
    for domain, spec in CANDIDATES.items():
        try: sparql, bindings = query(domain, spec); count_query, cardinality = query_cardinality(spec)
        except Exception as error:
            report["domains"][domain] = {"status": "query_failed", "error": str(error)}; report["rejected_domains"].append(domain); print(f"domain={domain} query_failed={error}"); continue
        (OUT / f"{domain}_wikidata_snapshot.json").write_text(json.dumps({"exact_one_query": sparql, "exact_one_bindings": bindings, "cardinality_audit_query": count_query, "cardinality_audit_bindings": cardinality}, ensure_ascii=False, indent=2), encoding="utf-8")
        retained, rejected, relation_rows = {}, [], defaultdict(dict)
        relations = list(spec["relations"])
        for row in cardinality:
            subject_id = entity_id(value(row, "subject"))
            for i, relation in enumerate(relations):
                count = int(value(row, f"n{i}"))
                if count != 1:
                    rejected.append({"subject_id": subject_id, "relation": relation, "reason": f"cardinality audit: expected one direct value; found {count}"})
        for row in bindings:
            subject_id, subject_label = entity_id(value(row, "subject")), value(row, "subjectLabel")
            values = []
            for i, relation in enumerate(relations):
                raw, label = value(row, f"value{i}"), value(row, f"value{i}Label")
                if missing_label(subject_label) or missing_label(label):
                    rejected.append({"subject_id": subject_id, "relation": relation, "reason": "missing English source label"}); break
                values.append((relation, entity_id(raw), label))
            else:
                retained[subject_id] = subject_label
                for relation, value_id, value_label in values:
                    relation_rows[relation][subject_id] = {"subject_id": subject_id, "subject_label": subject_label, "relation": relation, "value_id": value_id, "value_label": value_label, "qualifiers_status": "Wikidata direct best-rank value; query required exactly one value"}
        fields = ("subject_id", "subject_label", "relation", "value_id", "value_label", "qualifiers_status")
        for relation in relations:
            with (OUT / f"{domain}_{relation}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(relation_rows[relation].values())
        with (OUT / f"{domain}_rejections.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("subject_id", "relation", "reason")); writer.writeheader(); writer.writerows(rejected)
        stats = {}
        for relation in relations:
            counts = Counter(r["value_id"] for r in relation_rows[relation].values())
            stats[relation] = {"retained_subjects": len(relation_rows[relation]), "distinct_values": len(counts), "mean_subjects_per_value": len(counts) and len(relation_rows[relation]) / len(counts), "singleton_value_fraction": len(counts) and sum(n == 1 for n in counts.values()) / len(counts)}
        intersection = set.intersection(*(set(relation_rows[relation]) for relation in relations)) if relations else set()
        accepted = len(intersection) >= 50 and all(stats[relation]["retained_subjects"] >= 50 for relation in relations)
        report["domains"][domain] = {"status": "accepted" if accepted else "rejected", "relations": spec["relations"], "relation_stats": stats, "shared_subject_intersection": len(intersection), "rejected_records": len(rejected), "sample_limit": 2000}
        (report["accepted_domains"] if accepted else report["rejected_domains"]).append(domain)
        print(f"domain={domain} status={report['domains'][domain]['status']} shared_subjects={len(intersection)}")
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
