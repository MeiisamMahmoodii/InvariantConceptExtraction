"""Fetch and audit six factual relations; this script creates no text dataset."""

import csv
import html
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "relation_feasibility"
REPORT = ROOT / "Report"
ENDPOINT = "https://query.wikidata.org/sparql"
RELATIONS = {"capital_of": "P36", "continent_of": "P30", "currency_of": "P38", "atomic_number_of": "P1086", "period_of": None, "chemical_symbol_of": "P246"}
DOMAINS = {"geography": ("capital_of", "continent_of", "currency_of"), "science": ("atomic_number_of", "period_of", "chemical_symbol_of")}
QUERY = '''SELECT ?domain ?subject ?subjectLabel ?property ?value ?valueLabel WHERE {
  { BIND("geography" AS ?domain) ?subject wdt:P31 wd:Q3624078. VALUES ?property { wdt:P36 wdt:P30 wdt:P38 } ?subject ?property ?value. }
  UNION { BIND("science" AS ?domain) ?subject wdt:P31 wd:Q11344. VALUES ?property { wdt:P1086 wdt:P246 } ?subject ?property ?value. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}'''


def value(row, field):
    return row[field]["value"]


def canonical_value_id(row):
    raw = value(row, "value")
    return raw.rsplit("/", 1)[1] if raw.startswith("http://www.wikidata.org/entity/") else f"literal:{row['value'].get('datatype', 'string')}:{raw}"


def resolve_labels(ids):
    if not ids:
        return {}
    url = "https://www.wikidata.org/w/api.php?" + urlencode({"action": "wbgetentities", "ids": "|".join(sorted(ids)), "props": "labels", "languages": "en", "format": "json"})
    with urlopen(Request(url, headers={"User-Agent": "InvariantConceptExtraction/1.0"}), timeout=60) as response:
        entities = json.load(response)["entities"]
    return {entity_id: entity.get("labels", {}).get("en", {}).get("value") for entity_id, entity in entities.items()}


def rsc_period(number_and_record):
    number, record = number_and_record
    slug = re.sub(r"[^a-z0-9]+", "-", record["subject_label"].lower()).strip("-")
    url = f"https://periodic-table.rsc.org/element/{number}/{slug}"
    try:
        with urlopen(Request(url, headers={"User-Agent": "InvariantConceptExtraction/1.0"}), timeout=30) as response:
            text = html.unescape(re.sub(r"<[^>]+>", " ", response.read().decode("utf-8", errors="replace")))
    except OSError:
        return number, None, url
    match = re.search(r"Period\s+(\d+)\s+Boiling point", text, flags=re.S)
    if not match:
        return number, None, url
    return number, int(match.group(1)), url


def main():
    request = Request(f"{ENDPOINT}?{urlencode({'format': 'json', 'query': QUERY})}", headers={"User-Agent": "InvariantConceptExtraction/1.0"})
    with urlopen(request, timeout=120) as response:
        bindings = json.load(response)["results"]["bindings"]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "wikidata_relation_snapshot.json").write_text(json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8")
    unresolved_ids = {value(row, field).rsplit("/", 1)[1] for row in bindings for field in ("subject", "value") if value(row, field).startswith("http://www.wikidata.org/entity/") and value(row, f"{field}Label").startswith("Q") and value(row, f"{field}Label")[1:].isdigit()}
    resolved_labels = resolve_labels(unresolved_ids)
    with (DATA / "label_resolution_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("entity_id", "resolved_label", "source")); writer.writeheader(); writer.writerows({"entity_id": entity_id, "resolved_label": label or "", "source": f"https://www.wikidata.org/wiki/{entity_id}"} for entity_id, label in resolved_labels.items())
    property_map = {f"http://www.wikidata.org/prop/direct/{pid}": relation for relation, pid in RELATIONS.items() if pid}
    candidates = defaultdict(lambda: defaultdict(list))
    for row in bindings:
        relation = property_map[value(row, "property")]
        subject_id, value_id = value(row, "subject").rsplit("/", 1)[1], canonical_value_id(row)
        subject_label = resolved_labels.get(subject_id) or value(row, "subjectLabel")
        raw_value_label = value(row, "valueLabel")
        candidates[relation][subject_id].append({"subject_id": subject_id, "subject_label": subject_label, "relation": relation, "value_id": value_id, "value_label": resolved_labels.get(value_id) or raw_value_label, "qualifiers_status": "Wikidata direct best-rank claim; qualifiers not used"})
    clean, rejected = {relation: {} for relation in RELATIONS}, []
    for relation, subjects in candidates.items():
        for subject_id, rows in subjects.items():
            if len(rows) != 1:
                rejected.append({"subject_id": subject_id, "relation": relation, "reason": f"expected one direct best-rank value; found {len(rows)}"})
            elif any(rows[0][field].startswith("Q") and rows[0][field][1:].isdigit() for field in ("subject_label", "value_label")):
                rejected.append({"subject_id": subject_id, "relation": relation, "reason": "missing English source label"})
            else:
                clean[relation][subject_id] = rows[0]
    atomic_by_number = {int(row["value_label"]): row for row in clean["atomic_number_of"].values() if row["value_label"].isdigit() and 1 <= int(row["value_label"]) <= 118}
    with ThreadPoolExecutor(max_workers=8) as pool:
        periods = list(pool.map(rsc_period, atomic_by_number.items()))
    (DATA / "rsc_period_snapshot.json").write_text(json.dumps(periods, indent=2), encoding="utf-8")
    for number, period, url in periods:
        atomic = atomic_by_number[number]
        if period is None:
            rejected.append({"subject_id": atomic["subject_id"], "relation": "period_of", "reason": "Royal Society of Chemistry page did not expose one period"})
        else:
            clean["period_of"][atomic["subject_id"]] = {"subject_id": atomic["subject_id"], "subject_label": atomic["subject_label"], "relation": "period_of", "value_id": f"literal:period:{period}", "value_label": str(period), "qualifiers_status": f"Royal Society of Chemistry fact box; source={url}"}
    fields = ("subject_id", "subject_label", "relation", "value_id", "value_label", "qualifiers_status")
    for relation, rows in clean.items():
        with (DATA / f"{relation}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows.values())
    with (DATA / "rejected_records.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("subject_id", "relation", "reason")); writer.writeheader(); writer.writerows(rejected)
    stats = {}
    for relation, rows in clean.items():
        counts = Counter(row["value_id"] for row in rows.values())
        stats[relation] = {"retained_subjects": len(rows), "distinct_values": len(counts), "mean_subjects_per_value": len(rows) / len(counts) if counts else 0, "singleton_value_fraction": sum(count == 1 for count in counts.values()) / len(counts) if counts else 0}
    coverage = {domain: {"relations": relations, "shared_subject_count": len(set.intersection(*(set(clean[relation]) for relation in relations)))} for domain, relations in DOMAINS.items()}
    failures = [relation for relation, stat in stats.items() if stat["retained_subjects"] < 50]
    failures += [domain for domain, result in coverage.items() if result["shared_subject_count"] < 50]
    report = {"source": "Wikidata SPARQL direct best-rank claims; Royal Society of Chemistry element fact boxes for period_of", "relation_stats": stats, "domain_coverage": coverage, "rejected_record_count": len(rejected), "policy": {"country_continent": "reject if P30 count is not exactly one", "period_of": "retain one explicit Royal Society of Chemistry period value; no derivation from atomic number"}, "acceptance_passed": not failures, "acceptance_failures": failures}
    (REPORT / "relation_coverage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved six factual tables and {len(rejected)} rejected records to {DATA}.")
    if failures:
        print(f"FEASIBILITY FAILED: {', '.join(failures)}")


if __name__ == "__main__":
    main()
