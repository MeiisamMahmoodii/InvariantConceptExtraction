"""Classify controlled fact pairs for partial-C contrastive supervision."""

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "data" / "factual_matrix" / "factual_c_matrix.csv"
OUT = ROOT / "data" / "pair_taxonomy" / "fact_pair_taxonomy.csv"
REPORT = ROOT / "Report" / "partial_c_pair_taxonomy_report.json"


def category(a, b):
    if a["C_relation"] == b["C_relation"]:
        return "same_relation_different_subject"
    if (a["C_domain"], a["C_subject_id"]) == (b["C_domain"], b["C_subject_id"]):
        return "same_subject_different_relation"
    if a["C_domain"] == b["C_domain"]:
        return "same_domain_only"
    return "no_shared_C_factor"


def main():
    with FACTS.open(newline="", encoding="utf-8") as file: facts = list(csv.DictReader(file))
    OUT.parent.mkdir(exist_ok=True)
    counts = Counter()
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["fact_id_a", "fact_id_b", "pair_category"])
        writer.writeheader()
        for a, b in combinations(facts, 2):
            kind = category(a, b); counts[kind] += 1
            writer.writerow({"fact_id_a": a["fact_id"], "fact_id_b": b["fact_id"], "pair_category": kind})
    report = {"facts": len(facts), "unordered_distinct_fact_pairs": sum(counts.values()), "categories": {"same_exact_fact_different_S": "surface-row pairs within one fact; handled as strong positives during training", **dict(counts)}, "rule": {"strong_positive": "same exact fact, different S_family", "weak_positive": ["same relation, different subject", "same subject, different relation"], "neutral": "same domain only", "negative": "no shared C factor"}}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
