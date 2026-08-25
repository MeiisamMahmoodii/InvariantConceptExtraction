import csv
from collections import Counter, defaultdict
from pathlib import Path

rows = list(csv.DictReader((Path(__file__).resolve().parents[1] / "Report" / "surface_template_candidates.csv").open(encoding="utf-8")))
groups = defaultdict(set)
missing = 0
for row in rows:
    groups[row["fact_id"]].add(row["S_family"])
    missing += int(row["C_subject_label"] not in row["candidate_text"] or row["C_value_label"] not in row["candidate_text"])
families = {"declarative", "question", "paraphrase", "formal", "structured"}
failed = len(rows) != 90 or len(groups) != 18 or missing or any(value != families for value in groups.values()) or set(Counter(row["C_relation"] for row in rows).values()) != {15}
print(f"candidate_rows={len(rows)} candidate_facts={len(groups)} missing_factor_mentions={missing}")
if failed:
    raise SystemExit("VALIDATION FAILED")
print("VALIDATION PASSED")
