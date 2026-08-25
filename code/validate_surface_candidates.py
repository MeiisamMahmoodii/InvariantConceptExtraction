import csv
from collections import Counter, defaultdict
from pathlib import Path

rows = list(csv.DictReader((Path(__file__).resolve().parents[1] / "Report" / "surface_template_candidates.csv").open(encoding="utf-8")))
groups = defaultdict(set)
family_counts = Counter()
texts = defaultdict(set)
all_texts = defaultdict(set)
missing = 0
for row in rows:
    groups[row["fact_id"]].add(row["S_family"])
    family_counts[(row["fact_id"], row["S_family"])] += 1
    texts[(row["fact_id"], row["S_family"])] .add(row["candidate_text"])
    all_texts[row["fact_id"]].add(row["candidate_text"])
    missing += int(row["C_subject_label"].casefold() not in row["candidate_text"].casefold() or row["C_value_label"].casefold() not in row["candidate_text"].casefold())
families = {"declarative", "question", "paraphrase", "formal", "structured"}
expected = {"declarative": 3, "question": 3, "paraphrase": 3, "formal": 2, "structured": 1}
count_failures = sum(family_counts[(fact, family)] != expected[family] for fact in groups for family in families)
diversity_failures = sum(len(texts[(fact, family)]) != expected[family] for fact in groups for family in families)
cross_family_failures = sum(len(all_texts[fact]) != 12 for fact in groups)
failed = len(rows) != 216 or len(groups) != 18 or missing or any(value != families for value in groups.values()) or count_failures or diversity_failures or cross_family_failures or set(Counter(row["C_relation"] for row in rows).values()) != {36}
print(f"candidate_rows={len(rows)} candidate_facts={len(groups)} missing_factor_mentions={missing} variant_count_failures={count_failures} within_family_diversity_failures={diversity_failures} cross_family_distinction_failures={cross_family_failures}")
if failed:
    raise SystemExit("VALIDATION FAILED")
print("VALIDATION PASSED")
