"""Select a fixed source-only RAVEL prompt subset for the matched partition budget."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ravel_clean_subset"
OUT = ROOT / "data" / os.environ.get("RAVEL_PARTITION_DIR", "ravel_partition_layer8")
SEED = 20260825
PER_FACT = 3


def choose(ids: list[str], fact_id: str) -> list[str]:
    ranked = sorted(ids, key=lambda item: hashlib.blake2b(f"{SEED}:{fact_id}:{item}".encode(), digest_size=16).digest())
    return ranked[:PER_FACT]


def main() -> None:
    facts = [json.loads(line) for line in (DATA / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    templates = {row["template_id"]: row for row in (json.loads(line) for line in (DATA / "templates.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = []
    for fact in facts:
        ids = fact["training_template_ids"] if fact["C_split"] == "train" else fact["heldout_template_ids"] if fact["C_split"] == "test" else []
        for template_id in choose(ids, fact["fact_id"]):
            template = templates[template_id]
            prompt = template["template"] % fact["C_subject_label"]
            text = f"{prompt} {fact['C_value_label']}" if os.environ.get("RAVEL_PROMPT_ANSWER") == "1" else prompt
            rows.append({"example_id": f"{fact['fact_id']}:{template_id}", "fact_id": fact["fact_id"], "C_entity_type": fact["C_entity_type"], "C_subject_label": fact["C_subject_label"], "C_relation": fact["C_relation"], "C_value_label": fact["C_value_label"], "C_split": fact["C_split"], "template_id": template_id, "S_split": template["S_split"], "text": text})
    # The three source templates are selected per fact, so a rare selected
    # template can occur for only one fact. Remove only such sampled rows until
    # every retained row can form both requested pair types.
    for c_split in ("train", "test"):
        while True:
            split_rows = [row for row in rows if row["C_split"] == c_split]
            template_counts = Counter(row["template_id"] for row in split_rows)
            fact_counts = Counter(row["fact_id"] for row in split_rows)
            filtered = [row for row in rows if row["C_split"] != c_split or (template_counts[row["template_id"]] >= 2 and fact_counts[row["fact_id"]] >= 2)]
            if len(filtered) == len(rows): break
            rows = filtered
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = {"seed": SEED, "rows": len(rows), "per_fact": PER_FACT, "training_rows": sum(r["C_split"] == "train" for r in rows), "heldout_test_rows": sum(r["C_split"] == "test" for r in rows), "training_policy": "C_train facts with only published train/val templates", "evaluation_policy": "C_test facts with only published test templates", "text_definition": "rendered template + verbatim published value" if os.environ.get("RAVEL_PROMPT_ANSWER") == "1" else "rendered template only", "pairs_materialized": 0, "all_sampled_train_rows_have_negative_coverage": all(count >= 2 for count in Counter(r["template_id"] for r in rows if r["C_split"] == "train").values()), "all_sampled_train_facts_have_positive_coverage": all(count >= 2 for count in Counter(r["fact_id"] for r in rows if r["C_split"] == "train").values())}
    (OUT / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
