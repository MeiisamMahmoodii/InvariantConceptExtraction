"""Create and audit a fixed RAVEL prompt+published-answer sample; no training."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ravel_clean_subset"
OUT = ROOT / "data" / "ravel_prompt_answer_preflight"
REPORT = ROOT / "Report" / "ravel_prompt_answer_preflight_report.json"
SEED = 20260825
SAMPLE_FACTS = 200


def pick(rows, count, key):
    return sorted(rows, key=lambda row: hashlib.blake2b(f"{SEED}:{key}:{row['fact_id']}".encode(), digest_size=16).digest())[:count]


def value_in_prompt(value, prompt):
    return bool(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", prompt.casefold()))


def main():
    facts = [json.loads(line) for line in (DATA / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    templates = {row["template_id"]: row for row in (json.loads(line) for line in (DATA / "templates.jsonl").read_text(encoding="utf-8").splitlines())}
    by_relation_split = defaultdict(list)
    for fact in facts:
        enough_templates = len(fact["training_template_ids"]) >= 2 if fact["C_split"] == "train" else len(fact["heldout_template_ids"]) >= 2
        if fact["C_split"] in {"train", "test"} and enough_templates: by_relation_split[(fact["C_relation"], fact["C_split"])].append(fact)
    # Spread 200 facts across every retained relation and both entity splits.
    selected = []
    keys = sorted(by_relation_split)
    base, remainder = divmod(SAMPLE_FACTS, len(keys))
    for index, key in enumerate(keys): selected.extend(pick(by_relation_split[key], base + (index < remainder), ":".join(key)))
    assert len(selected) == SAMPLE_FACTS
    rows, failures = [], []
    negative_index = defaultdict(list)
    for fact in selected:
        template_ids = fact["training_template_ids"] if fact["C_split"] == "train" else fact["heldout_template_ids"]
        chosen = sorted(template_ids, key=lambda item: hashlib.blake2b(f"{SEED}:{fact['fact_id']}:{item}".encode(), digest_size=16).digest())[:2]
        if len(chosen) != 2: failures.append({"fact_id": fact["fact_id"], "reason": "fewer_than_two_available_templates"}); continue
        for template_id in chosen:
            template = templates[template_id]
            prompt = template["template"] % fact["C_subject_label"]
            text = f"{prompt} {fact['C_value_label']}"
            if fact["C_value_label"] not in text: failures.append({"fact_id": fact["fact_id"], "template_id": template_id, "reason": "published_value_missing_after_append"})
            if value_in_prompt(fact["C_value_label"], prompt): failures.append({"fact_id": fact["fact_id"], "template_id": template_id, "reason": "value_present_in_template_before_append"})
            if (fact["C_split"] == "train") != (template["S_split"] in {"train", "val"}): failures.append({"fact_id": fact["fact_id"], "template_id": template_id, "reason": "template_split_policy_violation"})
            rows.append({"fact_id": fact["fact_id"], "C_entity_type": fact["C_entity_type"], "C_subject_label": fact["C_subject_label"], "C_relation": fact["C_relation"], "C_value_label": fact["C_value_label"], "C_split": fact["C_split"], "template_id": template_id, "S_split": template["S_split"], "rendered_template": prompt, "text_prompt_answer": text})
            negative_index[(fact["C_split"], template_id)].append(fact["fact_id"])
    fact_rows = Counter(row["fact_id"] for row in rows)
    semantic_failures = [fact_id for fact_id, count in fact_rows.items() if count != 2]
    all_train_by_template = defaultdict(set)
    for fact in facts:
        for template_id in fact["training_template_ids"]: all_train_by_template[template_id].add(fact["fact_id"])
    negative_valid = all(len(all_train_by_template[row["template_id"]]) >= 2 for row in rows if row["C_split"] == "train")
    report = {"status": "pass" if not failures and not semantic_failures and negative_valid else "fail", "facts": len(selected), "views": len(rows), "per_relation_split": {f"{relation}:{split}": len(group) for (relation, split), group in sorted(by_relation_split.items())}, "checks": {"same_fact_two_distinct_templates": not semantic_failures, "published_value_present_in_every_view": not any(row["reason"] == "published_value_missing_after_append" for row in failures), "no_value_in_template_before_append": not any(row["reason"] == "value_present_in_template_before_append" for row in failures), "exact_template_training_negatives_valid_against_full_subset": negative_valid, "train_uses_only_published_train_val_templates": not any(row["reason"] == "template_split_policy_violation" and row["fact_id"] in {r["fact_id"] for r in rows if r["C_split"] == "train"} for row in failures), "test_uses_only_published_test_templates": not any(row["reason"] == "template_split_policy_violation" and row["fact_id"] in {r["fact_id"] for r in rows if r["C_split"] == "test"} for row in failures)}, "failures": failures, "semantic_failures": semantic_failures, "text_definition": "rendered published template + one space + verbatim published value", "training_performed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sample_views.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
