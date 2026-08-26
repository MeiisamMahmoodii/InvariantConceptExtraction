"""Build a strict RAVEL fact/template subset without materializing pairs.

Input is the official RAVEL JSON release. Output is facts, templates, a rejection
log, and an aggregate construction report. No prompts, pairs, or model outputs
are generated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_URL = "https://github.com/explanare/ravel/raw/main/data.tgz"
VALID_SPLITS = {"train", "val", "test"}


def pair_count(size: int) -> int:
    return size * (size - 1) // 2


def has_compound_or_missing_value(value: object) -> str | None:
    if value is None or not isinstance(value, str) or not value.strip():
        return "missing_or_non_scalar_value"
    if re.search(r"[,;/]", value):
        return "compound_or_delimited_value"
    return None


def value_in_prompt(value: str, prompt: str) -> bool:
    # Word boundaries prevent "1" matching an unrelated longer number.
    return bool(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", prompt.casefold()))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir, output_dir = args.source_dir, args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    facts, templates, rejections = [], [], []
    source_types = sorted(
        file.name.removeprefix("ravel_").removesuffix("_entity_attributes.json")
        for file in source_dir.glob("ravel_*_entity_attributes.json")
    )
    templates_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    raw_template_attributes: dict[tuple[str, str], set[str]] = defaultdict(set)
    entities_by_type: dict[str, dict[str, str]] = {}
    attributes_by_type: dict[str, dict[str, dict[str, object]]] = {}

    for entity_type in source_types:
        entity_values = json.loads((source_dir / f"ravel_{entity_type}_entity_attributes.json").read_text(encoding="utf-8"))
        entity_splits = json.loads((source_dir / f"ravel_{entity_type}_entity_to_split.json").read_text(encoding="utf-8"))
        attribute_prompts = json.loads((source_dir / f"ravel_{entity_type}_attribute_to_prompts.json").read_text(encoding="utf-8"))
        prompt_splits = json.loads((source_dir / f"ravel_{entity_type}_prompt_to_split.json").read_text(encoding="utf-8"))
        entities_by_type[entity_type] = entity_splits
        attributes_by_type[entity_type] = entity_values

        for attribute, prompt_list in attribute_prompts.items():
            for index, template in enumerate(prompt_list):
                raw_template_attributes[(entity_type, template)].add(attribute)
                split = prompt_splits.get(template)
                reason = None
                if split not in VALID_SPLITS:
                    reason = "missing_or_invalid_template_split"
                elif template.count("%s") != 1:
                    reason = "template_does_not_have_exactly_one_entity_placeholder"
                else:
                    try:
                        template % "RAVEL_ENTITY_CHECK"
                    except (TypeError, ValueError):
                        reason = "template_cannot_be_rendered_with_entity_placeholder"
                if reason:
                    rejections.append({"kind": "template", "entity_type": entity_type, "attribute": attribute, "template": template, "reason": reason})
                    continue
                template_id = f"ravel:{entity_type}:{attribute}:{index:03d}"
                row = {"template_id": template_id, "entity_type": entity_type, "attribute": attribute, "template": template, "S_split": split, "source_name": "RAVEL", "source_record_id": f"ravel_{entity_type}_attribute_to_prompts.json#{attribute}[{index}]", "source_provenance": SOURCE_URL}
                templates_by_key[(entity_type, attribute)].append(row)

        for entity, values in entity_values.items():
            c_split = entity_splits.get(entity)
            if c_split not in VALID_SPLITS:
                rejections.append({"kind": "entity", "entity_type": entity_type, "entity": entity, "reason": "missing_or_invalid_entity_split"})
                continue
            for attribute, value in values.items():
                reason = has_compound_or_missing_value(value)
                if reason:
                    rejections.append({"kind": "value", "entity_type": entity_type, "entity": entity, "attribute": attribute, "value": value, "reason": reason})
                    continue
                if not templates_by_key[(entity_type, attribute)]:
                    rejections.append({"kind": "fact", "entity_type": entity_type, "entity": entity, "attribute": attribute, "value": value, "reason": "no_structurally_valid_templates"})
                    continue
                fact_id = f"ravel:{entity_type}:{entity}:{attribute}"
                allowed_train, allowed_test = [], []
                for template in templates_by_key[(entity_type, attribute)]:
                    prompt = template["template"] % entity
                    if value_in_prompt(value, prompt):
                        rejections.append({"kind": "fact_template", "fact_id": fact_id, "template_id": template["template_id"], "reason": "rendered_prompt_contains_target_value"})
                        continue
                    (allowed_test if template["S_split"] == "test" else allowed_train).append(template["template_id"])
                if len(allowed_train) < 2:
                    rejections.append({"kind": "fact", "fact_id": fact_id, "reason": "fewer_than_two_clean_train_or_val_templates"})
                    continue
                if not allowed_test:
                    rejections.append({"kind": "fact", "fact_id": fact_id, "reason": "no_clean_heldout_test_template"})
                    continue
                facts.append({"fact_id": fact_id, "C_entity_type": entity_type, "C_subject_label": entity, "C_relation": attribute, "C_value_label": value, "C_split": c_split, "training_template_ids": allowed_train, "heldout_template_ids": allowed_test, "source_name": "RAVEL", "source_record_id": f"ravel_{entity_type}_entity_attributes.json#{entity}.{attribute}", "source_provenance": SOURCE_URL})

    # A template string assigned to multiple attributes is ambiguous as an S label.
    ambiguous_strings = {key for key, attributes in raw_template_attributes.items() if len(attributes) > 1}
    if ambiguous_strings:
        bad_ids = {t["template_id"] for group in templates_by_key.values() for t in group if (t["entity_type"], t["template"]) in ambiguous_strings}
        for fact in facts:
            fact["training_template_ids"] = [x for x in fact["training_template_ids"] if x not in bad_ids]
            fact["heldout_template_ids"] = [x for x in fact["heldout_template_ids"] if x not in bad_ids]
        templates = [t for group in templates_by_key.values() for t in group if t["template_id"] not in bad_ids]
        for entity_type, template in ambiguous_strings:
            rejections.append({"kind": "template", "entity_type": entity_type, "template": template, "reason": "same_exact_template_assigned_to_multiple_attributes"})
    else:
        templates = [t for group in templates_by_key.values() for t in group]

    # Apply coverage once more after removing globally ambiguous template strings.
    facts = [f for f in facts if len(f["training_template_ids"]) >= 2 and f["heldout_template_ids"]]
    used_template_ids = {template_id for fact in facts for template_id in fact["training_template_ids"] + fact["heldout_template_ids"]}
    templates = [template for template in templates if template["template_id"] in used_template_ids]
    templates_by_id = {template["template_id"]: template for template in templates}

    fact_ids_by_template_and_split: dict[tuple[str, str], list[str]] = defaultdict(list)
    for fact in facts:
        for template_id in fact["training_template_ids"]:
            fact_ids_by_template_and_split[(fact["C_split"], template_id)].append(fact["fact_id"])
    pairability = {}
    for c_split in sorted(VALID_SPLITS):
        split_facts = [f for f in facts if f["C_split"] == c_split]
        positive_count = sum(pair_count(len(f["training_template_ids"])) for f in split_facts)
        negative_count = sum(pair_count(len(ids)) for (split_name, _), ids in fact_ids_by_template_and_split.items() if split_name == c_split)
        pairable_templates = sum(len(ids) >= 2 for (split_name, _), ids in fact_ids_by_template_and_split.items() if split_name == c_split)
        pairability[c_split] = {"facts": len(split_facts), "positive_pairs_same_fact_different_training_template": positive_count, "negative_pairs_different_fact_same_training_template": negative_count, "negative_pairable_training_templates": pairable_templates}

    entity_splits = {entity_type: Counter(mapping.values()) for entity_type, mapping in entities_by_type.items()}
    retained_by_type_attribute = Counter((f["C_entity_type"], f["C_relation"]) for f in facts)
    template_counts = Counter((t["entity_type"], t["attribute"], t["S_split"]) for t in templates)
    entity_to_splits = defaultdict(set)
    for fact in facts:
        entity_to_splits[(fact["C_entity_type"], fact["C_subject_label"])].add(fact["C_split"])
    leakage = {"entity_split_leakage": sum(len(splits) != 1 for splits in entity_to_splits.values()), "heldout_template_ids_in_training_lists": sum(any(templates_by_id[t]["S_split"] == "test" for t in f["training_template_ids"]) for f in facts)}
    all_type_attributes = sorted((entity_type, attribute) for entity_type, entities in attributes_by_type.items() for attribute in {name for values in entities.values() for name in values})
    report = {"status": "complete", "scope": "RAVEL strict subset construction only; no prompt pairs or model training", "C_definition": "(entity_type, entity_label, attribute, published scalar value)", "S_definition": "published exact template scoped to entity type and attribute", "published_split_policy": "C_split follows RAVEL entity split; training templates are published train+val only; published test templates are held out", "retained": {"facts": len(facts), "templates": len(templates), "facts_per_entity_type_attribute": [{"entity_type": k[0], "attribute": k[1], "facts": retained_by_type_attribute[k]} for k in all_type_attributes], "templates_per_entity_type_attribute_split": [{"entity_type": k[0], "attribute": k[1], "S_split": k[2], "templates": v} for k, v in sorted(template_counts.items())]}, "entity_counts": {entity_type: dict(sorted(counts.items())) for entity_type, counts in sorted(entity_splits.items())}, "facts_with_both_training_and_heldout_template_coverage": len(facts), "pairability_by_entity_split": pairability, "rejections": {"counts": dict(sorted(Counter(row["reason"] for row in rejections).items())), "total": len(rejections)}, "validation": {**leakage, "all_retained_facts_have_two_training_and_one_heldout_template": all(len(f["training_template_ids"]) >= 2 and f["heldout_template_ids"] for f in facts), "no_heldout_template_leakage_into_training": leakage["heldout_template_ids_in_training_lists"] == 0}, "dynamic_sampling": {"positive": "choose two distinct IDs from one fact.training_template_ids", "negative": "choose one training template ID then two fact IDs sharing that exact ID within the requested C_split", "pairs_materialized": 0}, "caveats": ["RAVEL source records provide labels, not canonical entity IDs.", "Validation is structural: source attribute assignment, one renderable entity placeholder, no raw-template attribute collision, and no target value in the rendered prompt. It cannot prove every natural-language template has identical pragmatics."]}
    write_jsonl(output_dir / "facts.jsonl", facts)
    write_jsonl(output_dir / "templates.jsonl", templates)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    (output_dir / "construction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RAVEL strict subset construction complete")
    print(f"Retained facts: {len(facts)}")
    print(f"Retained templates: {len(templates)}")
    print(f"Facts with train+heldout coverage: {len(facts)}")
    print("Entity leakage:", leakage["entity_split_leakage"])
    print("Held-out template leakage:", leakage["heldout_template_ids_in_training_lists"])
    for split, values in pairability.items():
        print(f"{split}: facts={values['facts']}, positives={values['positive_pairs_same_fact_different_training_template']}, negatives={values['negative_pairs_different_fact_same_training_template']}")
    print(f"Report: {output_dir / 'construction_report.json'}")


if __name__ == "__main__":
    main()
