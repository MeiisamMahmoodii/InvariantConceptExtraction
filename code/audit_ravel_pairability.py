"""Audit RAVEL's released entity/template files for controlled C/S pairs.

This reads only RAVEL source JSON.  It does not create prompts or train a model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def pair_count(size: int) -> int:
    return size * (size - 1) // 2


def value_flags(value: object) -> list[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return ["missing_value"]
    if not isinstance(value, str):
        return ["non_scalar_value"]
    flags = []
    if re.search(r"[,;/]", value):
        flags.append("delimited_or_compound_value")
    return flags


def audit_type(source_dir: Path, entity_type: str) -> dict:
    entities = json.loads(
        (source_dir / f"ravel_{entity_type}_entity_attributes.json").read_text(encoding="utf-8")
    )
    by_attribute = json.loads(
        (source_dir / f"ravel_{entity_type}_attribute_to_prompts.json").read_text(encoding="utf-8")
    )
    prompt_splits = json.loads(
        (source_dir / f"ravel_{entity_type}_prompt_to_split.json").read_text(encoding="utf-8")
    )

    facts = []
    clean_facts = []
    templates = []
    flags = Counter()
    missing_split_templates = []
    for entity, values in entities.items():
        for attribute, value in values.items():
            if attribute not in by_attribute:
                flags["attribute_without_templates"] += 1
                continue
            fact = (entity_type, entity, attribute, value)
            facts.append(fact)
            fact_flags = value_flags(value)
            flags.update(fact_flags)
            if not {"missing_value", "non_scalar_value"}.intersection(fact_flags):
                clean_facts.append(fact)
    for attribute, prompt_list in by_attribute.items():
        for template in prompt_list:
            if template not in prompt_splits:
                missing_split_templates.append(template)
            templates.append((entity_type, attribute, template, prompt_splits.get(template)))

    # An S-family is scoped by entity type and queried attribute.  This prevents
    # an accidental cross-attribute string collision from being treated as the
    # same semantic template.
    facts_by_attribute = defaultdict(list)
    for fact in clean_facts:
        facts_by_attribute[fact[2]].append(fact)
    templates_by_attribute = defaultdict(list)
    for template in templates:
        templates_by_attribute[template[1]].append(template)

    usable_facts = []
    usable_templates = []
    positive_pairs = 0
    negative_pairs = 0
    heldout_templates = 0
    heldout_facts = set()
    per_attribute = {}
    for attribute, attribute_facts in facts_by_attribute.items():
        attribute_templates = templates_by_attribute[attribute]
        # A fact can form a positive iff the attribute has at least two templates.
        # A template can form a negative iff the attribute has at least two facts.
        usable_attribute_facts = attribute_facts if len(attribute_templates) >= 2 else []
        usable_attribute_templates = attribute_templates if len(attribute_facts) >= 2 else []
        usable_facts.extend(usable_attribute_facts)
        usable_templates.extend(usable_attribute_templates)
        positive_pairs += len(attribute_facts) * pair_count(len(attribute_templates))
        negative_pairs += len(attribute_templates) * pair_count(len(attribute_facts))
        heldout = [t for t in usable_attribute_templates if t[3] == "test"]
        heldout_templates += len(heldout)
        if heldout and any(t[3] == "train" for t in usable_attribute_templates):
            heldout_facts.update(usable_attribute_facts)
        per_attribute[attribute] = {
            "usable_facts_with_explicit_scalar_value": len(attribute_facts),
            "templates": len(attribute_templates),
            "template_splits": dict(sorted(Counter(t[3] or "missing" for t in attribute_templates).items())),
            "positive_pairs_same_C_different_S": len(attribute_facts) * pair_count(len(attribute_templates)),
            "negative_pairs_different_C_same_S": len(attribute_templates) * pair_count(len(attribute_facts)),
            "fully_heldout_test_templates": len(heldout),
            "facts_with_train_and_test_template_coverage": len(attribute_facts) if heldout and any(t[3] == "train" for t in attribute_templates) else 0,
        }

    raw_template_counts = Counter(t[2] for t in templates)
    return {
        "entity_type": entity_type,
        "source_entities": len(entities),
        "source_facts_entity_attribute_value": len(facts),
        "rejected_facts_missing_or_non_scalar_value": len(facts) - len(clean_facts),
        "usable_facts": len(usable_facts),
        "usable_templates_scoped_by_entity_type_attribute": len(usable_templates),
        "positive_pairs_same_C_different_S": positive_pairs,
        "negative_pairs_different_C_same_S": negative_pairs,
        "fully_heldout_test_templates": heldout_templates,
        "facts_with_train_and_test_template_coverage": len(heldout_facts),
        "attributes": per_attribute,
        "value_flags": dict(sorted(flags.items())),
        "templates_missing_published_split": len(missing_split_templates),
        "exact_template_strings_reused_across_attributes_or_types_within_type": sum(
            count > 1 for count in raw_template_counts.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir
    entity_types = sorted(
        path.name.removeprefix("ravel_").removesuffix("_entity_attributes.json")
        for path in source_dir.glob("ravel_*_entity_attributes.json")
    )
    reports = [audit_type(source_dir, entity_type) for entity_type in entity_types]
    totals = {
        key: sum(report[key] for report in reports)
        for key in (
            "source_entities",
            "source_facts_entity_attribute_value",
            "rejected_facts_missing_or_non_scalar_value",
            "usable_facts",
            "usable_templates_scoped_by_entity_type_attribute",
            "positive_pairs_same_C_different_S",
            "negative_pairs_different_C_same_S",
            "fully_heldout_test_templates",
            "facts_with_train_and_test_template_coverage",
        )
    }
    result = {
        "audit": "RAVEL controlled pairability; source JSON only; no model training",
        "C_definition": "(entity_type, entity, attribute, published_attribute_value)",
        "S_definition": "(entity_type, attribute, exact_published_template)",
        "pair_definitions": {
            "positive": "same C, two distinct S templates for the same queried attribute",
            "negative": "two distinct C facts with the same S template and queried attribute",
        },
        "source_dir": str(source_dir),
        "entity_types": reports,
        "totals": totals,
        "acceptance": {
            "both_pair_types_exist_at_meaningful_scale": totals["positive_pairs_same_C_different_S"] > 0
            and totals["negative_pairs_different_C_same_S"] > 0,
            "whole_template_holdout_possible": totals["fully_heldout_test_templates"] > 0
            and totals["facts_with_train_and_test_template_coverage"] > 0,
        },
        "caveats": [
            "RAVEL entity files use entity labels as keys rather than canonical source IDs. Provenance is archive/file-level, not fact-level canonical-ID provenance.",
            "Missing or non-scalar values are excluded. Delimited strings are retained as published values but flagged rather than split; these need relation-level filtering before any strict C-matrix is built.",
            "Some templates contain few-shot demonstrations or structured text. Template identity is therefore a controlled S label, not a guarantee of natural-language-only surface variation.",
            "This audit counts source combinatorics. It does not assert that every prompt is semantically equivalent, and it does not generate or train on any prompt.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RAVEL pairability audit complete")
    print(f"Entity types: {len(reports)}")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print("both_pair_types_exist_at_meaningful_scale:", result["acceptance"]["both_pair_types_exist_at_meaningful_scale"])
    print("whole_template_holdout_possible:", result["acceptance"]["whole_template_holdout_possible"])
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
