"""Dynamic RAVEL pair sampler; it never materializes a pair list."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


class RavelPairSampler:
    def __init__(self, facts_path: Path, seed: int = 0) -> None:
        self.random = random.Random(seed)
        self.facts = [json.loads(line) for line in facts_path.read_text(encoding="utf-8").splitlines()]
        self.by_id = {fact["fact_id"]: fact for fact in self.facts}
        self.positive = defaultdict(list)
        self.by_template = defaultdict(list)
        for fact in self.facts:
            split = fact["C_split"]
            if len(fact["training_template_ids"]) >= 2:
                self.positive[split].append(fact["fact_id"])
            for template_id in fact["training_template_ids"]:
                self.by_template[(split, template_id)].append(fact["fact_id"])
        self.negative = {split: [key for key, ids in self.by_template.items() if key[0] == split and len(ids) >= 2] for split in ("train", "val", "test")}

    def sample_positive(self, c_split: str = "train") -> tuple[dict, dict]:
        fact = self.by_id[self.random.choice(self.positive[c_split])]
        left, right = self.random.sample(fact["training_template_ids"], 2)
        return fact, {"left_template_id": left, "right_template_id": right}

    def sample_negative(self, c_split: str = "train") -> tuple[dict, dict, str]:
        _, template_id = self.random.choice(self.negative[c_split])
        left_id, right_id = self.random.sample(self.by_template[(c_split, template_id)], 2)
        return self.by_id[left_id], self.by_id[right_id], template_id


if __name__ == "__main__":
    path = Path("data/ravel_clean_subset/facts.jsonl")
    sampler = RavelPairSampler(path)
    fact, positive = sampler.sample_positive()
    left, right, template = sampler.sample_negative()
    assert positive["left_template_id"] != positive["right_template_id"]
    assert positive["left_template_id"] in fact["training_template_ids"] and positive["right_template_id"] in fact["training_template_ids"]
    assert left["fact_id"] != right["fact_id"] and template in left["training_template_ids"] and template in right["training_template_ids"]
    print("Dynamic RAVEL sampler check passed; no pairs were materialized.")
