"""Drop activation rows removed by the deterministic pairability filter."""

import hashlib
import json
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ravel_partition_layer8"


SEED = 20260825


def original_ids():
    facts = [json.loads(line) for line in (ROOT / "data" / "ravel_clean_subset" / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = []
    for fact in facts:
        template_ids = fact["training_template_ids"] if fact["C_split"] == "train" else fact["heldout_template_ids"] if fact["C_split"] == "test" else []
        ordered = sorted(template_ids, key=lambda item: hashlib.blake2b(f"{SEED}:{fact['fact_id']}:{item}".encode(), digest_size=16).digest())[:3]
        ids.extend(f"{fact['fact_id']}:{template_id}" for template_id in ordered)
    return ids


def main():
    before = original_ids()
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        after = [row["example_id"] for row in csv.DictReader(handle)]
    lookup = {example_id: index for index, example_id in enumerate(before)}
    indices = np.array([lookup[example_id] for example_id in after], dtype=np.int64)
    source = np.load(DATA / "gemma2_2b_layer8_mean.npy", mmap_mode="r")
    target = np.lib.format.open_memmap(DATA / "gemma2_2b_layer8_mean_filtered.npy", mode="w+", dtype=np.float32, shape=(len(indices), source.shape[1]))
    for start in range(0, len(indices), 1024): target[start:start + 1024] = source[indices[start:start + 1024]]
    del target, source
    (DATA / "gemma2_2b_layer8_mean.npy").unlink()
    (DATA / "gemma2_2b_layer8_mean_filtered.npy").replace(DATA / "gemma2_2b_layer8_mean.npy")
    print(f"Aligned activations: {len(before)} -> {len(after)} rows")


if __name__ == "__main__": main()
