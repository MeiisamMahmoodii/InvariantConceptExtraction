"""Subset existing template activations for the matched control without inference."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "three_domain_template_matched" / "controlled_surface_dataset.csv"
SOURCE = ROOT / "data" / "activations_three_domain"
OUT = ROOT / "data" / "activations_three_domain_template_matched"


def main():
    with DATASET.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    with (SOURCE / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: source_rows = list(csv.DictReader(file))
    index = {row["example_id"]: int(row["activation_row"]) for row in source_rows}; source_x = np.load(SOURCE / "gemma2_2b_layer8_mean" / "activations.npy")
    matrix = source_x[[index[row["example_id"]] for row in rows]]
    layer = OUT / "gemma2_2b_layer8_mean"; layer.mkdir(parents=True, exist_ok=True); np.save(layer / "activations.npy", matrix)
    fields = ("example_id", "fact_id", "C_domain", "C_relation", "C_subject_id", "C_value_id", "S_family", "S_variant", "C_split", "S_split", "activation_row")
    with (OUT / "gemma2_2b_layer_sweep_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key, "") for key in fields[:-1]} | {"activation_row": i} for i, row in enumerate(rows))
    result = {"rows": len(rows), "source": "frozen template activations", "new_model_inference": False, "training_performed": False}; (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
