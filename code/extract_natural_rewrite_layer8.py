"""Reuse unchanged layer-8 activations and extract only natural rewrites."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

_stub = torch.library.Library("torchvision", "DEF")
_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "three_domain_natural_rewrite" / "controlled_surface_dataset.csv"
SOURCE_ACT = ROOT / "data" / "activations_three_domain"
OUT = ROOT / "data" / "activations_three_domain_natural_rewrite"


def main():
    with DATASET.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    with (SOURCE_ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: source_rows = list(csv.DictReader(file))
    source_x = np.load(SOURCE_ACT / "gemma2_2b_layer8_mean" / "activations.npy")
    old_index = {row["example_id"]: int(row["activation_row"]) for row in source_rows}
    rewritten = [row for row in rows if row["S_family"] == "paraphrase"]
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained("google/gemma-2-2b", local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval(); new_x = {}
    with torch.inference_mode():
        for start in range(0, len(rewritten), 32):
            batch = rewritten[start:start + 32]; tokens = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            hidden = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states[8]; mask = tokens["attention_mask"].unsqueeze(-1)
            vectors = ((hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)).cpu().float().numpy(); new_x.update(zip((row["example_id"] for row in batch), vectors))
            if start % 1024 == 0 or start + len(batch) == len(rewritten): print(f"extracted_natural_rewrites={start + len(batch)}/{len(rewritten)}")
    matrix = np.stack([new_x[row["example_id"]] if row["S_family"] == "paraphrase" else source_x[old_index[row["example_id"]]] for row in rows])
    layer = OUT / "gemma2_2b_layer8_mean"; layer.mkdir(parents=True, exist_ok=True); np.save(layer / "activations.npy", matrix)
    fields = ("example_id", "fact_id", "C_domain", "C_relation", "C_subject_id", "C_value_id", "S_family", "S_variant", "C_split", "S_split", "activation_row")
    with (OUT / "gemma2_2b_layer_sweep_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key, "") for key in fields[:-1]} | {"activation_row": index} for index, row in enumerate(rows))
    result = {"total_rows": len(rows), "rewritten_rows_extracted": len(rewritten), "unchanged_rows_reused": len(rows) - len(rewritten), "training_performed": False}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
