"""Append frozen layer-8 activations for the held-out indirect family."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

_stub = torch.library.Library("torchvision", "DEF")
_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "controlled_surface_dataset_with_indirect.csv"
BASE_ACT = ROOT / "data" / "activations"
OUT = ROOT / "data" / "activations_with_indirect"
MODEL, DEVICE, BATCH = "google/gemma-2-2b", "cuda", 32


def main():
    with DATASET.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    indirect = [r for r in rows if r["S_family"] == "indirect"]
    base = np.load(BASE_ACT / "gemma2_2b_layer8_mean" / "activations.npy")
    assert len(rows) == len(base) + len(indirect)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(indirect), BATCH):
            batch = indirect[start:start + BATCH]; tokens = tokenizer([r["text"] for r in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            hidden = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states[8]; mask = tokens["attention_mask"].unsqueeze(-1)
            chunks.append(((hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)).cpu().float().numpy()); print(f"extracted={min(start + len(batch), len(indirect))}/{len(indirect)}")
    matrix = np.concatenate([base, np.concatenate(chunks)])
    out_layer = OUT / "gemma2_2b_layer8_mean"; out_layer.mkdir(parents=True, exist_ok=True); np.save(out_layer / "activations.npy", matrix)
    fields = ("example_id", "fact_id", "C_domain", "C_relation", "C_subject_id", "C_value_id", "S_family", "S_variant", "C_split", "S_split", "activation_row")
    with (OUT / "gemma2_2b_layer_sweep_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows({k: r.get(k, "") for k in fields[:-1]} | {"activation_row": i} for i, r in enumerate(rows))
    report = {"base_rows": len(base), "indirect_rows": len(indirect), "total_rows": len(matrix), "layer": 8, "model": MODEL, "device": DEVICE, "training_performed": False}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
