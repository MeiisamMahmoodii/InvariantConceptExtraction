"""Extract frozen Gemma 2 2B layer-13 masked-mean activations; no model training."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
# Work around an incompatible optional torchvision install; Gemma text inference does not use this operator.
_torchvision_stub = torch.library.Library("torchvision", "DEF")
_torchvision_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "controlled_surface_dataset.csv"
OUT = ROOT / "data" / "activations"
MODEL_ID = "google/gemma-2-2b"
LAYER_INDICES = (5, 13, 21)  # post-blocks 4, 12, 20; fixed before diagnostics
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = "cuda"


def main():
    layers = tuple(int(value) for value in sys.argv[1:]) or LAYER_INDICES
    if any(layer < 1 or layer > 26 for layer in layers):
        raise SystemExit("Layer indices must be between 1 and 26")
    OUT.mkdir(parents=True, exist_ok=True)
    with DATASET.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
    activations = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            tokens = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
            hidden_states = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states
            mask = tokens["attention_mask"].unsqueeze(-1)
            for layer in layers:
                hidden = hidden_states[layer]
                activations[layer].append(((hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)).cpu().float().numpy())
            if (start // BATCH_SIZE + 1) % 25 == 0 or start + len(batch) == len(rows):
                print(f"extracted={start + len(batch)}/{len(rows)}")
    matrices = {layer: np.concatenate(chunks) for layer, chunks in activations.items()}
    metadata_fields = ("example_id", "fact_id", "C_domain", "C_relation", "C_subject_id", "C_value_id", "S_family", "S_variant", "C_split", "S_split", "activation_row")
    with (OUT / "gemma2_2b_layer_sweep_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata_fields); writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in metadata_fields[:-1]} | {"activation_row": index} for index, row in enumerate(rows))
    for layer, matrix in matrices.items():
        layer_out = OUT / f"gemma2_2b_layer{layer}_mean"
        layer_out.mkdir(exist_ok=True)
        np.save(layer_out / "activations.npy", matrix)
    manifest = {"model": MODEL_ID, "layer_indices": list(layers), "layer_descriptions": {str(layer): f"residual hidden state after transformer block {layer - 1}" for layer in layers}, "pooling": "masked mean over all non-padding tokens", "max_length": MAX_LENGTH, "device": DEVICE, "model_dtype": "bfloat16", "dtype": str(next(iter(matrices.values())).dtype), "shape_per_layer": list(next(iter(matrices.values())).shape), "dataset_rows": len(rows), "training_performed": False}
    name = "gemma2_2b_layer_sweep_manifest.json" if layers == LAYER_INDICES else f"gemma2_2b_layer{'_'.join(map(str, layers))}_manifest.json"
    (OUT / name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
