"""Extract frozen masked-mean Gemma-2-2B layer-8 activations for RAVEL rows."""

import csv
import os
from pathlib import Path

import numpy as np
import torch

_stub = torch.library.Library("torchvision", "DEF")
_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / os.environ.get("RAVEL_PARTITION_DIR", "ravel_partition_layer8")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32


def main() -> None:
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained("google/gemma-2-2b", local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
    output = np.lib.format.open_memmap(DATA / "gemma2_2b_layer8_mean.npy", mode="w+", dtype=np.float32, shape=(len(rows), 2304))
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            tokens = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            hidden = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states[8]
            mask = tokens["attention_mask"].unsqueeze(-1)
            output[start:start + len(batch)] = ((hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)).float().cpu().numpy()
            if start % 512 == 0 or start + len(batch) == len(rows):
                print(f"extracted={start + len(batch)}/{len(rows)} device={DEVICE}")
    del output
    print(f"Saved frozen layer-8 activations: {DATA / 'gemma2_2b_layer8_mean.npy'}")


if __name__ == "__main__":
    main()
