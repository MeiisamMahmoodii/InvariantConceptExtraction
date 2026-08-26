"""Append frozen layer-8 activations for the validated Book expansion."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
_stub = torch.library.Library("torchvision", "DEF"); _stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "three_domain_diversity" / "controlled_surface_dataset.csv"
BASE = ROOT / "data" / "activations_with_indirect" / "gemma2_2b_layer8_mean" / "activations.npy"
OUT = ROOT / "data" / "activations_three_domain"


def main():
    with DATASET.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    new = rows[len(np.load(BASE)):]; base = np.load(BASE); assert len(rows) == len(base) + len(new)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", local_files_only=True); tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModel.from_pretrained("google/gemma-2-2b", local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval(); chunks=[]
    with torch.inference_mode():
        for start in range(0, len(new), 32):
            batch=new[start:start+32]; tokens=tok([r["text"] for r in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda"); h=model(**tokens, output_hidden_states=True, use_cache=False).hidden_states[8]; m=tokens["attention_mask"].unsqueeze(-1); chunks.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).cpu().float().numpy())
            if start % 1024 == 0 or start + len(batch) == len(new): print(f"extracted={start+len(batch)}/{len(new)}")
    matrix=np.concatenate([base,np.concatenate(chunks)]); layer=OUT/"gemma2_2b_layer8_mean"; layer.mkdir(parents=True,exist_ok=True); np.save(layer/"activations.npy",matrix)
    fields=("example_id","fact_id","C_domain","C_relation","C_subject_id","C_value_id","S_family","S_variant","C_split","S_split","activation_row")
    with (OUT/"gemma2_2b_layer_sweep_metadata.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,"") for k in fields[:-1]}|{"activation_row":i} for i,r in enumerate(rows))
    result={"base_rows":len(base),"book_rows":len(new),"total_rows":len(matrix),"training_performed":False};(OUT/"manifest.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
