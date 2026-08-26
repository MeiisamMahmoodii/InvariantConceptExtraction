"""Frozen Gemma, partition, and Top-k SAE extraction for extra held-out families."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_stub=torch.library.Library("torchvision","DEF");_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data"/"extra_heldout_surface_families";OUT=ROOT/"data"/"sae_activations";CKPT=ROOT/"checkpoint";DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class Partition(nn.Module):
    def __init__(self,width):super().__init__();self.c=nn.Linear(width,128);self.s=nn.Linear(width,128)
    def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)


class SAE(nn.Module):
    def __init__(self,width,k):super().__init__();self.k=k;self.encoder=nn.Linear(width,width*4);self.decoder=nn.Linear(width*4,width,bias=False);self.bias=nn.Parameter(torch.zeros(width))
    def encode(self,x):
        dense=F.relu(self.encoder(x));values,indices=torch.topk(dense,self.k,dim=-1);return indices,values,self.encoder.out_features


def load_sae(name):
    saved=torch.load(CKPT/f"topk_{name}_k64.pt",map_location=DEVICE,weights_only=False);model=SAE(saved["config"]["input_width"],saved["config"]["k"]).to(DEVICE);model.load_state_dict(saved["state_dict"]);model.eval();return model,torch.from_numpy(saved["input_mean"]).to(DEVICE),torch.from_numpy(saved["input_std"]).to(DEVICE)


def main():
    with (DATA/"controlled_surface_dataset.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    tokenizer=AutoTokenizer.from_pretrained("google/gemma-2-2b",local_files_only=True);tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    gemma=AutoModel.from_pretrained("google/gemma-2-2b",local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEVICE).eval();partition=Partition(2304).to(DEVICE);partition.load_state_dict(torch.load(CKPT/"cs_partition_layer8.pt",map_location=DEVICE,weights_only=False)["state_dict"],strict=False);partition.eval();raw_sae,raw_mean,raw_std=load_sae("raw_gemma_layer8");c_sae,c_mean,c_std=load_sae("z_C");collected={"raw_gemma_layer8":[],"z_C":[]}
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];tokens=tokenizer([r["text"] for r in batch],padding=True,truncation=True,max_length=128,return_tensors="pt").to(DEVICE);hidden=gemma(**tokens,output_hidden_states=True,use_cache=False).hidden_states[8];mask=tokens["attention_mask"].unsqueeze(-1);raw=((hidden*mask.to(hidden.dtype)).sum(1)/mask.sum(1)).float();z_c,_=partition(raw)
            for name,values,sae,mean,std in (("raw_gemma_layer8",raw,raw_sae,raw_mean,raw_std),("z_C",z_c,c_sae,c_mean,c_std)):
                indices,activations,width=sae.encode((values-mean)/std);collected[name].append((indices.cpu().numpy().astype(np.int32),activations.cpu().numpy().astype(np.float32),width))
            if start%256==0 or start+len(batch)==len(rows):print(f"extracted={start+len(batch)}/{len(rows)}")
    for name,chunks in collected.items():
        indices=np.concatenate([c[0] for c in chunks]);values=np.concatenate([c[1] for c in chunks]);np.savez_compressed(OUT/f"extra_heldout_{name}_k64_sparse_activations.npz",indices=indices,values=values,shape=np.array((len(rows),chunks[0][2]),dtype=np.int64))
    fields=("activation_row","example_id","fact_id","C_domain","C_relation","C_subject_id","C_value_id","S_family","S_variant","C_split","S_split")
    with (DATA/"sae_activation_metadata.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({key:row.get(key,"") for key in fields[1:]}|{"activation_row":i} for i,row in enumerate(rows))
    report={"rows":len(rows),"families":sorted({r["S_family"] for r in rows}),"raw_sparse_activations":str((OUT/"extra_heldout_raw_gemma_layer8_k64_sparse_activations.npz").relative_to(ROOT)),"z_C_sparse_activations":str((OUT/"extra_heldout_z_C_k64_sparse_activations.npz").relative_to(ROOT)),"partition_retrained":False,"SAE_retrained":False}
    (ROOT/"Report"/"extra_heldout_sae_extraction_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
