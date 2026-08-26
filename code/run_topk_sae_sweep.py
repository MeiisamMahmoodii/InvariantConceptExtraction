"""Matched Top-k SAE sweep on frozen raw Gemma and C-bottleneck activations."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations_three_domain_natural_rewrite"
CODES = ROOT / "checkpoint" / "fact_bottleneck_three_domain_natural_rewrite_balanced_layer8_encoded_all.npy"
REPORT = ROOT / "Report" / "topk_sae_sweep_report.json"
FEATURES = ROOT / "data" / "sae_features"
SEED, EPOCHS, BATCH, EXPANSION, KS, TOP, MIN_ACTIVE = 20260825, 30, 256, 4, (16, 32, 64, 128), 50, 10
FAMILIES = {"declarative", "question", "paraphrase", "formal", "structured"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TopKSAE(nn.Module):
    def __init__(self, width, k):
        super().__init__(); self.k = k; self.encoder = nn.Linear(width, width * EXPANSION); self.decoder = nn.Linear(width * EXPANSION, width, bias=False); self.bias = nn.Parameter(torch.zeros(width))
    def forward(self, x):
        dense = F.relu(self.encoder(x)); values, indices = torch.topk(dense, self.k, dim=-1); z = torch.zeros_like(dense).scatter(1, indices, values); return z, self.decoder(z) + self.bias
    def normalize_decoder(self):
        with torch.no_grad(): self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


def select(rows):
    rng = np.random.default_rng(SEED); candidates = [row for row in rows if row["C_split"] == "C_train" and row["S_family"] in FAMILIES]; subjects = defaultdict(set)
    for row in candidates: subjects[row["C_domain"]].add(row["C_subject_id"])
    limit = min(map(len, subjects.values())); chosen = set()
    for domain, values in sorted(subjects.items()): chosen.update((domain, subject) for subject in rng.choice(sorted(values), limit, replace=False))
    return [row for row in candidates if (row["C_domain"], row["C_subject_id"]) in chosen], {domain: limit for domain in sorted(subjects)}


def purity(values): return max(Counter(values).values()) / len(values)


def evaluate(model, matrix, mean, std, rows, indices, name, k):
    x = ((matrix[indices].astype(np.float32) - mean) / std); chunks, total = [], 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH):
            batch = torch.from_numpy(x[start:start + BATCH]).to(DEVICE); z, reconstruction = model(batch); chunks.append(z.cpu().numpy()); total += F.mse_loss(reconstruction, batch, reduction="sum").item()
    z = np.concatenate(chunks); held = [rows[index] for index in indices]; details=[]
    for feature in range(z.shape[1]):
        active=np.flatnonzero(z[:, feature] > 0)
        if len(active) < MIN_ACTIVE: continue
        top=active[np.argsort(-z[active, feature])[:TOP]]; relation=purity([held[i]["C_relation"] for i in top]); domain=purity([held[i]["C_domain"] for i in top]); family=purity([held[i]["S_family"] for i in top])
        details.append({"feature_id":feature,"active_examples":len(active),"C_relation_top50_purity":relation,"C_domain_top50_purity":domain,"S_family_top50_purity":family})
    path = FEATURES / f"topk_{name}_k{k}_feature_selectivity.csv"
    with path.open("w", newline="", encoding="utf-8") as file: writer=csv.DictWriter(file, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    rel=np.array([row["C_relation_top50_purity"] for row in details]); dom=np.array([row["C_domain_top50_purity"] for row in details]); fam=np.array([row["S_family_top50_purity"] for row in details]); width=z.shape[1]
    return {"standardized_reconstruction_mse":float(total/x.size),"mean_L0":float((z>0).sum(1).mean()),"active_feature_count":len(details),"active_feature_fraction":len(details)/width,"mean_top50_C_relation_purity":float(rel.mean()),"mean_top50_C_domain_purity":float(dom.mean()),"mean_top50_S_family_purity":float(fam.mean()),"C_relation_selective_features":int(np.sum((rel>=.8)&(fam<=.5))),"C_relation_selective_fraction":float(np.mean((rel>=.8)&(fam<=.5))),"S_family_selective_features":int(np.sum((fam>=.8)&(rel<=.5))),"S_family_selective_fraction":float(np.mean((fam>=.8)&(rel<=.5))),"feature_details":str(path.relative_to(ROOT))}


def run(name, matrix, train_indices, test_indices, rows):
    raw=matrix[train_indices].astype(np.float32); mean,std=raw.mean(0),raw.std(0).clip(1e-6); x=torch.from_numpy((raw-mean)/std).to(DEVICE); result={"input_width":x.shape[1],"feature_width":x.shape[1]*EXPANSION,"by_k":{}}
    for k in KS:
        torch.manual_seed(SEED); model=TopKSAE(x.shape[1],k).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        for epoch in range(EPOCHS):
            order=torch.randperm(len(x),device=DEVICE); total=0.; steps=0; model.train()
            for batch in order.split(BATCH):
                _,reconstruction=model(x[batch]); loss=F.mse_loss(reconstruction,x[batch]); opt.zero_grad(); loss.backward(); opt.step(); model.normalize_decoder(); total+=loss.item();steps+=1
            print(f"{name} k={k} epoch={epoch+1}/{EPOCHS} mse={total/steps:.5f}")
        result["by_k"][str(k)]=evaluate(model,matrix,mean,std,rows,test_indices,name,k)
    return result


def main():
    FEATURES.mkdir(exist_ok=True)
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as file: rows=list(csv.DictReader(file))
    train_rows,subjects=select(rows); train_indices=np.array([int(row["activation_row"]) for row in train_rows]); test_indices=np.array([i for i,row in enumerate(rows) if row["C_split"]=="C_test"])
    raw=np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy"); c=np.load(CODES)
    report={"device":DEVICE,"method":"Top-k ReLU SAE with unit-norm decoder columns","k_values":KS,"epochs":EPOCHS,"expansion_factor":EXPANSION,"training_rows":len(train_rows),"training_facts":len({row['fact_id'] for row in train_rows}),"training_subjects_per_domain":subjects,"same_rows_for_both_saes":True,"raw_gemma_layer8":run("raw_gemma_layer8",raw,train_indices,test_indices,rows),"c_bottleneck":run("c_bottleneck",c,train_indices,test_indices,rows),"contrastive_retrained":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({key:value for key,value in report.items() if key not in {'raw_gemma_layer8','c_bottleneck'}},indent=2))


if __name__=="__main__": main()
