"""Matched Top-k SAEs for frozen raw Gemma, non-adversarial z_C, and z_S."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];ACT=ROOT/"data"/"activations_three_domain_natural_rewrite";TOPK=ROOT/"Report"/"topk_sae_sweep_report.json";REPORT=ROOT/"Report"/"partition_topk_sae_report.json";FEATURES=ROOT/"data"/"sae_features"
SEED,EPOCHS,BATCH,K,EXPANSION,TOP,MIN_ACTIVE=20260825,30,256,64,4,50,10;FAMILIES={"declarative","question","paraphrase","formal","structured"};DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self,width):
        super().__init__();self.encoder=nn.Linear(width,width*EXPANSION);self.decoder=nn.Linear(width*EXPANSION,width,bias=False);self.bias=nn.Parameter(torch.zeros(width))
    def forward(self,x):
        dense=F.relu(self.encoder(x));values,indices=torch.topk(dense,K,dim=-1);z=torch.zeros_like(dense).scatter(1,indices,values);return z,self.decoder(z)+self.bias
    def normalize(self):
        with torch.no_grad():self.decoder.weight.div_(self.decoder.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))


def select(rows):
    rng=np.random.default_rng(SEED);candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES];groups=defaultdict(set)
    for r in candidates:groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values()));chosen=set()
    for domain,subjects in sorted(groups.items()):chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen],{d:n for d in sorted(groups)}


def purity(values):return max(Counter(values).values())/len(values)


def evaluate(model,matrix,mean,std,rows,indices,name):
    x=(matrix[indices].astype(np.float32)-mean)/std;zs=[];total=0.
    with torch.inference_mode():
        for start in range(0,len(x),BATCH):
            batch=torch.from_numpy(x[start:start+BATCH]).to(DEVICE);z,rec=model(batch);zs.append(z.cpu().numpy());total+=F.mse_loss(rec,batch,reduction="sum").item()
    z=np.concatenate(zs);held=[rows[i] for i in indices];details=[]
    for feature in range(z.shape[1]):
        active=np.flatnonzero(z[:,feature]>0)
        if len(active)<MIN_ACTIVE:continue
        top=active[np.argsort(-z[active,feature])[:TOP]];rel=purity([held[i]["C_relation"] for i in top]);dom=purity([held[i]["C_domain"] for i in top]);fam=purity([held[i]["S_family"] for i in top]);details.append({"feature_id":feature,"active_examples":len(active),"C_relation_top50_purity":rel,"C_domain_top50_purity":dom,"S_family_top50_purity":fam})
    path=FEATURES/f"partition_{name}_k{K}_feature_selectivity.csv"
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    rel=np.array([r["C_relation_top50_purity"] for r in details]);dom=np.array([r["C_domain_top50_purity"] for r in details]);fam=np.array([r["S_family_top50_purity"] for r in details]);c=int(np.sum((rel>=.8)&(fam<=.5)));s=int(np.sum((fam>=.8)&(rel<=.5)));width=z.shape[1]
    return {"input_width":matrix.shape[1],"feature_width":width,"standardized_reconstruction_mse":float(total/x.size),"mean_L0":float((z>0).sum(1).mean()),"effective_active_dictionary_size":len(details),"effective_active_dictionary_fraction":len(details)/width,"mean_top50_C_relation_purity":float(rel.mean()),"mean_top50_C_domain_purity":float(dom.mean()),"mean_top50_S_family_purity":float(fam.mean()),"C_relation_selective_features":c,"C_relation_selective_fraction_of_dictionary":c/width,"S_family_selective_features":s,"S_family_selective_fraction_of_dictionary":s/width,"feature_details":str(path.relative_to(ROOT))}


def train(name,matrix,train_indices,test_indices,rows):
    torch.manual_seed(SEED);raw=matrix[train_indices].astype(np.float32);mean,std=raw.mean(0),raw.std(0).clip(1e-6);x=torch.from_numpy((raw-mean)/std).to(DEVICE);model=SAE(x.shape[1]).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for epoch in range(EPOCHS):
        order=torch.randperm(len(x),device=DEVICE);total=0.;steps=0;model.train()
        for batch in order.split(BATCH):
            _,rec=model(x[batch]);loss=F.mse_loss(rec,x[batch]);opt.zero_grad();loss.backward();opt.step();model.normalize();total+=loss.item();steps+=1
        print(f"{name} k={K} epoch={epoch+1}/{EPOCHS} mse={total/steps:.5f}")
    return evaluate(model,matrix,mean,std,rows,test_indices,name)


def main():
    FEATURES.mkdir(exist_ok=True)
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    train_rows,subjects=select(rows);train_indices=np.array([int(r["activation_row"]) for r in train_rows]);test_indices=np.array([i for i,r in enumerate(rows) if r["C_split"]=="C_test"])
    raw_prior=json.loads(TOPK.read_text(encoding="utf-8"))["raw_gemma_layer8"]["by_k"][str(K)];zc=np.load(ROOT/"checkpoint"/"cs_partition_c_all.npy");zs=np.load(ROOT/"checkpoint"/"cs_partition_s_all.npy")
    report={"device":DEVICE,"method":"Top-k ReLU SAE with unit-norm decoder columns","fixed_k":K,"expansion_factor":EXPANSION,"epochs":EPOCHS,"training_rows":len(train_rows),"training_facts":len({r['fact_id'] for r in train_rows}),"training_subjects_per_domain":subjects,"raw_gemma_layer8_reused_from_matched_topk_sweep":raw_prior,"nonadversarial_z_C":train("z_C",zc,train_indices,test_indices,rows),"nonadversarial_z_S":train("z_S",zs,train_indices,test_indices,rows),"partition_retrained":False,"new_losses_added":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if not k.startswith('raw_') and not k.startswith('non')},indent=2))


if __name__=="__main__":main()
