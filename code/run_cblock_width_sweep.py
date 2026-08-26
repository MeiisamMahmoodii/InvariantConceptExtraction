"""Fixed-k C-block Top-k SAE width sweep; no encoder or data changes."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
ACT=ROOT/"data"/"activations_three_domain_natural_rewrite"
CODES=ROOT/"checkpoint"/"fact_bottleneck_three_domain_natural_rewrite_balanced_layer8_encoded_all.npy"
TOPK_REPORT=ROOT/"Report"/"topk_sae_sweep_report.json"
REPORT=ROOT/"Report"/"cblock_width_sweep_report.json"
FEATURES=ROOT/"data"/"sae_features"
SEED,EPOCHS,BATCH,K,WIDTHS,TOP,MIN_ACTIVE=20260825,30,256,64,(4,8,16),50,10
FAMILIES={"declarative","question","paraphrase","formal","structured"}; DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self, width, expansion):
        super().__init__(); self.encoder=nn.Linear(width,width*expansion); self.decoder=nn.Linear(width*expansion,width,bias=False); self.bias=nn.Parameter(torch.zeros(width))
    def forward(self,x):
        dense=F.relu(self.encoder(x)); values,indices=torch.topk(dense,K,dim=-1); z=torch.zeros_like(dense).scatter(1,indices,values); return z,self.decoder(z)+self.bias
    def normalize(self):
        with torch.no_grad(): self.decoder.weight.div_(self.decoder.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))


def select(rows):
    rng=np.random.default_rng(SEED); candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES]; groups=defaultdict(set)
    for r in candidates: groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values())); chosen=set()
    for domain,subjects in sorted(groups.items()): chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen],{d:n for d in sorted(groups)}


def purity(values): return max(Counter(values).values())/len(values)


def evaluate(model,matrix,mean,std,rows,indices,expansion):
    x=(matrix[indices].astype(np.float32)-mean)/std; zs=[]; total=0.
    with torch.inference_mode():
        for start in range(0,len(x),BATCH):
            batch=torch.from_numpy(x[start:start+BATCH]).to(DEVICE); z,reconstruction=model(batch); zs.append(z.cpu().numpy()); total+=F.mse_loss(reconstruction,batch,reduction="sum").item()
    z=np.concatenate(zs); held=[rows[i] for i in indices]; details=[]
    for feature in range(z.shape[1]):
        active=np.flatnonzero(z[:,feature]>0)
        if len(active)<MIN_ACTIVE: continue
        top=active[np.argsort(-z[active,feature])[:TOP]]; rel=purity([held[i]["C_relation"] for i in top]); dom=purity([held[i]["C_domain"] for i in top]); fam=purity([held[i]["S_family"] for i in top])
        details.append({"feature_id":feature,"active_examples":len(active),"C_relation_top50_purity":rel,"C_domain_top50_purity":dom,"S_family_top50_purity":fam})
    path=FEATURES/f"cblock_width_{expansion}x_k{K}_feature_selectivity.csv"
    with path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    rel=np.array([r["C_relation_top50_purity"] for r in details]);dom=np.array([r["C_domain_top50_purity"] for r in details]);fam=np.array([r["S_family_top50_purity"] for r in details]); count=int(np.sum((rel>=.8)&(fam<=.5))); scount=int(np.sum((fam>=.8)&(rel<=.5)))
    return {"feature_width":z.shape[1],"standardized_reconstruction_mse":float(total/x.size),"mean_L0":float((z>0).sum(1).mean()),"active_feature_count":len(details),"active_feature_fraction":len(details)/z.shape[1],"mean_top50_C_relation_purity":float(rel.mean()),"mean_top50_C_domain_purity":float(dom.mean()),"mean_top50_S_family_purity":float(fam.mean()),"C_relation_selective_features":count,"C_relation_selective_fraction_of_dictionary":count/z.shape[1],"S_family_selective_features":scount,"S_family_selective_fraction_of_dictionary":scount/z.shape[1],"feature_details":str(path.relative_to(ROOT))}


def train(expansion,matrix,train_indices,test_indices,rows):
    torch.manual_seed(SEED); raw=matrix[train_indices].astype(np.float32);mean,std=raw.mean(0),raw.std(0).clip(1e-6);x=torch.from_numpy((raw-mean)/std).to(DEVICE); model=SAE(x.shape[1],expansion).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for epoch in range(EPOCHS):
        order=torch.randperm(len(x),device=DEVICE);total=0.;steps=0;model.train()
        for batch in order.split(BATCH):
            _,rec=model(x[batch]);loss=F.mse_loss(rec,x[batch]);opt.zero_grad();loss.backward();opt.step();model.normalize();total+=loss.item();steps+=1
        print(f"c_bottleneck expansion={expansion}x k={K} epoch={epoch+1}/{EPOCHS} mse={total/steps:.5f}")
    return evaluate(model,matrix,mean,std,rows,test_indices,expansion)


def main():
    FEATURES.mkdir(exist_ok=True)
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    train_rows,subjects=select(rows);train_indices=np.array([int(r["activation_row"]) for r in train_rows]);test_indices=np.array([i for i,r in enumerate(rows) if r["C_split"]=="C_test"]);matrix=np.load(CODES)
    prior=json.loads(TOPK_REPORT.read_text(encoding="utf-8"))["c_bottleneck"]["by_k"][str(K)]
    report={"device":DEVICE,"representation":"frozen natural-rewrite C bottleneck","fixed_k":K,"expansions":WIDTHS,"epochs":EPOCHS,"training_rows":len(train_rows),"training_facts":len({r['fact_id'] for r in train_rows}),"training_subjects_per_domain":subjects,"expansion_4x_reused_from_matched_topk_sweep":prior,"expansion_8x":train(8,matrix,train_indices,test_indices,rows),"expansion_16x":train(16,matrix,train_indices,test_indices,rows),"contrastive_retrained":False,"new_data_generated":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if not k.startswith('expansion_')},indent=2))


if __name__=="__main__": main()
