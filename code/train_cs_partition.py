"""Learn separate C and S contrastive blocks from frozen layer-8 activations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
ACT=ROOT/"data"/"activations_three_domain_natural_rewrite"
OUT=ROOT/"checkpoint"; REPORT=ROOT/"Report"/"cs_partition_training_report.json"
SEED,EPOCHS,BATCH,DIM,TEMP=20260825,30,256,128,.07
FAMILIES={"declarative","question","paraphrase","formal","structured"}; DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class Partition(nn.Module):
    def __init__(self,width):
        super().__init__();self.c=nn.Linear(width,DIM);self.s=nn.Linear(width,DIM)
    def forward(self,x): return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)


def select(rows):
    rng=np.random.default_rng(SEED); candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES]; groups=defaultdict(set)
    for r in candidates: groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values()));chosen=set()
    for domain,subjects in sorted(groups.items()): chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen],{d:n for d in sorted(groups)}


def pair_loss(anchor,positive,negative):
    return F.cross_entropy(torch.stack([(anchor*positive).sum(-1),(anchor*negative).sum(-1)],1)/TEMP,torch.zeros(len(anchor),dtype=torch.long,device=DEVICE))


def main():
    torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    for r in rows:r["activation_row"]=int(r["activation_row"])
    train,subjects=select(rows);by_fact,by_template=defaultdict(list),defaultdict(list)
    for r in train:by_fact[r["fact_id"]].append(r);by_template[(r["S_family"],r["S_variant"])].append(r)
    x=torch.from_numpy(np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy")).to(DEVICE);model=Partition(x.shape[1]).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);history=[]
    for epoch in range(EPOCHS):
        total_c=total_s=0.;steps=0;model.train()
        for choices in np.array_split(rng.permutation(len(train)),max(1,len(train)//BATCH)):
            anchors=[train[i] for i in choices]
            c_pos=[rng.choice([r for r in by_fact[a["fact_id"]] if r["S_family"]!=a["S_family"]]) for a in anchors]
            c_neg=[rng.choice([r for r in by_template[(a["S_family"],a["S_variant"])] if r["fact_id"]!=a["fact_id"]]) for a in anchors]
            s_pos=[rng.choice([r for r in by_template[(a["S_family"],a["S_variant"])] if r["fact_id"]!=a["fact_id"]]) for a in anchors]
            s_neg=[rng.choice([r for r in by_fact[a["fact_id"]] if r["S_family"]!=a["S_family"]]) for a in anchors]
            idx=lambda group:torch.tensor([r["activation_row"] for r in group],device=DEVICE)
            c_a,s_a=model(x[idx(anchors)]);c_p,_=model(x[idx(c_pos)]);c_n,_=model(x[idx(c_neg)]);_,s_p=model(x[idx(s_pos)]);_,s_n=model(x[idx(s_neg)])
            lc=pair_loss(c_a,c_p,c_n);ls=pair_loss(s_a,s_p,s_n);loss=(lc+ls)/2;opt.zero_grad();loss.backward();opt.step();total_c+=lc.item();total_s+=ls.item();steps+=1
        history.append({"epoch":epoch+1,"C_triplet_loss":total_c/steps,"S_triplet_loss":total_s/steps});print(f"epoch={epoch+1}/{EPOCHS} C_loss={history[-1]['C_triplet_loss']:.4f} S_loss={history[-1]['S_triplet_loss']:.4f}")
    model.eval()
    with torch.no_grad():c,s=model(x)
    np.save(OUT/"cs_partition_c_all.npy",c.cpu().numpy());np.save(OUT/"cs_partition_s_all.npy",s.cpu().numpy())
    torch.save({"state_dict":model.state_dict(),"config":{"input_width":x.shape[1],"C_dim":DIM,"S_dim":DIM,"temperature":TEMP,"epochs":EPOCHS,"seed":SEED,"C_positive":"same fact, different S family","C_negative":"different fact, same S family and variant","S_positive":"same S family and variant, different fact","S_negative":"same fact, different S family"},"history":history},OUT/"cs_partition_layer8.pt")
    report={"device":DEVICE,"training_rows":len(train),"training_facts":len(by_fact),"training_subjects_per_domain":subjects,"S_train_families":sorted(FAMILIES),"held_out_S_family":"indirect","architecture":"2304 -> z_C(128) + z_S(128)","loss":"two matched one-positive/one-negative cosine contrastive objectives","decoder":False,"reconstruction":False,"SAE":False,"history":history};REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))


if __name__=="__main__":main()
