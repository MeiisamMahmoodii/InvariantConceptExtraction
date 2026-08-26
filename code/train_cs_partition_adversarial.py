"""Two-branch C/S partition with only gradient-reversal anti-leakage probes."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];ACT=ROOT/"data"/"activations_three_domain_natural_rewrite";OUT=ROOT/"checkpoint";REPORT=ROOT/"Report"/"cs_partition_adversarial_training_report.json"
SEED,EPOCHS,BATCH,DIM,TEMP,LAMBDA=20260825,30,256,128,.07,.1;FAMILIES=("declarative","question","paraphrase","formal","structured");DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):ctx.weight=weight;return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):return -ctx.weight*grad,None


class Model(nn.Module):
    def __init__(self,width,n_s,n_c):
        super().__init__();self.c=nn.Linear(width,DIM);self.s=nn.Linear(width,DIM);self.s_probe=nn.Linear(DIM,n_s);self.c_probe=nn.Linear(DIM,n_c)
    def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)


def select(rows):
    rng=np.random.default_rng(SEED);candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES];groups=defaultdict(set)
    for r in candidates:groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values()));chosen=set()
    for domain,subjects in sorted(groups.items()):chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen],{d:n for d in sorted(groups)}


def pair_loss(a,p,n):return F.cross_entropy(torch.stack([(a*p).sum(-1),(a*n).sum(-1)],1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEVICE))


def main():
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    for r in rows:r["activation_row"]=int(r["activation_row"])
    train,subjects=select(rows);by_fact,by_template=defaultdict(list),defaultdict(list)
    for r in train:by_fact[r["fact_id"]].append(r);by_template[(r["S_family"],r["S_variant"])].append(r)
    s_ids={label:i for i,label in enumerate(FAMILIES)};c_labels=sorted({r["C_relation"] for r in train});c_ids={label:i for i,label in enumerate(c_labels)}
    x=torch.from_numpy(np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy")).to(DEVICE);model=Model(x.shape[1],len(s_ids),len(c_ids)).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);history=[]
    for epoch in range(EPOCHS):
        tc=ts=ta=0.;steps=0;model.train()
        for choices in np.array_split(rng.permutation(len(train)),max(1,len(train)//BATCH)):
            anchors=[train[i] for i in choices];cpos=[rng.choice([r for r in by_fact[a["fact_id"]] if r["S_family"]!=a["S_family"]]) for a in anchors];cneg=[rng.choice([r for r in by_template[(a["S_family"],a["S_variant"])] if r["fact_id"]!=a["fact_id"]]) for a in anchors];spos=[rng.choice([r for r in by_template[(a["S_family"],a["S_variant"])] if r["fact_id"]!=a["fact_id"]]) for a in anchors];sneg=[rng.choice([r for r in by_fact[a["fact_id"]] if r["S_family"]!=a["S_family"]]) for a in anchors]
            idx=lambda group:torch.tensor([r["activation_row"] for r in group],device=DEVICE);ca,sa=model(x[idx(anchors)]);cp,_=model(x[idx(cpos)]);cn,_=model(x[idx(cneg)]);_,sp=model(x[idx(spos)]);_,sn=model(x[idx(sneg)])
            lc=pair_loss(ca,cp,cn);ls=pair_loss(sa,sp,sn);ys=torch.tensor([s_ids[r["S_family"]] for r in anchors],device=DEVICE);yc=torch.tensor([c_ids[r["C_relation"]] for r in anchors],device=DEVICE);adv=F.cross_entropy(model.s_probe(Reverse.apply(ca,LAMBDA)),ys)+F.cross_entropy(model.c_probe(Reverse.apply(sa,LAMBDA)),yc);loss=(lc+ls)/2+adv;opt.zero_grad();loss.backward();opt.step();tc+=lc.item();ts+=ls.item();ta+=adv.item();steps+=1
        history.append({"epoch":epoch+1,"C_triplet_loss":tc/steps,"S_triplet_loss":ts/steps,"adversarial_probe_loss":ta/steps});print(f"epoch={epoch+1}/{EPOCHS} C_loss={history[-1]['C_triplet_loss']:.4f} S_loss={history[-1]['S_triplet_loss']:.4f} adv_loss={history[-1]['adversarial_probe_loss']:.4f}")
    model.eval()
    with torch.no_grad():c,s=model(x)
    np.save(OUT/"cs_partition_adversarial_c_all.npy",c.cpu().numpy());np.save(OUT/"cs_partition_adversarial_s_all.npy",s.cpu().numpy());torch.save({"state_dict":model.state_dict(),"config":{"input_width":x.shape[1],"C_dim":DIM,"S_dim":DIM,"adversarial_weight":LAMBDA,"C_adversary_target":"S_family","S_adversary_target":"C_relation","epochs":EPOCHS,"seed":SEED},"history":history},OUT/"cs_partition_adversarial_layer8.pt")
    report={"device":DEVICE,"training_rows":len(train),"training_facts":len(by_fact),"training_subjects_per_domain":subjects,"S_train_families":list(FAMILIES),"held_out_S_family":"indirect","architecture":"2304 -> z_C(128) + z_S(128)","loss":"same two contrastive objectives plus two gradient-reversal probes","adversarial_weight":LAMBDA,"decoder":False,"reconstruction":False,"SAE":False,"history":history};REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))


if __name__=="__main__":main()
