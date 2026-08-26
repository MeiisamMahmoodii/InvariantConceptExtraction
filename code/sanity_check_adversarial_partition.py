"""Numerically verify GRL mechanics and saved adversary signal; no retraining of partition."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];ACT=ROOT/"data"/"activations_three_domain_natural_rewrite";CKPT=ROOT/"checkpoint"/"cs_partition_adversarial_layer8.pt";REPORT=ROOT/"Report"/"cs_partition_adversarial_sanity_check.json"
SEED,BATCH,LAMBDA=20260825,256,.1;FAMILIES=("declarative","question","paraphrase","formal","structured");DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):ctx.weight=weight;return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):return -ctx.weight*grad,None


class Model(nn.Module):
    def __init__(self,width,n_s,n_c):
        super().__init__();self.c=nn.Linear(width,128);self.s=nn.Linear(width,128);self.s_probe=nn.Linear(128,n_s);self.c_probe=nn.Linear(128,n_c)
    def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)


def select(rows):
    rng=np.random.default_rng(SEED);candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES];groups=defaultdict(set)
    for r in candidates:groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values()));chosen=set()
    for domain,subjects in sorted(groups.items()):chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen]


def gradient_dot(model,x,labels,branch,head):
    model.zero_grad();z=model(x)[branch];ordinary=F.cross_entropy(head(z),labels);ordinary.backward();g1=(model.c.weight if branch==0 else model.s.weight).grad.detach().flatten().clone()
    model.zero_grad();z=model(x)[branch];reversed_loss=F.cross_entropy(head(Reverse.apply(z,LAMBDA)),labels);reversed_loss.backward();g2=(model.c.weight if branch==0 else model.s.weight).grad.detach().flatten().clone()
    return {"ordinary_gradient_norm":float(g1.norm()),"GRL_gradient_norm":float(g2.norm()),"dot_product":float(g1@g2),"cosine":float((g1@g2)/(g1.norm()*g2.norm()))}


def frozen_probe(x,y,train,test):
    scaler=StandardScaler().fit(x[train]);model=LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(scaler.transform(x[train]),y[train]);return {"train_accuracy":float(np.mean(model.predict(scaler.transform(x[train]))==y[train])),"held_out_accuracy":float(np.mean(model.predict(scaler.transform(x[test]))==y[test]))}


def main():
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):r["activation_row"]=i
    train_rows=select(rows);saved=torch.load(CKPT,map_location=DEVICE,weights_only=False);model=Model(2304,5,9).to(DEVICE);model.load_state_dict(saved["state_dict"]);model.eval();xall=torch.from_numpy(np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy")).to(DEVICE)
    s_ids={label:i for i,label in enumerate(FAMILIES)};relations=sorted({r["C_relation"] for r in train_rows});c_ids={label:i for i,label in enumerate(relations)};batch=train_rows[:BATCH];idx=torch.tensor([r["activation_row"] for r in batch],device=DEVICE);ys=torch.tensor([s_ids[r["S_family"]] for r in batch],device=DEVICE);yc=torch.tensor([c_ids[r["C_relation"]] for r in batch],device=DEVICE)
    with torch.no_grad():z_c,z_s=model(xall);zc=z_c.cpu().numpy();zs=z_s.cpu().numpy()
    train=np.array([r["C_split"]=="C_train" for r in rows]);test=~train;labels_s=np.array([r["S_family"] for r in rows]);labels_c=np.array([r["C_relation"] for r in rows]);selected=np.array([(r["C_split"]=="C_train" and r["S_family"] in FAMILIES) for r in rows]);
    with torch.no_grad():joint_s=model.s_probe(z_c).argmax(1).cpu().numpy();joint_c=model.c_probe(z_s).argmax(1).cpu().numpy()
    target_s=np.array([s_ids.get(r["S_family"],-1) for r in rows]);target_c=np.array([c_ids.get(r["C_relation"],-1) for r in rows]);
    report={"implementation":{"gradient_reversal_weight":LAMBDA,"z_C_adversary_target":"S_family","z_S_adversary_target":"C_relation","S_classes":list(FAMILIES),"C_relation_classes":relations,"labels_verified_on_training_rows":all(r["S_family"] in s_ids and r["C_relation"] in c_ids for r in train_rows)},"numerical_gradient_check":{"z_C_S_adversary":gradient_dot(model,xall[idx],ys,0,model.s_probe),"z_S_C_adversary":gradient_dot(model,xall[idx],yc,1,model.c_probe)},"saved_joint_adversary_final_accuracy":{"z_C_to_S_family_training_rows":float(np.mean(joint_s[selected]==target_s[selected])),"z_S_to_C_relation_training_rows":float(np.mean(joint_c[selected]==target_c[selected]))},"fresh_frozen_block_probe":{"z_C_to_S_family":frozen_probe(zc,labels_s,train,test),"z_S_to_C_relation":frozen_probe(zs,labels_c,train,test)},"historical_joint_adversary_accuracy":"not logged during the completed training run; final checkpoint accuracy reported instead","partition_retrained":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
