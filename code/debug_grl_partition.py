"""Sanity-check GRL direction and adversary signal without retraining the partition."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from train_cs_partition_adversarial import Model, Reverse, FAMILIES, LAMBDA, SEED

ROOT=Path(__file__).resolve().parents[1];ACT=ROOT/"data"/"activations_three_domain_natural_rewrite";CKPT=ROOT/"checkpoint"/"cs_partition_adversarial_layer8.pt";REPORT=ROOT/"Report"/"cs_partition_adversarial_grl_debug.json";DEVICE="cuda" if torch.cuda.is_available() else "cpu"


def accuracy(logits,labels):return float((logits.argmax(1)==labels).float().mean().item())


def probe(x,y,train,test):
    model=LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(x[train],y[train]);return {"C_train_accuracy":float(np.mean(model.predict(x[train])==y[train])),"C_test_accuracy":float(np.mean(model.predict(x[test])==y[test]))}


def main():
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):r["activation_row"]=i
    saved=torch.load(CKPT,map_location=DEVICE,weights_only=False);state=saved["state_dict"];c_labels=sorted({r["C_relation"] for r in rows});s_ids={name:i for i,name in enumerate(FAMILIES)};c_ids={name:i for i,name in enumerate(c_labels)}
    model=Model(2304,len(s_ids),len(c_ids)).to(DEVICE);model.load_state_dict(state);model.eval();raw=np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy")
    train_rows=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in s_ids][:256];idx=torch.tensor([r["activation_row"] for r in train_rows],device=DEVICE);ys=torch.tensor([s_ids[r["S_family"]] for r in train_rows],device=DEVICE);yc=torch.tensor([c_ids[r["C_relation"]] for r in train_rows],device=DEVICE);x=torch.from_numpy(raw).to(DEVICE)
    ca,_=model(x[idx]);normal_s=torch.autograd.grad(F.cross_entropy(model.s_probe(ca),ys),model.c.weight,retain_graph=False)[0].flatten();ca,_=model(x[idx]);grl_s=torch.autograd.grad(F.cross_entropy(model.s_probe(Reverse.apply(ca,LAMBDA)),ys),model.c.weight,retain_graph=False)[0].flatten()
    _,sa=model(x[idx]);normal_c=torch.autograd.grad(F.cross_entropy(model.c_probe(sa),yc),model.s.weight,retain_graph=False)[0].flatten();_,sa=model(x[idx]);grl_c=torch.autograd.grad(F.cross_entropy(model.c_probe(Reverse.apply(sa,LAMBDA)),yc),model.s.weight,retain_graph=False)[0].flatten()
    c=np.load(ROOT/"checkpoint"/"cs_partition_adversarial_c_all.npy");s=np.load(ROOT/"checkpoint"/"cs_partition_adversarial_s_all.npy");train=np.array([r["C_split"]=="C_train" and r["S_family"] in s_ids for r in rows]);seen_test=np.array([r["C_split"]=="C_test" and r["S_family"] in s_ids for r in rows]);relation_test=np.array([r["C_split"]=="C_test" for r in rows]);sy=np.array([s_ids.get(r["S_family"],-1) for r in rows]);cy=np.array([c_ids[r["C_relation"]] for r in rows])
    with torch.no_grad():joint_s=model.s_probe(torch.from_numpy(c).to(DEVICE));joint_c=model.c_probe(torch.from_numpy(s).to(DEVICE))
    joint_s_metrics={"C_train":accuracy(joint_s[torch.from_numpy(train).to(DEVICE)],torch.from_numpy(sy[train]).to(DEVICE)),"C_test_seen":accuracy(joint_s[torch.from_numpy(seen_test).to(DEVICE)],torch.from_numpy(sy[seen_test]).to(DEVICE))}
    joint_c_metrics={"C_train":accuracy(joint_c[torch.from_numpy(train).to(DEVICE)],torch.from_numpy(cy[train]).to(DEVICE)),"C_test":accuracy(joint_c[torch.from_numpy(relation_test).to(DEVICE)],torch.from_numpy(cy[relation_test]).to(DEVICE))}
    report={
        "labels":{"z_C_adversary":"S_family (five observed training families)","z_S_adversary":"C_relation (nine canonical relation IDs)","target_validation":{"S_family_classes":list(FAMILIES),"C_relation_classes":c_labels}},
        "gradient_reversal":{"lambda":LAMBDA,"S_from_z_C":{"dot_product":float(normal_s@grl_s),"cosine":float(F.cosine_similarity(normal_s,grl_s,dim=0))},"C_relation_from_z_S":{"dot_product":float(normal_c@grl_c),"cosine":float(F.cosine_similarity(normal_c,grl_c,dim=0))}},
        "frozen_probe_accuracy":{"S_family_from_z_C":probe(c,sy,train,seen_test),"C_relation_from_z_S":probe(s,cy,train,relation_test)},
        "joint_adversary_head_accuracy":{"S_family_from_z_C":joint_s_metrics,"C_relation_from_z_S":joint_c_metrics},
        "partition_retrained":False,"lambda_sweep":False,
    }
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))
    if report["gradient_reversal"]["S_from_z_C"]["dot_product"]>=0 or report["gradient_reversal"]["C_relation_from_z_S"]["dot_product"]>=0:raise SystemExit("GRL sign check failed")


if __name__=="__main__":main()
