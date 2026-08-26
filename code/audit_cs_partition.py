"""Held-out C/S block decodability, rank, and controlled retrieval audit."""

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
ACT=ROOT/"data"/"activations_three_domain_natural_rewrite"
CODES=Path(os.environ.get("C_CODES",ROOT/"checkpoint"/"cs_partition_c_all.npy"));SCODES=Path(os.environ.get("S_CODES",ROOT/"checkpoint"/"cs_partition_s_all.npy"))
REPORT=Path(os.environ.get("AUDIT_REPORT",ROOT/"Report"/"cs_partition_audit.json"));SEED=20260825


def probe(x,rows,label):
    train=np.array([r["C_split"]=="C_train" for r in rows]);test=~train;y=np.array([r[label] for r in rows]);scaler=StandardScaler().fit(x[train]);model=LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(scaler.transform(x[train]),y[train]);return {"accuracy":float(np.mean(model.predict(scaler.transform(x[test]))==y[test])),"chance":1/len(set(y))}


def rank(x):
    centered=x-x.mean(0);values=np.linalg.svd(centered,compute_uv=False)**2;p=values/values.sum();return {"participation_ratio":float(values.sum()**2/(values**2).sum()),"entropy_effective_rank":float(np.exp(-(p*np.log(p+1e-30)).sum()))}


def c_retrieval(x,rows):
    test=[r for r in rows if r["C_split"]=="C_test"];queries=[r for r in test if r["S_family"]=="indirect"];bank=[r for r in test if r["S_family"]!="indirect"];bank_x=x[[r["activation_row"] for r in bank]];ranks=[]
    for row in queries:
        order=np.argsort(-(bank_x@x[row["activation_row"]]));ranks.append(next(i+1 for i,j in enumerate(order) if bank[j]["fact_id"]==row["fact_id"]))
    ranks=np.array(ranks);return {"queries":len(queries),"bank_rows":len(bank),"R@1":float(np.mean(ranks<=1)),"R@5":float(np.mean(ranks<=5)),"MRR":float(np.mean(1/ranks))}


def s_retrieval(x,rows):
    test=[r for r in rows if r["C_split"]=="C_test"];rng=np.random.default_rng(SEED);queries=[test[i] for i in rng.choice(len(test),size=min(500,len(test)),replace=False)];bank_x=torch.from_numpy(x[[r["activation_row"] for r in test]]).to("cuda");scores=[]
    for row in queries:
        similarity=bank_x@torch.from_numpy(x[row["activation_row"]]).to("cuda");similarity[torch.tensor([i for i,r in enumerate(test) if r["fact_id"]==row["fact_id"]],device="cuda")]=-float("inf");best=int(torch.argmax(similarity));scores.append(test[best]["S_family"]==row["S_family"] and test[best]["S_variant"]==row["S_variant"])
    return {"queries":len(queries),"target":"same S family and variant, different fact","R@1":float(np.mean(scores))}


def main():
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    for i,row in enumerate(rows):row["activation_row"]=i
    c=np.load(CODES);s=np.load(SCODES);test=np.array([r["C_split"]=="C_test" for r in rows])
    report={"evaluation":"held-out C_test subjects; S probe trained on C_train", "z_C":{"C_domain_probe":probe(c,rows,"C_domain"),"C_relation_probe":probe(c,rows,"C_relation"),"S_family_probe":probe(c,rows,"S_family"),"effective_rank":rank(c[test]),"same_C_different_S_retrieval":c_retrieval(c,rows)},"z_S":{"C_domain_probe":probe(s,rows,"C_domain"),"C_relation_probe":probe(s,rows,"C_relation"),"S_family_probe":probe(s,rows,"S_family"),"effective_rank":rank(s[test]),"same_S_different_C_retrieval":s_retrieval(s,rows)},"new_training":"none","SAE_trained":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
