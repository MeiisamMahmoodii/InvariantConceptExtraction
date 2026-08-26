"""Direct held-out comparison of the prior C bottleneck and new z_C block."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
ACT=ROOT/"data"/"activations_three_domain_natural_rewrite"
OLD=ROOT/"checkpoint"/"fact_bottleneck_three_domain_natural_rewrite_balanced_layer8_encoded_all.npy"
NEW=ROOT/"checkpoint"/"cs_partition_c_all.npy"
REPORT=ROOT/"Report"/"c_bottleneck_vs_partition_comparison.json";SEED=20260825


def probe(x,rows,label):
    train=np.array([r["C_split"]=="C_train" for r in rows]);test=~train;y=np.array([r[label] for r in rows]);scale=StandardScaler().fit(x[train]);model=LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(scale.transform(x[train]),y[train]);return float(np.mean(model.predict(scale.transform(x[test]))==y[test]))


def family_probe(x,rows):
    train=np.array([r["C_split"]=="C_train" for r in rows]);test=np.array([not flag for flag in train]);y=np.array([r["S_family"] for r in rows]);scale=StandardScaler().fit(x[train]);model=LogisticRegression(C=1.,max_iter=1000,random_state=SEED).fit(scale.transform(x[train]),y[train]);pred=model.predict(scale.transform(x[test]));actual=y[test];held=np.array([r["S_family"]=="indirect" for r in np.array(rows,dtype=object)[test]])
    return {"all_six_families":float(np.mean(pred==actual)),"seen_families":float(np.mean(pred[~held]==actual[~held])),"held_out_indirect":float(np.mean(pred[held]==actual[held]))}


def rank(x):
    x=x-x.mean(0);v=np.linalg.svd(x,compute_uv=False)**2;p=v/v.sum();return {"participation_ratio":float(v.sum()**2/(v**2).sum()),"entropy_effective_rank":float(np.exp(-(p*np.log(p+1e-30)).sum()))}


def retrieval(x,rows):
    test=[r for r in rows if r["C_split"]=="C_test"];queries=[r for r in test if r["S_family"]=="indirect"];bank=[r for r in test if r["S_family"]!="indirect"];bx=x[[r["activation_row"] for r in bank]];ranks=[]
    for row in queries:
        order=np.argsort(-(bx@x[row["activation_row"] ]));ranks.append(next(i+1 for i,j in enumerate(order) if bank[j]["fact_id"]==row["fact_id"]))
    ranks=np.array(ranks);return {"R@1":float(np.mean(ranks<=1)),"R@5":float(np.mean(ranks<=5)),"MRR":float(np.mean(1/ranks))}


def metrics(x,rows):
    test=np.array([r["C_split"]=="C_test" for r in rows]);return {"dimensions":x.shape[1],"C_relation_accuracy":probe(x,rows,"C_relation"),"S_family_accuracy":family_probe(x,rows),"same_C_different_S_retrieval":retrieval(x,rows),"effective_rank":rank(x[test])}


def main():
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):r["activation_row"]=i
    report={"evaluation":"same natural-rewrite data, C_test subjects, six-family probe, and indirect-to-seen retrieval", "C_bottleneck_256":metrics(np.load(OLD),rows),"two_route_z_C_128":metrics(np.load(NEW),rows),"new_training":"none"};REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
