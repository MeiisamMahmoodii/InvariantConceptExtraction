"""Frozen Top-k SAE activation consistency on controlled C-test pair types."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data"/"sae_activations";REPORT=ROOT/"Report"/"sae_activation_consistency_report.json";FEATURES=ROOT/"data"/"sae_features"
SEED,N,BATCH=20260825,10_000,512
MODELS={"raw_gemma_layer8":"raw_gemma_layer8","z_C":"z_C","z_S":"z_S"}


def stats(x):return {"n":len(x),"mean":float(x.mean()),"std":float(x.std()),"p05":float(np.quantile(x,.05)),"median":float(np.median(x)),"p95":float(np.quantile(x,.95))}


def pairs(rows):
    by_fact,by_template=defaultdict(list),defaultdict(list)
    for index,row in enumerate(rows):by_fact[row["fact_id"]].append(index);by_template[(row["S_family"],row["S_variant"])].append(index)
    rng=random.Random(SEED);c_pairs=[];s_pairs=[]
    while len(c_pairs)<N:
        group=by_fact[rng.choice(list(by_fact))];a,b=rng.sample(group,2)
        if rows[a]["S_family"]!=rows[b]["S_family"]:c_pairs.append((a,b))
    while len(s_pairs)<N:
        group=by_template[rng.choice(list(by_template))];a,b=rng.sample(group,2)
        if rows[a]["fact_id"]!=rows[b]["fact_id"]:s_pairs.append((a,b))
    return np.array(c_pairs),np.array(s_pairs)


def dense_sparse(path):
    saved=np.load(path);shape=tuple(saved["shape"]);x=np.zeros(shape,dtype=np.float32);x[np.arange(shape[0])[:,None],saved["indices"]]=saved["values"];return x


def consistency(x,pair_indices):
    difference=np.zeros(x.shape[1],dtype=np.float64);mass=np.zeros(x.shape[1],dtype=np.float64)
    for start in range(0,len(pair_indices),BATCH):
        p=pair_indices[start:start+BATCH];a=x[p[:,0]];b=x[p[:,1]];difference+=np.abs(a-b).sum(0);mass+=(a+b).sum(0)
    active=mass>0;result=np.full(x.shape[1],np.nan);result[active]=1-difference[active]/mass[active];return result,active


def audit(name,stem,rows,c_pairs,s_pairs):
    x=dense_sparse(DATA/f"{stem}_k64_c_test_sparse_activations.npz");c,active_c=consistency(x,c_pairs);s,active_s=consistency(x,s_pairs);active=active_c&active_s;delta=c[active]-s[active]
    details=[]
    for index in np.flatnonzero(active):details.append({"feature_id":int(index),"Cons_C_fixed":float(c[index]),"Cons_S_fixed":float(s[index]),"delta_C_minus_S":float(c[index]-s[index])})
    path=FEATURES/f"{stem}_k64_activation_consistency.csv"
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    return {"feature_width":x.shape[1],"evaluated_features":int(active.sum()),"inactive_in_either_pair_set":int((~active).sum()),"Cons_C_fixed":stats(c[active]),"Cons_S_fixed":stats(s[active]),"delta_C_minus_S":stats(delta),"C_invariant_oriented_features":int(np.sum(delta>0)),"C_invariant_oriented_fraction":float(np.mean(delta>0)),"S_invariant_oriented_features":int(np.sum(delta<0)),"S_invariant_oriented_fraction":float(np.mean(delta<0)),"ties":int(np.sum(delta==0)),"feature_details":str(path.relative_to(ROOT))}


def main():
    with (DATA/"topk_k64_c_test_example_ids.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    c_pairs,s_pairs=pairs(rows);report={"evaluation":"frozen persisted C-test sparse activations only","consistency_definition":"1 - sum_pair_abs_difference / sum_pair_activation_mass; cross-condition comparison excludes features with zero total activation in either pair set","same_C_different_S_pairs":len(c_pairs),"same_S_different_C_pairs":len(s_pairs),"pair_seed":SEED,"models":{name:audit(name,stem,rows,c_pairs,s_pairs) for name,stem in MODELS.items()},"new_training":"none"}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
