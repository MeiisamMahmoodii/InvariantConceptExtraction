"""Frozen concept-feature selection on seen S and evaluation on held-out indirect S."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data"/"sae_activations";SURFACE=ROOT/"data"/"three_domain_natural_rewrite"/"controlled_surface_dataset.csv";REPORT=ROOT/"Report"/"seen_to_indirect_feature_invariance_report.json"
TOP,MIN_ACTIVE=50,10;MODELS={"raw_gemma_layer8":"raw_gemma_layer8","z_C":"z_C"}
CONCEPTS={"capital_of":lambda r:r["C_relation"]=="capital_of","currency_of":lambda r:r["C_relation"]=="currency_of","atomic_number_of":lambda r:r["C_relation"]=="atomic_number_of","Europe":lambda r:r["C_value_id"]=="Q46","period_4":lambda r:r["C_value_id"]=="literal:period:4"}


def dense_sparse(stem):
    saved=np.load(DATA/f"{stem}_k64_c_test_sparse_activations.npz");x=np.zeros(tuple(saved["shape"]),dtype=np.float32);x[np.arange(x.shape[0])[:,None],saved["indices"]]=saved["values"];return x


def select_feature(x,seen,positive):
    best=None
    for feature in range(x.shape[1]):
        active=seen[x[seen,feature]>0]
        if len(active)<MIN_ACTIVE:continue
        top=active[np.argsort(-x[active,feature])[:TOP]];purity=float(np.mean(positive[top]));mean=float(x[top,feature].mean());candidate=(purity,mean,-feature,feature,len(top))
        if best is None or candidate>best:best=candidate
    if best is None:raise ValueError("no sufficiently active feature")
    return {"feature_id":best[3],"seen_top_activation_concept_purity":best[0],"seen_top_examples":best[4]}


def stability(x,feature,rows,seen,indirect):
    by_fact=defaultdict(list)
    for index in seen:by_fact[rows[index]["fact_id"]].append(index)
    indirect_by_fact={rows[index]["fact_id"]:index for index in indirect};left=[];right=[]
    for fact,indices in by_fact.items():
        if fact in indirect_by_fact:left.append(x[indices,feature].mean());right.append(x[indirect_by_fact[fact],feature])
    left=np.array(left);right=np.array(right);denom=(left+right).sum()
    return {"facts":len(left),"pearson_correlation":float(np.corrcoef(left,right)[0,1]) if left.std()>0 and right.std()>0 else None,"normalized_activation_consistency":float(1-np.abs(left-right).sum()/denom) if denom>0 else None,"seen_mean_activation":float(left.mean()),"indirect_mean_activation":float(right.mean())}


def audit(name,stem,rows,seen,indirect):
    x=dense_sparse(stem);result={}
    for concept,predicate in CONCEPTS.items():
        positive=np.array([predicate(row) for row in rows]);y=positive[indirect]
        if not positive[seen].any() or y.sum()==0 or (~y).sum()==0:
            result[concept]={"available":False,"seen_positive_examples":int(positive[seen].sum()),"indirect_positive_examples":int(y.sum()),"reason":"concept lacks both positive and negative support in the required seen-to-indirect C-test split"};continue
        selected=select_feature(x,seen,positive);f=selected["feature_id"];scores=x[indirect,f]
        selected.update({"available":True,"seen_positive_examples":int(positive[seen].sum()),"indirect_positive_examples":int(y.sum()),"indirect_negative_examples":int((~y).sum()),"unseen_indirect_AUC":float(roc_auc_score(y,scores)),"unseen_indirect_activation_margin":float(scores[y].mean()-scores[~y].mean()),"seen_to_indirect_stability":stability(x,f,rows,seen,indirect)})
        result[concept]=selected
    valid=[value for value in result.values() if value["available"]]
    return {"concepts":result,"mean_unseen_indirect_AUC":float(np.mean([value["unseen_indirect_AUC"] for value in valid])),"mean_normalized_stability":float(np.mean([value["seen_to_indirect_stability"]["normalized_activation_consistency"] for value in valid])),"available_concepts":len(valid)}


def main():
    with (DATA/"topk_k64_c_test_example_ids.csv").open(newline="",encoding="utf-8") as f:metadata=list(csv.DictReader(f))
    with SURFACE.open(newline="",encoding="utf-8") as f:surface={row["example_id"]:row for row in csv.DictReader(f)}
    rows=[surface[row["example_id"]] for row in metadata];assert [row["example_id"] for row in rows]==[row["example_id"] for row in metadata]
    seen=np.array([i for i,row in enumerate(rows) if row["S_family"]!="indirect"]);indirect=np.array([i for i,row in enumerate(rows) if row["S_family"]=="indirect"])
    report={"selection_split":"C-test rows from five seen S families only","evaluation_split":"C-test indirect S family only","top_active_seen_examples_for_selection":TOP,"minimum_active_seen_examples":MIN_ACTIVE,"concepts":list(CONCEPTS),"models":{name:audit(name,stem,rows,seen,indirect) for name,stem in MODELS.items()},"new_training":"none"}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
