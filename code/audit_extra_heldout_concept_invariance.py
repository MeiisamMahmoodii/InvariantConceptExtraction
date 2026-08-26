"""Evaluate seen-selected frozen SAE features on conversational and reordered S."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];SAE=ROOT/"data"/"sae_activations";BASE_SURFACE=ROOT/"data"/"three_domain_natural_rewrite"/"controlled_surface_dataset.csv";EXTRA=ROOT/"data"/"extra_heldout_surface_families";REPORT=ROOT/"Report"/"extra_heldout_concept_invariance_report.json";DETAILS=ROOT/"data"/"sae_features"/"extra_heldout_concept_invariance.csv"
TOP,MIN_ACTIVE,REL_MIN,VALUE_MIN_ALL,VALUE_MIN_EVAL,SUBJECT_MIN,BOOT,SEED=50,10,5,10,5,3,5000,20260825;MODELS={"raw_gemma_layer8":"raw_gemma_layer8","z_C":"z_C"}


def dense(path):
    saved=np.load(path);x=np.zeros(tuple(saved["shape"]),dtype=np.float32);x[np.arange(x.shape[0])[:,None],saved["indices"]]=saved["values"];return x


def top_seen(x,seen):
    out=[]
    for f in range(x.shape[1]):
        active=seen[x[seen,f]>0];out.append(None if len(active)<MIN_ACTIVE else active[np.argsort(-x[active,f])[:TOP]])
    return out


def select(x,tops,positive):
    best=None
    for f,indices in enumerate(tops):
        if indices is None:continue
        candidate=(float(positive[indices].mean()),float(x[indices,f].mean()),-f,f)
        if best is None or candidate>best:best=candidate
    return best[3],best[0]


def fact_support(rows,key):
    groups=defaultdict(set)
    for row in rows:groups[row[key]].add(row["fact_id"])
    return {key:len(value) for key,value in groups.items()}


def targets(rows):
    relation=fact_support(rows,"C_relation");value=fact_support(rows,"C_value_id");subject=fact_support(rows,"C_subject_id");labels={}
    for row in rows:labels.setdefault(("reusable_value",row["C_value_id"]),row["C_value_label"]);labels.setdefault(("subject",row["C_subject_id"]),row["C_subject_label"])
    result=[]
    result += [("relation",key,key,n) for key,n in sorted(relation.items()) if n>=REL_MIN]
    result += [("reusable_value",key,labels[("reusable_value",key)],n) for key,n in sorted(value.items()) if n>=VALUE_MIN_ALL and n>=VALUE_MIN_EVAL]
    result += [("subject",key,labels[("subject",key)],n) for key,n in sorted(subject.items()) if n>=SUBJECT_MIN]
    return result


def stability(x,f,seen_rows,seen_indices,eval_rows):
    groups=defaultdict(list)
    for i in seen_indices:groups[seen_rows[i]["fact_id"]].append(i)
    evaluate={row["fact_id"]:i for i,row in enumerate(eval_rows)};a=[];b=[]
    for fact,indices in groups.items():
        if fact in evaluate:a.append(x[indices,f].mean());b.append(eval_x_current[evaluate[fact],f])
    a=np.array(a);b=np.array(b);return float(1-np.abs(a-b).sum()/(a+b).sum()) if (a+b).sum()>0 else np.nan


def bootstrap(values):
    values=np.array(values);rng=np.random.default_rng(SEED);means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(BOOT)]);return {"mean_difference":float(values.mean()),"bootstrap_95_CI":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"concepts":len(values)}


def main():
    global eval_x_current
    with (SAE/"topk_k64_c_test_example_ids.csv").open(newline="",encoding="utf-8") as f:metadata=list(csv.DictReader(f))
    with BASE_SURFACE.open(newline="",encoding="utf-8") as f:base_by_id={r["example_id"]:r for r in csv.DictReader(f)}
    seen_rows=[base_by_id[r["example_id"]] for r in metadata];seen=np.array([i for i,r in enumerate(seen_rows) if r["S_family"]!="indirect"])
    with (EXTRA/"controlled_surface_dataset.csv").open(newline="",encoding="utf-8") as f:extra=list(csv.DictReader(f))
    base_x={name:dense(SAE/f"{stem}_k64_c_test_sparse_activations.npz") for name,stem in MODELS.items()};tops={name:top_seen(x,seen) for name,x in base_x.items()};all_details=[];summaries={}
    for family in ("conversational","reordered"):
        eval_rows=[r for r in extra if r["S_family"]==family];targets_for_family=targets(eval_rows);eval_xs={name:dense(SAE/f"extra_heldout_{stem}_k64_sparse_activations.npz")[[i for i,r in enumerate(extra) if r["S_family"]==family]] for name,stem in MODELS.items()};family_rows=[]
        for kind,concept_id,label,n in targets_for_family:
            key={"relation":"C_relation","reusable_value":"C_value_id","subject":"C_subject_id"}[kind];base_positive=np.array([r[key]==concept_id for r in seen_rows]);eval_positive=np.array([r[key]==concept_id for r in eval_rows]);entry={"heldout_family":family,"concept_type":kind,"concept_id":concept_id,"concept_label":label,"distinct_fact_support":n}
            for name in MODELS:
                f,purity=select(base_x[name],tops[name],base_positive);eval_x_current=eval_xs[name];scores=eval_x_current[:,f];entry[f"{name}_feature_id"]=f;entry[f"{name}_seen_top_purity"]=purity;entry[f"{name}_AUC"]=float(roc_auc_score(eval_positive,scores));entry[f"{name}_margin"]=float(scores[eval_positive].mean()-scores[~eval_positive].mean());entry[f"{name}_stability"]=stability(base_x[name],f,seen_rows,seen,eval_rows)
            family_rows.append(entry);all_details.append(entry)
        summaries[family]={}
        for group,group_rows in {"all":family_rows,**{kind:[r for r in family_rows if r["concept_type"]==kind] for kind in ("relation","reusable_value","subject")}}.items():
            if not group_rows:continue
            auc=[r["z_C_AUC"]-r["raw_gemma_layer8_AUC"] for r in group_rows];stab=[r["z_C_stability"]-r["raw_gemma_layer8_stability"] for r in group_rows]
            summaries[family][group]={"raw_mean_AUC":float(np.mean([r["raw_gemma_layer8_AUC"] for r in group_rows])),"z_C_mean_AUC":float(np.mean([r["z_C_AUC"] for r in group_rows])),"AUC_paired_difference":bootstrap(auc),"raw_mean_stability":float(np.mean([r["raw_gemma_layer8_stability"] for r in group_rows])),"z_C_mean_stability":float(np.mean([r["z_C_stability"] for r in group_rows])),"stability_paired_difference":bootstrap(stab)}
    with DETAILS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(all_details[0]));w.writeheader();w.writerows(all_details)
    report={"selection_split":"original C-test rows from five seen S families only","evaluation_families":["conversational","reordered"],"support_rules":{"relation_minimum_distinct_facts":REL_MIN,"reusable_value_minimum_distinct_facts":VALUE_MIN_ALL,"subject_minimum_distinct_facts":SUBJECT_MIN},"bootstrap_replicates":BOOT,"summaries":summaries,"per_concept_details":str(DETAILS.relative_to(ROOT)),"new_training":"none"};REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
