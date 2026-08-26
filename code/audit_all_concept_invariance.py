"""All-supported-concept frozen seen-to-indirect feature invariance audit."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data"/"sae_activations";SURFACE=ROOT/"data"/"three_domain_natural_rewrite"/"controlled_surface_dataset.csv";REPORT=ROOT/"Report"/"all_concept_seen_to_indirect_invariance_report.json";DETAILS=ROOT/"data"/"sae_features"/"all_concept_seen_to_indirect_invariance.csv"
TOP,MIN_ACTIVE,REL_MIN,VALUE_MIN_ALL,VALUE_MIN_IND,SUBJECT_MIN,BOOT,SEED=50,10,5,10,5,3,5000,20260825
MODELS={"raw_gemma_layer8":"raw_gemma_layer8","z_C":"z_C"}


def dense_sparse(stem):
    saved=np.load(DATA/f"{stem}_k64_c_test_sparse_activations.npz");x=np.zeros(tuple(saved["shape"]),dtype=np.float32);x[np.arange(x.shape[0])[:,None],saved["indices"]]=saved["values"];return x


def top_seen(x,seen):
    result=[]
    for feature in range(x.shape[1]):
        active=seen[x[seen,feature]>0]
        if len(active)<MIN_ACTIVE:result.append(None);continue
        result.append(active[np.argsort(-x[active,feature])[:TOP]])
    return result


def select_feature(x,tops,positive):
    best=None
    for feature,indices in enumerate(tops):
        if indices is None:continue
        purity=float(positive[indices].mean());mean=float(x[indices,feature].mean());candidate=(purity,mean,-feature,feature,len(indices))
        if best is None or candidate>best:best=candidate
    return {"feature_id":best[3],"seen_top_purity":best[0],"seen_top_examples":best[4]}


def stability(x,feature,rows,seen,indirect_by_fact):
    groups=defaultdict(list)
    for index in seen:groups[rows[index]["fact_id"]].append(index)
    a=[];b=[]
    for fact,indices in groups.items():
        if fact in indirect_by_fact:a.append(x[indices,feature].mean());b.append(x[indirect_by_fact[fact],feature])
    a=np.array(a);b=np.array(b);denom=(a+b).sum();return float(1-np.abs(a-b).sum()/denom) if denom>0 else np.nan


def support(rows,indices,key):return {label:len({rows[i]["fact_id"] for i in indices if rows[i][key]==label}) for label in {rows[i][key] for i in indices}}


def concepts(rows,seen,indirect):
    all_relation=support(rows,np.arange(len(rows)),"C_relation");ind_relation=support(rows,indirect,"C_relation");all_value=support(rows,np.arange(len(rows)),"C_value_id");ind_value=support(rows,indirect,"C_value_id");ind_subject=support(rows,indirect,"C_subject_id")
    labels={}
    for row in rows:labels.setdefault(("value",row["C_value_id"]),row["C_value_label"]);labels.setdefault(("subject",row["C_subject_id"]),row["C_subject_label"])
    out=[]
    for label,n in sorted(ind_relation.items()):
        if n>=REL_MIN:out.append(("relation",label,label,n))
    for label,n in sorted(ind_value.items()):
        if n>=VALUE_MIN_IND and all_value.get(label,0)>=VALUE_MIN_ALL:out.append(("reusable_value",label,labels[("value",label)],n))
    for label,n in sorted(ind_subject.items()):
        if n>=SUBJECT_MIN:out.append(("subject",label,labels[("subject",label)],n))
    return out


def bootstrap(values):
    values=np.array(values);rng=np.random.default_rng(SEED);means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(BOOT)]);return {"mean_difference":float(values.mean()),"bootstrap_95_CI":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"concepts":len(values)}


def main():
    with (DATA/"topk_k64_c_test_example_ids.csv").open(newline="",encoding="utf-8") as f:metadata=list(csv.DictReader(f))
    with SURFACE.open(newline="",encoding="utf-8") as f:surface={r["example_id"]:r for r in csv.DictReader(f)}
    rows=[surface[row["example_id"]] for row in metadata];seen=np.array([i for i,r in enumerate(rows) if r["S_family"]!="indirect"]);indirect=np.array([i for i,r in enumerate(rows) if r["S_family"]=="indirect"]);indirect_by_fact={rows[i]["fact_id"]:i for i in indirect};targets=concepts(rows,seen,indirect)
    matrices={name:dense_sparse(stem) for name,stem in MODELS.items()};tops={name:top_seen(x,seen) for name,x in matrices.items()};details=[]
    for kind,concept_id,label,ind_support in targets:
        key={"relation":"C_relation","reusable_value":"C_value_id","subject":"C_subject_id"}[kind];positive=np.array([r[key]==concept_id for r in rows]);entry={"concept_type":kind,"concept_id":concept_id,"concept_label":label,"indirect_distinct_fact_support":ind_support}
        for name,x in matrices.items():
            selected=select_feature(x,tops[name],positive);f=selected["feature_id"];y=positive[indirect];scores=x[indirect,f];entry[f"{name}_feature_id"]=f;entry[f"{name}_seen_top_purity"]=selected["seen_top_purity"];entry[f"{name}_unseen_AUC"]=float(roc_auc_score(y,scores));entry[f"{name}_activation_margin"]=float(scores[y].mean()-scores[~y].mean());entry[f"{name}_stability"]=stability(x,f,rows,seen,indirect_by_fact)
        details.append(entry)
    with DETAILS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    summaries={}
    for group,group_rows in {"all":details,**{kind:[r for r in details if r["concept_type"]==kind] for kind in ("relation","reusable_value","subject")}}.items():
        if not group_rows:continue
        auc_diff=[r["z_C_unseen_AUC"]-r["raw_gemma_layer8_unseen_AUC"] for r in group_rows];stab_diff=[r["z_C_stability"]-r["raw_gemma_layer8_stability"] for r in group_rows]
        summaries[group]={"raw_mean_AUC":float(np.mean([r["raw_gemma_layer8_unseen_AUC"] for r in group_rows])),"z_C_mean_AUC":float(np.mean([r["z_C_unseen_AUC"] for r in group_rows])),"AUC_paired_difference":bootstrap(auc_diff),"raw_mean_stability":float(np.mean([r["raw_gemma_layer8_stability"] for r in group_rows])),"z_C_mean_stability":float(np.mean([r["z_C_stability"] for r in group_rows])),"stability_paired_difference":bootstrap(stab_diff)}
    report={"selection_split":"C-test rows from five seen S families only","evaluation_split":"C-test indirect family only","support_rules":{"relation_minimum_indirect_distinct_facts":REL_MIN,"reusable_value_minimum_all_C_test_distinct_facts":VALUE_MIN_ALL,"reusable_value_minimum_indirect_distinct_facts":VALUE_MIN_IND,"subject_minimum_indirect_distinct_facts":SUBJECT_MIN},"top_active_seen_examples":TOP,"minimum_active_seen_examples":MIN_ACTIVE,"bootstrap_replicates":BOOT,"concept_count":len(details),"summary":summaries,"per_concept_details":str(DETAILS.relative_to(ROOT)),"new_training":"none"}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
