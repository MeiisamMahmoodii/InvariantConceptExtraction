"""Frozen multi-factor audit of persisted Top-k SAE C-test activations."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"data"/"sae_activations";REPORT=ROOT/"Report"/"sae_multifactor_feature_audit.json";FEATURES=ROOT/"data"/"sae_features"
TOP,MIN_ACTIVE,SUBJECT_MIN_FACTS,VALUE_MIN_FACTS,TOP_MIN_FACTS=50,10,3,10,3
MODELS={"raw_gemma_layer8":"raw_gemma_layer8","z_C":"z_C","z_S":"z_S"}
FACTORS=("C_domain","C_relation","C_subject_id","C_value_id","S_family")


def stats(values):return {"n":len(values),"mean":float(np.mean(values)),"max":float(np.max(values)),"p05":float(np.quantile(values,.05)),"median":float(np.median(values)),"p95":float(np.quantile(values,.95))}


def sparse_dense(stem):
    saved=np.load(DATA/f"{stem}_k64_c_test_sparse_activations.npz");x=np.zeros(tuple(saved["shape"]),dtype=np.float32);x[np.arange(x.shape[0])[:,None],saved["indices"]]=saved["values"];return x


def label_stats(rows,key):
    facts=defaultdict(set);examples=Counter()
    for row in rows:facts[row[key]].add(row["fact_id"]);examples[row[key]]+=1
    return {label:len(ids) for label,ids in facts.items()},{label:count/len(rows) for label,count in examples.items()}


def winner(rows,indices,key,eligible,baseline):
    labels=[rows[i][key] for i in indices];counts=Counter(labels);label,count=counts.most_common(1)[0];purity=count/len(indices);fact_support=len({rows[i]["fact_id"] for i in indices if rows[i][key]==label})
    valid=label in eligible and fact_support>=TOP_MIN_FACTS
    return {"label":label,"purity":purity if valid else 0.,"top_distinct_facts":fact_support,"evidence":purity-baseline.get(label,0.) if valid else -1.}


def audit(name,stem,rows,eligibility,baselines):
    x=sparse_dense(stem);details=[]
    keys={"domain":"C_domain","relation":"C_relation","subject":"C_subject_id","reusable_value":"C_value_id","S_family":"S_family"}
    for feature in range(x.shape[1]):
        active=np.flatnonzero(x[:,feature]>0)
        if len(active)<MIN_ACTIVE:continue
        top=active[np.argsort(-x[active,feature])[:TOP]];out={"feature_id":feature,"active_examples":len(active),"top_examples":len(top),"top_example_ids":";".join(rows[i]["example_id"] for i in top)}
        scores={}
        for factor,key in keys.items():
            result=winner(rows,top,key,eligibility[factor],baselines[key]);scores[factor]=result;out[f"{factor}_label"]=result["label"];out[f"{factor}_purity"]=result["purity"];out[f"{factor}_top_distinct_facts"]=result["top_distinct_facts"]
        out["primary_factor"]=max(scores,key=lambda factor:scores[factor]["evidence"]);details.append(out)
    path=FEATURES/f"{stem}_k64_multifactor_feature_audit.csv"
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    primary=Counter(row["primary_factor"] for row in details);purities={factor:[row[f"{factor}_purity"] for row in details] for factor in keys}
    return {"feature_width":x.shape[1],"evaluated_features":len(details),"excluded_low_activity_features":x.shape[1]-len(details),"primary_factor_counts":dict(primary),"primary_factor_fractions":{factor:primary[factor]/len(details) for factor in (*keys,"unclassified")},"purity_distributions":{factor:stats(values) for factor,values in purities.items()},"feature_details":str(path.relative_to(ROOT))}


def main():
    with (DATA/"topk_k64_c_test_example_ids.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    support={key:label_stats(rows,key)[0] for key in FACTORS};baselines={key:label_stats(rows,key)[1] for key in FACTORS}
    eligibility={"domain":set(support["C_domain"]),"relation":set(support["C_relation"]),"subject":{label for label,n in support["C_subject_id"].items() if n>=SUBJECT_MIN_FACTS},"reusable_value":{label for label,n in support["C_value_id"].items() if n>=VALUE_MIN_FACTS},"S_family":set(support["S_family"])}
    report={"evaluation":"frozen persisted C-test sparse activations only","top_active_examples_per_feature":TOP,"minimum_feature_active_examples":MIN_ACTIVE,"support_rules":{"subject_minimum_distinct_C_test_facts":SUBJECT_MIN_FACTS,"reusable_value_minimum_distinct_C_test_facts":VALUE_MIN_FACTS,"minimum_distinct_top_facts_for_subject_or_value_assignment":TOP_MIN_FACTS},"primary_assignment":"eligible factor with largest purity minus its C-test example-frequency baseline","models":{name:audit(name,stem,rows,eligibility,baselines) for name,stem in MODELS.items()},"new_training":"none"}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
