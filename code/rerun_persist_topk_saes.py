"""Deterministic persistence rerun of the three existing k=64 Top-k SAEs."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];ACT=ROOT/"data"/"activations_three_domain_natural_rewrite";OUT=ROOT/"checkpoint";DATA=ROOT/"data"/"sae_activations";REPORT=ROOT/"Report"/"topk_sae_persistence_rerun_report.json"
SEED,EPOCHS,BATCH,K,EXPANSION,TOP,MIN_ACTIVE,TOL=20260825,30,256,64,4,50,10,5e-4;FAMILIES={"declarative","question","paraphrase","formal","structured"};DEVICE="cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self,width):
        super().__init__();self.encoder=nn.Linear(width,width*EXPANSION);self.decoder=nn.Linear(width*EXPANSION,width,bias=False);self.bias=nn.Parameter(torch.zeros(width))
    def forward(self,x):
        dense=F.relu(self.encoder(x));values,indices=torch.topk(dense,K,dim=-1);z=torch.zeros_like(dense).scatter(1,indices,values);return z,self.decoder(z)+self.bias
    def normalize(self):
        with torch.no_grad():self.decoder.weight.div_(self.decoder.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))


def select(rows):
    rng=np.random.default_rng(SEED);candidates=[r for r in rows if r["C_split"]=="C_train" and r["S_family"] in FAMILIES];groups=defaultdict(set)
    for r in candidates:groups[r["C_domain"]].add(r["C_subject_id"])
    n=min(map(len,groups.values()));chosen=set()
    for domain,subjects in sorted(groups.items()):chosen.update((domain,s) for s in rng.choice(sorted(subjects),n,replace=False))
    return [r for r in candidates if (r["C_domain"],r["C_subject_id"]) in chosen],{d:n for d in sorted(groups)}


def purity(values):return max(Counter(values).values())/len(values)


def metrics_and_sparse(model,matrix,mean,std,rows,indices,name):
    x=(matrix[indices].astype(np.float32)-mean)/std;zs=[];total=0.
    with torch.inference_mode():
        for start in range(0,len(x),BATCH):
            batch=torch.from_numpy(x[start:start+BATCH]).to(DEVICE);z,rec=model(batch);zs.append(z.cpu().numpy());total+=F.mse_loss(rec,batch,reduction="sum").item()
    z=np.concatenate(zs);held=[rows[i] for i in indices];details=[]
    for feature in range(z.shape[1]):
        active=np.flatnonzero(z[:,feature]>0)
        if len(active)<MIN_ACTIVE:continue
        top=active[np.argsort(-z[active,feature])[:TOP]];rel=purity([held[i]["C_relation"] for i in top]);dom=purity([held[i]["C_domain"] for i in top]);fam=purity([held[i]["S_family"] for i in top]);details.append((rel,dom,fam))
    rel=np.array([v[0] for v in details]);dom=np.array([v[1] for v in details]);fam=np.array([v[2] for v in details]);c=int(np.sum((rel>=.8)&(fam<=.5)));s=int(np.sum((fam>=.8)&(rel<=.5)))
    values,indices_sparse=np.sort(z,axis=1)[:,-K:],np.argsort(z,axis=1)[:,-K:]
    np.savez_compressed(DATA/f"{name}_k{K}_c_test_sparse_activations.npz",indices=indices_sparse.astype(np.int32),values=values.astype(np.float32),shape=np.array(z.shape,dtype=np.int64))
    return {"input_width":matrix.shape[1],"feature_width":z.shape[1],"standardized_reconstruction_mse":float(total/x.size),"mean_L0":float((z>0).sum(1).mean()),"effective_active_dictionary_size":len(details),"effective_active_dictionary_fraction":len(details)/z.shape[1],"mean_top50_C_relation_purity":float(rel.mean()),"mean_top50_C_domain_purity":float(dom.mean()),"mean_top50_S_family_purity":float(fam.mean()),"C_relation_selective_features":c,"C_relation_selective_fraction_of_dictionary":c/z.shape[1],"S_family_selective_features":s,"S_family_selective_fraction_of_dictionary":s/z.shape[1],"sparse_activations":str((DATA/f"{name}_k{K}_c_test_sparse_activations.npz").relative_to(ROOT))}


def run(name,matrix,train_indices,test_indices,rows):
    torch.manual_seed(SEED);raw=matrix[train_indices].astype(np.float32);mean,std=raw.mean(0),raw.std(0).clip(1e-6);x=torch.from_numpy((raw-mean)/std).to(DEVICE);model=SAE(x.shape[1]).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for epoch in range(EPOCHS):
        order=torch.randperm(len(x),device=DEVICE);total=0.;steps=0;model.train()
        for batch in order.split(BATCH):
            _,rec=model(x[batch]);loss=F.mse_loss(rec,x[batch]);opt.zero_grad();loss.backward();opt.step();model.normalize();total+=loss.item();steps+=1
        print(f"{name} k={K} epoch={epoch+1}/{EPOCHS} mse={total/steps:.5f}")
    result=metrics_and_sparse(model,matrix,mean,std,rows,test_indices,name);torch.save({"state_dict":model.state_dict(),"input_mean":mean,"input_std":std,"config":{"input_width":x.shape[1],"feature_width":x.shape[1]*EXPANSION,"k":K,"expansion_factor":EXPANSION,"epochs":EPOCHS,"batch_size":BATCH,"seed":SEED,"preprocessing":"C_train per-dimension standardization"}},OUT/f"topk_{name}_k{K}.pt");return result


def compare(actual,expected):
    keys=("standardized_reconstruction_mse","mean_L0","effective_active_dictionary_size","mean_top50_C_relation_purity","mean_top50_C_domain_purity","mean_top50_S_family_purity","C_relation_selective_features","S_family_selective_features")
    differences={}
    for key in keys:
        expected_key="active_feature_count" if key=="effective_active_dictionary_size" and "active_feature_count" in expected else key
        expected_value=expected[expected_key];delta=abs(actual[key]-expected_value);differences[key]={"expected":expected_value,"actual":actual[key],"absolute_difference":delta,"within_tolerance":delta<=TOL}
    return differences,all(item["within_tolerance"] for item in differences.values())


def main():
    DATA.mkdir(exist_ok=True)
    with (ACT/"gemma2_2b_layer_sweep_metadata.csv").open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    train_rows,subjects=select(rows);train_indices=np.array([int(r["activation_row"]) for r in train_rows]);test_indices=np.array([i for i,r in enumerate(rows) if r["C_split"]=="C_test"])
    with (DATA/"topk_k64_c_test_example_ids.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=("activation_row","example_id","fact_id","C_domain","C_relation","C_subject_id","C_value_id","S_family","S_variant","C_split","S_split"));w.writeheader();w.writerows({key:rows[i].get(key,"") for key in w.fieldnames} | {"activation_row":i} for i in test_indices)
    raw=np.load(ACT/"gemma2_2b_layer8_mean"/"activations.npy");zc=np.load(OUT/"cs_partition_c_all.npy");zs=np.load(OUT/"cs_partition_s_all.npy");old_topk=json.loads((ROOT/"Report"/"topk_sae_sweep_report.json").read_text(encoding="utf-8"));old_partition=json.loads((ROOT/"Report"/"partition_topk_sae_report.json").read_text(encoding="utf-8"))
    results={"raw_gemma_layer8":run("raw_gemma_layer8",raw,train_indices,test_indices,rows),"nonadversarial_z_C":run("z_C",zc,train_indices,test_indices,rows),"nonadversarial_z_S":run("z_S",zs,train_indices,test_indices,rows)}
    expected={"raw_gemma_layer8":old_topk["raw_gemma_layer8"]["by_k"][str(K)],"nonadversarial_z_C":old_partition["nonadversarial_z_C"],"nonadversarial_z_S":old_partition["nonadversarial_z_S"]};checks={name:dict(zip(("differences","passed"),compare(results[name],expected[name]))) for name in results}
    report={"device":DEVICE,"purpose":"deterministic persistence rerun only","k":K,"expansion_factor":EXPANSION,"epochs":EPOCHS,"seed":SEED,"training_rows":len(train_rows),"training_facts":len({r['fact_id'] for r in train_rows}),"training_subjects_per_domain":subjects,"C_test_example_ids":str((DATA/"topk_k64_c_test_example_ids.csv").relative_to(ROOT)),"results":results,"reproduction_checks":checks,"all_reproduced":all(check["passed"] for check in checks.values()),"partition_retrained":False,"new_settings":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps({"all_reproduced":report["all_reproduced"],"reproduction_checks":checks,"C_test_example_ids":report["C_test_example_ids"]},indent=2))
    if not report["all_reproduced"]:raise SystemExit("aggregate reproduction check failed")


if __name__=="__main__":main()
