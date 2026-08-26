"""Matched Top-k FLORES SAEs and held-out Arabic/Chinese feature stability audit."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"data"/"flores_partition_artifacts";OUT=ROOT/"Report"/"flores_sae_feature_stability.json";PERSIST=ROOT/"Report"/"flores_sae_persistence_rerun.json";CKPT=ROOT/"checkpoint";DEV="cuda" if torch.cuda.is_available() else "cpu";K=64;EPOCHS=30;SEED=20260825
class SAE(nn.Module):
 def __init__(self,w):super().__init__();self.e=nn.Linear(w,w*4);self.d=nn.Linear(w*4,w,bias=False)
 def forward(self,x):
  h=F.relu(self.e(x));v,i=torch.topk(h,K,1);z=torch.zeros_like(h).scatter(1,i,v);return z,self.d(z)
def main():
 torch.manual_seed(SEED);raw=np.load(ART/"raw_layer8.npy");zc=np.load(ART/"z_C.npy");zs=np.load(ART/"z_S.npy");ids=np.load(ART/"sentence_ids.npy");langs=np.load(ART/"languages.npy");splits=np.load(ART/"splits.npy");train=(splits=="train")&np.isin(langs,["eng_Latn","fra_Latn","spa_Latn","deu_Latn","rus_Cyrl","hin_Deva","swh_Latn","tur_Latn"]);test=(splits=="test")&np.isin(langs,["arb_Arab","zho_Hans"]);report={"settings":{"top_k":K,"expansion":4,"epochs":EPOCHS,"training_rows":int(train.sum())},"SAE_trained":True}
 for name,x in {"raw":raw,"z_C":zc,"z_S":zs}.items():
  mean=x[train].mean(0);std=x[train].std(0)+1e-6;xt=torch.from_numpy(((x-mean)/std).astype(np.float32)).to(DEV);m=SAE(x.shape[1]).to(DEV);o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
  for _ in range(EPOCHS):
   for b in np.array_split(np.random.default_rng(SEED).permutation(np.where(train)[0]),max(1,train.sum()//256)):
    _,y=m(xt[b]);l=F.mse_loss(y,xt[b]);o.zero_grad();l.backward();o.step()
  with torch.no_grad():a,_=m(xt);a=a.cpu().numpy()
  pairs=[]
  for sid in np.unique(ids[test]):
   p=np.where(test&(ids==sid)&(langs=="arb_Arab"))[0];q=np.where(test&(ids==sid)&(langs=="zho_Hans"))[0]
   if len(p) and len(q):pairs.append((p[0],q[0]))
  left=np.stack([a[i] for i,j in pairs]);right=np.stack([a[j] for i,j in pairs]);cons=1-np.abs(left-right).sum(0)/(left.sum(0)+right.sum(0)+1e-8)
  report[name]={"reconstruction_mse":float(F.mse_loss(m(xt[train])[1],xt[train]).item()),"mean_L0":K,"mean_feature_cross_language_consistency":float(cons.mean()),"fraction_features_consistency_gt_0_5":float((cons>.5).mean()),"pairs":len(pairs)};print(name,report[name])
  with torch.no_grad():dense=F.relu(m.e(xt));values,indices=torch.topk(dense,K,1)
  np.savez_compressed(ART/f"sae_{name}_all_sparse.npz",indices=indices.cpu().numpy().astype(np.int32),values=values.cpu().numpy().astype(np.float32),shape=np.array(a.shape,dtype=np.int64));torch.save({"state_dict":m.state_dict(),"mean":mean,"std":std,"k":K,"expansion":4,"epochs":EPOCHS,"seed":SEED},CKPT/f"flores_topk_{name}_k64.pt")
 old=json.loads(OUT.read_text()) if OUT.exists() else {};checks={name:{metric:abs(report[name][metric]-old.get(name,{}).get(metric,float('inf'))) for metric in ("reconstruction_mse","mean_L0","mean_feature_cross_language_consistency","fraction_features_consistency_gt_0_5")} for name in ("raw","z_C","z_S")};passed=all(delta<=1e-6 for result in checks.values() for delta in result.values());report["reproduction"]={"passed":passed,"absolute_differences":checks};OUT.write_text(json.dumps(report,indent=2)+"\n");PERSIST.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));
 if not passed:raise RuntimeError("FLORES SAE reproduction mismatch")
if __name__=="__main__":main()
