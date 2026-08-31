"""Recover C2 per-ID SAE stability from frozen checkpoints; no training or probe fitting."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from numpy._core.multiarray import _reconstruct
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data'/'massive_partition_artifacts';S=ROOT/'data'/'massive_sae_artifacts';I=ROOT/'data'/'massive_b2_infonce_artifacts';C=ROOT/'checkpoint';O=ROOT/'Report';SEED=20260828;NBOOT=10000;D='cuda' if torch.cuda.is_available() else 'cpu'
class SAE(nn.Module):
 def __init__(self):super().__init__();self.e=nn.Linear(128,512);self.d=nn.Linear(512,128,bias=False);self.b=nn.Parameter(torch.zeros(128))
 def forward(self,x):a=F.relu(self.e(x));v,i=torch.topk(a,64,1);z=torch.zeros_like(a).scatter(1,i,v);return z,self.d(z)+self.b
class Partition(nn.Module):
 def __init__(self):super().__init__();self.c=nn.Linear(2304,128);self.s=nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def load(path):
 torch.serialization.add_safe_globals([_reconstruct,np.ndarray,np.dtype,np.dtypes.Float32DType]);q=torch.load(path,map_location=D,weights_only=True);s=q['state_dict'];s={'e.weight':s['encoder.weight'],'e.bias':s['encoder.bias'],'d.weight':s['decoder.weight'],'b':s['bias']} if 'encoder.weight' in s else s;m=SAE().to(D);m.load_state_dict(s);m.eval();return m,np.asarray(q['mean']),np.asarray(q['std'])
def act(m,x,mean,std):
 o=[]
 with torch.no_grad():
  for i in range(0,len(x),512):o.append(m(torch.from_numpy((np.asarray(x[i:i+512],np.float32)-mean)/std).to(D))[0].cpu().numpy())
 return np.concatenate(o)
def per_id(train,test,meta,ckpt):
 m,mean,std=load(ckpt);ix=np.random.default_rng(20260827).choice(len(train),20000,False);a=act(m,train[ix],mean,std);active=a.mean(0)>1e-6;b=act(m,test,mean,std);ar={str(meta.id.iloc[i]):i for i in np.where(meta.locale.to_numpy()=='ar-SA')[0]};zh={str(meta.id.iloc[i]):i for i in np.where(meta.locale.to_numpy()=='zh-CN')[0]};ids=sorted(ar.keys()&zh.keys());x=b[[ar[i] for i in ids]][:,active].astype(np.float64);y=b[[zh[i] for i in ids]][:,active].astype(np.float64);x=(x-x.mean(0))/x.std(0).clip(1e-6);y=(y-y.mean(0))/y.std(0).clip(1e-6);return ids,(x*y).mean(1),float((x*y).mean(0).mean()),int(active.sum())
def uncontrolled_codes():
 q=torch.load(C/'massive_matching_coverage'/'partition_uncontrolled_k49_seed20260827.pt',map_location=D,weights_only=True);m=Partition().to(D);m.load_state_dict(q['state_dict']);m.eval();train=np.load(A/'raw_train_layer8.npy',mmap_mode='r');test=np.load(A/'raw_test_layer8.npy',mmap_mode='r');ix=np.random.default_rng(20260827).choice(len(train),20000,False)
 def encode(x):
  out=[]
  with torch.no_grad():
   for i in range(0,len(x),512):out.append(m(torch.from_numpy(np.array(x[i:i+512],dtype=np.float32,copy=True)).to(D))[0].cpu().numpy())
  return np.concatenate(out)
 return encode(train[ix]),encode(test)
def main():
 meta=pd.read_csv(A/'test_metadata.csv');mi=np.load(S/'z_C_train.npy',mmap_mode='r');mt=np.load(A/'z_C_test.npy',mmap_mode='r');ni=np.load(I/'z_C_train.npy',mmap_mode='r');nt=np.load(I/'z_C_test.npy',mmap_mode='r');ids,ms,ma,mn=per_id(mi,mt,meta,C/'massive_topk_z_C_k64.pt');ids2,ns,na,nn=per_id(ni,nt,meta,I/'infonce_topk_z_C_k64.pt');ut,uv=uncontrolled_codes();ids3,us,ua,un=per_id(ut,uv,meta,C/'massive_matching_coverage'/'sae_uncontrolled_k49_seed20260827.pt');assert ids==ids2==ids3;delta=ms-ns;delta_u=ms-us;rng=np.random.default_rng(SEED);draws=rng.integers(0,len(delta),(NBOOT,len(delta)));boot=delta[draws].mean(1);boot_u=delta_u[draws].mean(1);csv=O/'massive_c2_stability_per_id.csv';pd.DataFrame({'id':ids,'matched_stability_contribution':ms,'infonce_stability_contribution':ns,'uncontrolled_negative_stability_contribution':us,'delta_matched_minus_infonce':delta,'delta_matched_minus_uncontrolled':delta_u}).to_csv(csv,index=False);report={'scope':{'C2_stability_only':True,'frozen_SAE_inference_only':True,'no_training':True,'no_probe_fitting':True,'inference_device':D,'n_bootstrap':NBOOT,'bootstrap_seed':SEED},'matched_vs_naive_infonce_zC_stability':{'n_semantic_ids':len(ids),'active_features_matched':mn,'active_features_infonce':nn,'aggregate_matched':ma,'aggregate_infonce':na,'mean_difference':float(delta.mean()),'bootstrap_95_CI':[float(x) for x in np.quantile(boot,[.025,.975])],'fraction_ids_delta_gt_0':float((delta>0).mean()),'resampling_unit':'held-out MASSIVE semantic id','per_id_csv':str(csv.relative_to(ROOT))},'matched_vs_uncontrolled_negative_zC_stability':{'n_semantic_ids':len(ids),'active_features_matched':mn,'active_features_uncontrolled':un,'aggregate_matched':ma,'aggregate_uncontrolled':ua,'mean_difference':float(delta_u.mean()),'bootstrap_95_CI':[float(x) for x in np.quantile(boot_u,[.025,.975])],'fraction_ids_delta_gt_0':float((delta_u>0).mean()),'resampling_unit':'held-out MASSIVE semantic id','per_id_csv':str(csv.relative_to(ROOT))},'C_swap_specificity':{'status':'BLOCKED: exact frozen intent probe checkpoint/per-case outcomes were not persisted; no probe was retrained.'},'S_swap_specificity':{'status':'BLOCKED: exact frozen locale probe checkpoint/per-case outcomes were not persisted; no probe was retrained.'}}
 (O/'massive_c2_statistical_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
