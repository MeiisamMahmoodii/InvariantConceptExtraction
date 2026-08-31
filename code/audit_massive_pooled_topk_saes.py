"""Read-only pooled SAE feature audit."""
import csv,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from numpy._core.multiarray import _reconstruct
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data'/'massive_partition_artifacts';S=ROOT/'data'/'massive_sae_artifacts';C=ROOT/'checkpoint';O=ROOT/'Report'/'massive_pooled_sae_threeway_audit.json';E=ROOT/'Report'/'massive_pooled_sae_top_examples.csv';D='cuda';SEED=20260827
class SAE(nn.Module):
 def __init__(self,w):super().__init__();self.e=nn.Linear(w,w*4);self.d=nn.Linear(w*4,w,bias=False);self.b=nn.Parameter(torch.zeros(w))
 def forward(self,x):a=F.relu(self.e(x));v,i=torch.topk(a,64,1);z=torch.zeros_like(a).scatter(1,i,v);return z,self.d(z)+self.b
class P(nn.Module):
 def __init__(self):super().__init__();self.c,self.s=nn.Linear(2304,128),nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def act(m,x,mean,std):
 out=[]
 with torch.no_grad():
  for i in range(0,len(x),512):out.append(m(torch.from_numpy((np.asarray(x[i:i+512],np.float32)-mean)/std).to(D))[0].cpu().numpy())
 return np.concatenate(out)
def main():
 rng=np.random.default_rng(SEED);tr,te=pd.read_csv(A/'train_metadata.csv'),pd.read_csv(A/'test_metadata.csv');raw,rawt=np.load(A/'raw_train_layer8.npy',mmap_mode='r'),np.load(A/'raw_test_layer8.npy',mmap_mode='r');zc,zct=np.load(S/'z_C_train.npy',mmap_mode='r'),np.load(A/'z_C_test.npy',mmap_mode='r');p=P().to(D);p.load_state_dict(torch.load(C/'massive_partition_layer8.pt',map_location=D,weights_only=True)['state_dict']);p.eval();zs=np.load(S/'z_S_train.npy',mmap_mode='r');zst=[]
 with torch.no_grad():
  for i in range(0,len(rawt),4096):zst.append(p(torch.from_numpy(np.asarray(rawt[i:i+4096])).to(D))[1].cpu().numpy())
 zst=np.concatenate(zst);idx=rng.choice(len(tr),20000,False);texts=pd.read_parquet(ROOT/'data'/'massive_all_train.parquet');lookup={(str(r.id),r.locale):r.utt for _,r in texts.iterrows()};report={};examples=[]
 for name,x,xt,file in [('H',raw,rawt,'massive_topk_raw_k64.pt'),('zC',zc,zct,'massive_topk_z_C_k64.pt'),('zS',zs,zst,'massive_topk_z_S_k64.pt')]:
  torch.serialization.add_safe_globals([_reconstruct,np.ndarray,np.dtype,np.dtypes.Float32DType]);q=torch.load(C/file,map_location=D,weights_only=True);m=SAE(x.shape[1]).to(D);m.load_state_dict(q['state_dict']);m.eval();a=act(m,x[idx],np.asarray(q['mean']),np.asarray(q['std']));b=act(m,xt,np.asarray(q['mean']),np.asarray(q['std']));mu=a.mean(0);sd=a.std(0).clip(1e-6);im=np.stack([a[tr.intent.to_numpy()[idx]==v].mean(0) for v in np.unique(tr.intent)]).max(0)-mu;lm=np.stack([a[tr.locale.to_numpy()[idx]==v].mean(0) for v in np.unique(tr.locale)]).max(0)-mu;im,lm=im/sd,lm/sd;active=mu>1e-6;intent=active&(im>1.1*lm);lang=active&(lm>1.1*im);mixed=active&~(intent|lang);ar={te.id.iloc[i]:i for i in np.where(te.locale=='ar-SA')[0]};zh={te.id.iloc[i]:i for i in np.where(te.locale=='zh-CN')[0]};pairs=[(ar[k],zh[k]) for k in ar.keys()&zh.keys()];st=np.nanmean([np.corrcoef(b[[i for i,j in pairs],f],b[[j for i,j in pairs],f])[0,1] for f in np.where(active)[0] if b[[i for i,j in pairs],f].std() and b[[j for i,j in pairs],f].std()]);report[name]={'mean_intent_selectivity_z':float(im[active].mean()),'mean_locale_selectivity_z':float(lm[active].mean()),'heldout_arabic_chinese_stability':float(st),'active_features':int(active.sum()),'intent_oriented_fraction':float(intent.sum()/active.sum()),'language_oriented_fraction':float(lang.sum()/active.sum()),'mixed_other_fraction':float(mixed.sum()/active.sum())}
  for label,f in [('intent',int(im.argmax())),('language',int(lm.argmax()))]:
   for r in np.argsort(-a[:,f])[:3]:examples.append({'representation':name,'orientation':label,'feature_id':f,'activation':float(a[r,f]),'intent_selectivity_z':float(im[f]),'locale_selectivity_z':float(lm[f]),'passes_orientation_criterion':bool(intent[f] if label=='intent' else lang[f]),'intent':int(tr.intent.iloc[idx[r]]),'locale':tr.locale.iloc[idx[r]],'utterance':lookup.get((str(tr.id.iloc[idx[r]]),tr.locale.iloc[idx[r]]),'')})
 O.write_text(json.dumps(report,indent=2)+'\n');pd.DataFrame(examples).to_csv(E,index=False);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
