"""B1 only: pooled MASSIVE partition baselines; no SAE, decoder, Scope, or swaps."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data'/'massive_partition_artifacts';OUT=ROOT/'Report'/'massive_b1_partition_baselines.json';SEED=20260827;DEV='cuda';BATCH=256;EPOCHS=30;TEMP=.07
class P(nn.Module):
 def __init__(self):super().__init__();self.c,self.s=nn.Linear(2304,128),nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v*v).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def retrieval(x,ids,langs):
 a=np.where(langs=='ar-SA')[0];b=np.where(langs=='zh-CN')[0];u=F.normalize(torch.from_numpy(x[a]),dim=1).numpy();v=F.normalize(torch.from_numpy(x[b]),dim=1).numpy();o=np.argsort(-(u@v.T),1);r=np.array([np.where(ids[b][row]==ids[a[n]])[0][0]+1 for n,row in enumerate(o)]);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def locprobe(x,langs):
 ix=np.random.default_rng(SEED).permutation(len(x));a,b=ix[:len(ix)//2],ix[len(ix)//2:];s=StandardScaler().fit(x[a]);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(s.transform(x[a]),langs[a]);return float((m.predict(s.transform(x[b]))==langs[b]).mean())
def ev(x,te):return {'semantic_retrieval':retrieval(x,te.id.to_numpy(),te.locale.to_numpy()),'locale_probe':locprobe(x,te.locale.to_numpy()),'effective_rank':rank(x)}
def encode(p,x,which):
 o=[]
 with torch.no_grad():
  for a in range(0,len(x),4096):o.append(p(torch.from_numpy(np.asarray(x[a:a+4096])).to(DEV))[which].cpu().numpy())
 return np.concatenate(o)
def infonce(raw,tr,ht):
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);p=P().to(DEV);op=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.0001);ids,_=pd.factorize(tr.id,sort=True);langs,_=pd.factorize(tr.locale,sort=True);grid=np.empty((tr.id.nunique(),49),np.int64);grid[ids,langs]=np.arange(len(tr))
 for e in range(EPOCHS):
  for b in np.array_split(rng.permutation(len(tr)),max(1,len(tr)//BATCH)):
   ib,lb=ids[b],langs[b];cp=grid[ib,(lb+rng.integers(1,49,len(b)))%49];sp=grid[(ib+rng.integers(1,len(grid),len(b)))%len(grid),lb];a=torch.from_numpy(np.asarray(raw[b])).to(DEV);pc=torch.from_numpy(np.asarray(raw[cp])).to(DEV);ps=torch.from_numpy(np.asarray(raw[sp])).to(DEV);zc,zs=p(a);zcp,_=p(pc);_,zsp=p(ps);t=torch.arange(len(b),device=DEV);loss=(F.cross_entropy(zc@zcp.T/TEMP,t)+F.cross_entropy(zs@zsp.T/TEMP,t))/2;op.zero_grad();loss.backward();op.step()
  print(f'InfoNCE {e+1}/{EPOCHS}',flush=True)
 return encode(p,ht,0),encode(p,ht,1)
def pca(raw,ht):
 mean=torch.zeros(2304,device=DEV);n=0
 for a in range(0,len(raw),4096):x=torch.from_numpy(np.asarray(raw[a:a+4096],np.float32)).to(DEV);mean+=x.sum(0);n+=len(x)
 mean/=n;cov=torch.zeros((2304,2304),device=DEV)
 for a in range(0,len(raw),4096):x=torch.from_numpy(np.asarray(raw[a:a+4096],np.float32)).to(DEV)-mean;cov.add_(x.T@x)
 _,v=torch.linalg.eigh(cov/(n-1));w=v[:,-128:];return np.concatenate([((torch.from_numpy(np.asarray(ht[a:a+4096],np.float32)).to(DEV)-mean)@w).cpu().numpy() for a in range(0,len(ht),4096)])
def main():
 assert torch.cuda.is_available();tr,te=pd.read_csv(A/'train_metadata.csv'),pd.read_csv(A/'test_metadata.csv');raw=np.load(A/'raw_train_layer8.npy',mmap_mode='r');ht=np.asarray(np.load(A/'raw_test_layer8.npy',mmap_mode='r'));prior=json.loads((ROOT/'Report'/'massive_partition_audit.json').read_text());r={'scope':{'B1_only':True,'fixed_data_split':True,'frozen_Gemma_activations':True,'SAE_trained':False,'decoder_trained':False,'GemmaScope_loaded':False,'interventions_run':False},'raw_H':{'semantic_retrieval':prior['same_ID_cross_language_retrieval']['raw'],'locale_probe':prior['heldout_language_probe_accuracy']['raw'],'effective_rank':prior['heldout_effective_rank']['raw']}}
 gs=[]
 for s in range(5):
  w=torch.randn((2304,128),generator=torch.Generator(device=DEV).manual_seed(SEED+s),device=DEV)/np.sqrt(128);x=np.concatenate([(torch.from_numpy(np.asarray(ht[a:a+4096],np.float32)).to(DEV)@w).cpu().numpy() for a in range(0,len(ht),4096)]);gs.append(ev(x,te))
 r['random_gaussian_128']={'seeds':gs};r['PCA_128']=ev(pca(raw,ht),te);zc,zs=infonce(raw,tr,ht);r['naive_inbatch_InfoNCE']={'zC':ev(zc,te),'zS':ev(zs,te)};r['matched_CS_partition']={'zC':{'semantic_retrieval':prior['same_ID_cross_language_retrieval']['z_C'],'locale_probe':prior['heldout_language_probe_accuracy']['z_C'],'effective_rank':prior['heldout_effective_rank']['z_C']},'zS':{'semantic_retrieval':prior['same_ID_cross_language_retrieval']['z_S'],'locale_probe':prior['heldout_language_probe_accuracy']['z_S'],'effective_rank':prior['heldout_effective_rank']['z_S']}};OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
