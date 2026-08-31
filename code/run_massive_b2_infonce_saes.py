"""B2: reproduce B1 InfoNCE artifacts, then audit only the requested Top-k SAEs."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from numpy._core.multiarray import _reconstruct

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data'/'massive_partition_artifacts';S=ROOT/'data'/'massive_sae_artifacts';C=ROOT/'checkpoint';OUT=ROOT/'data'/'massive_b2_infonce_artifacts';REPORT=ROOT/'Report'/'massive_b2_infonce_sae_comparison.json'
SEED,EPOCHS,BATCH,TEMP,K,EXP=20260827,30,256,.07,64,4;D='cuda';TOL=1e-3
EXPECTED={'zC':{'R@1':.4882075471698113,'R@5':.7324797843665768,'MRR':.6007511894153003,'locale_probe':.9235175202156334},'zS':{'R@1':.0006738544474393531,'locale_probe':.9996630727762803}}

class P(nn.Module):
 def __init__(self):super().__init__();self.c,self.s=nn.Linear(2304,128),nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
class SAE(nn.Module):
 def __init__(self,w):super().__init__();self.e=nn.Linear(w,w*EXP);self.d=nn.Linear(w*EXP,w,bias=False);self.b=nn.Parameter(torch.zeros(w))
 def forward(self,x):a=F.relu(self.e(x));v,i=torch.topk(a,K,1);z=torch.zeros_like(a).scatter(1,i,v);return z,self.d(z)+self.b
 def normalize(self):
  with torch.no_grad():self.d.weight.div_(self.d.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))

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
  for a in range(0,len(x),4096):o.append(p(torch.from_numpy(np.asarray(x[a:a+4096])).to(D))[which].cpu().numpy())
 return np.concatenate(o)
def save_array(path,x):
 y=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=x.shape);y[:]=x;y.flush()
def train_infonce(raw,tr,ht):
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);p=P().to(D);op=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.0001);ids,_=pd.factorize(tr.id,sort=True);langs,_=pd.factorize(tr.locale,sort=True);grid=np.empty((tr.id.nunique(),49),np.int64);grid[ids,langs]=np.arange(len(tr))
 for e in range(EPOCHS):
  for b in np.array_split(rng.permutation(len(tr)),max(1,len(tr)//BATCH)):
   ib,lb=ids[b],langs[b];cp=grid[ib,(lb+rng.integers(1,49,len(b)))%49];sp=grid[(ib+rng.integers(1,len(grid),len(b)))%len(grid),lb];a=torch.from_numpy(np.asarray(raw[b])).to(D);pc=torch.from_numpy(np.asarray(raw[cp])).to(D);ps=torch.from_numpy(np.asarray(raw[sp])).to(D);zc,zs=p(a);zcp,_=p(pc);_,zsp=p(ps);t=torch.arange(len(b),device=D);loss=(F.cross_entropy(zc@zcp.T/TEMP,t)+F.cross_entropy(zs@zsp.T/TEMP,t))/2;op.zero_grad();loss.backward();op.step()
  print(f'InfoNCE {e+1}/{EPOCHS}',flush=True)
 return p,encode(p,raw,0),encode(p,raw,1),encode(p,ht,0),encode(p,ht,1)
def train_sae(x,name):
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);mean,std=np.asarray(x).mean(0),np.asarray(x).std(0).clip(1e-6);m=SAE(x.shape[1]).to(D);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);hist=[]
 for e in range(EPOCHS):
  losses=[]
  for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
   q=torch.from_numpy((np.asarray(x[ids])-mean)/std).to(D);_,r=m(q);loss=F.mse_loss(r,q);opt.zero_grad();loss.backward();opt.step();m.normalize();losses.append(loss.item())
  hist.append(float(np.mean(losses)));print(f'{name} SAE {e+1}/{EPOCHS} mse={hist[-1]:.6f}',flush=True)
 torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'config':{'k':K,'expansion':EXP,'epochs':EPOCHS,'seed':SEED,'input_width':x.shape[1]}},OUT/f'infonce_topk_{name}_k64.pt')
 return m,mean,std,hist
def acts(m,x,mean,std,recon=False):
 a=[];sq=0.;n=0
 with torch.no_grad():
  for i in range(0,len(x),512):
   q=torch.from_numpy((np.asarray(x[i:i+512],np.float32)-mean)/std).to(D);z,r=m(q);a.append(z.cpu().numpy());sq+=F.mse_loss(r,q,reduction='sum').item();n+=q.numel()
 return np.concatenate(a),float(sq/n) if recon else None
def load_sae(path,w):
 torch.serialization.add_safe_globals([_reconstruct,np.ndarray,np.dtype,np.dtypes.Float32DType]);q=torch.load(path,map_location=D,weights_only=True);m=SAE(w).to(D);m.load_state_dict(q['state_dict']);m.eval();return m,np.asarray(q['mean']),np.asarray(q['std'])
def audit(name,x,xt,tr,te,m,mean,std):
 rng=np.random.default_rng(SEED);idx=rng.choice(len(tr),20000,False);a,_=acts(m,x[idx],mean,std);b,mse=acts(m,xt,mean,std,True);mu=a.mean(0);sd=a.std(0).clip(1e-6);im=np.stack([a[tr.intent.to_numpy()[idx]==v].mean(0) for v in np.unique(tr.intent)]).max(0)-mu;lm=np.stack([a[tr.locale.to_numpy()[idx]==v].mean(0) for v in np.unique(tr.locale)]).max(0)-mu;im,lm=im/sd,lm/sd;active=mu>1e-6;intent=active&(im>1.1*lm);lang=active&(lm>1.1*im);mixed=active&~(intent|lang);ar={te.id.iloc[i]:i for i in np.where(te.locale=='ar-SA')[0]};zh={te.id.iloc[i]:i for i in np.where(te.locale=='zh-CN')[0]};pairs=[(ar[k],zh[k]) for k in ar.keys()&zh.keys()];vals=[np.corrcoef(b[[i for i,j in pairs],f],b[[j for i,j in pairs],f])[0,1] for f in np.where(active)[0] if b[[i for i,j in pairs],f].std() and b[[j for i,j in pairs],f].std()]
 return {'mean_intent_selectivity_z':float(im[active].mean()),'mean_locale_selectivity_z':float(lm[active].mean()),'heldout_arabic_chinese_stability':float(np.nanmean(vals)),'active_feature_count':int(active.sum()),'intent_oriented_fraction':float(intent.sum()/active.sum()),'language_oriented_fraction':float(lang.sum()/active.sum()),'mixed_other_fraction':float(mixed.sum()/active.sum()),'heldout_reconstruction_mse':mse}
def main():
 assert torch.cuda.is_available();OUT.mkdir(exist_ok=True);tr,te=pd.read_csv(A/'train_metadata.csv'),pd.read_csv(A/'test_metadata.csv');raw=np.load(A/'raw_train_layer8.npy',mmap_mode='r');ht=np.asarray(np.load(A/'raw_test_layer8.npy',mmap_mode='r'));p,zc,zs,zct,zst=train_infonce(raw,tr,ht);save_array(OUT/'z_C_train.npy',zc);save_array(OUT/'z_S_train.npy',zs);save_array(OUT/'z_C_test.npy',zct);save_array(OUT/'z_S_test.npy',zst);torch.save({'state_dict':p.state_dict(),'config':{'seed':SEED,'epochs':EPOCHS,'batch_size':BATCH,'temperature':TEMP,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'architecture':'2304->128+128'}},OUT/'infonce_partition.pt')
 reprodu={'zC':ev(zct,te),'zS':ev(zst,te)};flat={'zC':{**reprodu['zC']['semantic_retrieval'],'locale_probe':reprodu['zC']['locale_probe'],'effective_rank':reprodu['zC']['effective_rank']},'zS':{**reprodu['zS']['semantic_retrieval'],'locale_probe':reprodu['zS']['locale_probe'],'effective_rank':reprodu['zS']['effective_rank']}};diffs=[abs(flat[k][m]-v) for k,d in EXPECTED.items() for m,v in d.items()];validation={'tolerance':TOL,'max_absolute_difference':max(diffs),'matches':max(diffs)<=TOL,'expected':EXPECTED,'observed':flat};(OUT/'infonce_reproduction.json').write_text(json.dumps(validation,indent=2)+'\n');assert validation['matches'],json.dumps(validation,indent=2)
 mc,msc,mss,hc=train_sae(zc,'z_C');ms,mssm,msss,hs=train_sae(zs,'z_S');matched_c=np.load(S/'z_C_train.npy',mmap_mode='r');matched_s=np.load(S/'z_S_train.npy',mmap_mode='r');matched_ct=np.load(A/'z_C_test.npy',mmap_mode='r');matched_st=np.load(A/'z_S_test.npy',mmap_mode='r');mC,meanC,stdC=load_sae(C/'massive_topk_z_C_k64.pt',128);mS,meanS,stdS=load_sae(C/'massive_topk_z_S_k64.pt',128);mH,meanH,stdH=load_sae(C/'massive_topk_raw_k64.pt',2304);prior=json.loads((ROOT/'Report'/'massive_pooled_sae_threeway_audit.json').read_text())
 report={'scope':{'B2_only':True,'frozen_Gemma_activations':True,'fixed_split':True,'heldout_locales':['ar-SA','zh-CN'],'no_ConCA':True,'no_GemmaScope':True,'no_interventions':True},'infonce_reproduction':validation,'sae_config':{'input_width':128,'dictionary_width':512,'expansion':EXP,'top_k':K,'epochs':EPOCHS,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'standardized_inputs':True},'branches':{'InfoNCE_zC':audit('InfoNCE_zC',zc,zct,tr,te,mc,msc,mss),'InfoNCE_zS':audit('InfoNCE_zS',zs,zst,tr,te,ms,mssm,msss),'Matched_zC':audit('Matched_zC',matched_c,matched_ct,tr,te,mC,meanC,stdC),'Matched_zS':audit('Matched_zS',matched_s,matched_st,tr,te,mS,meanS,stdS),'Raw_H':{**prior['H'],**{'heldout_reconstruction_mse':acts(mH,ht,meanH,stdH,True)[1]}}},'training_loss':{'InfoNCE_zC':hc,'InfoNCE_zS':hs}}
 REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
