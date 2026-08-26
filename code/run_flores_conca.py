"""Official ConCA BatchNorm + PAnnealSoftPlus on frozen FLORES raw/z_C arrays."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data'/'flores_partition_artifacts'; OUT=ROOT/'Report'/'flores_conca_audit.json'; CKPT=ROOT/'checkpoint'; DEV='cuda' if torch.cuda.is_available() else 'cpu'
SEED=20260826; EXPANSION=4; STEPS=24000; BATCH=256; LR=1e-4; WARMUP=200; SPARSITY_WARMUP=400; ALPHA=0.001
TRAIN_LANG=np.array(['eng_Latn','fra_Latn','spa_Latn','deu_Latn','rus_Cyrl','hin_Deva','swh_Latn','tur_Latn']); HOLD=np.array(['arb_Arab','zho_Hans'])

class ConstrainedAdam(torch.optim.Adam):
 def __init__(self,params,constrained_params,lr): super().__init__(params,lr=lr,betas=(.9,.999)); self.constrained_params=list(constrained_params)
 def step(self,closure=None):
  with torch.no_grad():
   for p in self.constrained_params:
    u=p/p.norm(dim=0,keepdim=True);p.grad-=(p.grad*u).sum(dim=0,keepdim=True)*u
  super().step(closure)
  with torch.no_grad():
   for p in self.constrained_params:p/=p.norm(dim=0,keepdim=True)

class ConCA(nn.Module):
 def __init__(self,d,m):
  super().__init__(); self.bias=nn.Parameter(torch.zeros(d)); self.encoder=nn.Linear(d,m); self.decoder=nn.Linear(m,d,bias=False); self.bn=nn.BatchNorm1d(m)
  w=torch.randn(d,m); w=w/w.norm(dim=0,keepdim=True)*.1; self.encoder.weight=nn.Parameter(w.T); self.decoder.weight=nn.Parameter(w)
 def forward(self,x):
  f=self.bn(self.encoder(x-self.bias)); return self.decoder(f)+self.bias,f

def auc(scores,y):
 order=np.argsort(scores,kind='stable'); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1); n1=y.sum(); n0=len(y)-n1
 return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
def rank(x):
 s=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=s/s.sum();return float(np.exp(-(p*np.log(p+1e-30)).sum()))
def language_probe(x,y):
 r=np.random.default_rng(SEED); i=np.arange(len(y));r.shuffle(i);cut=len(i)//2;sc=StandardScaler().fit(x[i[:cut]]);m=LogisticRegression(max_iter=1000,random_state=SEED).fit(sc.transform(x[i[:cut]]),y[i[:cut]]);return float((m.predict(sc.transform(x[i[cut:]]))==y[i[cut:]]).mean())
def feature_audit(a,ids,langs,splits):
 test=splits=='test'; seen=test&np.isin(langs,TRAIN_LANG); held=test&np.isin(langs,HOLD); sids=np.unique(ids[test]); out=[]
 for sid in sids:
  y=ids[seen]==sid; values=a[seen]; all_auc=np.array([auc(values[:,k],y) for k in range(a.shape[1])]); signed=np.abs(all_auc-.5); k=int(signed.argmax()); sign=1 if all_auc[k]>=.5 else -1
  yh=ids[held]==sid; au=auc(sign*a[held,k],yh)
  seen_mean=sign*a[seen][ids[seen]==sid,k].mean(); held_mean=sign*a[held][ids[held]==sid,k].mean(); stability=1-abs(seen_mean-held_mean)/(abs(seen_mean)+abs(held_mean)+1e-8)
  out.append({'sentence_id':int(sid),'feature':k,'seen_auc':float(.5+signed[k]),'heldout_auc':au,'stability':float(stability)})
 # Selectivity and language leakage use the same held-out C-test rows as the frozen SAE audit.
 at=a[test]; sid=ids[test]; lan=langs[test]
 def ratio(g):
  means=np.stack([at[g==v].mean(0) for v in np.unique(g)]); within=np.mean([at[g==v].var(0) for v in np.unique(g)],0);return means.var(0)/(within+1e-8)
 sr,lr=ratio(sid),ratio(lan); sentence=(sr>lr)&(sr>1); language=(lr>sr)&(lr>1)
 return {'concepts':out,'mean_seen_concept_auc':float(np.mean([x['seen_auc'] for x in out])),'mean_heldout_concept_auc':float(np.mean([x['heldout_auc'] for x in out])),'mean_component_stability':float(np.mean([x['stability'] for x in out])),'sentence_oriented_fraction':float(sentence.mean()),'language_oriented_fraction':float(language.mean()),'language_probe_accuracy':language_probe(at,lan),'effective_rank':rank(at)}
def main():
 torch.manual_seed(SEED);np.random.seed(SEED); raw=np.load(ART/'raw_layer8.npy'); zc=np.load(ART/'z_C.npy'); ids=np.load(ART/'sentence_ids.npy'); langs=np.load(ART/'languages.npy'); splits=np.load(ART/'splits.npy'); train=(splits=='train')&np.isin(langs,TRAIN_LANG)
 report={'source':'official ConCA repository: AutoEncodeBatchNorm + PAnnealSoftPlusTrainer','settings':{'objective':'official reconstruction plus alpha*Lp(softplus(features)); p_start=p_end=1','steps':STEPS,'batch_size':BATCH,'lr':LR,'warmup_steps':WARMUP,'sparsity_warmup_steps':SPARSITY_WARMUP,'alpha':ALPHA,'expansion':EXPANSION,'training_rows':int(train.sum()),'frozen_inputs':True,'partition_retrained':False}}
 for name,x in {'raw':raw,'z_C':zc}.items():
  mean=x[train].mean(0); std=x[train].std(0)+1e-6; xt=torch.from_numpy(((x-mean)/std).astype(np.float32)).to(DEV); m=ConCA(x.shape[1],x.shape[1]*EXPANSION).to(DEV); opt=ConstrainedAdam(m.parameters(),m.decoder.parameters(),LR); sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(step/WARMUP,1.0)); rng=np.random.default_rng(SEED)
  m.train()
  for step in range(STEPS):
   b=rng.choice(np.where(train)[0],BATCH,replace=True); rec,f=m(xt[b]); loss=F.mse_loss(rec,xt[b],reduction='none').sum(1).mean()+ALPHA*min(step/SPARSITY_WARMUP,1.0)*F.softplus(f).abs().sum(1).mean(); opt.zero_grad();loss.backward();opt.step();sched.step()
   if (step+1)%4000==0: print(f'{name} step={step+1} loss={loss.item():.6f}')
  m.eval()
  with torch.no_grad(): rec,f=m(xt); a=f.cpu().numpy(); mse=float(F.mse_loss(rec[train],xt[train]).item())
  torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'settings':report['settings']},CKPT/f'flores_conca_{name}.pt');np.save(ART/f'conca_{name}_features.npy',a)
  result=feature_audit(a,ids,langs,splits);result['reconstruction_mse']=mse;result['dictionary_size']=int(a.shape[1]);report[name]=result;print(name,json.dumps({k:v for k,v in result.items() if k!='concepts'},indent=2))
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
