"""Three-seed robustness audit for the fixed pooled MASSIVE core pipeline.

Gemma Scope is deliberately excluded: its historical R_C/R_S code and artifacts
are unavailable, so this script does not reconstruct or approximate them.
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data'/'massive_partition_artifacts'; OUT=ROOT/'data'/'massive_core_seed_robustness'; REPORT=ROOT/'Report'/'massive_core_seed_robustness.json'
SEEDS=(20260828,20260829,20260830); EPOCHS=30; BATCH=256; K=64; EXP=4; DEV='cuda'; HOLD=('ar-SA','zh-CN'); EVAL_SEED=20260827
class P(nn.Module):
 def __init__(self): super().__init__();self.c=nn.Linear(2304,128);self.s=nn.Linear(2304,128)
 def forward(self,x): return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
class SAE(nn.Module):
 def __init__(self): super().__init__();self.e=nn.Linear(128,512);self.d=nn.Linear(512,128,bias=False);self.b=nn.Parameter(torch.zeros(128))
 def forward(self,x):
  a=F.relu(self.e(x));v,i=torch.topk(a,K,1);z=torch.zeros_like(a).scatter(1,i,v);return z,self.d(z)+self.b
 def norm(self):
  with torch.no_grad(): self.d.weight.div_(self.d.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
def enc(p,x,which):
 o=np.empty((len(x),128),np.float32);p.eval()
 with torch.no_grad():
  for a in range(0,len(x),4096):o[a:a+4096]=p(torch.from_numpy(np.asarray(x[a:a+4096])).to(DEV))[which].cpu().numpy()
 return o
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;q=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v*v).sum()),'entropy_effective_rank':float(np.exp(-(q*np.log(q+1e-30)).sum()))}
def retrieval(x,ids,lang):
 a=np.where(lang==HOLD[0])[0];b=np.where(lang==HOLD[1])[0];u=F.normalize(torch.from_numpy(x[a]),dim=1).numpy();v=F.normalize(torch.from_numpy(x[b]),dim=1).numpy();order=np.argsort(-(u@v.T),1);r=np.array([np.where(ids[b][row]==ids[a[n]])[0][0]+1 for n,row in enumerate(order)])
 return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def probe(tx,ty,qx,qy):
 s=StandardScaler().fit(tx);m=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=1000,tol=1e-3,random_state=EVAL_SEED).fit(s.transform(tx),ty);return float((m.predict(s.transform(qx))==qy).mean()),s,m
def moments(x):
 s=np.zeros(128,np.float64);q=s.copy()
 for a in range(0,len(x),4096):z=np.asarray(x[a:a+4096],np.float32);s+=z.sum(0);q+=(z*z).sum(0)
 m=s/len(x);return m.astype(np.float32),np.sqrt(np.maximum(q/len(x)-m*m,1e-12)).astype(np.float32)
def train_sae(x,seed,path):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);mean,std=moments(x);m=SAE().to(DEV);op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);hist=[]
 for _ in range(EPOCHS):
  ls=[]
  for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
   t=torch.from_numpy((np.asarray(x[ids],np.float32)-mean)/std).to(DEV);_,r=m(t);loss=F.mse_loss(r,t);op.zero_grad();loss.backward();op.step();m.norm();ls.append(loss.item())
  hist.append(float(np.mean(ls)))
 torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'config':{'k':K,'expansion':EXP,'epochs':EPOCHS,'seed':seed}},path);return m,mean,std,hist
def act(m,x,mean,std):
 o=[]
 with torch.no_grad():
  for a in range(0,len(x),512):o.append(m(torch.from_numpy((np.asarray(x[a:a+512],np.float32)-mean)/std).to(DEV))[0].cpu().numpy())
 return np.concatenate(o)
def sae_audit(m,mean,std,x,xt,tr,te):
 rng=np.random.default_rng(EVAL_SEED);ix=rng.choice(len(tr),20000,False);a=act(m,x[ix],mean,std);b=act(m,xt,mean,std);mu=a.mean(0);sd=a.std(0).clip(1e-6);im=np.stack([a[tr.intent.to_numpy()[ix]==v].mean(0) for v in np.unique(tr.intent)]).max(0)-mu;lm=np.stack([a[tr.locale.to_numpy()[ix]==v].mean(0) for v in np.unique(tr.locale)]).max(0)-mu;im/=sd;lm/=sd;active=mu>1e-6;intent=active&(im>1.1*lm);lang=active&(lm>1.1*im);ar={te.id.iloc[i]:i for i in np.where(te.locale.to_numpy()==HOLD[0])[0]};zh={te.id.iloc[i]:i for i in np.where(te.locale.to_numpy()==HOLD[1])[0]};pairs=[(ar[k],zh[k]) for k in ar.keys()&zh.keys()]
 st=np.nanmean([np.corrcoef(b[[i for i,j in pairs],f],b[[j for i,j in pairs],f])[0,1] for f in np.where(active)[0] if b[[i for i,j in pairs],f].std() and b[[j for i,j in pairs],f].std()])
 return {'intent_oriented_fraction':float(intent.sum()/active.sum()),'language_oriented_fraction':float(lang.sum()/active.sum()),'heldout_arabic_chinese_stability':float(st),'active_features':int(active.sum())}
def train_partition(raw,tr,seed):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);p=P().to(DEV);op=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.0001);ids,_=pd.factorize(tr.id,sort=True);langs,_=pd.factorize(tr.locale,sort=True);grid=np.empty((tr.id.nunique(),49),np.int64);grid[ids,langs]=np.arange(len(tr));hist=[]
 for _ in range(EPOCHS):
  total=0.;n=0
  for batch in np.array_split(rng.permutation(len(tr)),max(1,len(tr)//BATCH)):
   ib,lb=ids[batch],langs[batch];cp=grid[ib,(lb+rng.integers(1,49,len(batch)))%49];cn=grid[(ib+rng.integers(1,len(grid),len(batch)))%len(grid),lb];a=torch.from_numpy(np.asarray(raw[batch])).to(DEV);pp=torch.from_numpy(np.asarray(raw[cp])).to(DEV);nnn=torch.from_numpy(np.asarray(raw[cn])).to(DEV);zc,zs=p(a);zcp,_=p(pp);zcn,_=p(nnn);_,zsp=p(nnn);_,zsn=p(pp);lc=F.cross_entropy(torch.stack(((zc*zcp).sum(-1),(zc*zcn).sum(-1)),1)/.07,torch.zeros(len(batch),device=DEV,dtype=torch.long));ls=F.cross_entropy(torch.stack(((zs*zsp).sum(-1),(zs*zsn).sum(-1)),1)/.07,torch.zeros(len(batch),device=DEV,dtype=torch.long));op.zero_grad();((lc+ls)/2).backward();op.step();total+=(lc+ls).item()/2;n+=1
  hist.append(total/n)
 return p,hist
def decoder(zc,zs,h,seed):
 torch.manual_seed(seed);rng=np.random.default_rng(seed);sq=count=0
 for a in range(0,len(h),8192):x=np.asarray(h[a:a+8192],np.float32);sq+=np.square(x,dtype=np.float32).sum(dtype=np.float64);count+=x.size
 scale=float((sq/count)**.5);m=nn.Linear(256,2304).to(DEV);op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 for _ in range(4):
  for ix in np.array_split(rng.permutation(len(h)),max(1,len(h)//1024)):
   x=np.concatenate((zc[ix],zs[ix]),1);y=np.asarray(h[ix])/scale;loss=F.mse_loss(m(torch.from_numpy(x).to(DEV)),torch.from_numpy(y).to(DEV));op.zero_grad();loss.backward();op.step()
 return m,scale
def swap_metrics(dec,scale,zc,zs,h,te,tr,raw):
 rng=np.random.default_rng(EVAL_SEED);by={(str(r.id),r.locale):i for i,r in te.iterrows()};ids=sorted(set(te.id.astype(str)));src=[];don=[];c_t=[];s_t=[];c_int=[];s_int=[];src_loc=[];don_loc=[]
 for a,b in ((HOLD[0],HOLD[1]),(HOLD[1],HOLD[0])):
  for i in ids:
   js=[]
   for j in rng.permutation(ids):
    if j!=i:js.append(j)
    if len(js)==3:break
   for j in js:src.append(by[i,a]);don.append(by[j,b]);c_t.append(by[j,a]);s_t.append(by[i,b]);c_int.append(te.iloc[by[j,a]].intent);s_int.append(te.iloc[by[i,b]].intent);src_loc.append(a);don_loc.append(b)
 src=np.array(src);don=np.array(don);c_int=np.array(c_int);s_int=np.array(s_int);src_loc=np.array(src_loc);don_loc=np.array(don_loc)
 def predict(c,s):
  out=[]
  with torch.no_grad():
   for a in range(0,len(c),1024):out.append((dec(torch.from_numpy(np.concatenate((c[a:a+1024],s[a:a+1024]),1)).to(DEV))*scale).cpu().numpy())
  return np.concatenate(out)
 co=predict(zc[don],zs[src]);so=predict(zc[src],zs[don]);ix=[]
 for lab in sorted(tr.intent.unique()):ix.extend(rng.choice(np.where(tr.intent.to_numpy()==lab)[0],100,False))
 _,isc,ic=probe(np.asarray(raw[np.array(ix)]),tr.intent.to_numpy()[np.array(ix)],co,c_int);perm=rng.permutation(len(h));_,lsc,lc=probe(h[perm[:len(h)//2]],te.locale.to_numpy()[perm[:len(h)//2]],co,src_loc);ci=ic.predict(isc.transform(co));si=ic.predict(isc.transform(so));cl=lc.predict(lsc.transform(co));sl=lc.predict(lsc.transform(so));labels=np.array(sorted(tr.intent.unique()));unrel=np.array([rng.choice(labels[labels!=x]) for x in c_int])
 return {'C_swap_donor_intent_success':float((ci==c_int).mean()),'C_swap_donor_locale_leakage':float((cl==don_loc).mean()),'S_swap_donor_locale_success':float((sl==don_loc).mean()),'S_swap_donor_intent_leakage':float((si==c_int).mean()),'C_unrelated_intent_control':float((ci==unrel).mean()),'S_unrelated_locale_control':float((sl==src_loc).mean())}
def main():
 assert torch.cuda.is_available();OUT.mkdir(exist_ok=True);tr=pd.read_csv(ART/'train_metadata.csv');te=pd.read_csv(ART/'test_metadata.csv');raw=np.load(ART/'raw_train_layer8.npy',mmap_mode='r');ht=np.asarray(np.load(ART/'raw_test_layer8.npy',mmap_mode='r'));results=[]
 for seed in SEEDS:
  sd=OUT/f'seed_{seed}';sd.mkdir(exist_ok=True);p,ph=train_partition(raw,tr,seed);torch.save({'state_dict':p.state_dict(),'history':ph},sd/'partition.pt');zc=enc(p,raw,0);zs=enc(p,raw,1);zct=enc(p,ht,0);zst=enc(p,ht,1);np.save(sd/'z_C_train.npy',zc);np.save(sd/'z_S_train.npy',zs);np.save(sd/'z_C_test.npy',zct);np.save(sd/'z_S_test.npy',zst)
  perm=np.random.default_rng(EVAL_SEED).permutation(len(ht));half=perm[:len(ht)//2];other=perm[len(ht)//2:];part={'zC':{'semantic_retrieval':retrieval(zct,te.id.to_numpy(),te.locale.to_numpy()),'locale_probe':probe(zct[half],te.locale.to_numpy()[half],zct[other],te.locale.to_numpy()[other])[0],'effective_rank':rank(zct)},'zS':{'semantic_retrieval':retrieval(zst,te.id.to_numpy(),te.locale.to_numpy()),'locale_probe':probe(zst[half],te.locale.to_numpy()[half],zst[other],te.locale.to_numpy()[other])[0],'effective_rank':rank(zst)}}
  mc,mcmean,mcstd,_=train_sae(zc,seed,sd/'sae_zC.pt');ms,msmean,msstd,_=train_sae(zs,seed,sd/'sae_zS.pt');sae={'zC':sae_audit(mc,mcmean,mcstd,zc,zct,tr,te),'zS':sae_audit(ms,msmean,msstd,zs,zst,tr,te)};dec,scale=decoder(zc,zs,raw,seed);torch.save({'state_dict':dec.state_dict(),'scale':scale},sd/'joint_decoder.pt');inter=swap_metrics(dec,scale,zct,zst,ht,te,tr,raw);row={'seed':seed,'partition':part,'sae':sae,'representation_intervention':inter};(sd/'report.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row);print(f'completed seed {seed}',flush=True);torch.cuda.empty_cache()
 def vals(path):
  out=[]
  for r in results:
   x=r
   for k in path:x=x[k]
   out.append(float(x))
  return {'values':out,'mean':float(np.mean(out)),'std':float(np.std(out,ddof=1))}
 paths={'zC_R1':['partition','zC','semantic_retrieval','R@1'],'zC_R5':['partition','zC','semantic_retrieval','R@5'],'zC_MRR':['partition','zC','semantic_retrieval','MRR'],'zC_locale':['partition','zC','locale_probe'],'zC_rank':['partition','zC','effective_rank','participation_ratio'],'zC_entropy_rank':['partition','zC','effective_rank','entropy_effective_rank'],'zS_R1':['partition','zS','semantic_retrieval','R@1'],'zS_R5':['partition','zS','semantic_retrieval','R@5'],'zS_MRR':['partition','zS','semantic_retrieval','MRR'],'zS_locale':['partition','zS','locale_probe'],'zS_rank':['partition','zS','effective_rank','participation_ratio'],'zS_entropy_rank':['partition','zS','effective_rank','entropy_effective_rank'],'zC_intent_fraction':['sae','zC','intent_oriented_fraction'],'zC_language_fraction':['sae','zC','language_oriented_fraction'],'zC_stability':['sae','zC','heldout_arabic_chinese_stability'],'zS_intent_fraction':['sae','zS','intent_oriented_fraction'],'zS_language_fraction':['sae','zS','language_oriented_fraction'],'zS_stability':['sae','zS','heldout_arabic_chinese_stability']}
 for k in next(iter(results))['representation_intervention']:paths[k]=['representation_intervention',k]
 report={'seeds':list(SEEDS),'per_seed':results,'mean_std':{k:vals(v) for k,v in paths.items()},'gemma_scope_seed_robustness':'NOT RUN — exact historical R_C/R_S artifacts unavailable; original frozen-SAE downstream result retained as single-run external validation.'};REPORT.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
