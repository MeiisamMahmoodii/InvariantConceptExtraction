"""Frozen Gemma layer-8 SALAD partition; AutoDAN fully held out. No SAE/ConCA."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'salad_attack_enhanced.parquet';ART=ROOT/'data'/'salad_autodan_partition_artifacts';CKPT=ROOT/'checkpoint';OUT=ROOT/'Report'/'salad_autodan_heldout_partition_audit.json';DEV='cuda';TRAIN=['gcg_llama','gptfuzz','jb'];HOLD='autodan';SEED=20260826;EPOCHS=30;BATCH=256;TEMP=.07
class P(nn.Module):
 def __init__(self,w):super().__init__();self.c=nn.Linear(w,128);self.s=nn.Linear(w,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEV))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v**2).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def retrieval(x,q,b,ids):
 score=x[q]@x[b].T;o=np.argsort(-score,1);r=np.array([np.where(ids[b][row]==ids[q[n]])[0][0]+1 for n,row in enumerate(o)]);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'R@10':float((r<=10).mean()),'MRR':float((1/r).mean())}
def method_retrieval(x,idx,methods,ids):
 r=[]
 for q in idx:
  b=np.array([j for j in idx if j!=q and ids[j]!=ids[q]]);o=b[np.argsort(-(x[q]@x[b].T))];r.append(np.where(methods[o]==methods[q])[0][0]+1)
 r=np.array(r);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def probe(tx,ty,vx,vy):
 sc=StandardScaler().fit(tx);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(sc.transform(tx),ty);return float((m.predict(sc.transform(vx))==vy).mean())
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);d=pd.read_parquet(DATA);n=d[d.method.isin(TRAIN)].groupby('qid').method.nunique();eligible=set(d.loc[d.method.eq(HOLD),'qid']) & set(n[n.ge(2)].index);d=d[d.qid.isin(eligible) & d.method.isin(TRAIN+[HOLD])].copy();rows=d[['qid','method','augq']].rename(columns={'qid':'id','augq':'text'}).to_dict('records');ids=np.array([r['id'] for r in rows]);methods=np.array([r['method'] for r in rows]);train=np.isin(methods,TRAIN);held=methods==HOLD;assert len(eligible)==140 and train.sum()==471 and held.sum()==140 and not np.any(methods[train]==HOLD)
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;g=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEV).eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(rows),8):
   t=tok([r['text'] for r in rows[start:start+8]],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEV);h=g(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);parts.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());print(f'extracted={min(start+8,len(rows))}/{len(rows)}')
 raw=np.concatenate(parts);x=torch.from_numpy(raw).to(DEV);byid=defaultdict(list);bym=defaultdict(list)
 for i in np.where(train)[0]:byid[ids[i]].append(i);bym[methods[i]].append(i)
 assert all(len(v)>=2 for v in byid.values())
 model=P(raw.shape[1]).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);ti=np.where(train)[0];hist=[]
 for e in range(EPOCHS):
  total=[0.,0.];steps=0
  for batch in np.array_split(rng.permutation(ti),max(1,len(ti)//BATCH)):
   cp=np.array([rng.choice([j for j in byid[ids[i]] if methods[j]!=methods[i]]) for i in batch]);cn=np.array([rng.choice([j for j in bym[methods[i]] if ids[j]!=ids[i]]) for i in batch]);zc,zs=model(x[batch]);zcp,_=model(x[cp]);zcn,_=model(x[cn]);_,zsp=model(x[cn]);_,zsn=model(x[cp]);lc,ls=loss(zc,zcp,zcn),loss(zs,zsp,zsn);opt.zero_grad();((lc+ls)/2).backward();opt.step();total[0]+=lc.item();total[1]+=ls.item();steps+=1
  hist.append({'epoch':e+1,'C_triplet_loss':total[0]/steps,'S_triplet_loss':total[1]/steps});print(f"epoch={e+1}/{EPOCHS} C_loss={hist[-1]['C_triplet_loss']:.4f} S_loss={hist[-1]['S_triplet_loss']:.4f}")
 model.eval()
 with torch.no_grad():zc,zs=model(x);zc,zs=zc.cpu().numpy(),zs.cpu().numpy()
 ART.mkdir(parents=True,exist_ok=True);np.save(ART/'raw_layer8.npy',raw);np.save(ART/'z_C.npy',zc);np.save(ART/'z_S.npy',zs);np.save(ART/'qids.npy',ids);np.save(ART/'methods.npy',methods);(ART/'metadata.json').write_text(json.dumps([{'qid':int(r['id']),'method':r['method']} for r in rows],indent=2));torch.save({'state_dict':model.state_dict(),'history':hist,'config':{'architecture':'2304 -> z_C(128) + z_S(128)','heldout_method':HOLD}},CKPT/'salad_autodan_heldout_partition_layer8.pt')
 seen=np.unique(ids[train]);s=set(rng.choice(seen,70,replace=False));pt=np.array([i for i in ti if ids[i] in s]);pv=np.array([i for i in ti if ids[i] not in s])
 report={'scope':{'dataset':'OpenSafetyLab/Salad-Data attack_enhanced_set','eligible_qids':140,'training_methods':TRAIN,'heldout_method':'AutoDAN','training_rows':int(train.sum()),'heldout_rows':int(held.sum()),'Gemma_layer':8,'pooling':'masked mean','partition_retrained':True,'SAE_trained':False,'ConCA_trained':False},'training':{'architecture':'2304 -> z_C(128) + z_S(128)','epochs':EPOCHS,'optimizer':'AdamW(lr=0.001, weight_decay=0.0001)','history':hist},'heldout_AutoDAN_goal_retrieval':{'raw':retrieval(raw,np.where(held)[0],ti,ids),'z_C':retrieval(zc,np.where(held)[0],ti,ids),'z_S':retrieval(zs,np.where(held)[0],ti,ids)},'heldout_AutoDAN_goal_probe_accuracy':{'raw':probe(raw[train],ids[train],raw[held],ids[held]),'z_C':probe(zc[train],ids[train],zc[held],ids[held]),'z_S':probe(zs[train],ids[train],zs[held],ids[held])},'seen_method_behavior_disjoint_probe_accuracy':{'raw':probe(raw[pt],methods[pt],raw[pv],methods[pv]),'z_C':probe(zc[pt],methods[pt],zc[pv],methods[pv]),'z_S':probe(zs[pt],methods[pt],zs[pv],methods[pv])},'seen_method_retrieval':{'raw':method_retrieval(raw,ti,methods,ids),'z_C':method_retrieval(zc,ti,methods,ids),'z_S':method_retrieval(zs,ti,methods,ids)},'heldout_AutoDAN_effective_rank':{'raw':rank(raw[held]),'z_C':rank(zc[held]),'z_S':rank(zs[held])},'note':'AutoDAN is unseen. Method metrics use behavior-disjoint train rows; goal metrics test AutoDAN directly.'};OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='training'},indent=2))
if __name__=='__main__':main()
