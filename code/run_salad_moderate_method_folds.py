"""Three frozen-Gemma leave-one-method-out SALAD partitions; no SAE/ConCA."""
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
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'salad_attack_enhanced.parquet';ART=ROOT/'data'/'salad_moderate_method_folds_artifacts';CKPT=ROOT/'checkpoint';OUT=ROOT/'Report'/'salad_moderate_method_folds.json';METHODS=['gcg_llama','gptfuzz','jb'];EPOCHS=30;BATCH=256;LR=1e-3;TEMP=.07;SEED=20260827;DEV='cuda'
class P(nn.Module):
 def __init__(self,w):super().__init__();self.c=nn.Linear(w,128);self.s=nn.Linear(w,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEV))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v**2).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def retrieval(x,q,b,ids):
 o=np.argsort(-(x[q]@x[b].T),1);r=np.array([np.where(ids[b][row]==ids[q[n]])[0][0]+1 for n,row in enumerate(o)]);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def probe(tx,ty,vx,vy):
 s=StandardScaler().fit(tx);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(s.transform(tx),ty);return float((m.predict(s.transform(vx))==vy).mean())
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def geometry(x,train,held,ids,methods):
 same=[cosine(x[i],x[j]) for i in held for j in train if ids[i]==ids[j] and methods[i]!=methods[j]];diff=[cosine(x[i],x[j]) for i,j in __import__('itertools').combinations(held,2) if ids[i]!=ids[j]]
 return {'same_goal_different_method':{'n':len(same),'mean':float(np.mean(same)),'std':float(np.std(same))},'different_goal_same_heldout_method':{'n':len(diff),'mean':float(np.mean(diff)),'std':float(np.std(diff))},'margin':float(np.mean(same)-np.mean(diff))}
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);d=pd.read_parquet(DATA);x=d[d.method.isin(METHODS)];sets=x.groupby('qid').method.agg(set);keep=sorted(sets[sets.map(lambda s:s==set(METHODS))].index);x=x[x.qid.isin(keep)].sort_values(['qid','method','aid']).reset_index(drop=True);rows=x[['qid','method','augq']].rename(columns={'augq':'text'});assert len(keep)==79 and len(rows)==347
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;g=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEV).eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(rows),8):
   t=tok(rows.text.iloc[start:start+8].tolist(),padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEV);h=g(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);parts.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());print(f'extracted={min(start+8,len(rows))}/{len(rows)}')
 raw=np.concatenate(parts);ids=rows.qid.to_numpy();methods=rows.method.to_numpy();tensor=torch.from_numpy(raw).to(DEV);ART.mkdir(parents=True,exist_ok=True);np.save(ART/'raw_layer8.npy',raw);rows[['qid','method']].to_csv(ART/'metadata.csv',index=False);report={'scope':{'qids':79,'source_rows':347,'rows_per_method':{k:int(v) for k,v in rows.method.value_counts().items()},'Gemma_layer':8,'pooling':'masked mean','architecture':'2304 -> z_C(128) + z_S(128)','epochs':EPOCHS,'AutoDAN_used':False,'SAE_trained':False,'ConCA_trained':False},'folds':{}}
 for fold,hold in enumerate(METHODS):
  torch.manual_seed(SEED+fold);fold_rng=np.random.default_rng(SEED+fold);train=np.where(methods!=hold)[0];held=np.where(methods==hold)[0];byid=defaultdict(list);bym=defaultdict(list)
  for i in train:byid[ids[i]].append(i);bym[methods[i]].append(i)
  assert all(len({methods[j] for j in v})==2 for v in byid.values())
  model=P(2304).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4);history=[]
  for epoch in range(EPOCHS):
   totals=[0.,0.];steps=0
   for batch in np.array_split(fold_rng.permutation(train),max(1,len(train)//BATCH)):
    cp=np.array([fold_rng.choice([j for j in byid[ids[i]] if methods[j]!=methods[i]]) for i in batch]);cn=np.array([fold_rng.choice([j for j in bym[methods[i]] if ids[j]!=ids[i]]) for i in batch]);zc,zs=model(tensor[batch]);zcp,_=model(tensor[cp]);zcn,_=model(tensor[cn]);_,zsp=model(tensor[cn]);_,zsn=model(tensor[cp]);lc,ls=loss(zc,zcp,zcn),loss(zs,zsp,zsn);opt.zero_grad();((lc+ls)/2).backward();opt.step();totals[0]+=lc.item();totals[1]+=ls.item();steps+=1
   history.append({'epoch':epoch+1,'C_triplet_loss':totals[0]/steps,'S_triplet_loss':totals[1]/steps})
  with torch.no_grad():zc,zs=model(tensor);zc,zs=zc.cpu().numpy(),zs.cpu().numpy()
  qids=np.array(sorted(set(ids)));split=set(fold_rng.choice(qids,len(qids)//2,replace=False));pt=np.array([i for i in train if ids[i] in split]);pv=np.array([i for i in train if ids[i] not in split])
  metrics={'training_methods':[m for m in METHODS if m!=hold],'heldout_method':hold,'training_rows':int(len(train)),'heldout_rows':int(len(held)),'heldout_goal_retrieval':{'raw':retrieval(raw,held,train,ids),'z_C':retrieval(zc,held,train,ids),'z_S':retrieval(zs,held,train,ids)},'heldout_goal_probe_accuracy':{'raw':probe(raw[train],ids[train],raw[held],ids[held]),'z_C':probe(zc[train],ids[train],zc[held],ids[held]),'z_S':probe(zs[train],ids[train],zs[held],ids[held])},'seen_method_behavior_disjoint_probe_accuracy':{'raw':probe(raw[pt],methods[pt],raw[pv],methods[pv]),'z_C':probe(zc[pt],methods[pt],zc[pv],methods[pv]),'z_S':probe(zs[pt],methods[pt],zs[pv],methods[pv])},'heldout_pair_geometry':{'raw':geometry(raw,train,held,ids,methods),'z_C':geometry(zc,train,held,ids,methods),'z_S':geometry(zs,train,held,ids,methods)},'heldout_effective_rank':{'raw':rank(raw[held]),'z_C':rank(zc[held]),'z_S':rank(zs[held])},'final_losses':history[-1]};report['folds'][hold]=metrics;torch.save({'state_dict':model.state_dict(),'history':history,'config':{'heldout_method':hold,'architecture':'2304 -> z_C(128) + z_S(128)'}},CKPT/f'salad_moderate_holdout_{hold}_partition_layer8.pt');np.save(ART/f'z_C_holdout_{hold}.npy',zc);np.save(ART/f'z_S_holdout_{hold}.npy',zs);print(f'fold={hold} R1_raw={metrics["heldout_goal_retrieval"]["raw"]["R@1"]:.3f} R1_zC={metrics["heldout_goal_retrieval"]["z_C"]["R@1"]:.3f}')
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
