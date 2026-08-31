"""Same JBB partition, adding canonical goals as vanilla training views; PAIR unseen."""
import json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];SRC=Path(os.environ['TEMP'])/'jbb_artifacts_audit'/'attack-artifacts';BASE=ROOT/'data'/'jbb_pair_partition_artifacts';ART=ROOT/'data'/'jbb_pair_anchored_partition_artifacts';CKPT=ROOT/'checkpoint';OUT=ROOT/'Report'/'jbb_pair_anchored_partition_audit.json';DEV='cuda' if torch.cuda.is_available() else 'cpu';TARGET='vicuna-13b-v1.5';ATTACKS=['DSN','GCG','JBC','prompt_with_random_search'];HOLD='PAIR';REJECT={3,40,43,47,67,82};SEED=20260826;EPOCHS=30;BATCH=256;TEMP=.07
class P(nn.Module):
 def __init__(self,w):super().__init__();self.c=nn.Linear(w,128);self.s=nn.Linear(w,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def pair_loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEV))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v**2).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def retrieve(x,q,b,ids):
 score=x[q]@x[b].T;order=np.argsort(-score,1);r=[]
 for n,o in enumerate(order):
  hit=np.where(ids[b][o]==ids[q[n]])[0];r.append(hit[0]+1 if len(hit) else len(b)+1)
 r=np.array(r);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'R@10':float((r<=10).mean()),'MRR':float((1/r).mean())}
def method_retrieve(x,idx,meth,ids):
 r=[]
 for q in idx:
  b=np.array([j for j in idx if j!=q and ids[j]!=ids[q]]);o=b[np.argsort(-(x[q]@x[b].T))];r.append(np.where(meth[o]==meth[q])[0][0]+1)
 r=np.array(r);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def probe(a,y,b,z):
 s=StandardScaler().fit(a);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(s.transform(a),y);return float((m.predict(s.transform(b))==z).mean())
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);old=np.load(BASE/'raw_layer8.npy');oldmeta=json.loads((BASE/'metadata.json').read_text());goals={}
 for method in ATTACKS:
  f=next((SRC/method).glob(f'*/*{TARGET}.json'))
  for r in json.loads(f.read_text(encoding='utf-8'))['jailbreaks']:
   if r.get('index') not in REJECT and r.get('goal'):goals.setdefault(r['index'],r['goal'])
 assert len(goals)==94
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;g=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEV).eval();parts=[];items=sorted(goals.items())
 with torch.inference_mode():
  for start in range(0,len(items),16):
   t=tok([v for _,v in items[start:start+16]],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEV);h=g(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);parts.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());print(f'canonical_extracted={min(start+16,len(items))}/{len(items)}')
 vanilla=np.concatenate(parts);rows=[{'behavior_id':i,'method':'vanilla'} for i,_ in items]+oldmeta;raw=np.concatenate([vanilla,old]);ids=np.array([r['behavior_id'] for r in rows]);meth=np.array([r['method'] for r in rows]);train=np.isin(meth,ATTACKS+['vanilla']);held=meth==HOLD;assert train.sum()==470 and held.sum()==78 and not np.any(meth[train]==HOLD)
 byid=defaultdict(list);bym=defaultdict(list)
 for i in np.where(train)[0]:byid[ids[i]].append(i);bym[meth[i]].append(i)
 assert all(len(v)==5 for v in byid.values()) and all(len(v)==94 for v in bym.values());x=torch.from_numpy(raw).to(DEV);model=P(raw.shape[1]).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);ti=np.where(train)[0];hist=[]
 for epoch in range(EPOCHS):
  totals=[0.,0.];steps=0
  for batch in np.array_split(rng.permutation(ti),max(1,len(ti)//BATCH)):
   cp=np.array([rng.choice([j for j in byid[ids[i]] if meth[j]!=meth[i]]) for i in batch]);cn=np.array([rng.choice([j for j in bym[meth[i]] if ids[j]!=ids[i]]) for i in batch]);zc,zs=model(x[batch]);zcp,_=model(x[cp]);zcn,_=model(x[cn]);_,zsp=model(x[cn]);_,zsn=model(x[cp]);lc,ls=pair_loss(zc,zcp,zcn),pair_loss(zs,zsp,zsn);opt.zero_grad();((lc+ls)/2).backward();opt.step();totals[0]+=lc.item();totals[1]+=ls.item();steps+=1
  hist.append({'epoch':epoch+1,'C_triplet_loss':totals[0]/steps,'S_triplet_loss':totals[1]/steps});print(f"epoch={epoch+1}/{EPOCHS} C_loss={hist[-1]['C_triplet_loss']:.4f} S_loss={hist[-1]['S_triplet_loss']:.4f}")
 model.eval()
 with torch.no_grad():zc,zs=model(x);zc,zs=zc.cpu().numpy(),zs.cpu().numpy()
 ART.mkdir(parents=True,exist_ok=True);np.save(ART/'raw_layer8.npy',raw);np.save(ART/'z_C.npy',zc);np.save(ART/'z_S.npy',zs);np.save(ART/'behavior_ids.npy',ids);np.save(ART/'methods.npy',meth);(ART/'metadata.json').write_text(json.dumps(rows,indent=2)+'\n');torch.save({'state_dict':model.state_dict(),'history':hist,'config':{'architecture':'2304 -> z_C(128) + z_S(128)','canonical_anchor':'vanilla goal is a training surface view','heldout_method':HOLD}},CKPT/'jbb_pair_anchored_partition_layer8.pt')
 behavior=np.unique(ids[train]);style_train=set(rng.choice(behavior,70,replace=False));sptr=np.array([i for i in ti if ids[i] in style_train]);spte=np.array([i for i in ti if ids[i] not in style_train]);report={'scope':{'target_model':TARGET,'canonical_anchor_rows':94,'training_methods':['vanilla']+ATTACKS,'heldout_method':HOLD,'training_rows':int(train.sum()),'heldout_PAIR_rows':int(held.sum()),'Gemma_layer':8,'pooling':'masked mean','SAE_trained':False,'ConCA_trained':False},'training':{'architecture':'2304 -> z_C(128) + z_S(128)','epochs':EPOCHS,'optimizer':'AdamW(lr=0.001, weight_decay=0.0001)','history':hist},'heldout_PAIR_goal_retrieval':{'raw':retrieve(raw,np.where(held)[0],ti,ids),'z_C':retrieve(zc,np.where(held)[0],ti,ids),'z_S':retrieve(zs,np.where(held)[0],ti,ids)},'heldout_PAIR_goal_probe_accuracy':{'raw':probe(raw[train],ids[train],raw[held],ids[held]),'z_C':probe(zc[train],ids[train],zc[held],ids[held]),'z_S':probe(zs[train],ids[train],zs[held],ids[held])},'seen_method_behavior_disjoint_probe_accuracy':{'raw':probe(raw[sptr],meth[sptr],raw[spte],meth[spte]),'z_C':probe(zc[sptr],meth[sptr],zc[spte],meth[spte]),'z_S':probe(zs[sptr],meth[sptr],zs[spte],meth[spte])},'seen_method_retrieval':{'raw':method_retrieve(raw,ti,meth,ids),'z_C':method_retrieve(zc,ti,meth,ids),'z_S':method_retrieve(zs,ti,meth,ids)},'heldout_PAIR_effective_rank':{'raw':rank(raw[held]),'z_C':rank(zc[held]),'z_S':rank(zs[held])},'note':'PAIR is an unseen style class. Style metrics therefore use behavior-disjoint held-out behaviors from the five seen styles; PAIR goal tests are direct.'};OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='training'},indent=2))
if __name__=='__main__':main()
