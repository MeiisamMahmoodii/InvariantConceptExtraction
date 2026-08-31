"""Frozen Gemma layer-8 JBB partition; PAIR is completely unseen in training."""
import json,os,re
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
ROOT=Path(__file__).resolve().parents[1];SRC=Path(os.environ['TEMP'])/'jbb_artifacts_audit'/'attack-artifacts';ART=ROOT/'data'/'jbb_pair_partition_artifacts';CKPT=ROOT/'checkpoint';OUT=ROOT/'Report'/'jbb_pair_heldout_partition_audit.json';DEV='cuda' if torch.cuda.is_available() else 'cpu';TARGET='vicuna-13b-v1.5';TRAIN_METHODS=['DSN','GCG','JBC','prompt_with_random_search'];HOLD='PAIR';REJECT={3,40,43,47,67,82};SEED=20260826;EPOCHS=30;BATCH=256;TEMP=.07
class P(nn.Module):
 def __init__(self,w):super().__init__();self.c=nn.Linear(w,128);self.s=nn.Linear(w,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEV))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v**2).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def retrieval(x,q,b,ids):
 score=x[q]@x[b].T;order=np.argsort(-score,1);r=[]
 for n,o in enumerate(order):
  hits=np.where(ids[b][o]==ids[q[n]])[0];r.append(hits[0]+1 if len(hits) else len(b)+1)
 r=np.array(r);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'R@10':float((r<=10).mean()),'MRR':float((1/r).mean())}
def method_retrieval(x,idx,methods,ids):
 out=[]
 for q in idx:
  b=np.array([j for j in idx if j!=q and ids[j]!=ids[q]])
  o=b[np.argsort(-(x[q]@x[b].T))];hit=np.where(methods[o]==methods[q])[0][0]+1;out.append(hit)
 r=np.array(out);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def probe(train_x,train_y,test_x,test_y=None):
 if test_y is None:test_y=METHOD_TEST_LABELS
 sc=StandardScaler().fit(train_x);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(sc.transform(train_x),train_y);return float((m.predict(sc.transform(test_x))==test_y).mean())
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);rows=[]
 for method in TRAIN_METHODS+[HOLD]:
  f=next((SRC/method).glob(f'*/*{TARGET}.json'))
  for r in json.loads(f.read_text(encoding='utf-8'))['jailbreaks']:
   if r.get('index') not in REJECT and all(r.get(k) not in (None,'') for k in ('index','behavior','goal','prompt')):rows.append({'method':method,'behavior_id':r['index'],'prompt':r['prompt']})
 train=np.array([r['method'] in TRAIN_METHODS for r in rows]);held=np.array([r['method']==HOLD for r in rows]);ids=np.array([r['behavior_id'] for r in rows]);methods=np.array([r['method'] for r in rows]);assert not np.any(methods[train]==HOLD) and train.sum()==376 and held.sum()==78
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;g=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEV).eval();vec=[]
 with torch.inference_mode():
  for start in range(0,len(rows),8):
   t=tok([r['prompt'] for r in rows[start:start+8]],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEV);h=g(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);vec.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());print(f'extracted={min(start+8,len(rows))}/{len(rows)}')
 raw=np.concatenate(vec);x=torch.from_numpy(raw).to(DEV);byid=defaultdict(list);bym=defaultdict(list)
 for i in np.where(train)[0]:byid[ids[i]].append(i);bym[methods[i]].append(i)
 assert all(len(v)==4 for v in byid.values()) and all(len(v)==94 for v in bym.values())
 model=P(raw.shape[1]).to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);ti=np.where(train)[0];hist=[]
 for e in range(EPOCHS):
  total=[0.,0.];steps=0
  for batch in np.array_split(rng.permutation(ti),max(1,len(ti)//BATCH)):
   cp=np.array([rng.choice([j for j in byid[ids[i]] if methods[j]!=methods[i]]) for i in batch]);cn=np.array([rng.choice([j for j in bym[methods[i]] if ids[j]!=ids[i]]) for i in batch]);zc,zs=model(x[batch]);zcp,_=model(x[cp]);zcn,_=model(x[cn]);_,zsp=model(x[cn]);_,zsn=model(x[cp]);lc,ls=loss(zc,zcp,zcn),loss(zs,zsp,zsn);opt.zero_grad();((lc+ls)/2).backward();opt.step();total[0]+=lc.item();total[1]+=ls.item();steps+=1
  hist.append({'epoch':e+1,'C_triplet_loss':total[0]/steps,'S_triplet_loss':total[1]/steps});print(f"epoch={e+1}/{EPOCHS} C_loss={hist[-1]['C_triplet_loss']:.4f} S_loss={hist[-1]['S_triplet_loss']:.4f}")
 model.eval()
 with torch.no_grad():zc,zs=model(x);zc,zs=zc.cpu().numpy(),zs.cpu().numpy()
 ART.mkdir(parents=True,exist_ok=True);np.save(ART/'raw_layer8.npy',raw);np.save(ART/'z_C.npy',zc);np.save(ART/'z_S.npy',zs);np.save(ART/'behavior_ids.npy',ids);np.save(ART/'methods.npy',methods);(ART/'metadata.json').write_text(json.dumps([{'behavior_id':int(r['behavior_id']),'method':r['method']} for r in rows],indent=2)+'\n');torch.save({'state_dict':model.state_dict(),'history':hist,'config':{'architecture':'2304 -> z_C(128) + z_S(128)','C_positive':'same behavior, different training method','C_negative':'different behavior, same training method','S_positive':'different behavior, same training method','S_negative':'same behavior, different training method','heldout_method':HOLD}},CKPT/'jbb_pair_heldout_partition_layer8.pt')
 # Goal classifier: train only seen-method rows, test only unseen PAIR rows.
 seen_behavior=np.unique(ids[train]);split=set(rng.choice(seen_behavior,70,replace=False));mp_train=np.array([i for i in np.where(train)[0] if ids[i] in split]);mp_test=np.array([i for i in np.where(train)[0] if ids[i] not in split]);global METHOD_TEST_LABELS;METHOD_TEST_LABELS=methods[mp_test]
 report={'scope':{'target_model':TARGET,'strict_behavior_count':94,'training_methods':TRAIN_METHODS,'heldout_method':HOLD,'training_rows':int(train.sum()),'heldout_pair_rows':int(held.sum()),'Gemma_layer':8,'pooling':'masked mean','partition_retrained':True,'SAE_trained':False,'ConCA_trained':False},'training':{'architecture':'2304 -> z_C(128) + z_S(128)','epochs':EPOCHS,'optimizer':'AdamW(lr=0.001, weight_decay=0.0001)','history':hist},'heldout_PAIR_goal_retrieval':{'raw':retrieval(raw,np.where(held)[0],np.where(train)[0],ids),'z_C':retrieval(zc,np.where(held)[0],np.where(train)[0],ids),'z_S':retrieval(zs,np.where(held)[0],np.where(train)[0],ids)},'heldout_PAIR_goal_probe_accuracy':{'raw':probe(raw[train],ids[train],raw[held],ids[held]),'z_C':probe(zc[train],ids[train],zc[held],ids[held]),'z_S':probe(zs[train],ids[train],zs[held],ids[held])},'seen_method_behavior_disjoint_probe_accuracy':{'raw':probe(raw[mp_train],methods[mp_train],raw[mp_test],methods[mp_test]),'z_C':probe(zc[mp_train],methods[mp_train],zc[mp_test],methods[mp_test]),'z_S':probe(zs[mp_train],methods[mp_train],zs[mp_test])},'seen_method_retrieval':{'raw':method_retrieval(raw,ti,methods,ids),'z_C':method_retrieval(zc,ti,methods,ids),'z_S':method_retrieval(zs,ti,methods,ids)},'heldout_PAIR_effective_rank':{'raw':rank(raw[held]),'z_C':rank(zc[held]),'z_S':rank(zs[held])},'note':'PAIR is an unseen method class, so attack-method probes/retrieval are evaluated on behavior-disjoint held-out rows from the four seen methods; goal tests are evaluated directly on PAIR.'}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('training',)},indent=2))
if __name__=='__main__':main()
