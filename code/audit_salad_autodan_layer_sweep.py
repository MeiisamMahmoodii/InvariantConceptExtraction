"""Frozen Gemma layer sweep for AutoDAN goal separability; no partition training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'salad_attack_enhanced.parquet';ART=ROOT/'data'/'salad_autodan_partition_artifacts';OUT=ROOT/'Report'/'salad_autodan_layer_sweep.json';LAYERS=[5,8,13,21];TRAIN={'gcg_llama','gptfuzz','jb'};HOLD='autodan';DEVICE='cuda';SEED=20260827
def ranks(query,bank,ids):
 q=F.normalize(torch.from_numpy(query),dim=1).numpy();b=F.normalize(torch.from_numpy(bank),dim=1).numpy();o=np.argsort(-(q@b.T),axis=1);r=np.array([np.where(ids[o[i]]==ids[i])[0][0]+1 for i in range(len(ids))]);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'R@10':float((r<=10).mean()),'MRR':float((1/r).mean())}
def probe(train,train_y,test,test_y):
 s=StandardScaler().fit(train);m=LogisticRegression(max_iter=2000,random_state=SEED).fit(s.transform(train),train_y);return float((m.predict(s.transform(test))==test_y).mean())
def stat(v):
 v=np.asarray(v);return {'n':int(len(v)),'mean':float(v.mean()),'std':float(v.std()),'p05':float(np.quantile(v,.05)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95))}
def main():
 d=pd.read_parquet(DATA);n=d[d.method.isin(TRAIN)].groupby('qid').method.nunique();eligible=sorted(set(d.loc[d.method.eq(HOLD),'qid']) & set(n[n.ge(2)].index));assert len(eligible)==140
 attacks=d[d.qid.isin(eligible) & d.method.isin(TRAIN|{HOLD})][['qid','method','augq']].rename(columns={'augq':'text'}).sort_values(['qid','method']).reset_index(drop=True);base=d[d.qid.isin(eligible)].drop_duplicates('qid').set_index('qid').loc[eligible].reset_index()[['qid','baseq']].rename(columns={'baseq':'text'});rows=pd.concat([attacks.assign(kind='attack'),base.assign(method='baseq',kind='base')],ignore_index=True);tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE).eval();vec={layer:[] for layer in LAYERS}
 with torch.inference_mode():
  for start in range(0,len(rows),4):
   t=tok(rows.text.iloc[start:start+4].tolist(),padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEVICE);hs=model(**t,output_hidden_states=True,use_cache=False).hidden_states;m=t.attention_mask.unsqueeze(-1)
   for layer in LAYERS:vec[layer].append(((hs[layer]*m.to(hs[layer].dtype)).sum(1)/m.sum(1)).float().cpu().numpy())
   print(f'extracted={min(start+4,len(rows))}/{len(rows)}')
 qids=rows.qid.to_numpy();methods=rows.method.to_numpy();base_ix=np.where(rows.kind.eq('base'))[0];auto_ix=np.where(methods==HOLD)[0];train_ix=np.where(np.isin(methods,list(TRAIN)))[0];assert len(base_ix)==len(auto_ix)==140 and len(train_ix)==471 and np.array_equal(qids[base_ix],qids[auto_ix])
 report={'scope':{'qids':140,'training_methods':sorted(TRAIN),'heldout_method':'autodan','layers':LAYERS,'pooling':'masked mean','frozen_diagnostic_only':True,'partition_retrained':False},'metric_definition':{'goal_retrieval':'canonical baseq queries retrieve the matching AutoDAN qid','goal_probe':'linear probe trained on GCG/GPTFuzz/JB qids and tested on AutoDAN','cosine_margin':'baseq-to-matching-AutoDAN cosine minus different-qid AutoDAN-to-AutoDAN cosine'},'by_layer':{}}
 for layer in LAYERS:
  x=np.concatenate(vec[layer]);base=x[base_ix];auto=x[auto_ix];same=F.cosine_similarity(torch.from_numpy(base),torch.from_numpy(auto)).numpy();diff=[]
  for i in range(len(auto)):
   diff.extend(F.cosine_similarity(torch.from_numpy(auto[i:i+1]),torch.from_numpy(np.delete(auto,i,axis=0))).numpy())
  report['by_layer'][str(layer)]={'goal_retrieval_baseq_to_AutoDAN':ranks(base,auto,qids[auto_ix]),'goal_probe_train_methods_to_AutoDAN':probe(x[train_ix],qids[train_ix],x[auto_ix],qids[auto_ix]),'same_goal_baseq_to_AutoDAN_cosine':stat(same),'different_goal_same_AutoDAN_cosine':stat(diff),'mean_margin_same_goal_minus_different_goal':float(same.mean()-np.mean(diff))}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
