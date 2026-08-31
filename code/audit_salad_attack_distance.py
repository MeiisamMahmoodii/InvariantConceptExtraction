"""Frozen Gemma layer-8 distance audit for aligned SALAD attack methods."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'salad_attack_enhanced.parquet';ART=ROOT/'data'/'salad_autodan_partition_artifacts';OUT=ROOT/'Report'/'salad_attack_distance_audit.json';BASE=ART/'baseq_layer8.npy';DEVICE='cuda';METHODS=['gcg_llama','gptfuzz','jb','autodan']
def summary(x):
 x=np.asarray(x);return {'n':int(len(x)),'mean':float(x.mean()),'std':float(x.std()),'p05':float(np.quantile(x,.05)),'median':float(np.median(x)),'p95':float(np.quantile(x,.95))}
def main():
 meta=json.loads((ART/'metadata.json').read_text());raw=np.load(ART/'raw_layer8.npy');qids=np.array([r['qid'] for r in meta]);methods=np.array([r['method'] for r in meta]);assert len(raw)==len(qids)==611
 train={'gcg_llama','gptfuzz','jb'};d=pd.read_parquet(DATA);n=d[d.method.isin(train)].groupby('qid').method.nunique();eligible=set(d.loc[d.method.eq('autodan'),'qid']) & set(n[n.ge(2)].index);assert len(eligible)==140
 base_text=d[d.qid.isin(eligible)].drop_duplicates('qid').set_index('qid').baseq.loc[sorted(eligible)]
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE).eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(base_text),8):
   t=tok(base_text.iloc[start:start+8].tolist(),padding=True,truncation=True,max_length=1024,return_tensors='pt').to(DEVICE);h=model(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);parts.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());print(f'base_extracted={min(start+8,len(base_text))}/{len(base_text)}')
 base=np.concatenate(parts);np.save(BASE,base);base_by_qid=dict(zip(base_text.index,base));attack={}
 for method in METHODS:
  attack[method]={q:raw[i] for i,(q,m) in enumerate(zip(qids,methods)) if m==method}
  assert len(attack[method])==140 if method=='autodan' else len(attack[method])>0
 base_to_attack={m:summary([F.cosine_similarity(torch.from_numpy(base_by_qid[q])[None],torch.from_numpy(h)[None]).item() for q,h in values.items()]) for m,values in attack.items()}
 attacked_pairs={}
 for a,b in combinations(METHODS,2):
  shared=sorted(set(attack[a]) & set(attack[b]));attacked_pairs[f'{a}__{b}']=summary([F.cosine_similarity(torch.from_numpy(attack[a][q])[None],torch.from_numpy(attack[b][q])[None]).item() for q in shared])
 report={'scope':{'dataset':'OpenSafetyLab/Salad-Data attack_enhanced_set','qids':140,'Gemma_layer':8,'pooling':'masked mean','frozen_diagnostic_only':True,'partition_retrained':False},'canonical_baseq_to_attack_cosine':base_to_attack,'same_qid_attacked_to_attacked_cosine':attacked_pairs,'AutoDAN_minus_training_base_cosine_mean':{m:float(base_to_attack['autodan']['mean']-base_to_attack[m]['mean']) for m in METHODS if m!='autodan'}}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
