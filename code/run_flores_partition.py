"""One fixed FLORES language-split partition run; no SAE."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_stub=torch.library.Library("torchvision","DEF");_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"Report"/"flores_partition_audit.json"; CKPT=ROOT/"checkpoint"; ART=ROOT/"data"/"flores_partition_artifacts"; DEV="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_LANG=("eng_Latn","fra_Latn","spa_Latn","deu_Latn","rus_Cyrl","hin_Deva","swh_Latn","tur_Latn"); HOLD=("arb_Arab","zho_Hans"); SEED=20260825; TEMP=.07; EPOCHS=30; BATCH=256
class P(nn.Module):
 def __init__(self): super().__init__();self.c=nn.Linear(2304,128);self.s=nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEV))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {"participation_ratio":float(v.sum()**2/(v**2).sum()),"entropy_effective_rank":float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def lang_probe(x,labels):
 rng=np.random.default_rng(SEED);idx=np.arange(len(labels));rng.shuffle(idx);cut=len(idx)//2;sc=StandardScaler().fit(x[idx[:cut]]);m=LogisticRegression(max_iter=1000,random_state=SEED).fit(sc.transform(x[idx[:cut]]),labels[idx[:cut]]);return float(np.mean(m.predict(sc.transform(x[idx[cut:]]))==labels[idx[cut:]]))
def retrieve(x, ids, languages):
 a=np.array([i for i,l in enumerate(languages) if l==HOLD[0]]);b=np.array([i for i,l in enumerate(languages) if l==HOLD[1]]);scores=x[a]@x[b].T;order=np.argsort(-scores,1);r=np.array([np.where(ids[b][o]==ids[a][i])[0][0]+1 for i,o in enumerate(order)]);return {"R@1":float(np.mean(r==1)),"R@5":float(np.mean(r<=5)),"MRR":float(np.mean(1/r))}
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);ds=load_dataset("facebook/flores","all",split="dev").select(range(200)); langs=TRAIN_LANG+HOLD; texts=[r[f"sentence_{l}"] for r in ds for l in langs]; ids=np.array([i for i in range(200) for _ in langs]); labels=np.array([l for _ in ds for l in langs]); splits=np.array(["train" if i<140 else "val" if i<170 else "test" for i in range(200) for _ in langs])
 tok=AutoTokenizer.from_pretrained("google/gemma-2-2b",local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;g=AutoModel.from_pretrained("google/gemma-2-2b",local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval();vec=[]
 with torch.inference_mode():
  for s in range(0,len(texts),32):
   t=tok(texts[s:s+32],padding=True,truncation=True,max_length=128,return_tensors="pt").to(DEV);h=g(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);vec.append(((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy());
   if s%320==0:print(f"extracted={s}/{len(texts)}")
 x=np.concatenate(vec);train=np.array([(splits[i]=="train" and labels[i] in TRAIN_LANG) for i in range(len(x))]);byid={};bylang={}
 for i in np.where(train)[0]:byid.setdefault(ids[i],[]).append(i);bylang.setdefault(labels[i],[]).append(i)
 model=P().to(DEV);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);trainidx=np.where(train)[0]
 for e in range(EPOCHS):
  for batch in np.array_split(rng.permutation(trainidx),max(1,len(trainidx)//BATCH)):
   cp=np.array([rng.choice([j for j in byid[ids[i]] if labels[j]!=labels[i]]) for i in batch]);cn=np.array([rng.choice([j for j in bylang[labels[i]] if ids[j]!=ids[i]]) for i in batch]);z,ss=model(torch.from_numpy(x[batch]).to(DEV));zp,_=model(torch.from_numpy(x[cp]).to(DEV));zn,_=model(torch.from_numpy(x[cn]).to(DEV));_,sp=model(torch.from_numpy(x[cn]).to(DEV));_,sn=model(torch.from_numpy(x[cp]).to(DEV));L=(loss(z,zp,zn)+loss(ss,sp,sn))/2;opt.zero_grad();L.backward();opt.step()
  print(f"epoch={e+1}/{EPOCHS} loss={L.item():.4f}")
 with torch.no_grad():zc,zs=model(torch.from_numpy(x).to(DEV));zc,zs=zc.cpu().numpy(),zs.cpu().numpy()
 ART.mkdir(parents=True,exist_ok=True);np.save(ART/"raw_layer8.npy",x);np.save(ART/"z_C.npy",zc);np.save(ART/"z_S.npy",zs);np.save(ART/"sentence_ids.npy",ids);np.save(ART/"languages.npy",labels);np.save(ART/"splits.npy",splits)
 test=np.array([(splits[i]=="test" and labels[i] in HOLD) for i in range(len(x))]);report={"language_split":{"train":TRAIN_LANG,"heldout":HOLD},"sentence_split":{"train":140,"val":30,"test":30},"training":"same sentence/different language positive; different sentence/same language negative","z_C":{"heldout_language_probe":lang_probe(zc[test],labels[test]),"sentence_retrieval":retrieve(zc[test],ids[test],labels[test]),"rank":rank(zc[test])},"z_S":{"heldout_language_probe":lang_probe(zs[test],labels[test]),"sentence_retrieval":retrieve(zs[test],ids[test],labels[test]),"rank":rank(zs[test])},"raw":{"heldout_language_probe":lang_probe(x[test],labels[test])},"SAE_trained":False};OUT.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
