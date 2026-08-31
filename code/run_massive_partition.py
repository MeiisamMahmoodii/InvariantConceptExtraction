"""One full MASSIVE frozen-Gemma layer-8 partition run; no SAE or ConCA."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];TRAIN_FILE=ROOT/'data'/'massive_all_train.parquet';TEST_FILE=ROOT/'data'/'massive_all_test.parquet';ART=ROOT/'data'/'massive_partition_artifacts';CKPT=ROOT/'checkpoint'/'massive_partition_layer8.pt';OUT=ROOT/'Report'/'massive_partition_audit.json';SEED=20260827;EPOCHS=30;BATCH=256;LR=1e-3;TEMP=.07;DEVICE='cuda';HOLD=['ar-SA','zh-CN'];STABLE=set(range(60))-{29,37}
class P(nn.Module):
 def __init__(self):super().__init__();self.c=nn.Linear(2304,128);self.s=nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
def loss(a,p,n):return F.cross_entropy(torch.stack(((a*p).sum(-1),(a*n).sum(-1)),1)/TEMP,torch.zeros(len(a),dtype=torch.long,device=DEVICE))
def rank(x):
 v=np.linalg.svd(x-x.mean(0),compute_uv=False)**2;p=v/v.sum();return {'participation_ratio':float(v.sum()**2/(v**2).sum()),'entropy_effective_rank':float(np.exp(-(p*np.log(p+1e-30)).sum()))}
def extract(texts,path):
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE).eval();out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(texts),2304))
 with torch.inference_mode():
  for start in range(0,len(texts),64):
   t=tok(texts[start:start+64],padding=True,truncation=True,max_length=128,return_tensors='pt').to(DEVICE);h=model(**t,output_hidden_states=True,use_cache=False).hidden_states[8];m=t.attention_mask.unsqueeze(-1);out[start:start+len(t.input_ids)]=((h*m.to(h.dtype)).sum(1)/m.sum(1)).float().cpu().numpy()
   if start%4096==0 or start+64>=len(texts):print(f'extracted={min(start+64,len(texts))}/{len(texts)}')
 del model;torch.cuda.empty_cache();return np.load(path,mmap_mode='r')
def encode(model,raw):
 out=np.empty((len(raw),128),dtype='float32');model.eval()
 with torch.no_grad():
  for start in range(0,len(raw),4096):out[start:start+4096]=model(torch.from_numpy(np.asarray(raw[start:start+4096])).to(DEVICE))[0].cpu().numpy()
 return out
def linear_probe(train_x,train_y,test_x,test_y,kind):
 scale=StandardScaler().fit(train_x)
 if kind=='intent':model=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=1000,tol=1e-3,random_state=SEED)
 else:model=LogisticRegression(max_iter=2000,random_state=SEED)
 model.fit(scale.transform(train_x),train_y);return float((model.predict(scale.transform(test_x))==test_y).mean())
def cross_retrieval(x,ids,langs):
 a=np.where(langs==HOLD[0])[0];b=np.where(langs==HOLD[1])[0];q=F.normalize(torch.from_numpy(x[a]),dim=1).numpy();k=F.normalize(torch.from_numpy(x[b]),dim=1).numpy();o=np.argsort(-(q@k.T),1);r=np.array([np.where(ids[b][row]==ids[a[n]])[0][0]+1 for n,row in enumerate(o)]);return {'R@1':float((r==1).mean()),'R@5':float((r<=5).mean()),'MRR':float((1/r).mean())}
def same_language_nn(x,ids,langs):
 z=F.normalize(torch.from_numpy(x),dim=1).numpy();score=z@z.T;np.fill_diagonal(score,-np.inf);score[ids[:,None]==ids[None,:]]=-np.inf;nn=score.argmax(1);return float((langs[nn]==langs).mean())
def main():
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED);train=pd.read_parquet(TRAIN_FILE);test=pd.read_parquet(TEST_FILE);train=train[(~train.locale.isin(HOLD))&train.intent.isin(STABLE)].sort_values(['id','locale']).reset_index(drop=True);test=test[test.locale.isin(HOLD)&test.intent.isin(STABLE)].sort_values(['id','locale']).reset_index(drop=True);assert len(train)==563108 and len(test)==5936 and train.groupby('id').size().eq(49).all() and test.groupby('id').size().eq(2).all();ART.mkdir(parents=True,exist_ok=True);train[['id','locale','intent']].to_csv(ART/'train_metadata.csv',index=False);test[['id','locale','intent']].to_csv(ART/'test_metadata.csv',index=False)
 raw_train=extract(train.utt.tolist(),ART/'raw_train_layer8.npy');raw_test=extract(test.utt.tolist(),ART/'raw_test_layer8.npy');id_code,_=pd.factorize(train.id,sort=True);lang_code,_=pd.factorize(train.locale,sort=True);grid=np.empty((train.id.nunique(),49),dtype=np.int64);grid[id_code,lang_code]=np.arange(len(train));assert np.unique(grid).size==len(train)
 model=P().to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4);history=[]
 for epoch in range(EPOCHS):
  totals=[0.,0.];steps=0
  for batch in np.array_split(rng.permutation(len(train)),max(1,len(train)//BATCH)):
   ids_b=id_code[batch];langs_b=lang_code[batch];other_lang=(langs_b+rng.integers(1,49,len(batch)))%49;other_id=(ids_b+rng.integers(1,len(grid),len(batch)))%len(grid);cp=grid[ids_b,other_lang];cn=grid[other_id,langs_b];a=torch.from_numpy(np.asarray(raw_train[batch])).to(DEVICE);p=torch.from_numpy(np.asarray(raw_train[cp])).to(DEVICE);n=torch.from_numpy(np.asarray(raw_train[cn])).to(DEVICE);zc,zs=model(a);zcp,_=model(p);zcn,_=model(n);_,zsp=model(n);_,zsn=model(p);lc,ls=loss(zc,zcp,zcn),loss(zs,zsp,zsn);opt.zero_grad();((lc+ls)/2).backward();opt.step();totals[0]+=lc.item();totals[1]+=ls.item();steps+=1
  history.append({'epoch':epoch+1,'C_triplet_loss':totals[0]/steps,'S_triplet_loss':totals[1]/steps});print(f'epoch={epoch+1}/{EPOCHS} C_loss={history[-1]["C_triplet_loss"]:.4f} S_loss={history[-1]["S_triplet_loss"]:.4f}')
 zc_test=encode(model,raw_test);model.eval()
 with torch.no_grad():
  zs_test=np.empty_like(zc_test)
  for start in range(0,len(raw_test),4096):zs_test[start:start+4096]=model(torch.from_numpy(np.asarray(raw_test[start:start+4096])).to(DEVICE))[1].cpu().numpy()
 np.save(ART/'z_C_test.npy',zc_test);np.save(ART/'z_S_test.npy',zs_test);ids_test=test.id.to_numpy();langs_test=test.locale.to_numpy();intent_test=test.intent.to_numpy()
 # Fixed, balanced diagnostic probe set: 100 seen-language train examples per intent.
 probe=[]
 for intent in sorted(STABLE):probe.extend(rng.choice(np.where(train.intent.to_numpy()==intent)[0],100,replace=False))
 probe=np.array(probe);raw_probe=np.asarray(raw_train[probe]);zc_probe=encode(model,raw_probe);model.eval()
 with torch.no_grad():zs_probe=model(torch.from_numpy(raw_probe).to(DEVICE))[1].cpu().numpy()
 half=rng.permutation(len(test))[:len(test)//2];other=np.setdiff1d(np.arange(len(test)),half);report={'scope':{'dataset':'AmazonScience/massive','seen_languages':49,'heldout_languages':HOLD,'shared_intents':58,'train_rows':len(train),'heldout_test_rows':len(test),'Gemma_layer':8,'pooling':'masked mean','architecture':'2304 -> z_C(128) + z_S(128)','epochs':EPOCHS,'SAE_trained':False,'ConCA_trained':False},'pair_training':{'C':'exact MASSIVE id','S':'locale','C_positive':'same id, different seen locale','C_negative':'different id, same seen locale'},'intent_probe_accuracy_seen_train_to_heldout_test':{'raw':linear_probe(raw_probe,train.intent.to_numpy()[probe],np.asarray(raw_test),intent_test,'intent'),'z_C':linear_probe(zc_probe,train.intent.to_numpy()[probe],zc_test,intent_test,'intent'),'z_S':linear_probe(zs_probe,train.intent.to_numpy()[probe],zs_test,intent_test,'intent')},'heldout_language_probe_accuracy':{'raw':linear_probe(np.asarray(raw_test)[half],langs_test[half],np.asarray(raw_test)[other],langs_test[other],'language'),'z_C':linear_probe(zc_test[half],langs_test[half],zc_test[other],langs_test[other],'language'),'z_S':linear_probe(zs_test[half],langs_test[half],zs_test[other],langs_test[other],'language')},'same_ID_cross_language_retrieval':{'raw':cross_retrieval(np.asarray(raw_test),ids_test,langs_test),'z_C':cross_retrieval(zc_test,ids_test,langs_test),'z_S':cross_retrieval(zs_test,ids_test,langs_test)},'same_language_different_ID_nearest_neighbor_fraction':{'raw':same_language_nn(np.asarray(raw_test),ids_test,langs_test),'z_C':same_language_nn(zc_test,ids_test,langs_test),'z_S':same_language_nn(zs_test,ids_test,langs_test)},'heldout_effective_rank':{'raw':rank(np.asarray(raw_test)),'z_C':rank(zc_test),'z_S':rank(zs_test)},'training':{'optimizer':'AdamW(lr=0.001, weight_decay=0.0001)','loss':'two matched cosine contrastive objectives','history':history,'intent_probe_train':'fixed balanced 100 examples per intent from seen-language train rows'}};torch.save({'state_dict':model.state_dict(),'history':history,'config':report['scope']},CKPT);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('training',)},indent=2))
if __name__=='__main__':main()
