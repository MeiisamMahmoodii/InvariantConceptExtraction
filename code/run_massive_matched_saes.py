"""Matched frozen MASSIVE Top-k SAEs and held-out Arabic/Chinese intent-feature audit."""
import csv,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'data'/'massive_partition_artifacts'; OUT=ROOT/'data'/'massive_sae_artifacts'; CKPT=ROOT/'checkpoint'; REPORT=ROOT/'Report'/'massive_sae_intent_monitoring_audit.json'; DETAIL=ROOT/'Report'/'massive_sae_intent_feature_details'
SEED,EPOCHS,BATCH,K,EXPANSION=20260827,30,256,64,4; DEVICE='cuda'

class Partition(nn.Module):
 def __init__(self): super().__init__();self.c=nn.Linear(2304,128);self.s=nn.Linear(2304,128)
 def forward(self,x): return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
class TopKSAE(nn.Module):
 def __init__(self,w): super().__init__();self.e=nn.Linear(w,w*EXPANSION);self.d=nn.Linear(w*EXPANSION,w,bias=False);self.b=nn.Parameter(torch.zeros(w))
 def forward(self,x):
  dense=F.relu(self.e(x));v,i=torch.topk(dense,K,1);z=torch.zeros_like(dense).scatter(1,i,v);return z,self.d(z)+self.b
 def normalize(self):
  with torch.no_grad():self.d.weight.div_(self.d.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))

def moments(x):
 total=np.zeros(x.shape[1],np.float64);sq=total.copy()
 for start in range(0,len(x),4096):
  a=np.asarray(x[start:start+4096],np.float32);total+=a.sum(0);sq+=(a*a).sum(0)
 mean=total/len(x);return mean.astype(np.float32),np.sqrt(np.maximum(sq/len(x)-mean*mean,1e-12)).astype(np.float32)
def zc_train(raw):
 path=OUT/'z_C_train.npy'
 if path.exists():return np.load(path,mmap_mode='r')
 saved=torch.load(CKPT/'massive_partition_layer8.pt',map_location=DEVICE,weights_only=False);m=Partition().to(DEVICE);m.load_state_dict(saved['state_dict']);m.eval();out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(raw),128))
 with torch.inference_mode():
  for start in range(0,len(raw),4096):
   out[start:start+4096]=m(torch.from_numpy(np.asarray(raw[start:start+4096])).to(DEVICE))[0].cpu().numpy()
   if start%65536==0:print(f'z_C_train={min(start+4096,len(raw))}/{len(raw)}')
 return np.load(path,mmap_mode='r')
def run(model,x,mean,std,indices):
 pieces=[];mse=0.;model.eval()
 with torch.inference_mode():
  for start in range(0,len(indices),BATCH):
   a=(np.asarray(x[indices[start:start+BATCH]],np.float32)-mean)/std;t=torch.from_numpy(a).to(DEVICE);z,r=model(t);pieces.append(z.cpu().numpy());mse+=F.mse_loss(r,t,reduction='sum').item()
 return np.concatenate(pieces),mse/(len(indices)*x.shape[1])

def audit(name,x,train_intent,test_intent,test_id,test_lang,intent_labels):
 mean,std=moments(x);torch.manual_seed(SEED);rng=np.random.default_rng(SEED);m=TopKSAE(x.shape[1]).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 for epoch in range(EPOCHS):
  total=0.;steps=0
  for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
   a=(np.asarray(x[ids],np.float32)-mean)/std;t=torch.from_numpy(a).to(DEVICE);_,r=m(t);loss=F.mse_loss(r,t);opt.zero_grad();loss.backward();opt.step();m.normalize();total+=loss.item();steps+=1
  print(f'{name} epoch={epoch+1}/{EPOCHS} mse={total/steps:.6f}')
 width=x.shape[1]*EXPANSION;all_ids=np.arange(len(x));sums=np.zeros((58,width),np.float64);total=np.zeros(width,np.float64);counts=np.bincount(train_intent,minlength=58);train_mse=0.
 # Candidate selection is seen-language only. Class-versus-rest activation margin avoids scanning all 9,216 raw features with 58 full AUCs.
 for start in range(0,len(x),4096):
  ids=all_ids[start:start+4096];z,mse=run(m,x,mean,std,ids);total+=z.sum(0);train_mse+=mse*len(ids)
  for c in np.unique(train_intent[ids]):sums[c]+=z[train_intent[ids]==c].sum(0)
 pos=sums/counts[:,None];neg=(total[None,:]-sums)/(len(x)-counts)[:,None];features=(pos-neg).argmax(1)
 selected=np.empty((len(x),58),np.float32)
 for start in range(0,len(x),4096):
  ids=all_ids[start:start+4096];z,_=run(m,x,mean,std,ids);selected[start:start+len(ids)]=z[:,features]
 tx=np.load(ART/('raw_test_layer8.npy' if name=='raw' else 'z_C_test.npy'),mmap_mode='r');test_z,test_mse=run(m,tx,mean,std,np.arange(len(tx)))
 sparse=np.argpartition(test_z,-K,1)[:,-K:];np.savez_compressed(OUT/f'{name}_test_sparse.npz',indices=sparse.astype(np.int32),values=np.take_along_axis(test_z,sparse,1).astype(np.float32),shape=np.array(test_z.shape,dtype=np.int64))
 ar={test_id[i]:i for i in np.where(test_lang=='ar-SA')[0]};zh={test_id[i]:i for i in np.where(test_lang=='zh-CN')[0]};pairs=[(ar[k],zh[k]) for k in ar.keys()&zh.keys()];rows=[]
 for c,f in enumerate(features):
  yt=train_intent==c;st=selected[:,c];fp,tp,th=roc_curve(yt,st);threshold=th[np.argmax(tp-fp)];yh=test_intent==c;sh=test_z[:,f];pred=sh>=threshold;la=roc_auc_score(test_lang=='ar-SA',sh);left=np.array([test_z[i,f] for i,j in pairs]);right=np.array([test_z[j,f] for i,j in pairs]);stab=float(np.corrcoef(left,right)[0,1]) if left.std() and right.std() else 0.
  rows.append({'intent':int(intent_labels[c]),'feature_id':int(f),'seen_selection_margin':float(pos[c,f]-neg[c,f]),'heldout_auc':float(roc_auc_score(yh,sh)),'heldout_balanced_accuracy':float((pred[yh].mean()+(~pred[~yh]).mean())/2),'heldout_false_positive_rate':float(pred[~yh].mean()),'heldout_language_auc':float(la),'heldout_language_leakage':float(abs(la-.5)*2),'arabic_chinese_stability_correlation':stab})
 with (DETAIL/f'{name}_intent_features.csv').open('w',newline='',encoding='utf-8') as file:w=csv.DictWriter(file,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'config':{'k':K,'expansion':EXPANSION,'epochs':EPOCHS,'seed':SEED,'input_width':x.shape[1]}},CKPT/f'massive_topk_{name}_k{K}.pt')
 def avg(key):return float(np.mean([r[key] for r in rows]))
 return {'input_width':x.shape[1],'dictionary_width':width,'k':K,'train_reconstruction_mse':float(train_mse/len(x)),'heldout_reconstruction_mse':float(test_mse),'mean_L0':K,'selected_intents':58,'mean_heldout_intent_auc':avg('heldout_auc'),'mean_heldout_one_vs_rest_balanced_accuracy':avg('heldout_balanced_accuracy'),'mean_heldout_false_positive_rate':avg('heldout_false_positive_rate'),'mean_selected_feature_language_leakage':avg('heldout_language_leakage'),'mean_arabic_chinese_feature_stability':avg('arabic_chinese_stability_correlation'),'feature_details':str((DETAIL/f'{name}_intent_features.csv').relative_to(ROOT))}

def main():
 assert torch.cuda.is_available(),'GPU required';OUT.mkdir(exist_ok=True);DETAIL.mkdir(exist_ok=True);tr=pd.read_csv(ART/'train_metadata.csv');te=pd.read_csv(ART/'test_metadata.csv');raw=np.load(ART/'raw_train_layer8.npy',mmap_mode='r');zc=zc_train(raw);labels=np.sort(tr.intent.unique());assert len(labels)==58 and np.array_equal(labels,np.sort(te.intent.unique()));train_codes=np.searchsorted(labels,tr.intent.to_numpy());test_codes=np.searchsorted(labels,te.intent.to_numpy())
 report={'scope':{'dataset':'AmazonScience/massive','frozen_partition':'checkpoint/massive_partition_layer8.pt','train_languages':49,'heldout_languages':['ar-SA','zh-CN'],'Gemma_layer':8,'pooling':'masked mean','representations':['raw','z_C'],'TopK':K,'expansion':EXPANSION,'epochs':EPOCHS,'SAE_trained':True,'ConCA_trained':False},'selection':'One feature per intent is selected exclusively from seen-language training rows by class-versus-rest mean activation margin. Held-out Arabic/Chinese affect neither feature selection nor thresholds. Internal contiguous class codes preserve the original 58 MASSIVE intent labels in the output.','raw':audit('raw',raw,train_codes,test_codes,te.id.to_numpy(),te.locale.to_numpy(),labels),'z_C':audit('z_C',zc,train_codes,test_codes,te.id.to_numpy(),te.locale.to_numpy(),labels),'notes':'ConCA was not run: it is optional and would introduce a separate objective; this stage isolates the matched Top-k SAE test.'}
 REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
