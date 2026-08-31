"""Frozen MASSIVE baselines: raw H -> English H and mean-English H."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data'/'massive_partition_artifacts';DECART=ROOT/'data'/'massive_decoder_artifacts';CKPT=ROOT/'checkpoint'/'massive_raw_to_english_h_baseline_layer8.pt';REPORT=ROOT/'Report'/'massive_raw_to_english_baselines_audit.json'
SEED,EPOCHS,BATCH,LR=20260827,100,512,1e-3;DEVICE='cuda'
def stats(pred,target,raw):
 cos=F.cosine_similarity(pred,target).cpu().numpy();base=F.cosine_similarity(raw,target).cpu().numpy();mse=((pred-target)**2).mean(1).cpu().numpy()
 def norms(x):
  a=x.norm(dim=1).cpu().numpy();return {'mean':float(a.mean()),'std':float(a.std())}
 return {'MSE':{'mean':float(mse.mean()),'std':float(mse.std())},'cosine':{'mean':float(cos.mean()),'std':float(cos.std())},'raw_to_English_cosine':{'mean':float(base.mean()),'std':float(base.std())},'cosine_improvement':{'mean':float((cos-base).mean()),'std':float((cos-base).std()),'fraction_positive':float((cos>base).mean())},'activation_norms':{'raw_H':norms(raw),'English_canonical_H_C':norms(target),'prediction':norms(pred)}}
def main():
 assert torch.cuda.is_available(),'GPU required';torch.manual_seed(SEED);np.random.seed(SEED)
 tr=pd.read_csv(ART/'train_metadata.csv');te=pd.read_csv(ART/'test_metadata.csv');raw_tr=np.load(ART/'raw_train_layer8.npy',mmap_mode='r');raw_hold=np.load(ART/'raw_test_layer8.npy',mmap_mode='r');targets=np.load(DECART/'english_canonical_targets_aligned.npy');zc_pred=np.load(DECART/'decoded_arabic_chinese_to_english.npy')
 en=tr[tr.locale.eq('en-US')];assert en.id.is_unique and len(en)==tr.id.nunique();row_for_id=pd.Series(en.index.to_numpy(),index=en.id);target_rows=tr.id.map(row_for_id).to_numpy();mask=tr.locale.ne('en-US').to_numpy();x=torch.from_numpy(np.asarray(raw_tr[mask])).to(DEVICE);y=torch.from_numpy(np.asarray(raw_tr[target_rows[mask]])).to(DEVICE);model=nn.Linear(2304,2304).to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=LR);rng=np.random.default_rng(SEED);history=[]
 for epoch in range(EPOCHS):
  losses=[]
  for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
   loss=F.mse_loss(model(x[batch]),y[batch]);opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  history.append(float(np.mean(losses)));print(f'epoch={epoch+1}/{EPOCHS} train_MSE={history[-1]:.6f}')
 model.eval();th=torch.from_numpy(np.asarray(raw_hold)).to(DEVICE);ty=torch.from_numpy(targets).to(DEVICE)
 with torch.no_grad():pred=model(th);mean_pred=torch.from_numpy(np.asarray(raw_tr[en.index]).mean(0,keepdims=True)).to(DEVICE).expand_as(th);zc=torch.from_numpy(zc_pred).to(DEVICE)
 result={'raw_linear_D_H':stats(pred,ty,th),'mean_English_constant':stats(mean_pred,ty,th),'existing_D_C_z_C':stats(zc,ty,th)}
 for lang in ('ar-SA','zh-CN'):
  m=torch.from_numpy(te.locale.to_numpy()==lang).to(DEVICE);result[lang]={'raw_linear_D_H':stats(pred[m],ty[m],th[m]),'mean_English_constant':stats(mean_pred[m],ty[m],th[m]),'existing_D_C_z_C':stats(zc[m],ty[m],th[m])}
 torch.save({'state_dict':model.state_dict(),'config':{'architecture':'Linear(2304,2304)','input':'frozen raw Gemma layer-8 activation','target':'frozen H(id,en-US)','epochs':EPOCHS,'lr':LR},'history':history},CKPT)
 report={'scope':{'frozen_Gemma_layer':8,'pooling':'masked mean','frozen_partition_used_only_for_existing_D_C':True,'baseline_decoder':'Linear(2304,2304)','decoder_only_training':True,'SAE_trained':False,'ConCA_trained':False,'GemmaScope_trained':False},'split':{'train':'same 49-language published train rows excluding English','heldout':'same Arabic/Chinese published test rows','train_rows':int(mask.sum()),'heldout_rows':len(te)},'training':{'epochs':EPOCHS,'optimizer':'Adam(lr=0.001)','loss':'MSE(D_H(H(id,other seen language)), H(id,en-US))','final_train_MSE':history[-1]},'heldout_metrics':result}
 REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
