"""Train only a frozen MASSIVE z_C -> English canonical layer-8 decoder."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
_stub=torch.library.Library('torchvision','DEF');_stub.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
from transformers import AutoModel,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data'/'massive_partition_artifacts';OUT=ROOT/'data'/'massive_decoder_artifacts';CKPT=ROOT/'checkpoint'/'massive_zc_to_english_h_decoder_layer8.pt';REPORT=ROOT/'Report'/'massive_zc_to_english_decoder_audit.json';TEST=ROOT/'data'/'massive_all_test.parquet'
SEED,EPOCHS,BATCH,LR=20260827,100,512,1e-3;DEVICE='cuda';HOLD=['ar-SA','zh-CN'];STABLE=set(range(60))-{29,37}

def english_activations(texts):
 path=OUT/'english_canonical_test_layer8.npy'
 if path.exists():return np.load(path,mmap_mode='r')
 tok=AutoTokenizer.from_pretrained('google/gemma-2-2b',local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;model=AutoModel.from_pretrained('google/gemma-2-2b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(DEVICE).eval();out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(texts),2304))
 with torch.inference_mode():
  for start in range(0,len(texts),64):
   t=tok(texts[start:start+64],padding=True,truncation=True,max_length=128,return_tensors='pt').to(DEVICE);h=model(**t,output_hidden_states=True,use_cache=False).hidden_states[8];mask=t.attention_mask.unsqueeze(-1);out[start:start+len(t.input_ids)]=((h*mask.to(h.dtype)).sum(1)/mask.sum(1)).float().cpu().numpy()
   if start%1024==0:print(f'english_targets={min(start+64,len(texts))}/{len(texts)}')
 del model;torch.cuda.empty_cache();return np.load(path,mmap_mode='r')
def stats(pred,target,raw):
 cos=F.cosine_similarity(pred,target).cpu().numpy();base=F.cosine_similarity(raw,target).cpu().numpy();mse=((pred-target)**2).mean(1).cpu().numpy()
 def norms(x):
  a=x.norm(dim=1).cpu().numpy();return {'mean':float(a.mean()),'std':float(a.std())}
 return {'decoded_MSE':{'mean':float(mse.mean()),'std':float(mse.std())},'decoded_cosine':{'mean':float(cos.mean()),'std':float(cos.std())},'raw_to_English_cosine':{'mean':float(base.mean()),'std':float(base.std())},'cosine_improvement':{'mean':float((cos-base).mean()),'std':float((cos-base).std()),'fraction_positive':float((cos>base).mean())},'activation_norms':{'raw_H':norms(raw),'English_canonical_H_C':norms(target),'decoded_H_C_prime':norms(pred)}}
def main():
 assert torch.cuda.is_available(),'GPU required';torch.manual_seed(SEED);np.random.seed(SEED);OUT.mkdir(exist_ok=True)
 tr=pd.read_csv(ART/'train_metadata.csv');te=pd.read_csv(ART/'test_metadata.csv');raw_tr=np.load(ART/'raw_train_layer8.npy',mmap_mode='r');zc_tr=np.load(OUT.parent/'massive_sae_artifacts'/'z_C_train.npy',mmap_mode='r');raw_hold=np.load(ART/'raw_test_layer8.npy',mmap_mode='r');zc_hold=np.load(ART/'z_C_test.npy',mmap_mode='r')
 en_train=tr[tr.locale.eq('en-US')];assert en_train.id.is_unique and len(en_train)==tr.id.nunique();english_row=pd.Series(en_train.index.to_numpy(),index=en_train.id);target_rows=tr.id.map(english_row).to_numpy();train_mask=tr.locale.ne('en-US').to_numpy();assert train_mask.sum()==len(tr)-len(en_train)
 full=pd.read_parquet(TEST);eng=full[(full.locale=='en-US')&full.intent.isin(STABLE)].sort_values('id').reset_index(drop=True);assert len(eng)==te.id.nunique();english=english_activations(eng.utt.tolist());eng_pos=pd.Series(np.arange(len(eng)),index=eng.id.astype(str));held_english_rows=te.id.astype(str).map(eng_pos);assert held_english_rows.notna().all();target_hold=np.asarray(english[held_english_rows.to_numpy(dtype=np.int64)])
 x=torch.from_numpy(np.asarray(zc_tr[train_mask])).to(DEVICE);y=torch.from_numpy(np.asarray(raw_tr[target_rows[train_mask]])).to(DEVICE);decoder=nn.Linear(128,2304).to(DEVICE);opt=torch.optim.Adam(decoder.parameters(),lr=LR);rng=np.random.default_rng(SEED);history=[]
 for epoch in range(EPOCHS):
  losses=[]
  for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
   loss=F.mse_loss(decoder(x[batch]),y[batch]);opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  history.append(float(np.mean(losses)));print(f'epoch={epoch+1}/{EPOCHS} train_MSE={history[-1]:.6f}')
 decoder.eval();tx=torch.from_numpy(np.asarray(zc_hold)).to(DEVICE);ty=torch.from_numpy(target_hold).to(DEVICE);th=torch.from_numpy(np.asarray(raw_hold)).to(DEVICE)
 with torch.no_grad():pred=decoder(tx)
 result={'all_heldout':stats(pred,ty,th)}
 for lang in HOLD:
  mask=torch.from_numpy(te.locale.to_numpy()==lang).to(DEVICE);result[lang]=stats(pred[mask],ty[mask],th[mask])
 np.save(OUT/'decoded_arabic_chinese_to_english.npy',pred.cpu().numpy());np.save(OUT/'english_canonical_targets_aligned.npy',target_hold);te[['id','locale','intent']].to_csv(OUT/'heldout_metadata.csv',index=False);torch.save({'state_dict':decoder.state_dict(),'config':{'architecture':'Linear(128,2304)','input':'frozen MASSIVE z_C','target':'frozen Gemma layer-8 masked-mean H(id,en-US)','epochs':EPOCHS,'lr':LR},'history':history},CKPT)
 report={'scope':{'frozen_Gemma_layer':8,'pooling':'masked mean','frozen_partition':'checkpoint/massive_partition_layer8.pt','z_C_dimension':128,'decoder':'Linear(128,2304)','decoder_only_training':True,'SAE_trained':False,'ConCA_trained':False,'GemmaScope_trained':False},'canonical_surface':{'locale':'en-US'},'split':{'decoder_train':'49 seen-language published train rows except en-US','heldout_evaluation':'Arabic and Chinese published test rows','train_rows':int(train_mask.sum()),'heldout_rows':len(te),'heldout_ids':int(te.id.nunique())},'training':{'epochs':EPOCHS,'optimizer':'Adam(lr=0.001)','loss':'MSE(D_C(z_C(id,other seen language)), H(id,en-US))','final_train_MSE':history[-1]},'heldout_metrics':result,'artifacts':{'checkpoint':str(CKPT.relative_to(ROOT)),'arrays':str(OUT.relative_to(ROOT))}}
 REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
