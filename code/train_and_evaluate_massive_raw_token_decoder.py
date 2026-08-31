"""Raw-token 2304→2304 Sinkhorn control."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]; TRAIN,CANON=ROOT/'data'/'massive_token_layer8_bridge_partition',ROOT/'data'/'massive_token_layer8_canonical_heldout'; CKPT,OUT=ROOT/'checkpoint'/'massive_raw_token_sinkhorn_decoder.pt',ROOT/'Report'/'massive_raw_token_sinkhorn_decoder_eval.json'; DEVICE,ENGLISH,HOLD,SEED,EPOCHS,BATCH,EPSILON,ITERS='cuda','en-US',('ar-SA','zh-CN'),20260827,5,16,5000.,30
def sinkhorn(decoded,target,source_mask,target_mask):
 cost=(decoded[:,:,None]-target[:,None,:]).square().sum(-1); kernel=(-cost/EPSILON).exp()*(source_mask[:,:,None]&target_mask[:,None,:]); a=source_mask/source_mask.sum(1,keepdim=True);b=target_mask/target_mask.sum(1,keepdim=True);u,v=torch.ones_like(a),torch.ones_like(b)
 for _ in range(ITERS):u=a/(kernel@v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12);v=b/(kernel.transpose(1,2)@u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
 loss=((u[:,:,None]*kernel*v[:,None,:])*cost).sum((1,2)).mean();assert torch.isfinite(loss);return loss
def score(query,target):
 q=query/np.linalg.norm(query,axis=1,keepdims=True);t=target/np.linalg.norm(target,axis=1,keepdims=True);return float((q@t.T).max(1).mean())
def main():
 assert torch.cuda.is_available();torch.manual_seed(SEED);rng=np.random.default_rng(SEED);raw,md=np.load(TRAIN/'train_tokens.npy',mmap_mode='r'),pd.read_csv(TRAIN/'train_metadata.csv');groups={k:v.index.to_numpy() for k,v in md.groupby(['id','locale'])};pairs=[(groups[(i,l)],groups[(i,ENGLISH)]) for i in sorted(set(md.id)) for l in sorted(set(md.loc[md.id==i,'locale'])-{ENGLISH})];assert len(pairs)==5568
 decoder=nn.Linear(2304,2304).to(DEVICE);opt=torch.optim.AdamW(decoder.parameters(),lr=3e-4,weight_decay=1e-4);history=[]
 for epoch in range(EPOCHS):
  losses=[]
  for chosen in np.array_split(rng.permutation(len(pairs)),max(1,len(pairs)//BATCH)):
   batch=[pairs[i] for i in chosen];ns,nt=max(len(p[0]) for p in batch),max(len(p[1]) for p in batch);source,target=np.zeros((len(batch),ns,2304),np.float32),np.zeros((len(batch),nt,2304),np.float32);sm,tm=np.zeros((len(batch),ns),bool),np.zeros((len(batch),nt),bool)
   for i,(si,ti) in enumerate(batch):source[i,:len(si)],target[i,:len(ti)],sm[i,:len(si)],tm[i,:len(ti)]=raw[si],raw[ti],True,True
   loss=sinkhorn(decoder(torch.from_numpy(source).to(DEVICE)),torch.from_numpy(target).to(DEVICE),torch.from_numpy(sm).to(DEVICE),torch.from_numpy(tm).to(DEVICE));opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  history.append(float(np.mean(losses)));print(f'epoch={epoch+1}/{EPOCHS} cost={history[-1]:.3f}')
 english,enmd=np.load(CANON/'en-US_tokens.npy'),pd.read_csv(CANON/'en-US_metadata.csv',dtype={'id':str});eg={k:v.index.to_numpy() for k,v in enmd.groupby('id')};rows=[];preds=[]
 with torch.no_grad():
  for locale in HOLD:
   x,meta=np.load(CANON/f'{locale}_tokens.npy'),pd.read_csv(CANON/f'{locale}_metadata.csv',dtype={'id':str});pred=np.concatenate([decoder(torch.from_numpy(x[i:i+4096]).to(DEVICE)).cpu().numpy() for i in range(0,len(x),4096)]);preds.append(pred)
   for i,g in meta.groupby('id'):idx=g.index.to_numpy();t=english[eg[i]];rows.append({'raw':score(x[idx],t),'pred':score(pred[idx],t),'coverage':score(t,pred[idx])})
 result=pd.DataFrame(rows);prediction=np.concatenate(preds);unit=prediction[:1000]/np.linalg.norm(prediction[:1000],axis=1,keepdims=True);cosine=unit@unit.T;np.fill_diagonal(cosine,0.)
 report={'raw_foreign_to_english_similarity':float(result.raw.mean()),'D_H_to_english_similarity':float(result.pred.mean()),'reverse_coverage':float(result.coverage.mean()),'prediction_norm':{'mean':float(np.linalg.norm(prediction,axis=1).mean()),'std':float(np.linalg.norm(prediction,axis=1).std())},'coordinate_std':float(prediction.std(0).mean()),'random_pair_cosine':float(cosine.sum()/(len(unit)*(len(unit)-1)))};torch.save({'state_dict':decoder.state_dict(),'sinkhorn_cost_history':history},CKPT);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
