"""Train only a frozen-z_C to canonical-Gemma-activation linear decoder."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
ACT=ROOT/'data'/'activations_three_domain_natural_rewrite'
META=ACT/'gemma2_2b_layer_sweep_metadata.csv'; RAW=ACT/'gemma2_2b_layer8_mean'/'activations.npy'; ZC=ROOT/'checkpoint'/'cs_partition_c_all.npy'
CKPT=ROOT/'checkpoint'/'z_c_to_canonical_h_decoder_layer8.pt'; OUT=ROOT/'Report'/'z_c_canonical_activation_decoder_audit.json'; ART=ROOT/'data'/'z_c_canonical_activation_decoder_artifacts'
SEED=20260826; EPOCHS=100; BATCH=512; LR=1e-3; DEVICE='cuda'

def metrics(pred,target,raw):
    cos=F.cosine_similarity(pred,target).cpu().numpy(); base=F.cosine_similarity(raw,target).cpu().numpy(); mse=((pred-target)**2).mean(1).cpu().numpy()
    def norm(x):
        a=x.norm(dim=1).cpu().numpy();return {'mean':float(a.mean()),'std':float(a.std())}
    delta=cos-base
    return {'decoded_to_canonical_MSE':{'mean':float(mse.mean()),'std':float(mse.std())},'decoded_to_canonical_cosine':{'mean':float(cos.mean()),'std':float(cos.std())},'raw_heldout_to_canonical_cosine':{'mean':float(base.mean()),'std':float(base.std())},'cosine_improvement':{'mean':float(delta.mean()),'std':float(delta.std()),'fraction_positive':float((delta>0).mean())},'activation_norms':{'raw_heldout_H':norm(raw),'canonical_H_C':norm(target),'decoded_H_C_prime':norm(pred)}}

def main():
    torch.manual_seed(SEED);np.random.seed(SEED)
    meta=pd.read_csv(META);raw=np.load(RAW,mmap_mode='r');zc=np.load(ZC,mmap_mode='r')
    assert len(meta)==len(raw)==len(zc)==51324
    canonical=meta[(meta.S_family=='declarative') & (meta.S_variant=='v1')].set_index('fact_id').activation_row
    assert canonical.index.is_unique and len(canonical)==meta.fact_id.nunique()
    target_rows=meta.fact_id.map(canonical).to_numpy()
    canonical_row=meta.activation_row.to_numpy()==target_rows
    train_mask=(meta.C_split.eq('C_train') & meta.S_split.eq('S_train') & ~canonical_row).to_numpy()
    test_mask=(meta.C_split.eq('C_test') & meta.S_family.eq('indirect') & meta.S_variant.eq('v1')).to_numpy()
    assert train_mask.sum()>0 and test_mask.sum()>0 and not (meta.S_family[test_mask]=='declarative').any()
    x=torch.from_numpy(np.asarray(zc[train_mask])).to(DEVICE);y=torch.from_numpy(np.asarray(raw[target_rows[train_mask]])).to(DEVICE)
    tx=torch.from_numpy(np.asarray(zc[test_mask])).to(DEVICE);ty=torch.from_numpy(np.asarray(raw[target_rows[test_mask]])).to(DEVICE);th=torch.from_numpy(np.asarray(raw[meta.activation_row.to_numpy()[test_mask]])).to(DEVICE)
    decoder=nn.Linear(128,2304).to(DEVICE);opt=torch.optim.Adam(decoder.parameters(),lr=LR);history=[];rng=np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        losses=[]
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//BATCH)):
            loss=F.mse_loss(decoder(x[batch]),y[batch]);opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        if epoch in (0,9,49,99):print(f'epoch={epoch+1}/{EPOCHS} train_MSE={np.mean(losses):.6f}')
        history.append(float(np.mean(losses)))
    decoder.eval()
    with torch.no_grad():pred=decoder(tx);result=metrics(pred,ty,th)
    ART.mkdir(parents=True,exist_ok=True);np.save(ART/'decoded_heldout_indirect.npy',pred.cpu().numpy());np.save(ART/'heldout_canonical_targets.npy',ty.cpu().numpy());np.save(ART/'heldout_raw_indirect.npy',th.cpu().numpy());meta.loc[test_mask,['example_id','fact_id','C_split','S_family','S_variant','activation_row']].to_csv(ART/'heldout_metadata.csv',index=False)
    torch.save({'state_dict':decoder.state_dict(),'config':{'architecture':'Linear(128,2304)','source':'frozen cs_partition z_C','target':'frozen Gemma layer-8 masked-mean H(C,declarative/v1)','epochs':EPOCHS,'lr':LR},'train_mse_history':history},CKPT)
    report={'scope':{'frozen_Gemma_layer':8,'pooling':'masked mean','frozen_partition':'checkpoint/cs_partition_layer8.pt','z_C_dimension':128,'decoder':'Linear(128,2304)','decoder_only_training':True,'SAE_trained':False,'ConCA_trained':False},'canonical_surface':{'S_family':'declarative','S_variant':'v1'},'split':{'decoder_train':'C_train, S_train rows excluding canonical declarative/v1','heldout_evaluation':'C_test, indirect/v1','train_rows':int(train_mask.sum()),'heldout_rows':int(test_mask.sum()),'train_facts':int(meta.loc[train_mask,'fact_id'].nunique()),'heldout_facts':int(meta.loc[test_mask,'fact_id'].nunique())},'training':{'epochs':EPOCHS,'optimizer':'Adam(lr=0.001)','loss':'MSE(decoded z_C, canonical H_C)','final_train_MSE':history[-1]},'heldout_indirect_metrics':result,'artifacts':{'checkpoint':str(CKPT.relative_to(ROOT)),'heldout_arrays':str(ART.relative_to(ROOT))}}
    OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
