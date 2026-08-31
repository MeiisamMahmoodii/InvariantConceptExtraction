"""Train pooled MASSIVE z_S Top-k SAE."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data'/'massive_partition_artifacts';OUT=ROOT/'data'/'massive_sae_artifacts';PART=ROOT/'checkpoint'/'massive_partition_layer8.pt';CKPT=ROOT/'checkpoint'/'massive_topk_z_S_k64.pt';REPORT=ROOT/'Report'/'massive_zs_topk_training.json';SEED,EPOCHS,BATCH,K,EXP=20260827,30,256,64,4;DEVICE='cuda'
class Partition(nn.Module):
 def __init__(self):super().__init__();self.c,self.s=nn.Linear(2304,128),nn.Linear(2304,128)
 def forward(self,x):return F.normalize(self.c(x),dim=-1),F.normalize(self.s(x),dim=-1)
class SAE(nn.Module):
 def __init__(self):super().__init__();self.e=nn.Linear(128,512);self.d=nn.Linear(512,128,bias=False);self.b=nn.Parameter(torch.zeros(128))
 def forward(self,x):
  a=F.relu(self.e(x));v,i=torch.topk(a,K,1);z=torch.zeros_like(a).scatter(1,i,v);return z,self.d(z)+self.b
 def normalize(self):
  with torch.no_grad():self.d.weight.div_(self.d.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
def main():
 assert torch.cuda.is_available();torch.manual_seed(SEED);rng=np.random.default_rng(SEED);OUT.mkdir(exist_ok=True);raw=np.load(ART/'raw_train_layer8.npy',mmap_mode='r');p=Partition().to(DEVICE);p.load_state_dict(torch.load(PART,map_location=DEVICE,weights_only=True)['state_dict']);p.eval();z=np.lib.format.open_memmap(OUT/'z_S_train.npy',mode='w+',dtype='float32',shape=(len(raw),128))
 with torch.no_grad():
  for i in range(0,len(raw),4096):z[i:i+4096]=p(torch.from_numpy(np.asarray(raw[i:i+4096])).to(DEVICE))[1].cpu().numpy()
 mean,std=np.asarray(z).mean(0),np.asarray(z).std(0).clip(1e-6);m=SAE().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);history=[]
 for e in range(EPOCHS):
  losses=[]
  for ids in np.array_split(rng.permutation(len(z)),max(1,len(z)//BATCH)):
   x=torch.from_numpy((np.asarray(z[ids])-mean)/std).to(DEVICE);_,r=m(x);loss=F.mse_loss(r,x);opt.zero_grad();loss.backward();opt.step();m.normalize();losses.append(loss.item())
  history.append(float(np.mean(losses)));print(f'epoch={e+1}/{EPOCHS} mse={history[-1]:.6f}')
 torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'config':{'k':K,'expansion':EXP,'epochs':EPOCHS,'seed':SEED,'input_width':128}},CKPT);REPORT.write_text(json.dumps({'input_width':128,'dictionary_width':512,'k':K,'expansion':EXP,'epochs':EPOCHS,'history':history},indent=2)+'\n')
if __name__=='__main__':main()
