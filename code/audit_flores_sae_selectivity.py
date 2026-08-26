"""Frozen FLORES SAE feature selectivity: sentence identity versus language."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"data"/"flores_partition_artifacts";OUT=ROOT/"Report"/"flores_sae_selectivity_audit.json"
def load(name):
 q=np.load(ART/f"sae_{name}_all_sparse.npz");a=np.zeros(tuple(q["shape"]),np.float32);np.put_along_axis(a,q["indices"],q["values"],1);return a
def score(a,ids,langs):
 def ratio(groups):
  means=np.stack([a[groups==g].mean(0) for g in np.unique(groups)]);within=np.mean([a[groups==g].var(0) for g in np.unique(groups)],0);return means.var(0)/(within+1e-8)
 return ratio(ids),ratio(langs)
def main():
 ids=np.load(ART/"sentence_ids.npy");langs=np.load(ART/"languages.npy");splits=np.load(ART/"splits.npy");test=splits=="test";report={"evaluation":"frozen SAE activations; 30 C-test sentences across all ten languages","SAE_trained":False}
 for name in ("raw","z_C","z_S"):
  a=load(name)[test];sid,lan=ids[test],langs[test];sr,lr=score(a,sid,lan);sentence=(sr>lr)&(sr>1);language=(lr>sr)&(lr>1);pairs=[]
  for s in np.unique(sid):
   i=np.where((sid==s)&(lan=="arb_Arab"))[0][0];j=np.where((sid==s)&(lan=="zho_Hans"))[0][0];pairs.append((i,j))
  left=np.stack([a[i] for i,j in pairs]);right=np.stack([a[j] for i,j in pairs]);cons=1-np.abs(left-right).sum(0)/(left.sum(0)+right.sum(0)+1e-8);report[name]={"sentence_oriented_fraction":float(sentence.mean()),"language_oriented_fraction":float(language.mean()),"mixed_or_unselective_fraction":float((~(sentence|language)).mean()),"sentence_oriented_mean_cross_language_consistency":float(cons[sentence].mean()) if sentence.any() else None,"sentence_oriented_feature_count":int(sentence.sum()),"language_oriented_feature_count":int(language.sum())};print(name,report[name])
 OUT.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
