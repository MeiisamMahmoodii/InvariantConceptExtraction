"""Frozen pair-geometry audit: same qid/different method vs different qid/same method."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data'/'salad_autodan_partition_artifacts';OUT=ROOT/'Report'/'salad_partition_pair_geometry_audit.json';SEED=20260827
def summary(values):
 a=np.asarray(values);return {'n':int(len(a)),'mean':float(a.mean()),'std':float(a.std()),'p05':float(np.quantile(a,.05)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95))}
def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def all_pairs(items):
 return list(combinations(items,2))
def main():
 meta=json.loads((ART/'metadata.json').read_text());qids=np.array([r['qid'] for r in meta]);methods=np.array([r['method'] for r in meta]);vectors={'raw':np.load(ART/'raw_layer8.npy'),'z_C':np.load(ART/'z_C.npy'),'z_S':np.load(ART/'z_S.npy')};assert len(qids)==611
 by_qid={q:np.where(qids==q)[0].tolist() for q in np.unique(qids)};by_method={m:np.where(methods==m)[0].tolist() for m in np.unique(methods)}
 same=[]
 for q,idx in by_qid.items():
  same += [(a,b,f'{methods[a]}__{methods[b]}') for a,b in all_pairs(idx) if methods[a]!=methods[b]]
 different=[]
 for m,idx in by_method.items(): different += [(a,b,m) for a,b in all_pairs(idx) if qids[a]!=qids[b]]
 def evaluate(pairs):
  return {name:summary([cosine(x[a],x[b]) for a,b,_ in pairs]) for name,x in vectors.items()}
 def strata(pairs):
  result={}
  for key in sorted(set(k for *_,k in pairs)):
   group=[(a,b,k) for a,b,k in pairs if k==key];result[key]=evaluate(group)
  return result
 same_eval,diff_eval=evaluate(same),evaluate(different)
 auto_same=[p for p in same if (methods[p[0]]=='autodan') != (methods[p[1]]=='autodan')]
 auto_diff=[p for p in different if p[2]=='autodan']
 auto_same_eval,auto_diff_eval=evaluate(auto_same),evaluate(auto_diff)
 report={'scope':{'dataset':'OpenSafetyLab/Salad-Data aligned AutoDAN population','qids':len(by_qid),'rows':len(qids),'frozen_diagnostic_only':True,'partition_retrained':False},'pair_definition':{'same_goal_different_method':'same qid, distinct method','different_goal_same_method':'distinct qid, same method'},'same_goal_different_method_cosine':same_eval,'different_goal_same_method_cosine':diff_eval,'same_goal_by_method_pair':strata(same),'different_goal_by_method':strata(different),'mean_margin_same_goal_minus_different_goal':{name:float(same_eval[name]['mean']-diff_eval[name]['mean']) for name in vectors},'AutoDAN_heldout_focus':{'same_goal_AutoDAN_to_training_method':auto_same_eval,'different_goal_same_AutoDAN_method':auto_diff_eval,'mean_margin_same_goal_minus_different_goal':{name:float(auto_same_eval[name]['mean']-auto_diff_eval[name]['mean']) for name in vectors}}}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
