"""Create a blinded, stratified semantic-review packet; prints counts only."""
import csv,json,os
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; SRC=Path(os.environ['TEMP'])/'jbb_artifacts_audit'/'attack-artifacts'; OUT=ROOT/'data'/'jailbreakbench_semantic_audit'; TARGET='vicuna-13b-v1.5'; METHODS=['DSN','GCG','JBC','PAIR','prompt_with_random_search']; SEED=20260826; QUOTAS={'PAIR':50,'prompt_with_random_search':50}
def main():
 raw=[]
 for method in METHODS:
  f=next((SRC/method).glob(f'*/*{TARGET}.json'))
  for r in json.loads(f.read_text(encoding='utf-8'))['jailbreaks']:
   if all(r.get(k) not in (None,'') for k in ('index','behavior','goal','prompt')): raw.append({'method':method,**{k:r[k] for k in ('index','behavior','goal','prompt')}})
 grouped=defaultdict(list)
 for r in raw: grouped[r['index']].append(r)
 rejected={i for i,rs in grouped.items() if len({(r['behavior'],r['goal']) for r in rs})>1}
 candidates=[r for r in raw if r['index'] not in rejected and r['goal'] not in r['prompt']]
 rng=np.random.default_rng(SEED); selected=[]
 for method,n in QUOTAS.items():
  pool=[r for r in candidates if r['method']==method]
  if len(pool)<n: raise RuntimeError(f'{method} has only {len(pool)} non-verbatim eligible records')
  selected.extend(rng.choice(pool,n,replace=False).tolist())
 rng.shuffle(selected);OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'blinded_records.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['review_id','canonical_goal','adversarial_prompt','reviewer_label','reviewer_note']);w.writeheader()
  for n,r in enumerate(selected,1):w.writerow({'review_id':f'JBB-{n:03d}','canonical_goal':r['goal'],'adversarial_prompt':r['prompt'],'reviewer_label':'','reviewer_note':''})
 with (OUT/'key.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['review_id','behavior_index','attack_method']);w.writeheader()
  for n,r in enumerate(selected,1):w.writerow({'review_id':f'JBB-{n:03d}','behavior_index':r['index'],'attack_method':r['method']})
 available={m:sum(r['method']==m for r in candidates) for m in METHODS};manifest={'status':'awaiting_independent_semantic_review','sample_size':len(selected),'sampling':'deterministic seed 20260826; 50 PAIR and 50 prompt-with-random-search non-verbatim records; six canonical-conflict behavior IDs excluded','nonverbatim_eligible_by_method':available,'not_estimable_from_nonverbatim_sample':['DSN','GCG','JBC'],'reviewer_visible_fields':['canonical_goal','adversarial_prompt'],'allowed_labels':['SAME_INTENT','PARTIAL_DRIFT','CHANGED_INTENT','AMBIGUOUS'],'blinding':'attack method and behavior index are in key.csv only; do not show key.csv to reviewer','acceptance_rule':{'record_for_positive_pair':'SAME_INTENT only','method_eligible_for_training_or_full_holdout':'at least 95% SAME_INTENT and zero CHANGED_INTENT in its blinded sample','partial_drift_or_ambiguous':'reject those records from positive-pair construction'},'training_performed':False};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'sample_size':len(selected),'per_method':{m:sum(r['method']==m for r in selected) for m in METHODS},'nonverbatim_eligible_by_method':available,'rejected_canonical_conflict_ids':sorted(rejected),'output':str(OUT)},indent=2))
if __name__=='__main__':main()
