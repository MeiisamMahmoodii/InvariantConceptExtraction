"""Read-only structural mechanism audit; outputs no jailbreak text."""
import csv,json,os,re
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SRC=Path(os.environ['TEMP'])/'jbb_artifacts_audit'/'attack-artifacts';OUT=ROOT/'Report'/'jailbreakbench_mechanism_audit.json';ROWS=ROOT/'data'/'jailbreakbench_mechanism_audit.csv';TARGET='vicuna-13b-v1.5';METHODS=['DSN','GCG','JBC','PAIR','prompt_with_random_search'];SEED=20260826;N=20
def norm(x):return re.sub(r'\s+',' ',x).strip().casefold()
def main():
 raw=[]
 for method in METHODS:
  f=next((SRC/method).glob(f'*/*{TARGET}.json'))
  for r in json.loads(f.read_text(encoding='utf-8'))['jailbreaks']:
   if all(r.get(k) not in (None,'') for k in ('index','behavior','goal','prompt')):raw.append({'method':method,**{k:r[k] for k in ('index','behavior','goal','prompt')}})
 by=defaultdict(list)
 for r in raw:by[r['index']].append(r)
 reject={i for i,rs in by.items() if len({(r['behavior'],r['goal']) for r in rs})>1};rng=np.random.default_rng(SEED);audit=[]
 for method in METHODS:
  pool=[r for r in raw if r['method']==method and r['index'] not in reject];chosen=rng.choice(pool,N,replace=False)
  for r in chosen:
   p,g=norm(r['prompt']),norm(r['goal']);i=p.find(g);exact=i>=0;before=p[:i] if exact else '';after=p[i+len(g):] if exact else ''
   letters=sum(c.isalpha() for c in r['prompt']); odd=sum(not(c.isalnum() or c.isspace() or c in '.,;:!?\'\"-()[]{}') for c in r['prompt']); role=bool(re.search(r'\b(role.?play|pretend|act as|you are|fictional|hypothetical|imagine|character)\b',p)); obfuscation=(odd/max(len(r['prompt']),1)>.08 or sum(ord(c)>127 for c in r['prompt'])>5)
   audit.append({'method':method,'behavior_index':r['index'],'goal_verbatim':exact,'prefix_added':exact and bool(before.strip()),'suffix_added':exact and bool(after.strip()),'prefix_characters':len(before.strip()) if exact else 0,'suffix_characters':len(after.strip()) if exact else 0,'prompt_characters':len(r['prompt']),'rewritten_or_paraphrased':not exact,'role_play_marker':role,'encoded_or_obfuscated_heuristic':obfuscation,'syntax_substantially_changed':not exact,'intent_status':'SOURCE_DECLARED_SAME__INDEPENDENT_REVIEW_REQUIRED'})
 with ROWS.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
 report={'scope':{'target_model':TARGET,'samples_per_method':N,'strictly_rejected_behavior_ids':sorted(reject),'source_text_not_redistributed':True},'method_results':{},'semantic_limit':'All records retain a source-declared canonical behavior/goal. This structural audit does not claim an independent SAME_INTENT judgment; use the blinded review packet for that decision.','training_performed':False}
 for method in METHODS:
  x=[r for r in audit if r['method']==method];report['method_results'][method]={k:sum(bool(r[k]) for r in x) for k in ('goal_verbatim','prefix_added','suffix_added','rewritten_or_paraphrased','role_play_marker','encoded_or_obfuscated_heuristic','syntax_substantially_changed')};report['method_results'][method].update({'sample_size':len(x),'mean_prefix_characters':float(np.mean([r['prefix_characters'] for r in x])),'mean_suffix_characters':float(np.mean([r['suffix_characters'] for r in x])),'mean_prompt_characters':float(np.mean([r['prompt_characters'] for r in x]))})
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
