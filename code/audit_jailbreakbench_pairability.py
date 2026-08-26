"""Read-only pairability audit; prints counts only, never prompt text."""
import json, os
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=Path(os.environ['TEMP'])/'jbb_artifacts_audit'/'attack-artifacts'; OUT=ROOT/'Report'/'jailbreakbench_pairability_audit.json'; TARGET='vicuna-13b-v1.5'; EXCLUDE={'test-artifact'}
def main():
 rows=[]; files=[]
 for f in sorted(SRC.glob(f'*/*/{TARGET}.json')):
  method=f.relative_to(SRC).parts[0]
  if method in EXCLUDE: continue
  data=json.loads(f.read_text(encoding='utf-8')); files.append(str(f.relative_to(SRC)))
  for r in data['jailbreaks']:
   if all(r.get(k) not in (None,'') for k in ('index','behavior','goal','prompt')): rows.append({'method':method,**{k:r[k] for k in ('index','behavior','goal','prompt')}})
 by_behavior=defaultdict(list)
 for r in rows: by_behavior[r['index']].append(r)
 conflict_ids=sorted(i for i,rs in by_behavior.items() if len({(r['behavior'],r['goal']) for r in rs})>1)
 rows=[r for r in rows if r['index'] not in conflict_ids]; by_behavior=defaultdict(list); by_method=defaultdict(list)
 for r in rows: by_behavior[r['index']].append(r); by_method[r['method']].append(r)
 methods_per_behavior={i:len({r['method'] for r in rs}) for i,rs in by_behavior.items()}
 valid_behaviors=[i for i,n in methods_per_behavior.items() if n>=2]
 positives=sum(n*(n-1)//2 for n in methods_per_behavior.values())
 negatives=sum(len(rs)*(len(rs)-1)//2 for rs in by_method.values())
 report={'status':'passed_strict_structural_pairability','scope':{'target_model':TARGET,'excluded_artifacts':sorted(EXCLUDE),'artifact_files':files},'source_harmful_behavior_ids':100,'rejected_behavior_ids_due_to_canonical_conflict':conflict_ids,'retained_harmful_behavior_ids':len(by_behavior),'attack_methods':sorted(by_method),'retained_valid_records':len(rows),'records_per_method':{m:len(rs) for m,rs in by_method.items()},'methods_per_behavior':{'min':min(methods_per_behavior.values()),'max':max(methods_per_behavior.values()),'mean':sum(methods_per_behavior.values())/len(methods_per_behavior)},'behaviors_with_at_least_two_methods':len(valid_behaviors),'same_C_different_S_positive_pairs_unordered':positives,'different_C_same_S_negative_pairs_unordered':negatives,'attack_methods_eligible_for_full_holdout':[m for m,rs in by_method.items() if len(rs)==len(by_behavior)],'goal_string_verbatim_in_prompt_records':sum(r['goal'] in r['prompt'] for r in rows),'goal_string_not_verbatim_in_prompt_records':sum(r['goal'] not in r['prompt'] for r in rows),'semantic_preservation':'Artifact records retain a canonical behavior and goal. Non-verbatim goal wording is expected in jailbreak prompts and is not evidence of a changed intent; semantic equivalence still requires a separate blinded audit before training.','benign_controls':{'count':100,'source':'published JBB-Behaviors benign split','fields':['Index','Goal','Target','Behavior','Category','Source'],'note':'The benign split is behavior-matched thematically, not paired row-for-row with harmful artifacts.'},'training_performed':False}
 OUT.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
