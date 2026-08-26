"""Export the already sampled PAIR rows for blinded semantic annotation."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'/'jailbreakbench_semantic_audit';SRC=D/'blinded_records.csv';KEY=D/'key.csv';OUT=D/'pair_blinded_records.csv'
def main():
 with SRC.open(encoding='utf-8',newline='') as f:rows={r['review_id']:r for r in csv.DictReader(f)}
 with KEY.open(encoding='utf-8',newline='') as f:ids=[r['review_id'] for r in csv.DictReader(f) if r['attack_method']=='PAIR']
 with OUT.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['review_id','canonical_goal','adversarial_prompt','intent_label','reviewer_note']);w.writeheader()
  for n,i in enumerate(ids,1):r=rows[i];w.writerow({'review_id':f'PAIR-{n:03d}','canonical_goal':r['canonical_goal'],'adversarial_prompt':r['adversarial_prompt'],'intent_label':'','reviewer_note':''})
 print(f'Wrote {len(ids)} blinded PAIR rows to {OUT}')
if __name__=='__main__':main()
