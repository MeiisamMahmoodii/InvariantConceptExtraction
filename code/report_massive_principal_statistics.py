"""Statistical reporting from saved MASSIVE artifacts only; no model execution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'Report';OUT=R/'massive_principal_statistical_report.json';SEED=20260828;NBOOT=10000
def ci(x):
 return [float(v) for v in np.quantile(x,[.025,.975])]
def main():
 raw=pd.read_csv(R/'massive_sae_intent_feature_details'/'raw_intent_features.csv').sort_values('intent')
 zc=pd.read_csv(R/'massive_sae_intent_feature_details'/'z_C_intent_features.csv').sort_values('intent')
 assert raw.intent.tolist()==zc.intent.tolist() and len(raw)==58
 d=zc.heldout_auc.to_numpy()-raw.heldout_auc.to_numpy();rng=np.random.default_rng(SEED);ix=rng.integers(0,len(d),(NBOOT,len(d)));boot=d[ix].mean(1)
 b2=json.loads((R/'massive_b2_infonce_sae_comparison.json').read_text())['branches'];specific=json.loads((R/'massive_joint_cs_decoder_swap_specificity.json').read_text());seeds=json.loads((R/'massive_core_seed_robustness.json').read_text())['mean_std']
 def unavailable(a,b,label):
  return {'swap_rate':a,'unrelated_control_rate':b,'rate_difference':a-b,'n_cases':17808,'resampling_unit':'matched intervention case','bootstrap_95_CI':None,'mcnemar_test':None,'status':'NOT COMPUTABLE from saved artifacts: aggregate rates were persisted but paired per-case outcomes/2x2 discordant counts were not.'}
 report={'scope':{'statistical_reporting_only':True,'model_training_or_inference_run':False,'bootstrap_resamples':NBOOT,'bootstrap_seed':SEED},'matched_vs_naive_infonce_stability':{'matched_zC_stability':b2['Matched_zC']['heldout_arabic_chinese_stability'],'naive_infonce_zC_stability':b2['InfoNCE_zC']['heldout_arabic_chinese_stability'],'difference':b2['Matched_zC']['heldout_arabic_chinese_stability']-b2['InfoNCE_zC']['heldout_arabic_chinese_stability'],'resampling_unit':'held-out MASSIVE semantic id','bootstrap_95_CI':None,'status':'NOT COMPUTABLE from saved artifacts: SAE dictionaries are independently trained and unpaired; per-semantic-ID stability contributions were not persisted, so aggregate stability cannot be recomputed on bootstrap resamples without SAE inference.'},'raw_vs_matched_zC_concept_auc':{'raw_H_mean_auc':float(raw.heldout_auc.mean()),'matched_zC_mean_auc':float(zc.heldout_auc.mean()),'mean_difference_zC_minus_H':float(d.mean()),'bootstrap_95_CI':ci(boot),'fraction_concepts_improved':float((d>0).mean()),'n':len(d),'resampling_unit':'MASSIVE intent / shared concept identity','test':'paired nonparametric bootstrap percentile CI'},'C_swap_specificity':unavailable(specific['specificity_matrix']['zC_swap']['intent_follows_donor'],specific['random_mismatched_controls']['zC_code_unrelated_intent_accuracy'],'donor-intent success'),'S_swap_specificity':unavailable(specific['specificity_matrix']['zS_swap']['locale_follows_donor'],specific['random_mismatched_controls']['zS_code_unrelated_locale_accuracy'],'donor-locale success'),'seed_robustness':{'n_seeds':3,'significance_tests':'NOT RUN: n=3, per instruction','existing_mean_std':seeds}}
 OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
