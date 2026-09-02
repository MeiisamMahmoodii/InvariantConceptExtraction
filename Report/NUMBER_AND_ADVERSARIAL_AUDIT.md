# Number and Adversarial Audit

Date: 2026-09-02

## Outcome

- Every reported table value was checked against the canonical result CSVs.
- Every numerical improvement stated in the prose was recomputed from those CSVs.
- The 58-row feature catalogue was compared row by row with `factor_sae_feature_catalogue.csv`; all 58 rows match.
- No table-to-CSV numerical mismatch was found.
- Four claims were narrowed because their previous wording was broader than the evidence, especially for Pythia-160M.

## Canonical files checked

- `factor_sae_step4_definitive_test_summary.csv` and per-seed results
- `factor_sae_step5_width8_test_summary.csv` and per-seed results
- `factor_sae_step6_mtop_test_summary.csv` and per-seed results
- `factor_sae_pythia160m_transfer_summary.csv` and per-seed results
- `factor_sae_feature_interpretability_per_seed.csv`
- `factor_sae_feature_catalogue.csv`
- Figure 3 and Figure 4 data JSON files

## Recomputed headline claims

| Claim | Recomputed value | Paper value | Status |
|---|---:|---:|---|
| MASSIVE AUC gain over block control | .2117 | .212 | Match |
| MASSIVE stability gain over block control | .0746 | .075 | Match |
| Relative MASSIVE stability gain | 49.71% | 49.7% | Match |
| AUC gain over global BatchTopK | .1185 | .1185 | Match |
| Stability gain over global BatchTopK | .0970 | .0970 | Match |
| Stability gain over Matryoshka | .0863 | .0863 | Match |
| Relative FVE difference from block control | 7.98% | 8.0% | Match |
| Held-out purity gain over block control | .2362 | .236 | Match |
| Held-out coverage gain over block control | .3841 | .384 | Match |
| Eightfold relative stability gain | 25.56% | 25.6% | Match |
| Eightfold relative FVE difference | 6.27% | 6.3% | Match |
| MTOP AUC gain over block control | .2748 | .275/.2748 | Match |
| MTOP stability gain over block control | .1277 | .128/.1277 | Match |
| MTOP margin gain over global BatchTopK | .3101 | .3101 | Match |
| Pythia stability ratio over global BatchTopK | 3.236x | 3.2x | Match |
| Pythia margin gain over block control | .4797 | .4797 | Match |
| Reciprocal reduction in zS intent leakage vs one-sided | .0787 | .0787 | Match |

## Wording corrected

1. The abstract now says that the advantage **over the exact blockwise control** persists across width and transfer. It no longer implies that every ordering against every baseline transfers to Pythia.
2. The introduction now claims consistent stability improvement over the exact blockwise control and limits the cross-setting intent-orientation claim to MASSIVE.
3. “Factor recovery and disentanglement” was changed to “Factor recovery and route organization,” because the probe results show dominance rather than clean statistical separation.
4. The complete-route probe sentence now says the method retains accuracy comparable to the global sparse baseline; it no longer implies that no intent information is lost relative to raw activations.
5. “Designated invariant factor” was changed to “designated factor across nuisance variation,” matching the measured result.

## Places a reviewer could use against the paper

### Highest-risk results

1. **The intent route is not locale-invariant.** Table 1 reports zC locale-probe accuracy of .8944. This is lower than the exact control’s .9665, but the locale remains highly decodable.
2. **The locale route retains substantial intent information.** Table 1 reports zS intent accuracy of .3892. The method creates dominant routes, not cleanly disentangled variables.
3. **One-sided training wins intent AUC.** Its .9165 is slightly above ours at .9124. Ours wins the joint stability/orientation profile, not every metric.
4. **Triplet wins two columns.** It has lower zS intent leakage (.3818 vs .3892) and higher reconstruction FVE (.6427 vs .6098).
5. **Reconstruction is worse.** Ours has .6098 FVE versus .6627 for the exact control, an 8.0% relative difference. The paper correctly frames this as a trade-off for organization.
6. **Raw activations have higher MASSIVE stability.** Raw H has .3013 versus .2248 for ours. Raw H is not a sparse dictionary and has very high locale leakage, but the absolute stability column can still attract criticism.
7. **Full-representation intent accuracy is below raw H.** The complete-route balanced accuracy is .4625 for ours versus .6519 for raw H. The main text compares ours only with the global sparse baseline (.4680).

### Transfer weaknesses

8. **MTOP leakage remains high.** zC locale accuracy is .9278 and zS intent accuracy is .4639.
9. **Raw MTOP remains stronger on two absolute metrics.** Raw H has stability .5443 versus .4978 and retrieval R@1 .7998 versus .7809 for ours.
10. **Pythia AUC is weak.** Ours reaches .5227, only .0048 above the block control, below global BatchTopK (.5349), and below raw H (.5824).
11. **Pythia stability is low in absolute terms.** The improvement from .0075 to .0211 is consistent and 3.2x global BatchTopK, but .0211 remains small.
12. **Pythia relation margin remains negative.** It improves from -.5677 to -.0879, but the average positive relation still does not outrank the negative relation.
13. **Pythia remains locale-dominant in its feature orientation.** The intent-oriented fraction is .1792 and the locale-oriented fraction is .7604, although both improve over the exact control.

### Robustness and interpretability weaknesses

14. **Eightfold absolute stability falls.** It drops from .2248 at fourfold width to .1063 at eightfold width, despite remaining above the matched control.
15. **Eightfold locale leakage and orientation worsen.** Locale-probe accuracy rises from .8944 to .9397; intent orientation falls from .7864 to .6708; locale orientation rises from .1646 to .2530.
16. **Test-time activity is above the training budget.** The model trains at mean L0=64 but reaches 104.46 on held-out locales. Controls rise similarly, so this does not explain the comparative gain, but the shift is visible.
17. **Reliable coverage is .725, not complete.** About 27.5% of eligible intents do not meet the paper’s purity-at-least-.8 criterion.
18. **The full catalogue exposes failed concepts.** `general_greet`, `iot_hue_lighton`, `iot_wemo_on`, `music_dislikeness`, and `music_settings` have purity 0 and AUC near .5 for the reported seed.
19. **Figure 4 includes a visibly weaker example.** `takeaway_order` has purity .60, although its AUC (.970) and stability (.692) are strong.

### Presentation choices that make criticism easier

20. **The main table visibly bolds baseline wins.** One-sided AUC and triplet leakage/FVE are bold, so the paper must keep describing a joint organization objective rather than universal dominance.
21. **The Pythia table is weaker than the main transfer wording can sound.** The revised abstract and introduction now specify comparison to the exact blockwise control.
22. **The complete catalogue is scientifically useful but contains counterexamples.** Keeping it improves reproducibility; it also lets reviewers focus on individual failed intents.
23. **The scope paragraph narrows the claim.** It explicitly says the method does not establish coordinate-level independence or universal causal decomposition. This is accurate, but it limits how broadly the contribution can be advertised.

## Safe central claim after the audit

With the same dictionary, activity allocation, optimization, and reconstruction objective, reciprocal controlled relations improve intent-feature AUC, cross-language stability, and route orientation over the exact blockwise reconstruction control. The effect persists across dictionary width and remains directionally consistent on MTOP and Pythia-160M. The evidence supports **factor-dominant route organization**, not complete factor independence.
