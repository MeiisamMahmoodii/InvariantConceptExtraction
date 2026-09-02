# Scientific object

This project studies whether controlled relations can organize one overcomplete sparse dictionary into two factor-dominant routes.

- `C` is the preserved factor. On MASSIVE, `C = intent`.
- `S` is the surface factor. On MASSIVE, `S = locale/language`.
- `z_C` is the intent-dominant sparse route.
- `z_S` is the locale-dominant sparse route.

## Definitive method

The primary model is a direct sparse autoencoder over frozen Gemma 2 2B layer-8 pooled activations:

`H (2304) -> linear encoder -> blockwise BatchTopK -> [z_C, z_S] (9216) -> linear decoder -> H_hat`.

- Dictionary width: `9216` (4x).
- Route widths: `z_C = 2765`, `z_S = 6451`.
- Mean training activity: `k_C = 13`, `k_S = 51`, total `L0 = 64`.
- Objective: reconstruction plus the two binary controlled contrastive losses.
- Temperature: `0.07`.
- Contrastive weight: `1.0`.
- Training: 30 epochs, batch size 128, AdamW, learning rate `1e-4`.
- Seeds: `20260827`, `20260828`, `20260829`.
- No dense bottleneck, DCL, adversary, orthogonality, independence, prototype, or swap-training loss.

## MASSIVE relations

For `z_C`:

- positive: same intent, different locale;
- opposing example: different intent, anchor locale.

Exactly half of the positives are exact translations. The other half use a different utterance ID with the same intent. `z_S` reverses the two relations.

## Evaluation target

The primary result is improved cross-locale stability and few-feature intent recovery in `z_C`, together with strong locale recovery in `z_S`, while reconstruction and sparsity remain comparable to BatchTopK and Matryoshka SAE baselines.

All definitive metrics use `data/canonical_evaluation_manifest.json` (SHA-256 `6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439`). Feature orientation is selected on a balanced validation sample spanning all 49 seen locales; cross-locale validation stability uses paired `en-US`/`ja-JP` rows. The held-out `ar-SA`/`zh-CN` test activations are reserved for the final three-seed table.

The previous dense-pipeline scientific object is preserved at `archive/plans/2026-09-02_pre_factor_sae/SCIENTIFIC_OBJECT.md`.

## Definitive MASSIVE result

The frozen three-seed Arabic/Chinese test supports the scientific object. Relative to the exact blockwise reconstruction control, the reciprocal method improves mean intent concept AUC from `0.7007` to `0.9124`, cross-locale stability from `0.1501` to `0.2248`, and intent-oriented feature fraction from `0.4830` to `0.7864`, while reducing locale-oriented feature fraction from `0.4465` to `0.1646`. The `z_S` route reaches `0.9925` locale accuracy and reduces intent balanced accuracy to `0.3892`.

The definitive artifacts are `Report/factor_sae_step4_definitive_test_per_seed.csv`, `Report/factor_sae_step4_definitive_test_summary.csv`, and `Report/factor_sae_step4_definitive_test.json`.

## Sparse-width robustness

The result persists when the dictionary is doubled to `18432` features and the mean training activity is doubled to `L0=128`, preserving the active fraction. At 8x width, the reciprocal method reaches intent concept AUC `0.9156` and cross-locale stability `0.1063`, compared with `0.7373` and `0.0847` for the exact blockwise control. It also retains the best feature orientation and lowest `z_C` locale leakage among the width-matched methods, while reconstruction FVE stays within `6.3%` of the control at effectively identical observed test activity.

The 4x model remains the primary configuration. The 8x result demonstrates width robustness, so no 16x sweep is required. Artifacts are `Report/factor_sae_step5_width8_test_per_seed.csv`, `Report/factor_sae_step5_width8_test_summary.csv`, and `Report/factor_sae_step5_width8_test.json`.

## MTOP dataset transfer

On MTOP, the factor definitions remain unchanged except that `S = language`. Training uses German, English, Spanish, and French; the untouched test uses Hindi and Thai. The inventory contains 50 intents. Because exact translation identifiers are unavailable, every preserved relation is a same-intent/different-language pair and every opposing relation changes intent while retaining the anchor language.

The architecture, fourfold width, route proportions, activity budgets, optimizer, epochs, and seeds are unchanged from MASSIVE. All sparse models are evaluated at exact mean `L0=64`, and blockwise routes retain exact `L0(z_C)=13`.

Across three seeds, the reciprocal factor SAE reaches intent concept AUC `0.8707` and cross-language stability `0.4978`. The exact blockwise reconstruction control reaches `0.5959` and `0.3701`; global BatchTopK reaches `0.6529` and `0.3605`. Reciprocal supervision also produces the largest intent relation margin (`0.3441`) and lowers `z_C` language leakage while improving `z_S` language recovery and reducing its intent leakage relative to the exact control.

This result supports dataset transfer without parallel test IDs or a dataset-specific method change. Artifacts are `Report/factor_sae_step6_mtop_test_per_seed.csv`, `Report/factor_sae_step6_mtop_test_summary.csv`, and `Report/factor_sae_step6_mtop_test.json`.

## Pythia-160M model-family transfer

The same direct reciprocal factor SAE is trained on masked-mean Pythia-160M hidden-state index 8 activations. Sparse capacity remains fixed at 9,216 features with route widths 2,765/6,451 and exact evaluation activity `L0=64`. The MASSIVE manifest, relations, splits, epochs, optimizer, and three seeds are unchanged from the Gemma experiment.

Relative to the exact blockwise reconstruction control, reciprocal supervision increases cross-locale stability from `0.0075` to `0.0211`, increases the intent relation margin by `0.4797`, lowers locale leakage, improves feature orientation, and produces cleaner `z_S` diagnostics. Reconstruction FVE remains within `6.9%` of the control. These results support transfer across Gemma 2 2B and Pythia-160M without changing the method.

Artifacts are `Report/factor_sae_pythia160m_transfer_per_seed.csv`, `Report/factor_sae_pythia160m_transfer_summary.csv`, and `Report/factor_sae_pythia160m_transfer.json`.

## Feature-level stability evidence

The T-SAE-style stability figure uses the frozen seed `20260827` checkpoint and selects features only from disjoint validation fit/score semantic IDs. The held-out Arabic/Chinese activations do not determine the feature, representative intent, comparison intent, or displayed examples. The selected Gemma feature reaches held-out intent AUC `0.97` and cross-locale stability `0.79`, while remaining active across both locales for the target intent and rejecting every displayed comparison example.

The main evidence remains the three-seed all-feature statistic: reciprocal supervision raises stability from `0.1501` to `0.2248` relative to the exact Gemma blockwise control and from `0.0075` to `0.0211` for Pythia-160M. Figure and source data are `paper/Figures/factor_sae_figure3_stability.pdf` and `paper/figure_data/figure3_factor_stability.json`.

## Feature interpretability

The frozen three-seed feature catalogue selects one feature per intent on validation data and evaluates its top 20 unique semantic IDs on held-out Arabic/Chinese. Among the 46 intents with at least 20 held-out IDs, the reciprocal factor SAE reaches mean purity `0.8171` and reliable coverage `0.7246`, compared with `0.6526`/`0.3696` for global BatchTopK and `0.5808`/`0.3406` for the exact blockwise control. Its 58 selected features also have the highest mean cross-locale stability (`0.5626`) and the lowest mean top-ID overlap (`0.0022`).

The complete results are `Report/factor_sae_feature_interpretability.json`, `Report/factor_sae_feature_interpretability_per_seed.csv`, and `Report/factor_sae_feature_catalogue.csv`.
