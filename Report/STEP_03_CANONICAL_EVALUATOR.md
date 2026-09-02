# Step 3 — Canonical factor-SAE evaluator

## Outcome

One frozen evaluator now measures every Step 2 representation with identical semantic-ID splits, feature-selection rows, probe code, activity rules, and bootstrap code. All eight saved SAE checkpoints and raw Gemma activations were evaluated without retraining and without loading the untouched Arabic/Chinese test activations.

The reciprocal factor SAE is the strongest sparse model for the paper's primary task. It has the highest cross-locale sparse-feature stability (`0.3921`) and the lowest locale-probe accuracy (`0.8373`) among the evaluated SAEs, while preserving a high mean intent-concept AUC (`0.8723`).

## Frozen manifest

Manifest: `data/canonical_evaluation_manifest.json`

SHA-256:

```text
6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439
```

The manifest fixes:

- 10,343 representation-training semantic IDs;
- 1,149 validation semantic IDs;
- 2,968 untouched test semantic IDs;
- 49 seen training locales;
- held-out test locales `ar-SA` and `zh-CN`;
- validation stability locales `en-US` and `ja-JP`;
- 5,800 balanced intent-probe rows, 100 per intent;
- 5,220 balanced feature-selection rows, 90 per intent across the 49 seen locales;
- disjoint intent-stratified validation-ID halves for feature selection and scoring;
- representation seeds `20260827`, `20260828`, and `20260829`;
- seed `20260827` for splits, probes, feature selection, relations, and bootstrap.

The MASSIVE test relation file contains 5,936 triples and satisfies every relation exactly. Singleton-intent anchors are forced to use exact translations, and the sampler compensates on eligible anchors so the global positive mixture remains exactly 50% exact translations and 50% different-ID/same-intent examples.

## Canonical metrics

For each representation, the evaluator computes:

- intent probes using `k={1,5,10,20,all}` selected coordinates;
- mean one-vs-rest intent concept AUC across 58 intents;
- paired English/Japanese cross-locale feature stability;
- locale-probe accuracy;
- intent- and locale-oriented active-feature fractions;
- `z_S` locale recovery and intent leakage for blockwise models;
- reconstruction MSE, fraction of variance explained, and cosine similarity;
- fraction alive and observed validation `L0`.

Coordinate selection for each `k` uses training-only one-way ANOVA scores. The selected coordinates are passed to the same `StandardScaler(with_mean=False)` and `SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3)` probe. Feature selection never uses the scoring half of the validation IDs.

Shared numerical definitions:

| Definition | Frozen value |
|---|---:|
| Variance floor | `1e-12` |
| Activation epsilon | `1e-8` |
| Minimum activity rate | `1e-3` |
| Orientation ratio | `1.1` |
| Bootstrap resamples | `10,000` |
| Bootstrap seed | `20260827` |

## Canonical one-seed validation result

For global BatchTopK and Matryoshka baselines, the full 9,216-feature code is evaluated. For blockwise models, the primary columns evaluate `z_C`, while the two `z_S` columns evaluate the locale-dominant route. Lower locale and `z_S` intent probe scores are better.

| Method | Intent AUC ↑ | Stability ↑ | Locale probe ↓ | Intent-feature frac. ↑ | Locale-feature frac. ↓ | `z_S` locale ↑ | `z_S` intent ↓ | Rec. FVE ↑ | Total `L0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw `H` | .6991 | .4209 | .9992 | .1671 | .6693 | — | — | — | — |
| BatchTopK SAE | .7264 | .2300 | .9712 | .5068 | .3914 | — | — | .7867 | 69.85 |
| Blockwise SAE control | .6663 | .2683 | .9051 | .5434 | .3546 | .9619 | .6075 | .7866 | 69.54 |
| Matryoshka SAE | .7568 | .2496 | .9737 | .5197 | .3720 | — | — | .7742 | 69.21 |
| One-sided factor SAE | **.8845** | .3239 | .8390 | .7162 | .1903 | .9653 | .5660 | .7680 | 70.43 |
| **Reciprocal factor SAE (ours)** | .8723 | **.3921** | **.8373** | **.7814** | **.1387** | .9644 | .4668 | .7467 | 69.74 |
| Triplet, `m=.1` | .8288 | .3622 | .8958 | .7137 | .1804 | .9746 | .5019 | .7753 | 69.75 |
| Triplet, `m=.2` | .8616 | .3537 | .8653 | .7348 | .1703 | **.9873** | .4740 | .7739 | 70.59 |
| Triplet, `m=.4` | .8625 | .3183 | .8746 | .7595 | .1515 | .9763 | **.4613** | .7696 | 71.68 |

Raw `H` is a dense-coordinate reference rather than a sparse-concept model. Its high coordinate correlation comes with nearly perfect locale decodability (`0.9992`). The relevant feature-stability comparison is therefore between learned sparse dictionaries under the same width and activity budget.

Relative to the exact blockwise reconstruction control, reciprocal supervision:

- raises intent concept AUC by `0.2060`;
- raises cross-locale stability by `0.1238` (`46.1%` relative);
- lowers the `z_C` locale probe by `0.0678`;
- raises the intent-oriented feature fraction from `0.5434` to `0.7814`;
- lowers the locale-oriented feature fraction from `0.3546` to `0.1387`;
- lowers `z_S` intent leakage from `0.6075` to `0.4668` while retaining `0.9644` locale accuracy.

Relative to the standard global BatchTopK SAE, ours raises AUC by `0.1459`, raises stability by `0.1621`, and reduces locale-probe accuracy by `0.1339`. Relative to the one-sided model, reciprocal supervision trades `0.0122` AUC for `0.0682` additional stability and a more intent-muted `z_S` route. This isolates the benefit of supervising both routes.

## Few-feature probes

| Method | `k=1` | `k=5` | `k=10` | `k=20` | All features |
|---|---:|---:|---:|---:|---:|
| Raw `H` | .0345 | .0978 | .1578 | .3015 | .7608 |
| BatchTopK SAE | .0345 | .0929 | .1278 | .2132 | .5794 |
| Blockwise SAE control | .0345 | .0947 | .1642 | .2362 | .4625 |
| Matryoshka SAE | .0340 | .1171 | .1616 | .2899 | .5433 |
| One-sided factor SAE | .0345 | .0934 | .1372 | .2687 | .5420 |
| Reciprocal factor SAE | .0345 | .0771 | .1218 | .2497 | .5850 |
| Triplet, `m=.1` | .0338 | .1033 | .1417 | .2451 | .5754 |
| Triplet, `m=.2` | .0345 | .0905 | .1674 | .2792 | **.5856** |
| Triplet, `m=.4` | .0345 | .1040 | **.1789** | **.3355** | .5846 |

These probes measure multiclass intent recovery from only the selected coordinates. Intent concept AUC remains the primary individual-feature statistic; the `k`-feature curves are complementary evidence about compact supervised readout.

## Decision

- Keep the smooth binary reciprocal objective as the primary method because it gives the strongest cross-locale sparse-feature stability and the lowest `z_C` locale leakage.
- Carry only triplet margin `m=.2` into the final three-seed ablation. It is the strongest reciprocal triplet balance: best full-code intent recovery and best `z_S` locale recovery, with lower reconstruction error than the binary objective.
- Keep the one-sided row as the component ablation because it shows that reciprocal supervision—not only `z_C` supervision—produces the strongest stability and cleaner `z_S` specialization.

## Reproduction

Rebuild and audit the frozen manifest:

```powershell
python code\canonical_evaluator.py build-manifest
python code\canonical_evaluator.py audit-manifest
```

Evaluate every saved Step 2 checkpoint:

```powershell
python code\evaluate_massive_factor_sae.py
```

Evaluate only our method:

```powershell
python code\evaluate_massive_factor_sae.py --method "Reciprocal factor SAE"
```

Machine-readable outputs:

- `Report/factor_sae_step3_canonical_validation.csv`
- `Report/factor_sae_step3_canonical_validation.json`

## Verification

- The manifest audit passes all split, balance, locale, and relation checks.
- Thirteen focused evaluator, BatchTopK, and relation-sampler tests pass.
- All nine representations produced finite canonical metrics.
- Every SAE checkpoint was evaluated through the same code path.
- No model was retrained.
- The untouched MASSIVE test activation file was not loaded.

## Archive action

The superseded dense-pipeline Step 3 report and its earlier manifest were copied to `archive/plans/2026-09-02_pre_factor_sae_step3/`. No result or checkpoint was deleted.
