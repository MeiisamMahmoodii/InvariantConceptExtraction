# Step 4 — Definitive Three-Seed Factor-SAE Comparison

## Outcome

The proposed reciprocal factor SAE is the strongest sparse model for the study's target: organizing a 4x overcomplete dictionary into a cross-locale intent route and an opposing locale route. Across three paired seeds on untouched Arabic and Chinese, it has the highest cross-locale sparse-feature stability, the highest intent-oriented feature fraction, the lowest locale-oriented feature fraction, and the lowest `z_C` locale probe among the learned sparse models. Its complete-route intent balanced accuracy remains within 0.6 percentage points of the standard global BatchTopK SAE.

The result is stable across seeds. The binary reciprocal loss also outperforms the selected triplet ablation on every primary `z_C` feature metric. Triplet learning retains more reconstruction variance, so it remains a useful loss-form tradeoff rather than the main method.

## Frozen protocol

- Frozen activation: Gemma 2 2B, layer 8, masked-mean pooled, width 2304.
- Sparse dictionary: 9216 features (4x).
- Proposed routes: `z_C=2765`, `z_S=6451`.
- Blockwise BatchTopK training budgets: `k_C=13`, `k_S=51`, mean training `L0=64`.
- Training semantic IDs: 10,343.
- Validation semantic IDs: 1,149.
- Untouched test semantic IDs: 2,968.
- Held-out test locales: `ar-SA`, `zh-CN`.
- Epochs: 30; batch size: 128; AdamW learning rate: `1e-4`.
- Paired seeds: `20260827`, `20260828`, `20260829`.
- Canonical manifest SHA-256: `6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439`.

All learned methods use the same normalized inputs, representation split, initialization procedure, batches, optimizer, epoch budget, dictionary width, decoder normalization, evaluator, and seeds. The blockwise reconstruction control exactly matches the proposed route widths and activity budgets, isolating the effect of the controlled relations.

## Definitive result

Values are mean +/- sample standard deviation over three paired seeds. Raw `H` is one deterministic dense reference.

| Method | Intent bal. acc. | Intent AUC | Stability | `z_C` locale probe | Intent feature frac. | Locale feature frac. | `z_S` locale | `z_S` intent | Rec. FVE | Test total L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw `H` | .6519 | .7703 | .3013 | .9997 | .1159 | .7409 | — | — | — | — |
| BatchTopK SAE | .4680 +/-.0026 | .7940 +/-.0014 | .1278 +/-.0011 | .9871 +/-.0048 | .4579 +/-.0026 | .4723 +/-.0030 | — | — | .6618 +/-.0017 | 107.16 +/- .91 |
| Blockwise SAE control | .3527 +/-.0119 | .7007 +/-.0109 | .1501 +/-.0080 | .9665 +/-.0106 | .4830 +/-.0122 | .4465 +/-.0122 | .9878 +/-.0014 | .4798 +/-.0151 | .6627 +/-.0002 | 105.99 +/- .81 |
| Matryoshka SAE | .4112 +/-.0131 | .8352 +/-.0150 | .1385 +/-.0017 | .9821 +/-.0009 | .4686 +/-.0018 | .4652 +/-.0033 | — | — | .6482 +/-.0014 | 102.86 +/- 1.02 |
| One-sided factor SAE | .4188 +/-.0228 | **.9165 +/-.0083** | .1813 +/-.0061 | .8969 +/-.0143 | .6923 +/-.0198 | .2427 +/-.0163 | .9861 +/-.0029 | .4679 +/-.0004 | .6415 +/-.0015 | 100.38 +/- .87 |
| **Reciprocal factor SAE — ours** | **.4625 +/-.0081** | .9124 +/-.0027 | **.2248 +/-.0035** | **.8944 +/-.0034** | **.7864 +/-.0144** | **.1646 +/-.0065** | **.9925 +/-.0008** | .3892 +/-.0079 | .6098 +/-.0011 | 104.46 +/- .07 |
| Triplet `m=.2` | .4590 +/-.0100 | .9063 +/-.0190 | .1983 +/-.0029 | .9096 +/-.0049 | .7274 +/-.0013 | .2043 +/-.0011 | .9916 +/-.0029 | **.3818 +/-.0171** | **.6427 +/-.0012** | 109.87 +/- 1.53 |

Bold values mark the strongest result within the relevant learned sparse comparison. The proposed method's intent balanced accuracy is bold among the three relational models; global BatchTopK is slightly higher at `.4680` but does not organize its features around intent.

## Evidence for the main claim

### Controlled relations versus the exact architecture control

Reciprocal supervision changes the organization of the same blockwise SAE:

- Intent concept AUC: `.7007 -> .9124` (`+0.2117`).
- Cross-locale feature stability: `.1501 -> .2248` (`+0.0746`, `+49.7%`).
- Intent-oriented feature fraction: `.4830 -> .7864` (`+0.3035`).
- Locale-oriented feature fraction: `.4465 -> .1646` (`-0.2819`).
- `z_C` locale probe: `.9665 -> .8944` (`-0.0721`).
- `z_S` locale recovery: `.9878 -> .9925`.
- `z_S` intent balanced accuracy: `.4798 -> .3892` (`-0.0906`).

This is the clean causal comparison in the table because no architectural, capacity, sparsity-budget, initialization, batch, or optimization difference separates the two rows.

### Proposed method versus standard SAE baselines

Against global BatchTopK:

- AUC improves by `0.1185`.
- Stability improves by `0.0970` (`75.9%`).
- Intent-oriented features increase by `0.3286`.
- Locale-oriented features decrease by `0.3077`.
- `z_C` locale probe decreases by `0.0927`.
- Intent balanced accuracy remains almost unchanged: `.4625` versus `.4680`.

Against Matryoshka:

- AUC improves by `0.0772`.
- Stability improves by `0.0863`.
- Intent-oriented features increase by `0.3178`.
- Locale-oriented features decrease by `0.3007`.
- Intent balanced accuracy improves by `0.0513`.

### Why both routes are trained

The one-sided model shows that supervising `z_C` alone is already useful. Reciprocal supervision then:

- increases stability by `0.0434`;
- increases the intent-oriented fraction by `0.0941`;
- decreases the locale-oriented fraction by `0.0781`;
- lowers `z_S` intent leakage by `0.0787`;
- increases `z_S` locale recovery by `0.0064`.

Its AUC is `0.0040` below the one-sided model, while the remaining factor-organization metrics improve. The reciprocal objective therefore supplies the most complete two-route organization.

### Binary loss versus triplet loss

The binary objective is the main method because it beats triplet `m=.2` on:

- intent AUC: `.9124` versus `.9063`;
- stability: `.2248` versus `.1983`;
- `z_C` locale probe: `.8944` versus `.9096`;
- intent-oriented fraction: `.7864` versus `.7274`;
- locale-oriented fraction: `.1646` versus `.2043`.

Triplet learning has higher reconstruction FVE (`.6427` versus `.6098`) and slightly lower `z_S` intent accuracy. It is retained as the reconstruction-oriented loss tradeoff.

## Sparsity and reconstruction

BatchTopK fixes mean training activity at 64. Final inference uses thresholds calibrated only from training activations, so held-out locale shift can change observed activity without using the test set for calibration. The observed total test `L0` remains closely matched for the central comparison: `104.46` for ours, `105.99` for the blockwise control, and `107.16` for global BatchTopK.

The proposed model explains `60.98%` of the standardized activation variance. This is below the reconstruction controls, but its FVE remains within 8.0% relative of the exact blockwise control while the target sparse-feature metrics improve substantially. The triplet ablation shows the available reconstruction/organization operating point.

## Evaluator implementation check

The final evaluator uses the same standardized SGD logistic probe for every representation. Sparse codes are passed to the existing CSR computation path rather than materialized as dense matrices. This changes storage and runtime, not labels, scaling, classifier hyperparameters, seeds, splits, or metric definitions. The complete definitive evaluation was restarted after this correction so every reported row uses the same implementation.

Thirteen focused tests cover the canonical manifest, sparse evaluation, BatchTopK protocol, and relation sampler.

The updated Method and Results fragments compile successfully together. The temporary compilation PDF and log were moved to `archive/build_checks/2026-09-02_step4/`; no result or checkpoint was deleted.

## Machine-readable artifacts

- Per-seed table: `Report/factor_sae_step4_definitive_test_per_seed.csv`.
- Mean/std table: `Report/factor_sae_step4_definitive_test_summary.csv`.
- Complete metrics, selected features, and bootstrap intervals: `Report/factor_sae_step4_definitive_test.json`.
- Paper table and interpretation: `paper/factor_sae_validation_results.tex`.

## Reproduction

Train all missing definitive checkpoints for the two new seeds:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_massive_factor_sae.py --final --experiment batchtopk --experiment block_control --experiment matryoshka --experiment one_sided --experiment reciprocal --experiment triplet_m0p2 --seed 20260828 --seed 20260829
```

Evaluate all three seeds on the frozen test protocol:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\evaluate_massive_factor_sae.py --final
```

Recompute only the proposed method:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\evaluate_massive_factor_sae.py --final --method "Reciprocal factor SAE"
```

## Step decision

Step 4 supports the paper's central claim: controlled reciprocal relations reorganize a fixed-capacity overcomplete sparse dictionary into stable, factor-dominant routes. The binary reciprocal SAE remains the proposed method. Triplet `m=.2` remains one ablation. The next experiment should test whether the conclusion persists at 8x dictionary width before moving to MTOP transfer.
