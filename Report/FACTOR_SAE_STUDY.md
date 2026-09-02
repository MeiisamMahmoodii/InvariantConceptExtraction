# Factor-Contrastive SAE Study

This is the cumulative source of truth for the T-SAE-aligned direct sparse study. Earlier exploratory reports remain preserved but do not override the protocol recorded here.

## Step 1 — Definitive method and MASSIVE protocol

**Status: complete.**

### Scientific object

- Preserved factor `C`: intent.
- Surface factor `S`: locale/language.
- `z_C`: intent-dominant sparse route.
- `z_S`: locale-dominant sparse route.
- Main goal: stable, factor-specialized sparse features with comparable reconstruction.

### Frozen architecture

| Component | Frozen choice |
|---|---|
| Frozen activation | Gemma 2 2B, layer 8, masked-mean pooled |
| Input width | 2304 |
| Sparse dictionary | 9216 features (4x) |
| Sparse activation | blockwise BatchTopK |
| `z_C` width/activity | 2765 features, mean `k_C=13` |
| `z_S` width/activity | 6451 features, mean `k_S=51` |
| Total activity | mean `L0=64` |
| Decoder | linear 9216 to 2304 |
| Loss | reconstruction + binary `z_C` relation + binary `z_S` relation |
| Temperature / relation weight | 0.07 / 1.0 |
| Optimizer | AdamW, learning rate `1e-4` |
| Schedule | 30 epochs, 45,968 anchors per epoch, batch 128 |
| Seeds | 20260827, 20260828, 20260829 |

No dense bottleneck, DCL, adversarial classifier, orthogonality, independence, prototype, multi-positive, or swap-training term is part of the definitive method.

### Locked loss ablation

Triplet/max-margin learning is retained as one paper ablation. It replaces only the binary route loss with `max(0, margin + negative_similarity - positive_similarity)`. Margins `0.1`, `0.2`, and `0.4` are compared on validation data. The architecture, relation pairs, BatchTopK budgets, reconstruction objective, optimizer, training examples, epochs, and seeds remain identical. The smooth binary loss remains the primary method unless the triplet variant improves the target sparse-feature metrics across the three fixed seeds.

### Frozen MASSIVE relation sampler

For `z_C`, positives share intent and change locale. Exactly 50% are translations of the same semantic ID; the other 50% use a different utterance ID with the same intent. Opposing examples always have a different intent and the anchor locale. `z_S` reverses the relations.

A deterministic 10,000-pair audit produced:

| Check | Result |
|---|---:|
| `z_C` same intent | 100% |
| `z_C` different locale | 100% |
| exact translation positives | 50% |
| different-ID same-intent positives | 50% |
| `z_S` same locale | 100% |
| `z_S` different intent | 100% |

### Implementation

- Definitive launcher: `code/run_massive_factor_sae.py`.
- Reused trainer: `code/run_massive_sparse_partition_pilot.py`.
- Shared relation sampler: `code/intent_locale_relations.py`.
- Paper Method section: `paper/factor_sae_method.tex`.

Protocol check without training:

```powershell
python code\run_massive_factor_sae.py
```

Run all three definitive seeds:

```powershell
python code\run_massive_factor_sae.py --train
```

Run one definitive seed:

```powershell
python code\run_massive_factor_sae.py --train --seed 20260827
```

### Paper update

The new Method section defines the direct 2304-to-9216 sparse architecture, reciprocal relations, blockwise BatchTopK operation, reconstruction path, binary contrastive equations, and frozen MASSIVE hyperparameters. The previous compiled dense-route manuscript is retained while the new paper is rebuilt section by section.

### Archive action

The previous dense-pipeline `SCIENTIFIC_OBJECT.md` was moved to `archive/plans/2026-09-02_pre_factor_sae/SCIENTIFIC_OBJECT.md`. No experimental result, checkpoint, or dataset was deleted or moved.

## Step 2 — T-SAE-style comparison protocol

**Status: complete.**

### Comparison rows

| Row | Sparse operation | Training objective | Purpose |
|---|---|---|---|
| Raw `H` | none | none | no-SAE reference |
| BatchTopK SAE | one global 9216-feature dictionary, mean `L0=64` | reconstruction | standard SAE baseline |
| Blockwise SAE control | 30/70 blocks, `k_C=13`, `k_S=51` | reconstruction | exact architecture control |
| Matryoshka SAE | global BatchTopK; prefixes 576, 1152, 2304, 4608, 9216 | mean nested reconstruction | hierarchical SAE baseline |
| One-sided factor SAE | 30/70 blockwise BatchTopK | reconstruction + `z_C` relation | tests intent-route supervision alone |
| Reciprocal factor SAE | 30/70 blockwise BatchTopK | reconstruction + `z_C` + `z_S` relations | proposed method |
| Triplet reciprocal SAE | same as proposed | reconstruction + reciprocal triplet relations | loss-form ablation |

Triplet margins `0.1`, `0.2`, and `0.4` are validation-only candidates. The selected margin is the only triplet variant that will proceed to the final three-seed comparison.

### Fairness controls

- Every SAE uses width 9216 and mean `L0=64`.
- Every paired row uses the same seed, normalized inputs, training IDs, batches, optimizer, learning rate, epochs, and decoder normalization.
- Batch ordering and relation sampling use separate deterministic random generators, so relational sampling cannot alter the batches seen in later epochs.
- All definitive runs start from paired fresh initialization. The old reconstruction SAE supplies only the saved input mean and standard deviation; none of its learned encoder or decoder weights are reused.
- Fresh initialization sets the encoder biases and output bias to zero, ties the decoder to the transposed encoder, and normalizes decoder columns.
- The blockwise reconstruction row isolates relational supervision without changing the route widths or activity budgets.

### Runtime verification

- Python compilation passed for the shared trainer, launcher, and protocol test.
- Seven deterministic tests passed, including global BatchTopK activity, Matryoshka group coverage, fresh tied initialization, triplet-margin behavior, and the MASSIVE relation sampler.
- All eight trainable configurations completed a real CUDA forward pass, backward pass, optimizer update, decoder normalization, and inference-threshold calibration on the RTX 3090.
- Every smoke run reported mean `L0=64`.
- Reconstruction controls reported zero relation losses; one-sided training activated only `z_C`; reciprocal training activated both relation losses.

### One-seed validation outcome

All eight configurations completed 30 epochs on the same 90% training-ID split and were evaluated on the same untouched 10% validation IDs in `en-US` and `ja-JP`.

| Method | Reconstruction MSE ↓ | Held-out L0 | `z_C` stability ↑ | Intent feature fraction ↑ | Locale feature fraction ↓ | `z_C` locale probe ↓ | `z_S` locale probe ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BatchTopK SAE | 0.2882 | 78.07 | 0.2030 | 0.5171 | 0.4145 | 0.8294 | 0.9330 |
| Blockwise SAE control | 0.2885 | 77.34 | 0.2362 | 0.5226 | 0.3996 | 0.7189 | 0.8268 |
| Matryoshka SAE | 0.3068 | 77.95 | 0.2622 | 0.5341 | 0.4078 | 0.8877 | 0.6353 |
| One-sided factor SAE | 0.3115 | 78.33 | 0.2269 | 0.6938 | 0.2423 | 0.7441 | 0.8085 |
| Reciprocal factor SAE | 0.3379 | 79.83 | **0.2920** | 0.7553 | 0.1815 | 0.6936 | **0.9591** |
| Triplet `m=0.1` | 0.3024 | 78.54 | 0.2767 | 0.7171 | 0.2144 | 0.7171 | 0.7171 |
| Triplet `m=0.2` | 0.3034 | 81.37 | 0.2795 | 0.7574 | 0.1758 | **0.6580** | 0.8547 |
| Triplet `m=0.4` | 0.3077 | 84.74 | 0.2450 | **0.7929** | **0.1504** | 0.7415 | 0.9556 |

The binary reciprocal model best matches the paper goal: it has the highest cross-locale `z_C` stability and the strongest `z_S` locale probe, while also producing a strongly intent-oriented `z_C` dictionary. Relative to the exact blockwise control, stability increases by 0.0558 (23.6%), the intent-oriented fraction increases by 0.2327, and the locale-oriented fraction decreases by 0.2181. Triplet margin `0.2` is the strongest reconstruction-oriented alternative and remains the candidate margin for canonical evaluation.

These are one-seed validation diagnostics, not the final test table. The canonical evaluator will recompute the complete T-SAE-style probe and reconstruction suite from the saved checkpoints before the triplet margin and final three-seed configurations are frozen.

Validation CSV: `Report/factor_sae_step2_validation_provisional.csv`.

Paper validation draft: `paper/factor_sae_validation_results.tex`.

### Implementation

- Shared trainer: `code/run_massive_sparse_partition_pilot.py`.
- Locked comparison launcher: `code/run_massive_factor_sae.py`.
- Protocol tests: `tests/test_factor_sae_protocol.py` and `tests/test_intent_locale_relations.py`.
- Paper protocol: `paper/factor_sae_method.tex`.

Audit all comparison commands without training:

```powershell
uv run python code\run_massive_factor_sae.py --experiment batchtopk --experiment block_control --experiment matryoshka --experiment one_sided --experiment reciprocal --experiment triplet_m0p1 --experiment triplet_m0p2 --experiment triplet_m0p4
```

Run the one-seed validation suite:

```powershell
uv run python code\run_massive_factor_sae.py --validation --experiment batchtopk --experiment block_control --experiment matryoshka --experiment one_sided --experiment reciprocal --experiment triplet_m0p1 --experiment triplet_m0p2 --experiment triplet_m0p4
```

### Archive action

The eight fresh-initialization smoke checkpoints and reports were moved to `archive/smoke/2026-09-02_step2_factor_sae_fresh/`. The earlier pretrained-initialization smoke diagnostics remain in `archive/smoke/2026-09-02_step2_factor_sae/`. No scientific result was deleted.

## Step 3 — Canonical evaluator

**Status: complete.**

The evaluator is frozen to manifest SHA-256 `6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439`. It uses 10,343 training IDs, 1,149 validation IDs, 2,968 untouched test IDs, 5,800 balanced probe rows, and 5,220 balanced feature-selection rows spanning all 49 seen locales. English/Japanese pairs measure validation stability; Arabic/Chinese remain untouched for final testing.

| Method | Intent AUC ↑ | Stability ↑ | `z_C` locale probe ↓ | Intent feature frac. ↑ | Locale feature frac. ↓ | `z_S` locale ↑ | `z_S` intent ↓ | Rec. FVE ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchTopK SAE | .7264 | .2300 | .9712 | .5068 | .3914 | — | — | .7867 |
| Blockwise SAE control | .6663 | .2683 | .9051 | .5434 | .3546 | .9619 | .6075 | .7866 |
| Matryoshka SAE | .7568 | .2496 | .9737 | .5197 | .3720 | — | — | .7742 |
| One-sided factor SAE | **.8845** | .3239 | .8390 | .7162 | .1903 | .9653 | .5660 | .7680 |
| **Reciprocal factor SAE** | .8723 | **.3921** | **.8373** | **.7814** | **.1387** | .9644 | .4668 | .7467 |
| Triplet `m=.1` | .8288 | .3622 | .8958 | .7137 | .1804 | .9746 | .5019 | .7753 |
| Triplet `m=.2` | .8616 | .3537 | .8653 | .7348 | .1703 | **.9873** | .4740 | .7739 |
| Triplet `m=.4` | .8625 | .3183 | .8746 | .7595 | .1515 | .9763 | **.4613** | .7696 |

The reciprocal model wins the primary sparse-feature task. Against the exact blockwise control it improves AUC by 0.2060, stability by 0.1238 (46.1%), lowers locale-probe accuracy by 0.0678, and reduces `z_S` intent leakage by 0.1407. Against global BatchTopK it improves AUC by 0.1459 and stability by 0.1621 while reducing locale-probe accuracy by 0.1339.

Triplet margin `0.2` is selected as the only max-margin ablation for the final three-seed experiment. The binary objective remains the primary method because it has the strongest sparse-feature stability and lowest `z_C` locale leakage.

Full report: `Report/STEP_03_CANONICAL_EVALUATOR.md`.

Machine-readable results:

- `Report/factor_sae_step3_canonical_validation.csv`
- `Report/factor_sae_step3_canonical_validation.json`

Run the evaluator:

```powershell
python code\evaluate_massive_factor_sae.py
```

Thirteen focused tests pass. The previous dense-pipeline Step 3 report and manifest are preserved in `archive/plans/2026-09-02_pre_factor_sae_step3/`.

### Next step

Run the definitive paired three-seed table for BatchTopK, the exact blockwise control, Matryoshka, one-sided supervision, reciprocal supervision, and triplet margin `0.2`, then evaluate each seed with this frozen evaluator on the untouched test set.

## Step 4 — Definitive paired three-seed test

**Status: complete.**

All six learned configurations were trained for seeds `20260827`, `20260828`, and `20260829` and evaluated once on the untouched `ar-SA`/`zh-CN` test split. The proposed reciprocal SAE gives the strongest combined sparse-feature result.

| Method | Intent bal. acc. | Intent AUC | Stability | `z_C` locale probe | Intent frac. | Locale frac. | `z_S` locale | `z_S` intent | Rec. FVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchTopK SAE | .4680 | .7940 | .1278 | .9871 | .4579 | .4723 | — | — | .6618 |
| Blockwise SAE control | .3527 | .7007 | .1501 | .9665 | .4830 | .4465 | .9878 | .4798 | .6627 |
| Matryoshka SAE | .4112 | .8352 | .1385 | .9821 | .4686 | .4652 | — | — | .6482 |
| One-sided factor SAE | .4188 | **.9165** | .1813 | .8969 | .6923 | .2427 | .9861 | .4679 | .6415 |
| **Reciprocal factor SAE** | **.4625** | .9124 | **.2248** | **.8944** | **.7864** | **.1646** | **.9925** | .3892 | .6098 |
| Triplet `m=.2` | .4590 | .9063 | .1983 | .9096 | .7274 | .2043 | .9916 | **.3818** | **.6427** |

The table reports three-seed means; the full report and CSVs contain standard deviations. Relative to the exact blockwise control, reciprocal supervision raises intent AUC by `.2117`, raises stability by `.0746` (`49.7%`), raises the intent-oriented fraction by `.3035`, lowers the locale-oriented fraction by `.2819`, and lowers `z_S` intent leakage by `.0906`. Relative to global BatchTopK, stability rises by `75.9%` while intent balanced accuracy remains within `0.0055`.

The binary objective remains the main method because it beats the triplet ablation on AUC, stability, `z_C` locale leakage, and both feature-orientation fractions. Triplet learning is retained as the reconstruction-oriented ablation.

Full report: `Report/STEP_04_DEFINITIVE_FACTOR_SAE.md`.

Machine-readable results:

- `Report/factor_sae_step4_definitive_test_per_seed.csv`
- `Report/factor_sae_step4_definitive_test_summary.csv`
- `Report/factor_sae_step4_definitive_test.json`

Run the complete definitive evaluator:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\evaluate_massive_factor_sae.py --final
```

### Next step

Run the 8x width robustness comparison using the same reciprocal method and matched controls. The current 4x result is the main table; 8x tests whether the conclusion survives a larger dictionary before MTOP transfer.

## Step 5 — Sparse-width robustness

**Status: complete and passed.**

The reciprocal factor SAE preserves its advantage when the dictionary is doubled from 9,216 features (4x) to 18,432 features (8x) while keeping the same active fraction. At 8x, it reaches intent AUC `.9156`, compared with `.7373` for the exact blockwise control and `.8094` for global BatchTopK. It also has the highest cross-locale feature stability (`.1063`), lowest `z_C` locale probe (`.9397`), highest intent-oriented fraction (`.6708`), and lowest locale-oriented fraction (`.2530`).

Against the exact 8x blockwise control, reciprocal supervision improves intent AUC by `.1783` and stability by `25.6%`, lowers locale-probe accuracy by `.0363`, and lowers `z_S` intent leakage by `.0695`. Reconstruction FVE remains within `6.3%`, and observed test activity is effectively matched (`224.53` versus `225.16`). The proposed model's AUC is stable across widths (`.9124` at 4x and `.9156` at 8x). Although absolute per-feature stability decreases when the larger dictionary distributes evidence across more coordinates, the proposed method remains best at both widths.

Full report: `Report/STEP_05_SPARSE_WIDTH_ROBUSTNESS.md`.

Machine-readable results:

- `Report/factor_sae_step5_width8_test_per_seed.csv`
- `Report/factor_sae_step5_width8_test_summary.csv`
- `Report/factor_sae_step5_width8_test.json`

Run the 8x experiment and evaluator:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_massive_factor_sae.py --final --width-multiplier 8 --experiment batchtopk --experiment block_control --experiment reciprocal
uv run python code\evaluate_massive_factor_sae.py --width8
```

A 16x sweep is unnecessary because 4x and 8x agree on the method ranking and conclusion.

### Next step

Run the MTOP dataset-transfer experiment with the same direct blockwise BatchTopK architecture and reciprocal controlled relations.

## Step 6 — MTOP dataset transfer

**Status: complete and passed.**

The same fourfold direct reciprocal factor SAE was trained on German, English, Spanish, and French MTOP activations and tested on untouched Hindi and Thai activations. MTOP uses 50 intents and no exact-translation identifiers; all relations therefore use same-intent/different-language positives and different-intent/anchor-language opposing examples.

At exactly matched evaluation activity (`L0=64`, with `L0(z_C)=13` for blockwise methods), reciprocal supervision reaches intent concept AUC `.8707` and cross-language stability `.4978`. The exact blockwise reconstruction control reaches `.5959` AUC and `.3701` stability; global BatchTopK reaches `.6529` and `.3605`. Relative to the exact architectural control, the proposed method improves AUC by `.2748`, stability by `34.5%`, intent relation margin by `.3214`, and retrieval R@1 by `.2910`, while lowering language leakage. The `z_S` route also increases language recovery and lowers intent leakage relative to the blockwise control.

Full report: `Report/STEP_06_MTOP_DATASET_TRANSFER.md`.

Machine-readable results:

- `Report/factor_sae_step6_mtop_test_per_seed.csv`
- `Report/factor_sae_step6_mtop_test_summary.csv`
- `Report/factor_sae_step6_mtop_test.json`

Reproduce the transfer table:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_mtop_factor_sae.py --train
uv run python code\evaluate_mtop_factor_sae.py --evaluate
```

### Next step

Run the staged second-model-family transfer on MASSIVE with the unchanged reciprocal factor SAE.

## Step 7 — Pythia-160M model-family transfer

**Status: complete and passed.**

The unchanged 9,216-feature reciprocal factor SAE was transferred from Gemma 2 2B to Pythia-160M hidden-state index 8. MASSIVE splits, controlled relations, route widths, BatchTopK budgets, optimizer, training length, evaluator, and three seeds were held fixed. All learned methods evaluate at exact `L0=64`.

Relative to the exact blockwise reconstruction control, reciprocal supervision raises Arabic--Chinese feature stability from `.0075` to `.0211` (`181.5%`), raises intent relation margin by `.4797`, increases mean intent concept AUC by `.0048`, lowers locale leakage by `.0090`, improves both feature-orientation fractions, and strengthens both `z_S` diagnostics. Reconstruction FVE remains within `6.9%` of the exact control. The primary stability and route-organization improvements occur in all three paired seeds.

Full report: `Report/STEP_07_PYTHIA160M_MODEL_TRANSFER.md`.

Machine-readable results:

- `Report/factor_sae_pythia160m_transfer_per_seed.csv`
- `Report/factor_sae_pythia160m_transfer_summary.csv`
- `Report/factor_sae_pythia160m_transfer.json`

### Next step

Build the T-SAE-style qualitative feature-stability figure across controlled locale and intent changes.

## Step 8 — T-SAE-style feature stability

**Status: complete and passed.**

The final figure combines a validation-selected Gemma feature trace with three-seed aggregate evidence from Gemma 2 2B and Pythia-160M. No retraining was performed. Features, the representative intent, and the comparison intent are selected only from the frozen validation split; held-out activations do not affect selection, and displayed test IDs use deterministic numeric order.

For Gemma, the selected feature from the reciprocal factor SAE reaches held-out intent AUC `.97` and cross-locale stability `.79`. It activates across both Arabic and Chinese target examples while rejecting every displayed comparison example. The claim is supported by the all-feature averages: reciprocal supervision raises stability from `.1501` to `.2248` relative to the exact blockwise control for Gemma and from `.0075` to `.0211` for Pythia. It also produces the strongest intent-minus-locale feature-orientation gap in both model families.

Full report: `Report/STEP_08_FEATURE_STABILITY.md`.

Artifacts:

- `paper/Figures/factor_sae_figure3_stability.pdf`
- `paper/Figures/factor_sae_figure3_stability.png`
- `paper/figure_data/figure3_factor_stability.json`
- `code/build_factor_stability_figure.py`

Reproduce without training:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\build_factor_stability_figure.py
```

## Step 9 — Sparse-feature interpretability

**Status: complete and passed; the final Diagram Design figure is included in the paper.**

Using the frozen Gemma checkpoints, one feature per intent is selected on validation data and evaluated on unique held-out Arabic/Chinese semantic IDs. Across three seeds, the reciprocal factor SAE reaches mean top-20 intent purity `.817`, reliable-intent coverage `.725`, and selected-feature stability `.563`. Global BatchTopK reaches `.653`, `.370`, and `.457`; the exact blockwise control reaches `.581`, `.341`, and `.375`.

The proposed features also retain near-zero overlap between different intents' top example sets (`.0022`) and high two-locale entropy (`.633`). This shows that the gain comes from distinct, cross-locale intent features rather than duplicated generic detectors or one-language-only activation.

Full report: `Report/STEP_09_FEATURE_INTERPRETABILITY.md`.

Reproduce without training:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
$env:PYTHONIOENCODING='utf-8'
uv run python code\evaluate_feature_interpretability.py
```
