# Step 5 — Sparse-width robustness

## Outcome

**Passed.** The reciprocal factor SAE preserves its sparse-feature advantage when the dictionary is doubled from fourfold to eightfold width. At 8x, the proposed method has the highest intent concept AUC, the highest cross-locale feature stability, the lowest `z_C` locale probe, the highest intent-oriented feature fraction, and the lowest locale-oriented feature fraction among the three width-matched models. Its reconstruction FVE is within 6.3% of both controls, and observed test activity is closely matched.

The 4x configuration remains the main result because it provides the clearest factor organization with the smaller dictionary. The 8x result establishes that the conclusion is not specific to 9,216 features. A 16x sweep is not needed because 4x and 8x agree on the method ranking and scientific conclusion.

## Frozen protocol

- Dataset/model: MASSIVE with frozen Gemma 2 2B layer-8 pooled activations.
- Manifest: `data/canonical_evaluation_manifest.json`.
- Manifest SHA-256: `6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439`.
- Test locales: untouched `ar-SA` and `zh-CN` semantic IDs.
- Seeds: `20260827`, `20260828`, and `20260829`.
- Training: 30 epochs, batch size 128, AdamW, learning rate `1e-4`.
- 4x: width 9,216; mean training `L0=64`; block budgets `k_C=13`, `k_S=51`.
- 8x: width 18,432; mean training `L0=128`; block budgets `k_C=26`, `k_S=102`.
- Both widths use the same active fraction and 30/70 route allocation.
- Compared methods: global BatchTopK, the exact blockwise reconstruction control, and reciprocal factor SAE.

All methods at a given width use the same input activations, training IDs, batches, optimizer, epochs, seed, dictionary capacity, and mean activity budget. The exact blockwise control also shares the proposed method's route widths and route-specific BatchTopK budgets.

## Eightfold-width result

Mean ± standard deviation over three paired end-to-end seeds:

| Method | Intent bal. acc. | Intent AUC ↑ | Stability ↑ | `z_C` locale ↓ | Intent frac. ↑ | Locale frac. ↓ | `z_S` locale ↑ | `z_S` intent ↓ | Rec. FVE ↑ | Test `L0` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global BatchTopK | .4835±.0100 | .8094±.0037 | .0797±.0005 | .9908±.0022 | .4408±.0037 | .4637±.0026 | — | — | .7152±.0005 | 225.52±.57 |
| Blockwise control | .3615±.0127 | .7373±.0184 | .0847±.0010 | .9760±.0014 | .4494±.0086 | .4562±.0062 | .9915±.0026 | .4766±.0112 | .7154±.0004 | 225.16±.85 |
| **Reciprocal factor SAE** | .4292±.0092 | **.9156±.0050** | **.1063±.0010** | **.9397±.0106** | **.6708±.0139** | **.2530±.0136** | **.9923±.0012** | **.4071±.0134** | .6705±.0002 | 224.53±1.63 |

## Controlled effect at eightfold width

Relative to the exact blockwise reconstruction control, reciprocal relations:

- raise intent concept AUC by `0.1783`;
- raise cross-locale stability by `0.0216`, or `25.6%` relative;
- lower `z_C` locale-probe accuracy by `0.0363`;
- raise the intent-oriented feature fraction by `0.2214`;
- lower the locale-oriented feature fraction by `0.2032`;
- lower `z_S` intent leakage by `0.0695` while retaining `0.9923` locale accuracy;
- raise intent balanced accuracy by `0.0678`;
- keep reconstruction FVE within `6.3%` of the control;
- differ in observed test `L0` by only `0.63` features (`224.53` versus `225.16`).

Relative to global BatchTopK, the proposed model raises intent AUC by `0.1062`, raises stability by `0.0266` (`33.4%`), lowers locale-probe accuracy by `0.0511`, and produces a substantially cleaner intent/locale feature orientation. Reconstruction FVE remains within `6.3%`, and observed test `L0` differs by less than one feature.

## Fourfold versus eightfold

| Width | Method | Intent AUC ↑ | Stability ↑ | `z_C` locale ↓ | Intent frac. ↑ | Locale frac. ↓ | Rec. FVE ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| 4x | Global BatchTopK | .7940 | .1278 | .9871 | .4579 | .4723 | .6618 |
| 4x | Blockwise control | .7007 | .1501 | .9665 | .4830 | .4465 | .6627 |
| 4x | **Reciprocal factor SAE** | **.9124** | **.2248** | **.8944** | **.7864** | **.1646** | .6098 |
| 8x | Global BatchTopK | .8094 | .0797 | .9908 | .4408 | .4637 | .7152 |
| 8x | Blockwise control | .7373 | .0847 | .9760 | .4494 | .4562 | .7154 |
| 8x | **Reciprocal factor SAE** | **.9156** | **.1063** | **.9397** | **.6708** | **.2530** | .6705 |

Intent concept AUC is unchanged to within 0.004 for the proposed method (`.9124` at 4x and `.9156` at 8x), while reconstruction improves at 8x. Absolute coordinate-level stability falls for every method when the dictionary doubles because intent evidence can be distributed over more features. The relevant controlled comparison nevertheless remains positive: ours improves stability over the exact blockwise control by 49.7% at 4x and 25.6% at 8x. It also retains the best feature orientation and the least `z_C` locale leakage at both widths.

## Reproduction

Train all 8x configurations:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_massive_factor_sae.py --final --width-multiplier 8 --experiment batchtopk --experiment block_control --experiment reciprocal
```

Evaluate the frozen 8x checkpoints:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\evaluate_massive_factor_sae.py --width8
```

## Machine-readable artifacts

- `Report/factor_sae_step5_width8_test_per_seed.csv`
- `Report/factor_sae_step5_width8_test_summary.csv`
- `Report/factor_sae_step5_width8_test.json`

The nine full checkpoints are in `checkpoints/`. Width-8 smoke outputs were moved, not deleted, to `archive/smoke/2026-09-02_step5_width8/`.

## Verification

- All nine full checkpoints exist and contain the requested 18,432-feature configuration.
- Every training epoch reports exact mean BatchTopK `L0=128`.
- The evaluator produced 10 per-seed rows (raw reference plus nine learned checkpoints) and four summary rows.
- Fourteen focused protocol, relation-sampler, and evaluator tests pass.
- The updated method/results fragments compile without LaTeX errors or unresolved references. All four pages were rendered and visually inspected; the width table is full-width and legible.
- Temporary LaTeX and render artifacts were moved to `archive/build_checks/2026-09-02_step5/`.

## Next step

Replicate the direct reciprocal factor SAE on MTOP using intent as `C` and language as `S`. Exact translation IDs are not required; positives are same-intent/different-language and opposing examples are different-intent in the anchor language.
