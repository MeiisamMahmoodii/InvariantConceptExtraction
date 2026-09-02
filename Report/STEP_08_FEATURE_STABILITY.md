# Step 8 — T-SAE-style feature-stability evidence

**Status: complete and passed.**

## What this step tests

This step asks whether the quantitative stability gains correspond to recognizable sparse-feature behavior. It does not retrain any model. It reads the frozen Gemma 2 2B and Pythia-160M checkpoints and uses the same canonical evaluator, splits, batches, thresholds, and seed `20260827` used in the main experiments.

## Selection protocol

- Feature selection uses validation activations only.
- Validation semantic IDs are split into the evaluator's fixed fit and score halves.
- One feature is selected per intent on the fit half by standardized class-mean activation.
- The representative target is the highest validation-AUC intent among selected features at or above median validation cross-locale stability.
- Test labels are used only to require six display semantic IDs.
- No held-out activation is inspected when selecting the feature, intent, comparison intent, or examples.
- The same target intent is used for global BatchTopK, the exact blockwise control, and our method.
- The six target and six comparison test IDs are chosen by deterministic numeric order.
- Full test splits are encoded in their canonical batches before extracting the displayed rows, so BatchTopK behavior is not changed by rebatching the examples.

## Qualitative result

The fixed procedure selects MASSIVE intent `55` and comparison intent `30` for Gemma 2 2B.

| Method | Feature | Validation AUC | Validation stability | Held-out AUC | Held-out stability |
|---|---:|---:|---:|---:|---:|
| Global BatchTopK | 2615 | .9997 | .7493 | .8219 | .4992 |
| Exact blockwise control | 684 | 1.0000 | .7973 | .8484 | .7269 |
| **Reciprocal factor SAE** | **1250** | **1.0000** | **.8445** | **.8722** | **.6563** |

Our selected feature remains active for target examples in both held-out locales and is inactive for every displayed comparison-intent example. The qualitative panel illustrates the intended behavior; the claim rests on the aggregate all-feature results below.

## Aggregate result

| Backbone | Method | Mean feature stability | Intent fraction minus locale fraction |
|---|---|---:|---:|
| Gemma 2 2B | Global BatchTopK | .1278 | -.0144 |
| Gemma 2 2B | Exact blockwise control | .1501 | .0365 |
| Gemma 2 2B | **Reciprocal factor SAE** | **.2248** | **.6219** |
| Pythia-160M | Global BatchTopK | .0065 | -.6448 |
| Pythia-160M | Exact blockwise control | .0075 | -.6518 |
| Pythia-160M | **Reciprocal factor SAE** | **.0211** | **-.5813** |

The proposed method has the highest mean feature stability in both model families. Relative to the exact blockwise control, stability improves by 49.7% for Gemma and 181.5% for Pythia. Its dictionary is also more intent-oriented relative to locale orientation in both backbones.

## Artifacts

- Figure: `paper/Figures/factor_sae_figure3_stability.pdf`
- PNG preview: `paper/Figures/factor_sae_figure3_stability.png`
- Exact feature IDs, semantic IDs, activations, and aggregate values: `paper/figure_data/figure3_factor_stability.json`
- Builder: `code/build_factor_stability_figure.py`
- Paper text: `paper/factor_sae_validation_results.tex`

## Reproduce

No training is required.

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\build_factor_stability_figure.py
```

The focused protocol check is:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python -m pytest tests\test_factor_sae_protocol.py -q
```

## Simple outcome

- Our method makes the average intent feature more stable when language changes.
- The improvement appears in both Gemma 2 2B and Pythia-160M.
- The shown examples were selected without looking at their held-out activations.
- The paper now has a T-SAE-style feature visualization backed by the complete three-seed result.
