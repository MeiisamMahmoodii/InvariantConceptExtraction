# Step 9 — Sparse-feature interpretability and purity

**Status: complete and passed; the final Diagram Design figure is included in the paper.**

## Goal

This step tests whether the stable sparse coordinates correspond to recognizable intent concepts. It uses the frozen Gemma 2 2B checkpoints from the definitive three-seed comparison. No model is retrained.

## Protocol

- Methods: global BatchTopK, exact blockwise reconstruction control, and reciprocal factor SAE.
- Seeds: `20260827`, `20260828`, and `20260829`.
- One feature per intent is selected using only the frozen validation split.
- Validation semantic IDs use the evaluator's disjoint feature-fit and feature-score halves.
- Test evaluation uses the untouched Arabic and Chinese semantic IDs.
- Each semantic ID receives the larger activation across its two exact translations, preventing duplicated translations from occupying two top positions.
- Purity@20 is the fraction of the top 20 semantic IDs that have the selected feature's intended intent.
- The headline purity and coverage calculation includes the 46 intents with at least 20 test semantic IDs.
- Reliable coverage is the fraction of eligible intents with Purity@20 of at least `0.80`.
- Locale entropy measures whether top activations come from both held-out locales; `1.0` is perfectly balanced.
- Selected-feature stability is the Arabic--Chinese activation correlation averaged over all 58 selected intent features.
- Top-ID overlap is the mean pairwise Jaccard overlap between different intent features' top semantic IDs.

## Three-seed results

| Method | Purity@20 ↑ | Reliable coverage ↑ | Feature stability ↑ | Locale entropy ↑ | Top-ID overlap ↓ | Unique feature fraction ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Global BatchTopK | .653 ± .020 | .370 ± .022 | .457 ± .004 | .636 ± .031 | .0025 ± .0005 | 1.000 |
| Blockwise control | .581 ± .068 | .341 ± .091 | .375 ± .040 | .595 ± .047 | .0034 ± .0012 | .977 ± .026 |
| **Reciprocal factor SAE** | **.817 ± .012** | **.725 ± .045** | **.563 ± .003** | .633 ± .040 | **.0022 ± .0001** | **1.000** |

## What happened

- Our method raises mean top-20 intent purity by `.165` over global BatchTopK and by `.236` over the exact blockwise control.
- It raises reliable intent coverage from `.370` to `.725` relative to global BatchTopK.
- It raises selected-feature Arabic--Chinese stability from `.457` to `.563` relative to global BatchTopK.
- Every intent selects a distinct feature for our method in every seed.
- The top semantic sets of different intent features barely overlap (`.0022`), so the features are not all retrieving the same generic examples.
- Locale entropy remains high (`.633`), showing that the top examples are drawn from both Arabic and Chinese rather than one language dominating the catalogue.

## Validation-selected examples

The qualitative panel uses seed `20260827` and selects its three representative intents by validation AUC plus validation stability. Test purity and test activations do not choose the intents.

| Intent feature | Feature | Purity@20 | Test stability | Representative English-aligned examples |
|---|---:|---:|---:|---|
| `takeaway_order` | 299 | .60 | .692 | “order pizza for delivery”; “does pizza hut have delivery”; “order two wings with french fries” |
| `iot_coffee` | 1423 | 1.00 | .906 | “turn on the coffee machine”; “make a coffee”; “coffee” |
| `qa_currency` | 650 | .95 | .891 | “dollar to euro exchange rate”; “exchange rate between US and Mexico”; “US dollar to pound sterling” |

## Artifacts

- Summary JSON: `Report/factor_sae_feature_interpretability.json`
- Per-seed table: `Report/factor_sae_feature_interpretability_per_seed.csv`
- Complete 3-method × 3-seed feature catalogue: `Report/factor_sae_feature_catalogue.csv`
- Paper appendix catalogue: `paper/factor_sae_feature_catalogue_appendix.tex`
- Qualitative figure data: `paper/figure_data/figure4_feature_examples.json`
- Evaluator: `code/evaluate_feature_interpretability.py`

## Reproduce

```powershell
$env:UV_CACHE_DIR='.uv-cache'
$env:PYTHONIOENCODING='utf-8'
uv run python code\evaluate_feature_interpretability.py
```

## Simple outcome

- Our features are more often about one clear intent.
- More than 72% of sufficiently supported intents receive a feature with at least 80% top-example purity.
- Those features remain more consistent when the same meaning is translated between Arabic and Chinese.
- The result supports the T-SAE-style claim that the method improves usable sparse feature organization.
