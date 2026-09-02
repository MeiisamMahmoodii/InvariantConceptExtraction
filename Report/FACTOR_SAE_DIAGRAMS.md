# Current Factor-SAE Diagram Set

## Scope

This set belongs only to the current 9,216-feature factor-contrastive sparse-autoencoder paper. It does not reuse or update the old 128-dimensional bottleneck/upstream-routing figures.

The project is pinned to the Diagram Design plugin's `default` profile through `.diagram-design`. All diagrams use its paper, ink, muted, link-blue, and atomic-tangerine tokens; its Instrument Serif, Geist, and Geist Mono typography; accessible inline SVG; and a clean editorial layout without shadows.

## Generated figures

1. `factor_sae_figure1_method`
   - Shows the current architecture: frozen 2,304-dimensional Gemma activation, shared 9,216-feature encoder, blockwise BatchTopK routes, `B×13` and `B×51` activity budgets, concatenated sparse code with mean training `L0=64`, and reconstruction decoder.
   - Source: `paper/factor_sae_method.tex` and the frozen method configuration.

2. `factor_sae_figure2_evidence`
   - Shows paired comparisons against the exact blockwise reconstruction control for MASSIVE intent AUC, MASSIVE stability, MTOP intent AUC, and Pythia stability.
   - Adds the selected 8× robustness result without creating a separate width-sweep figure.
   - Sources: the frozen Step 4, Step 5, Step 6, and Pythia summary CSVs under `Report/`.

3. `factor_sae_figure3_stability`
   - Shows the validation-selected held-out activation example and the aggregate Gemma/Pythia results.
   - The requested top-row order is `Global BatchTopK → Ours → Blockwise control`.
   - The displayed intent is selected using validation data only: among high-AUC candidates where ours improves both AUC and stability over the exact blockwise control, it maximizes the combined validation gain. On held-out data, the selected example gives ours AUC `.97` and stability `.79`, versus control AUC `.52` and stability approximately zero.
   - Source: `paper/figure_data/figure3_factor_stability.json`.

4. `factor_sae_figure4_examples`
   - Shows held-out `takeaway_order`, `iot_coffee`, and `qa_currency` feature examples with their purity and cross-locale stability.
   - Adds the three-seed purity, reliable-coverage, and selected-feature-stability summary.
   - Source: `paper/figure_data/figure4_feature_examples.json`.

## Paper integration

- Figure 1 was inserted into `paper/factor_sae_method.tex`.
- Figures 2, 3, and 4 were inserted into `paper/factor_sae_validation_results.tex`.
- `paper/main.tex` is the complete current factor-SAE manuscript and references only these four figures.

## Reproduction

Run:

```powershell
C:\Python314\python.exe code\build_factor_sae_diagrams.py
```

Each figure is generated as `.html`, `.svg`, `.png`, and `.pdf` in `paper/Figures/`.

## Verification

- All four HTML diagrams pass the Diagram Design `self_check.py` accessibility/single-file check.
- Figure 1 passes `verify-geometry.py` with zero findings.
- All four PDF files are single-page, unencrypted Chromium vector exports with the requested dimensions.
- All four PNG previews were visually inspected. The only detected clipping issue in Figure 1 was corrected before final export.
