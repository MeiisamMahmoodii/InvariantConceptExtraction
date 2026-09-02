# Factor-SAE Paper Build Report

Date: 2026-09-02

## Outcome

The manuscript now presents the current direct sparse factor-SAE study rather than the earlier 128-dimensional bottleneck study. The compiled paper is:

- `output/pdf/InvariantConceptExtraction_factor_sae.pdf`
- 15 A4 pages in ACL review format
- 11 pages of main paper and references
- 4 pages of experimental appendix and the complete 58-intent feature catalogue

## Scientific story used

The paper makes one focused claim: controlled relations can organize a single overcomplete sparse dictionary around a designated invariant factor.

The method is defined as:

1. Freeze the language-model activation `H`.
2. Encode it directly into one 9,216-coordinate sparse dictionary.
3. Divide the dictionary into an intent-dominant route `zC` with 2,765 coordinates and a locale-dominant route `zS` with 6,451 coordinates.
4. Apply blockwise BatchTopK with batch budgets `B×13` and `B×51`, for mean training `L0=64`.
5. Reconstruct the original activation with one ordinary linear decoder.
6. Use reciprocal controlled relations: `zC` aligns same-intent/different-locale examples and opposes different-intent/same-locale examples; `zS` reverses the relations.

The exact blockwise reconstruction control has the same encoder, decoder, widths, sparsity budgets, initialization, optimizer, batches, and training duration. This makes the controlled relational objective the isolated difference.

## Evidence included

### MASSIVE main result

Across three paired seeds, reciprocal supervision improves over the exact blockwise control:

- Intent concept AUC: `.7007 → .9124`
- Arabic–Chinese feature stability: `.1501 → .2248`
- Intent-oriented feature fraction: `.4830 → .7864`
- Locale-oriented feature fraction: `.4465 → .1646`
- Locale probe accuracy on `zC`: `.9665 → .8944`

The main comparison also includes raw activations as a dense reference, global BatchTopK, Matryoshka reconstruction, one-sided supervision, and a validation-selected triplet-loss ablation.

### Robustness and transfer

- At eightfold width, reciprocal supervision retains the strongest AUC and stability among the width controls.
- On MTOP held-out Hindi and Thai, AUC improves from `.5959` to `.8707` and stability from `.3701` to `.4978` relative to the exact blockwise control.
- With Pythia-160M, stability improves from `.0075` to `.0211`, and the relation margin and feature orientation move in the intended direction.

### Feature interpretability

Validation-selected intent features are evaluated on held-out Arabic and Chinese semantic IDs:

- Purity@20: `.581 → .817` relative to the exact blockwise control
- Reliable intent coverage: `.341 → .725`
- Selected-feature stability: `.375 → .563`
- Top-ID overlap remains near zero, so distinct intent features are not retrieving the same generic examples.

The appendix contains the complete seed-20260827 catalogue for all 58 intents.

## Paper artifacts

- `paper/main.tex`: abstract, introduction, related work, discussion, scope, conclusion, and document assembly
- `paper/factor_sae_method.tex`: method, objective, architecture, evaluator, ablations, and transfer configurations
- `paper/factor_sae_validation_results.tex`: all main experimental results and interpretations
- `paper/factor_sae_appendix.tex`: split audit, metrics, hyperparameters, controls, transfer details, and reproduction commands
- `paper/factor_sae_feature_catalogue_appendix.tex`: complete intent-feature catalogue
- `paper/references.bib`: bibliography

## Figures

Only figures for the current 9,216-feature direct sparse method are used:

1. `factor_sae_figure1_method.pdf`: direct sparse architecture and blockwise BatchTopK routes
2. `factor_sae_figure2_evidence.pdf`: main MASSIVE result plus MTOP and Pythia transfer
3. `factor_sae_figure3_stability.pdf`: validation-selected held-out activation example and aggregate stability
4. `factor_sae_figure4_examples.pdf`: validation-selected recognizable intent-feature examples

All four follow the saved Diagram Design default profile. Old bottleneck figures are not referenced by the new manuscript.

## Appendix contents

The appendix records:

- exact MASSIVE split sizes and the 50/50 positive construction;
- metric definitions, variance floor, activity definition, bootstrap count, and random seeds;
- the complete fourfold training configuration;
- the eightfold configuration;
- precise definitions of every comparison and ablation;
- MTOP and Pythia transfer protocols;
- commands that reproduce the reported result tables;
- paths to the five machine-readable summary CSV files;
- the complete 58-intent feature catalogue.

## Reproduction entry points

```powershell
uv run python code/run_massive_factor_sae.py --final
uv run python code/evaluate_massive_factor_sae.py --final
uv run python code/run_massive_factor_sae.py --final --width-multiplier 8 --experiment batchtopk --experiment block_control --experiment reciprocal
uv run python code/evaluate_massive_factor_sae.py --width8
uv run python code/run_mtop_factor_sae.py --train
uv run python code/evaluate_mtop_factor_sae.py --evaluate
uv run python code/run_pythia_massive_factor_sae.py --train
uv run python code/run_pythia_massive_factor_sae.py --evaluate
uv run python code/evaluate_feature_interpretability.py
uv run python code/build_factor_stability_figure.py
```

All referenced scripts and CSV summaries were checked and exist in the repository.

## Archive

The old bottleneck manuscript was preserved in:

- `archive/old_bottleneck_paper_2026-09-02/main.tex`
- `archive/old_bottleneck_paper_2026-09-02/motivation.tex`
- `archive/old_bottleneck_paper_2026-09-02/InvariantConceptExtraction_old_bottleneck.pdf`

Nothing in this archive was deleted.

## Build and quality checks

- Built with `pdflatex`, `bibtex`, and three resolving LaTeX passes under the stable job name `factor_sae_paper`.
- All citations and cross-references resolve.
- The final log contains no overfull boxes, undefined citations, undefined references, or LaTeX errors.
- Extracted PDF text contains no unresolved `??` markers and none of the stale bottleneck-study terms searched during the audit.
- Every one of the 15 rendered pages was inspected. Figures and tables fit their pages; no content is clipped, overlapped, or missing.
- The main numerical claims in the abstract and introduction agree with the frozen result tables.

## Files intentionally left alone

The worktree already contains many older experiments and user changes. They were not deleted, reset, or rewritten as part of the paper build. The new paper references only the current factor-SAE files and results.
