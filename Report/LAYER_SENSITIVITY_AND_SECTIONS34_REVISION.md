# Layer Sensitivity, Sections 3--4, and PDF Layout Revision

Date: 2026-09-02

## Outcome

The requested layer sensitivity experiment was run and added to the appendix. The exact blockwise reconstruction control and the proposed relational model were trained from paired fresh initializations at Gemma hidden states 4, 8, 16, and 24. The proposed model improves intent concept AUC, cross-locale feature stability, and locale suppression at every tested layer.

Sections 3 and 4 were rewritten around the decisions a reader needs to understand: what relations define each route, why blockwise BatchTopK is required, what the exact control isolates, what success means, and how the evaluation answers the paper's questions. Training configuration remains in the appendix.

The review line-number overlap is fixed. A converged ACL build places line numbers outside both text columns, including pages containing equations, floats, and column transitions.

## Layer sensitivity protocol

- Dataset: MASSIVE.
- Model: Gemma 2 2B.
- Hidden states: 4, 8, 16, and 24.
- Methods: exact blockwise reconstruction control and proposed reciprocal relational SAE.
- Dictionary: 2,304 to 9,216 features.
- Route widths: 2,765 and 6,451.
- Blockwise BatchTopK budgets: 13B and 51B.
- Training: 30 epochs, batch size 128, AdamW, learning rate 1e-4.
- Seed: 20260827.
- Pairing: fresh identical initialization within each layer.
- Evaluation: the same frozen manifest, splits, feature selection, probes, and held-out Arabic--Chinese evaluator used by the main study.

## Layer sensitivity results

Each cell is `blockwise control / ours`.

| Hidden state | Intent AUC up | Stability up | Locale probe down | FVE up | Held-out L0 |
|---:|---:|---:|---:|---:|---:|
| 4 | .6225 / **.8989** | .0277 / **.0594** | .9720 / **.9053** | .4879 / .4367 | 109.13 / 105.03 |
| 8 | .7132 / **.9097** | .1509 / **.2184** | .9639 / **.8908** | .6631 / .6098 | 107.57 / 105.45 |
| 16 | .6847 / **.8956** | .1613 / **.2223** | .9879 / **.9464** | .7222 / .6680 | 117.40 / 118.60 |
| 24 | .6771 / **.9173** | .0857 / **.1463** | .9808 / **.9609** | .6392 / .6030 | 139.18 / 132.72 |

Paired improvements across the four layers:

- Intent AUC: +.1965 to +.2764.
- Cross-locale stability: +.0317 to +.0675.
- Locale-probe accuracy: -.0199 to -.0731.
- FVE: -.0362 to -.0542.

The useful conclusion is not that one layer is uniquely best. The relational advantage has the same direction at early, middle, and late hidden states. Hidden state 8 remains the main operating point because its main result is estimated over three paired seeds and it gives the best locale suppression in the sensitivity run.

## Section 3 revision

Reverse outline:

1. State the two-stage design in plain language: relations define route semantics; a sparse autoencoder realizes them.
2. Define the intent relation using one anchor, one same-intent/different-locale positive, and one different-intent/same-locale opposing example.
3. Reverse the relation for the locale route.
4. Explain why nuisance matching prevents a locale-only solution before presenting the proposition.
5. Map the activation directly into one overcomplete dictionary.
6. Explain why a global activity budget can starve one route, then define blockwise BatchTopK.
7. Introduce the exact architecture-matched reconstruction control.
8. Define reconstruction and the reciprocal relational loss.

This ordering removes the earlier jump from notation to architecture and makes the role of each component explicit.

## Section 4 revision

Reverse outline:

1. Open with four experimental questions: feature organization, reconstruction/activity cost, held-out interpretability, and transfer.
2. Describe models and data.
3. Explain fair comparisons and identify the exact blockwise control as the decisive comparison.
4. Define what a successful intent route and reciprocal locale route should do.
5. Explain validation-before-test selection.
6. Present route organization, reconstruction/activity, stability, interpretability, and transfer in that order.

Exact optimizer, dictionary, route-budget, threshold, bootstrap, and software values remain in the appendix instead of interrupting the main experimental story.

## Line-number fix and visual verification

The ACL review build uses `lineno`, whose two-column positions settle through repeated LaTeX passes. The manuscript now uses a narrow line-number separation so a transient marker remains in the gutter rather than crossing into text. The Overleaf package includes converged `main.aux` and `main.bbl` files.

Visual verification covered main-paper pages 2, 3, 4, and 8 and appendix pages 11 and 12. Line numbers remain outside the text on both columns. Page 8 begins the references immediately after the conclusion. The layer table fits cleanly on appendix page 11.

## Build verification

- PDF build: successful.
- Bibliography build: successful.
- Total pages: 15.
- Main paper including references start: 8 pages.
- References begin after the conclusion on page 8.
- Undefined citations: none.
- Undefined references: none.
- Overfull boxes: none.
- Remaining LaTeX warnings: two harmless `h` to `ht` float-placement adjustments.
- Canonical PDF SHA-256: `2AF71520DFCC7A86B1B5295FE58535E66152EB2BF2D25608460FD6E0AE7310EA`.
- Overleaf ZIP SHA-256: `8E8A0DF800A4A3CB3E3F234E29C20DE40991FDFC9B547052B83B0AE1FE95BFD0`.

## Claim--evidence check

| Claim | Evidence | Status |
|---|---|---|
| Controlled relations reorganize sparse features beyond architecture alone | Exact blockwise control versus ours, three MASSIVE seeds | Pass |
| The method improves cross-locale stability | Main three-seed comparison, width test, MTOP transfer, and four-layer paired test | Pass |
| The result is not tied to one hidden state | Layers 4, 8, 16, and 24 all improve AUC, stability, and locale suppression | Pass |
| Reconstruction remains functional | Held-out FVE reported for every main and sensitivity comparison | Pass |
| The method transfers beyond one dataset and model family | MTOP and Pythia-160M experiments | Pass |

## Reproduction

```powershell
python code\run_massive_factor_sae_layer_sensitivity.py
```

Machine-readable outputs:

- `Report/factor_sae_layer_sensitivity.csv`
- `Report/factor_sae_layer_sensitivity.json`

Layer checkpoints and detailed evaluation records are stored at:

- `D:\data\InvariantConceptExtraction\factor_sae_layer_sensitivity`

No prior checkpoint or result was deleted for this revision.
