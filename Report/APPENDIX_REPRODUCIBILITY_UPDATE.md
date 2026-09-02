# Appendix Reproducibility Update

Date: 2026-09-02

## Added to the appendix

- A deterministic six-step description of relation sampling, feature selection, and untouched-test evaluation.
- Exact dependency versions from `uv.lock`.
- Activation precision, sequence length, pooling, GPU, parameter counts, final run counts, and checkpoint sizes.
- A frozen validation-selection table covering global BatchTopK, the exact blockwise control, Matryoshka, one-sided, reciprocal binary, and all three triplet margins.
- A paired per-seed table for the exact blockwise control and reciprocal model on:
  - MASSIVE 4x;
  - MASSIVE 8x;
  - MTOP;
  - Pythia-160M.
- Expanded reproduction commands, including Pythia activation extraction and the protocol test suite.
- Direct filenames for every summary and per-seed CSV.
- A supplementary artifact declaration and checksum sidecar.

## Supplementary artifact

- Archive: `output/artifact/factor-sae.zip`
- SHA-256: `6fbaf50e23c4879a0f428608840a96496095ea1f15453491d0c8a2b9cc909eab`
- Checksum file: `output/artifact/factor-sae.zip.sha256`
- Contents: frozen training/evaluation code, dependency lock, deterministic manifests, protocol tests, validation CSV, all per-seed/summary CSVs, interpretability CSV, and the complete feature catalogue.
- Large activation arrays and checkpoints are not duplicated in the compact artifact.

## Verification

- All appendix values were transcribed from the canonical validation and per-seed CSVs.
- `uv run pytest -q`: 22 tests passed.
- LaTeX build: 16 pages total; the main paper remains 8 pages.
- References still begin in the remaining space on page 8.
- No undefined citations, undefined references, or overfull boxes.
- Pages 11--16 were rendered and visually checked; the new tables are legible and unclipped.
- The Overleaf package was rebuilt with the updated appendix.

## Material deliberately excluded

- old dense bottleneck experiments;
- SupCon development comparisons;
- swaps and generation steering;
- RAVEL, SALAD, and FLORES studies;
- discarded loss variants and old coverage proofs.
