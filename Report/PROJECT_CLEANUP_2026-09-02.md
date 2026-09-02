# Current-Project Cleanup Report

## Outcome

The active repository now contains only the direct factor-contrastive sparse-autoencoder study. Earlier dense bottlenecks, matched-DCL routing, Llama transfer, swaps, synthetic experiments, RAVEL/SALAD/FLORES work, old sparse-placement studies, superseded papers, old figures, caches, and development logs were moved to `archive/project_cleanup_2026-09-02/`.

Nothing was deleted. The archive preserves original relative paths for recovery.

## Active code

The `code/` directory now contains 13 files:

- MASSIVE training and evaluation
- MTOP training and evaluation
- Pythia-160M training and evaluation
- the canonical evaluator and relation sampler
- the shared sparse model/training core
- feature interpretability and stability evaluation
- current Diagram Design figure generation
- the MTOP downloader

## Active evidence

The active `Report/` directory contains only `factor_sae_*` machine-readable results, the cumulative factor-SAE study/build/diagram reports, Steps 3–9 of the current study, and current MASSIVE/MTOP run records.

## Active paper

The active `paper/` directory contains:

- the current manuscript and its four section/appendix files;
- ACL style and bibliography files;
- only the four current figures, each in HTML/SVG/PNG/PDF form;
- only the two JSON data files consumed by the current stability and example figures.

The only PDF in `output/pdf/` is `InvariantConceptExtraction_factor_sae.pdf`.

## Active data and checkpoints

Retained data:

- canonical MASSIVE manifest and frozen MASSIVE train/validation/test inputs;
- MASSIVE layer-8 frozen activation artifacts;
- MTOP raw parquet shards, frozen activation artifacts, and manifest.

Retained checkpoints:

- `checkpoint/sparse_partition_pilot/` — current Gemma fourfold/eightfold models and ablations;
- `checkpoint/mtop_factor_sae/` — current MTOP transfer models;
- `checkpoint/pythia160m_factor_sae/` — current Pythia transfer models.

## Small consistency fixes

- Updated `pyproject.toml` to describe the factor-SAE project and point pytest to `tests/`.
- Added a root `README.md` describing the active tree and main commands.
- Updated stale figure paths and Step 9 completion text in current reports.
- Restored a current-only `paper/figure_data/README.md`.
- Updated the retained representative-feature test to match the final validation-only comparison against the exact blockwise control.

## Verification

- All 22 retained tests pass.
- The MASSIVE entry point completes its protocol audit and confirms the exact 50/50 positive construction.
- All 16 required manifests, activation inputs, checkpoint directories, summary tables, current figures, and final PDF are present.
- Test caches produced during verification were moved into the cleanup archive.

## Recovery

Archived material can be recovered by moving its path from `archive/project_cleanup_2026-09-02/` back to the repository root. The existing older archive directories were left unchanged.
