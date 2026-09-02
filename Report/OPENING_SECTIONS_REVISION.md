# Opening Sections Revision

## What was wrong

- The abstract described components and benchmark names before establishing the research problem.
- The introduction mixed motivation, prior work, experimental design, and detailed results.
- Every contribution began with "We," which made the list read as actions rather than research contributions.
- Related Work named papers without clearly organizing what each line of work contributes and what gap remains.
- The method opened by saying it followed T-SAE's organization, which could make the framework sound derivative.

## What changed

### Abstract

- Opens with the scientific failure mode: an SAE can reconstruct accurately while learning a feature that follows language instead of meaning.
- States the missing ingredient: reconstruction and sparsity do not specify what should remain stable.
- Introduces reciprocal controlled relations as the solution and explains the nuisance-only shortcut in plain language.
- Keeps only the main MASSIVE comparison and the compact transfer/interpretability conclusion.

### Introduction

- Now follows one argument: stable features are desirable; reconstruction underdetermines factor orientation; matched relations resolve that ambiguity; the proposed SAE implements those relations; the experiments test the resulting feature organization.
- Removes named discussion of T-SAE and geometry-invariant SAEs.
- Removes the detailed inventory of baselines and per-dataset result deltas.
- Retains a short qualitative evidence statement. This is appropriate in an ML introduction because it tells the reader whether the proposed idea is empirically supported without duplicating the Results section.

### Contributions

- Replaces action-led "We" sentences with three claim-led contributions:
  1. relation-defined factor organization;
  2. a direct blockwise sparse implementation;
  3. feature-level evidence across datasets, models, and widths.

### Related Work

- Organizes the literature into three research lines: sparse feature learning and evaluation, structured SAEs, and contrastive factor isolation.
- Explains what Top-k, BatchTopK, Matryoshka SAE, T-SAE, geometry-invariant SAE, and controlled contrastive learning contribute.
- Ends each line with the unresolved issue or the precise distinction of this paper.
- T-SAE now appears only here, as prior work, rather than as the organizing authority for our method or experiments.

### Method opening

- Renames the section from "Framework: Factor-Contrastive Sparse Autoencoders" to "Factor-Contrastive Sparse Autoencoder."
- Opens directly with the two components of our framework: the controlled relation sampler and the blockwise sparse autoencoder.
- Removes the sentence stating that the method follows T-SAE's organization.

## Verification

- Rebuilt the complete ACL paper successfully: 16 pages including references and appendix.
- Main content still ends on page 8.
- The conclusion finishes on page 8 and the references begin in the remaining right-column space.
- No undefined citations, undefined references, LaTeX errors, or overfull boxes.
- Visually checked pages 1--3 and page 8 for section flow, clipping, overlap, and reference placement.
- Confirmed the reported headline values against the canonical result tables: AUC .701 to .912, stability .150 to .225, purity .817, and reliable coverage .725.

## Updated artifacts

- `paper/main.tex`
- `paper/factor_sae_method.tex`
- `paper/factor_sae_validation_results.tex`
- `output/pdf/InvariantConceptExtraction_factor_sae.pdf`
- `output/overleaf/InvariantConceptExtraction_Overleaf.zip`

The previous opening and PDF are preserved in `archive/paper_before_opening_revision_2026-09-02/`.
