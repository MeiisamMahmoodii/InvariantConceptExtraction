# Citation Deduplication Audit

Date: 2026-09-02

## Rule applied

- Cite a work when it is first introduced in the paper.
- Do not repeat the citation for routine later uses of the same method, dataset, model, or metric name.
- Repeat a citation only when the text directly attributes an exact operator, training rule, equation, or reproduced baseline construction to that source.

## Changes

- Removed 12 routine citation commands across the main paper and appendix (38 before, 26 after).
- Removed repeated citations from the Related Work overview after the SAE, Top-k, BatchTopK, and evaluation papers had already been introduced.
- Removed the second BatchTopK citation from the method's inference sentence and the appendix citation from the global BatchTopK baseline description.
- Removed repeated Matryoshka citations from the experimental comparison list, while retaining the citation beside the exact appendix baseline construction.
- Removed repeated AUC, Pearson correlation, bootstrap, and Pythia citations from the appendix because their sources are already cited at first use in the main evaluation section.

## Deliberate repeated citations

The following repetitions remain because they accompany exact borrowed technical content rather than a routine name mention:

- BatchTopK: first introduced in the Introduction and cited again where its minibatch operator is formally defined.
- Top-k SAE training: cited where decoder-column normalization is specified.
- Contrastive learning: cited where the exact softmax loss is written.
- Matryoshka SAE: cited in Related Work and beside the exact prefix-reconstruction baseline used in the appendix.
- Triplet loss: cited at first comparison and beside the exact margin equation in the appendix.

## Build verification

- Clean ACL build completed with BibTeX and four post-BibTeX LaTeX passes.
- Final PDF: 15 pages total; the main paper remains 8 pages.
- The conclusion finishes on page 8 and References begins immediately afterward in the remaining left-column space.
- No undefined citations, undefined references, overfull boxes, or line-number convergence warnings remain in the final log.
- Main-page line numbers were visually checked on pages 1, 3, and 8 and remain outside the text columns.

## Files updated

- `paper/main.tex`
- `paper/factor_sae_method.tex`
- `paper/factor_sae_validation_results.tex`
- `paper/factor_sae_appendix.tex`
- `paper/main.pdf`
- `output/pdf/InvariantConceptExtraction_factor_sae.pdf`
- `output/overleaf/InvariantConceptExtraction_Overleaf.zip`

The pre-edit sources and PDF are preserved in `archive/paper_before_citation_dedup_2026-09-02/`.
