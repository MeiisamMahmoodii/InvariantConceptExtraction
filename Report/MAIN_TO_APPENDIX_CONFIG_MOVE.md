# Main-to-Appendix Configuration Move

Date: 2026-09-02

## What changed

The main paper's **Training and comparisons** paragraph now contains only information needed to understand the experimental argument:

- which comparisons are evaluated;
- which controls are matched across learned methods; and
- where the complete configuration can be found.

The following implementation details were removed from the main paper because they are configuration rather than experimental evidence:

- 2,304-dimensional input;
- 9,216-feature dictionary;
- 2,765 / 6,451 route allocation;
- BatchTopK budgets of $13B / 51B$;
- mean training $L_0=64$;
- 30 epochs;
- AdamW optimizer;
- batch size 128;
- learning rate $10^{-4}$;
- temperature $.07$;
- relational weight $1$; and
- the three-seed schedule.

These settings are consolidated in Appendix A under **Training configuration**. The AdamW citation was moved with the optimizer description so that the citation appears where the configuration is now defined.

## Main-paper role after revision

The revised paragraph explains the comparison set and fairness controls. It does not duplicate numerical settings from the appendix.

## Verification

- Clean ACL/BibTeX build completed successfully.
- Final PDF remains 15 pages total with an 8-page main paper.
- References still begin on page 8 immediately after the conclusion.
- No undefined citations, undefined references, overfull boxes, or line-number convergence warnings remain.
- Pages 4 and 8 were visually inspected after the move; columns, citations, line numbers, and section transitions remain correctly placed.

## Files changed

- `paper/factor_sae_validation_results.tex`
- `paper/factor_sae_appendix.tex`
- `paper/main.pdf`
- `output/pdf/InvariantConceptExtraction_factor_sae.pdf`
- `output/overleaf/InvariantConceptExtraction_Overleaf.zip`

The previous version is preserved in `archive/paper_before_config_move_2026-09-02/`.
