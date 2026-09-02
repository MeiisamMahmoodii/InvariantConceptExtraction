# ACL Line-Number Layout Fix

## Problem

Review line numbers near two-column breaks were rendered over the manuscript text, including lines 218--221 on page 3.

## Root cause

The ACL review style uses the `lineno` package with the `switch` option. Column-side markers are written to the auxiliary file during page output. After a clean build or a bibliography/layout change, those markers can lag behind the current column breaks and require an additional LaTeX pass.

## Fix

- Preserved the official ACL style and its 1.6 cm margin ruler.
- Removed experimental spacing and float-suppression workarounds.
- Rebuilt from an isolated directory with four consecutive pdfLaTeX passes so the column markers converged.
- Added the extra-pass instruction to the Overleaf package README.

## Verification

- Visually checked all eight main-paper pages.
- Line numbers now remain in the left and right page margins with no overlap in either text column.
- The paper remains 16 pages including references and appendix; main content still ends on page 8.
- No undefined citations, undefined references, `lineno` warnings, LaTeX errors, or overfull boxes remain.
