# Reference and Narrative Audit

Date: 2026-09-02

## Outcome

The paper now has a complete citation trail for every external dataset, model, optimizer, SAE architecture, learning objective, and named statistical procedure used in the study. The Introduction and Related Work were rewritten as one argument that leads directly to the Factor-Contrastive SAE.

The final build contains:

- 8 pages of main paper;
- 2 pages of references;
- 4 pages of appendix;
- 25 cited sources;
- 0 missing bibliography keys;
- 0 unresolved citations or cross-references;
- 0 overfull boxes.

## Narrative structure

The Introduction now follows this progression:

1. SAEs make language-model activations available for feature-level analysis.
2. Reconstruction and sparsity do not determine which observed factor organizes a feature.
3. T-SAE and geometry-invariant SAEs show that relations between examples can guide feature discovery.
4. The remaining problem is to preserve two observed factors while making one sparse route invariant to the other.
5. Our reciprocal relations and blockwise BatchTopK construction solve that specific problem.
6. The evaluation isolates the relational objective with an exact architectural control and tests feature recovery, stability, interpretation, transfer, and sparse-width robustness.

The Related Work section now mirrors that logic:

1. sparse feature learning and SAE evaluation;
2. structured and relational SAEs;
3. contrastive factor isolation and the need for explicit relational assumptions;
4. the precise distinction of our two-route sparse method.

## Citation coverage

| Paper component | External source now cited |
|---|---|
| Standard SAEs and interpretable feature dictionaries | Huben et al. (2024); Gao et al. (2025); Gemma Scope |
| SAE evaluation beyond reconstruction | SAEBench |
| Top-k and decoder normalization | Gao et al. (2025) |
| BatchTopK training and fixed-threshold inference | Bussmann et al. (2024) |
| Matryoshka representation principle | Kusupati et al. (2022) |
| Matryoshka SAE baseline | SAEBench |
| Relational SAE precedent and paper organization | T-SAE; geometry-invariant SAE |
| Binary contrastive softmax | Contrastive Predictive Coding; Zimmermann et al. (2021) |
| Controlled content/style isolation | von Kuegelgen et al. (2021); CLAP |
| General non-identifiability of unsupervised disentanglement | Locatello et al. (2019) |
| Triplet-loss ablation | FaceNet |
| Optimizer | AdamW |
| Models | Gemma 2; Pythia |
| Datasets | MASSIVE; MTOP |
| AUC | Fawcett (2006) |
| Pearson correlation | Pearson (1895) |
| Logistic probes | Cox (1958) |
| One-way ANOVA | Fisher (1925) |
| Balanced accuracy | Brodersen et al. (2010) |
| Bootstrap confidence intervals | Efron and Tibshirani (1993) |

## Corrected attribution issues

- The paper no longer implies that the original Matryoshka Representation Learning paper introduced a Matryoshka SAE. It cites the original representation principle and SAEBench's SAE adaptation separately.
- The route loss is identified as a standard contrastive softmax form; the novelty is the reciprocal matched relation and its placement inside the sparse code.
- The triplet objective, bootstrap, BatchTopK inference, and decoder normalization now have direct citations.
- Pythia's author metadata, venue, PMLR volume, pages, publisher, and official URL were corrected.
- BatchTopK, Matryoshka, and Pythia bibliography entries now include their authoritative URLs and complete publication metadata.

## Claim-to-evidence map

| Main claim | Supporting evidence in the paper |
|---|---|
| Controlled relations improve intent-oriented sparse organization | MASSIVE Table 1: intent AUC, stability, orientation, leakage, and opposing-route probes against the exact blockwise control |
| The gain is not caused by the blockwise architecture alone | Exact blockwise reconstruction-only control with matched capacity, budgets, initialization, batches, optimizer, and epochs |
| The sparse features recover recognizable intent concepts | Figure 3 and Table 2: held-out purity, reliable coverage, selected-feature stability, and top-ID overlap |
| The result is not tied to one dictionary scale | Table 3: fourfold and eightfold widths with matched active fractions |
| The objective transfers across datasets | MTOP Table 4 using held-out Hindi and Thai without translation IDs |
| The objective transfers across model families | Pythia-160M result in Appendix Table 6 and Figure 4 |
| The reciprocal construction organizes both routes | One-sided and triplet ablations plus the opposing-route probes in Table 1 |

## Layout decision

The full Pythia transfer table was moved from a stranded ninth main-paper page to Appendix C. Its result remains summarized in the main text and visualized in Figure 4. This restores the requested eight-page ACL main paper without removing evidence.

## Verification

- LaTeX was rebuilt through BibTeX and repeated cross-reference passes.
- Every citation key in the TeX sources exists in `references.bib`.
- The final log contains no unresolved citation, missing reference, or overfull-box warning.
- All 14 rendered pages were visually checked for clipping, overlaps, broken tables, missing glyphs, and figure placement.
- The pre-audit paper and sources remain recoverable in `archive/paper_before_reference_audit_2026-09-02/`.

## Primary sources checked

- T-SAE: https://openreview.net/forum?id=bojVI4l9Kn
- SAEBench: https://proceedings.mlr.press/v267/karvonen25a.html
- Gemma Scope: https://aclanthology.org/2024.blackboxnlp-1.19/
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka Representation Learning: https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html
- Pythia: https://proceedings.mlr.press/v202/biderman23a.html
- MASSIVE: https://aclanthology.org/2023.acl-long.235/
- MTOP: https://aclanthology.org/2021.eacl-main.257/
- FaceNet: https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html
