# ACL eight-page paper revision

## Outcome

- Main text is exactly eight ACL pages.
- References begin on page 9.
- Appendix begins on page 10.
- The paper uses the current 9,216-feature factor-contrastive BatchTopK SAE only; obsolete bottleneck and SupCon narratives are absent.
- The previous paper snapshot is preserved in `archive/paper_before_8page_restructure_2026-09-02/`.

## Structure aligned with T-SAE

1. **Introduction**: problem, relational SAE motivation, method summary, evidence, contributions.
2. **Related Work**: sparse feature learning, structured SAEs, contrastive factor isolation.
3. **Framework**: factor-controlled data-generating relation, then the factor-contrastive sparse autoencoder.
4. **Experimental Evaluation**: implementation details, factor recovery, standard SAE quality, cross-locale consistency, feature interpretability, transfer, and ablations.
5. **Discussion and Conclusions**: mechanism, width robustness, transfer, and scope.

This follows T-SAE's progression from the structured relation, to the SAE objective, to probing/SAE quality/consistency/case studies/ablations.

## Numerical audit

All paper tables and headline deltas were checked against the current machine-readable result CSVs.

| Claim | CSV-derived value | Paper value | Status |
|---|---:|---:|---|
| MASSIVE AUC: ours vs blockwise control | .912443 vs .700695 | .9124 vs .7007 | matched |
| MASSIVE stability: ours vs control | .224772 vs .150140 | .2248 vs .1501 | matched |
| Relative stability increase | 49.708% | 49.7% | matched |
| Relative FVE gap | 7.979% | 8.0% | matched |
| 8x AUC gain vs control | .178309 | .1783 | matched |
| 8x relative stability increase | 25.557% | 25.6% | matched |
| 8x relative FVE gap | 6.275% | 6.3% | matched |
| MTOP AUC / stability gains | .274770 / .127715 | .2748 / .1277 | matched |
| MTOP margin / R@1 gains | .321363 / .290987 | .3214 / .2910 | matched |
| Pythia stability ratio vs global | 3.2359x | 3.2x | matched |
| Pythia margin gain vs control | .479724 | .4797 | matched |
| Interpretability purity / coverage | .817 / .725 | .817 / .725 | matched |

Audited sources:

- `Report/factor_sae_step4_definitive_test_summary.csv`
- `Report/factor_sae_step5_width8_test_summary.csv`
- `Report/factor_sae_step6_mtop_test_summary.csv`
- `Report/factor_sae_pythia160m_transfer_summary.csv`
- `Report/factor_sae_feature_interpretability_per_seed.csv`

## Appendix coverage

The appendix now contains:

- exact MASSIVE split and relation audit;
- canonical manifest hashes;
- formal shortcut proposition;
- metric definitions and fixed evaluator thresholds;
- complete training configuration and hardware;
- exact baseline and validation-selection definitions;
- MTOP and Pythia transfer protocol details;
- complete representative-seed 58-intent feature catalogue;
- reproduction commands and result CSV entry points.

## Recommended additions before submission

These require metadata or measurements that are not currently stored, so they were not invented:

1. Exact Hugging Face model revision hashes and dataset revision hashes.
2. Per-run wall-clock time and peak GPU memory for the main methods.
3. Paired confidence intervals or paired permutation tests for the headline method-control deltas.
4. A release manifest with SHA-256 hashes for checkpoints, manifests, and result CSVs.

The appendix is otherwise sufficient to understand and rerun the reported study.
