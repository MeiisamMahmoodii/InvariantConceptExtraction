# 256-D C-dominant bottleneck audit

## Training rule

The one bottleneck maps frozen Gemma layer-8 activations to 256 normalized dimensions. It used only C-train and declarative/question/paraphrase rows.

- Positive: same `fact_id`, different `S_family`.
- Negative: different `fact_id`, same exact `S_family` and `S_variant`.
- Formal and structured: not used during training.
- No reconstruction, SAE, relation-aware objective, or partial-C loss.

## Four held-out checks

| Check | Result |
|---|---:|
| S-family probe, all C-test rows ↓ | 0.575 |
| S-family probe, seen-S C-test rows ↓ | 0.433 |
| S-family probe, unseen formal/structured C-test rows ↓ | 1.000 |
| Held-out-S → train-S fact R@1 ↑ | 0.926 |
| Held-out-S → train-S MRR ↑ | 0.941 |
| Positive cosine: same fact, different S ↑ | 0.796 |
| Negative cosine: different fact, same template variant ↓ | 0.115 |
| Positive cosine exceeds matched negative cosine ↑ | 0.970 |
| Participation ratio / 256 ↑ | 21.06 |
| Entropy effective rank / 256 ↑ | 32.76 |

The fixed L2-normalization makes every output norm approximately 1.0; rank, rather than norm variation, is the collapse check.

## Interpretation

This simple controlled bottleneck preserves exact factual retrieval across unseen S forms slightly better than raw Gemma (R@1 0.926 versus 0.915) and clearly separates matched different-fact/same-template negatives. It does not generalize S muting to unseen formal/structured forms: their five-way S-family accuracy is 1.000. Therefore it meets the C-preservation and no-collapse checks, but fails the stated nuisance-invariance gate. Do not train an SAE yet.
