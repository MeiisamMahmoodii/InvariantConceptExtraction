# Raw Gemma versus contrastive retrieval

This is the matched C-test retrieval comparison. Both representations use the same 117 subject-disjoint test facts, the same query rows, and the same candidate banks. Raw Gemma is the frozen layer-8 masked-mean activation. The contrastive representation is the existing frozen encoder output. No new training occurred.

| Query → bank | Raw Gemma R@1 | Contrastive R@1 | Raw MRR | Contrastive MRR |
|---|---:|---:|---:|---:|
| Formal → train-S | 0.940 | 0.902 | 0.965 | 0.923 |
| Structured → train-S | 0.863 | 0.598 | 0.907 | 0.656 |
| All held-out → train-S | 0.915 | 0.801 | 0.946 | 0.834 |
| Train-S → all held-out | 0.992 | 0.962 | 0.996 | 0.979 |

| Query → bank | Raw R@5 / R@10 | Contrastive R@5 / R@10 |
|---|---:|---:|
| Formal → train-S | 1.000 / 1.000 | 0.944 / 0.974 |
| Structured → train-S | 0.966 / 1.000 | 0.684 / 0.812 |
| All held-out → train-S | 0.989 / 1.000 | 0.858 / 0.920 |
| Train-S → all held-out | 1.000 / 1.000 | 1.000 / 1.000 |

## What we found

The contrastive encoder reduced its own S-family decodability, but it worsened cross-surface fact retrieval relative to raw Gemma layer 8 in every requested direction. The biggest regression is structured-to-train-S retrieval: R@1 falls from 0.863 to 0.598. On this matched treatment comparison, the current contrastive objective does not improve held-out surface-form invariance.
