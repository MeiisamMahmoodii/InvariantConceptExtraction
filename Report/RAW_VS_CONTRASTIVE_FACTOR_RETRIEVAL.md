# Raw Gemma versus contrastive factor retrieval

This matched audit uses every C-test surface row as a query (1,404 queries). For every query, the bank excludes all rows with the same `fact_id` and all rows in the same `S_family`. No training occurred.

| Retrieval target | Representation | R@1 | R@5 | MRR |
|---|---|---:|---:|---:|
| Same relation, different subject | Raw Gemma layer 8 | 0.883 | 0.984 | 0.928 |
| Same relation, different subject | Frozen contrastive | 0.961 | 0.986 | 0.972 |
| Same subject, different relation | Raw Gemma layer 8 | 0.117 | 0.189 | 0.169 |
| Same subject, different relation | Frozen contrastive | 0.032 | 0.071 | 0.067 |

## What we found

The contrastive encoder improves retrieval of relation-sharing facts, but it damages retrieval of subject-sharing facts with different relations. The training objective used facts with the same subject and different relations as in-batch negatives, so this trade-off is an expected consequence of the current objective rather than evidence that subject information vanished from raw Gemma.
