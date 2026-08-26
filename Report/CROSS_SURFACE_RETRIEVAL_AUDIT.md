# Frozen cross-surface retrieval audit

The layer-8 contrastive encoder checkpoint was frozen. This audit used only C-test subjects: 117 facts, 1,404 rows. No encoder retraining or SAE training occurred. The retrieval bank excludes formal and structured rows.

## Same-fact cosine similarity

All distinct C-test row pairs with the same fact are included. Train-S is declarative, question, or paraphrase; held-out-S is formal or structured.

| Pair group | Pairs | Mean | Median | 5th–95th percentile |
|---|---:|---:|---:|---:|
| Train-S / train-S | 4,212 | 0.950 | 0.958 | 0.885–0.987 |
| Train-S / held-out-S | 3,159 | 0.771 | 0.801 | 0.490–0.935 |
| Held-out-S / held-out-S | 351 | 0.818 | 0.844 | 0.610–0.955 |

## C-test S-family confusion matrix

Rows are the true family; columns are the diagnostic probe prediction. The diagnostic probe was trained on C-train subjects and all five labels. The contrastive encoder itself never trained on formal or structured rows.

| Actual \ Predicted | Declarative | Question | Paraphrase | Formal | Structured |
|---|---:|---:|---:|---:|---:|
| Declarative | 147 | 90 | 65 | 40 | 9 |
| Question | 7 | 281 | 14 | 41 | 8 |
| Paraphrase | 21 | 83 | 197 | 41 | 9 |
| Formal | 0 | 2 | 0 | 214 | 18 |
| Structured | 0 | 0 | 0 | 0 | 117 |

## Fact retrieval across surface families

The rank is the first representation in the bank with the query’s `fact_id`.

| Query → bank | Queries / bank rows | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Formal or structured → declarative/question/paraphrase | 351 / 1,053 | 0.801 | 0.858 | 0.920 | 0.834 |
| Declarative/question/paraphrase → formal or structured | 1,053 / 351 | 0.962 | 1.000 | 1.000 | 0.979 |

## What we found

The encoder matches facts well across unseen surface forms, but it still retains strong surface-family information, especially for formal and structured formats. Retrieval and linear S-family invariance therefore give different answers: factual matching transfers well, while full format suppression does not.
