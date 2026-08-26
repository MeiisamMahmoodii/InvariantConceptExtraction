# Layer-8 contrastive encoder report

## What was trained

Input: frozen masked-mean Gemma 2 2B layer-8 activations (2,304 dimensions).

The encoder trained on 5,049 rows from `C_train` subjects and only the `declarative`, `question`, and `paraphrase` families. Its 30,294 positive pairs were two different allowed surface families for the same fact. `formal` and `structured` did not enter the training rows or pairs. Training ran for 30 epochs on CUDA; final InfoNCE loss was 0.3050. No SAE was trained.

## Held-out C-test diagnostics

The encoder was frozen. Diagnostic linear probes trained on C-train subjects; unlike the encoder, they can use all five labels so formal and structured can be measured fairly. Evaluation contains only 1,404 C-test rows from subject-disjoint test subjects.

| Measurement | Result | Chance |
|---|---:|---:|
| S family, all C-test rows | 0.681 | 0.200 |
| S family, C-test S-train rows | 0.594 | 0.200 |
| S family, C-test formal/structured rows | 0.943 | 0.200 |
| C domain, all C-test rows | 1.000 | 0.500 |
| C relation, all C-test rows | 1.000 | 0.167 |

## C-test cosine distributions

Values are mean cosine similarity over 10,000 fixed-seed pairs per type.

| Pair type | Mean cosine |
|---|---:|
| Same fact, different surface family | 0.858 |
| Different fact, same relation | 0.362 |
| Different relation, same subject | 0.227 |
| Different domain | -0.005 |

## What we found

The encoder preserved held-out C-domain and C-relation decodability while reducing surface-family decodability overall. It did not generalize that reduction to the completely unseen `formal` and `structured` families: their S-family accuracy is still 0.943. The next decision should therefore treat this as partial S suppression, not full surface-form invariance.
