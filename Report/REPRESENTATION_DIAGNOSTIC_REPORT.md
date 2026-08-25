# Frozen Gemma 2 representation diagnostic

## Fixed extraction

Model: `google/gemma-2-2b`, frozen. Layers: 5, 8, 13, and 21, fixed before the audit. Pooling: masked mean over non-padding tokens. Dataset: all 9,576 strict V1 rows. No InfoNCE or SAE training occurred.

## Held-out subject probes

Probes train on `C_train` subjects (6,732 rows) and test on disjoint `C_test` subjects (1,404 rows).

| Layer | S family (chance 0.20) | C domain (chance 0.50) | C relation (chance 0.167) |
|---:|---:|---:|---:|
| 5 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 |
| 13 | 0.999 | 1.000 | 1.000 |
| 21 | 0.999 | 1.000 | 1.000 |

These are fixed linear logistic-regression diagnostic probes, not the proposed method.

## Cosine-similarity distributions

Each type contains 10,000 fixed-seed sampled pairs. Values are mean / median / 5th–95th percentile.

| Layer | A: same fact, different S | B: different fact, same relation | C: different relation, same subject | D: different domain |
|---:|---|---|---|---|
| 5 | 0.965 / 0.970 / 0.931–0.986 | 0.953 / 0.955 / 0.917–0.982 | 0.951 / 0.954 / 0.918–0.974 | 0.923 / 0.925 / 0.893–0.948 |
| 8 | 0.969 / 0.973 / 0.943–0.987 | 0.962 / 0.964 / 0.935–0.987 | 0.957 / 0.959 / 0.931–0.976 | 0.922 / 0.923 / 0.888–0.952 |
| 13 | 0.981 / 0.984 / 0.965–0.992 | 0.979 / 0.980 / 0.962–0.994 | 0.975 / 0.977 / 0.960–0.987 | 0.962 / 0.963 / 0.943–0.977 |
| 21 | 0.966 / 0.970 / 0.939–0.985 | 0.957 / 0.959 / 0.929–0.984 | 0.953 / 0.955 / 0.926–0.975 | 0.920 / 0.921 / 0.885–0.950 |

## Gate result

PASS for representation audit. C structure is strongly decodable on held-out subjects; S family is also strongly decodable; and same-fact/different-S representations are similar but not identical. The frozen representation therefore contains measurable desired and nuisance structure.

Layer choice for any later training remains a separate pre-registration decision. This report does not select a layer or begin contrastive/SAE training.

## Linear subspace-removal diagnostic

For each layer, an S-family logistic probe was fitted on `C_train`; its learned raw-coordinate weight span was projected out before fitting fresh held-out C probes. The reverse diagnostic projects out the combined C-domain and C-relation probe span before fitting a fresh S-family probe.

| Layer | S-subspace rank | C-subspace rank | C domain after removing S | C relation after removing S | S family after removing C |
|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 7 | 1.000 | 1.000 | 1.000 |
| 8 | 5 | 7 | 1.000 | 1.000 | 1.000 |
| 13 | 5 | 7 | 1.000 | 1.000 | 1.000 |
| 21 | 5 | 7 | 1.000 | 1.000 | 0.999 |

At this linear-probe level, C and S are separable: removing the learned S span does not damage held-out C decoding, and removing the learned C span does not materially damage S decoding. This does **not** support a claim of meaningful linear C/S entanglement. Under the stated gate, contrastive training is not yet justified merely to mute a linearly separable S direction.
