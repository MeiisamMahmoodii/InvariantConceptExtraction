# Partial-C contrastive final audit

## Collapse check

The output is L2-normalized by design, so its norm is always 1.0 and cannot itself establish quality. Its centered C-test geometry does establish collapse:

| Measurement | Result |
|---|---:|
| Output dimensions | 1,024 |
| Participation ratio | 1.03 |
| Entropy effective rank | 1.11 |
| Mean cosine, unrelated cross-domain examples | -0.721 |

The representation is effectively one-dimensional: it separates the two domains in opposite directions.

## Pair geometry on C-test

| Pair relationship | Mean cosine | Median |
|---|---:|---:|
| Same exact fact, different S | 0.994 | 0.996 |
| Same relation, different subject | 0.993 | 0.996 |
| Same subject, different relation | 0.986 | 0.988 |
| Same domain only | 0.986 | 0.988 |
| No shared C factor | -0.721 | -0.728 |

The first four groups are all nearly identical. The learned geometry does not distinguish fact, relation, subject, and domain-only structure; it only separates domains.

## Matched comparison

| Metric | Raw Gemma | Plain InfoNCE | Multi-relation |
|---|---:|---:|---:|
| S-family accuracy ↓ | 1.000 | 0.681 | 0.998 |
| Held-out S-family accuracy ↓ | 1.000 | 0.943 | 1.000 |
| C-domain accuracy ↑ | 1.000 | 1.000 | 1.000 |
| C-relation accuracy ↑ | 1.000 | 1.000 | 1.000 |
| Held-out-S → train-S fact R@1 ↑ | 0.915 | 0.801 | 0.444 |
| Same relation/different subject R@1 ↑ | 0.883 | 0.961 | 0.989 |
| Same subject/different relation R@1 ↑ | 0.117 | 0.032 | 0.004 |

## Decision

Fail. The multi-relation objective does not meet the gate for SAE work. It collapses C-test geometry into an almost one-dimensional domain split, does not mute S, and makes both exact unseen-surface retrieval and subject-level reuse worse. No more contrastive variants or SAE training should be run without revising the contrastive design.
