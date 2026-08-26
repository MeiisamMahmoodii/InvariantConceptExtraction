# Partial-C contrastive supervision result

## Fixed supervision rule

| Pair type | Treatment |
|---|---|
| Same exact fact, different S family | Strong positive |
| Same relation, different subject | Weak positive |
| Same subject, different relation | Weak positive |
| Same domain only | Neutral: excluded from positive and negative terms |
| No shared C factor | Negative |

The full factual matrix contains 318,003 distinct fact pairs: 53,343 same-relation pairs, 798 same-subject pairs, 106,686 domain-only pairs, and 157,176 no-shared-factor pairs. The encoder trained once on 5,049 C-train/S-train rows. Formal and structured remained outside training. The weak-positive loss weight was fixed at 0.5; no cosine target was assigned. No SAE was trained.

## Held-out result

| Measurement | Raw Gemma | Original contrastive | Partial-C contrastive |
|---|---:|---:|---:|
| S-family accuracy, C-test | 1.000 | 0.681 | 0.998 |
| Held-out-S → train-S fact R@1 | 0.915 | 0.801 | 0.444 |
| Train-S → held-out-S fact R@1 | 0.992 | 0.962 | 0.364 |
| Same relation/different subject R@1 | 0.883 | 0.961 | 0.989 |
| Same subject/different relation R@1 | 0.117 | 0.032 | 0.004 |

Same-fact cosine means were 0.994 for train-S/train-S, 0.994 for train-S/held-out-S, and 0.997 for held-out-S/held-out-S. However, different facts in the same relation were nearly as similar (mean 0.993), while different-domain facts were strongly separated (mean -0.721).

## Decision

This trial fails the intended gate. It improved relation-level grouping, but worsened subject-level retrieval and cross-surface fact retrieval; it also left S-family information nearly fully decodable. The loss reached approximately zero after the first epoch because cross-domain negatives create an easy domain-separation solution. Do not proceed to SAE training or add another contrastive variant until the contrastive objective is reconsidered.
