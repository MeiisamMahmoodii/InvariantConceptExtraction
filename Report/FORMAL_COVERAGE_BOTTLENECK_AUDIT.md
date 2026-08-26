# Formal-coverage bottleneck audit

## What changed

The 256-D bottleneck architecture, matched-pair loss, optimizer, and all hyperparameters were unchanged. Only the S-family coverage changed:

- Training: declarative, question, paraphrase, formal.
- Held out: structured only.

## Held-out results

| Metric | Three train families | Four train families |
|---|---:|---:|
| S-family accuracy, all C-test rows ↓ | 0.575 | 0.395 |
| S-family accuracy, trained-family C-test rows ↓ | 0.433 | 0.340 |
| S-family accuracy, wholly unseen family ↓ | 1.000 (formal + structured) | 1.000 (structured) |
| Held-out-S → train-S fact R@1 ↑ | 0.926 | 0.880 |
| Positive cosine exceeds matched negative cosine ↑ | 0.970 | 0.988 |
| Entropy effective rank / 256 ↑ | 32.76 | 30.93 |

## What we found

Adding formal improves S suppression for the families included in training while maintaining non-collapsed geometry and strong matched-pair separation. It does not generalize to the wholly unseen structured family, which remains perfectly decodable. This supports the coverage hypothesis for seen forms, but it does not yet demonstrate S-family invariance outside the observed family set. No SAE was trained.
