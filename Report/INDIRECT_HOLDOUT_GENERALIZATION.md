# Indirect-family holdout generalization

## Design

One new natural-language `indirect` family was added separately from V1. It contains one indirect/reordered realization per fact, such as “To identify Zirconium by atomic number, use 40.” The canonical C fields, sources, and original V1 dataset were not changed.

The existing 256-D bottleneck and matched-pair loss were kept unchanged.

- Training families: declarative, question, paraphrase, formal, structured.
- Held out completely: indirect.
- No SAE, reconstruction, or loss change.

## Held-out result

| Measurement | Result |
|---|---:|
| Six-way S-family accuracy, all C-test rows ↓ | 0.377 |
| S-family accuracy, trained-family C-test rows ↓ | 0.326 |
| S-family accuracy, unseen indirect rows ↓ | 0.991 |
| Indirect → trained-family fact R@1 ↑ | 0.991 |
| Indirect → trained-family MRR ↑ | 0.996 |
| Positive cosine exceeds matched negative cosine ↑ | 0.998 |
| Entropy effective rank / 256 ↑ | 31.53 |

## Conclusion

The bottleneck gives excellent fact retrieval for a genuinely new surface family and does not collapse. However, the new family remains almost perfectly identifiable by the S-family probe. Increasing coverage therefore improves invariance within the specified training family set, but does not produce universal surface-form removal. This is a useful boundary: the learned C-dominant block should be described as invariant relative to the nuisance families included in its training support. Do not train an SAE yet.
