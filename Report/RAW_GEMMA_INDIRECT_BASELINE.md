# Raw Gemma indirect-family baseline

Both rows use the same expanded C-test set: all five original families are the seen bank, and `indirect` is the held-out query family. The raw row uses frozen Gemma layer-8 activations; the bottleneck row uses the frozen 256-D block. No training occurred for this baseline.

| Metric | Raw Gemma | C-bottleneck |
|---|---:|---:|
| Seen-S decodability ↓ | 1.000 | 0.326 |
| Unseen indirect decodability ↓ | 1.000 | 0.991 |
| Indirect → seen fact R@1 ↑ | 0.983 | 0.991 |
| Indirect → seen fact MRR ↑ | 0.991 | 0.996 |
| Effective rank | — | 31.53 / 256 |

## Conclusion

The bottleneck substantially reduces decodability of the nuisance families supplied during training, preserves the new indirect family’s residual surface information, and slightly improves indirect-to-seen factual retrieval. This supports relative C specialization rather than a claim that all S information has been erased. The C-block stage can now be frozen under that scoped definition; no SAE has been trained in this step.
