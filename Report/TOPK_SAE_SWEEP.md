# Top-k SAE sparsity sweep

What we did: trained matched Top-k SAEs on the frozen raw layer-8 and C-bottleneck activations. Both use the same 8,964 C-train rows, 747 facts, 4x expansion, seed, optimizer, and 30 epochs. Only k changes.

Why: Top-k directly fixes the number of active features, avoiding an arbitrary L1 coefficient.

| k | representation | reconstruction MSE | mean L0 | relation purity | domain purity | S-family purity | C-selective / dictionary | S-selective / dictionary |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | Raw Gemma | 0.285 | 16 | 0.681 | 0.879 | 0.626 | 57/9216 (0.6%) | 25/9216 (0.3%) |
| 16 | C bottleneck | 0.346 | 16 | 0.569 | 0.860 | 0.283 | 129/1024 (12.6%) | 0/1024 (0.0%) |
| 32 | Raw Gemma | 0.256 | 32 | 0.626 | 0.869 | 0.567 | 130/9216 (1.4%) | 53/9216 (0.6%) |
| 32 | C bottleneck | 0.283 | 32 | 0.550 | 0.850 | 0.282 | 158/1024 (15.4%) | 1/1024 (0.1%) |
| 64 | Raw Gemma | 0.245 | 64 | 0.579 | 0.853 | 0.517 | 198/9216 (2.1%) | 68/9216 (0.7%) |
| 64 | C bottleneck | 0.217 | 64 | 0.548 | 0.849 | 0.283 | 150/1024 (14.6%) | 0/1024 (0.0%) |
| 128 | Raw Gemma | 0.243 | 128 | 0.539 | 0.841 | 0.483 | 265/9216 (2.9%) | 111/9216 (1.2%) |
| 128 | C bottleneck | 0.152 | 128 | 0.566 | 0.845 | 0.287 | 182/1024 (17.8%) | 0/1024 (0.0%) |

What we found: across every k, C-bottleneck features have much lower surface-family purity (about 0.28 versus 0.48–0.63 for raw Gemma) and a higher dictionary-normalized fraction of relation-selective features (12.6–17.8% versus 0.6–2.9%). Raw Gemma retains 0.3–1.2% S-selective features; the C bottleneck has 0% except 1 feature at k=32. Reconstruction improves as k grows, with no contrastive retraining.
