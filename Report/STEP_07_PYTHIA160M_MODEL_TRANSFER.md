# Step 7 — Pythia-160M model-family transfer

**Status: complete and passed.**

## Purpose

This experiment tests whether the direct reciprocal factor SAE transfers from Gemma 2 2B to a different language-model family. Pythia-160M was selected because T-SAE also evaluates both Gemma 2 2B and Pythia-160M.

## Frozen protocol

- Dataset: MASSIVE.
- Preserved factor: `C = intent`.
- Surface factor: `S = locale/language`.
- Model: `EleutherAI/pythia-160m`.
- Activation: masked-mean hidden state at index 8, corresponding to the output of `GPTNeoXModel.layers[7]`.
- Input width: 768.
- Dictionary width: 9,216 for every learned method.
- Route widths: `z_C = 2,765`, `z_S = 6,451`.
- BatchTopK budgets: `k_C = 13`, `k_S = 51`, total mean `L0 = 64`.
- Positive relations: same intent and different locale, with 50% exact IDs and 50% different IDs.
- Opposing examples: different intent and the anchor locale.
- Training: 30 epochs, 45,968 anchors per epoch, batch size 128, AdamW, learning rate `1e-4`.
- Seeds: `20260827`, `20260828`, and `20260829`.
- Canonical MASSIVE manifest SHA-256: `6c33455d91ccd15fc6054c3f077cb2393f387ab136db71fcea76f581d7344439`.

The 9,216-feature width and `L0=64` are held fixed across Gemma and Pythia. This follows the two-model comparison style used by T-SAE and isolates the backbone rather than changing sparse capacity with hidden-state width.

## Frozen activations

The official Pythia-160M weights are stored at `D:\data\pythia-160m`. The resumable extractor produced:

- 563,108 training activations with shape `(563108, 768)`;
- 5,936 untouched Arabic/Chinese test activations with shape `(5936, 768)`;
- exact row alignment with the canonical MASSIVE metadata;
- finite values in the complete test set and a 10,000-row random training audit.

The activation cache is stored at `D:\data\InvariantConceptExtraction\pythia160m_massive_factor_sae`, keeping the large arrays out of the repository.

## Definitive results

Means and standard deviations are over three paired end-to-end seeds. All learned representations have exact total `L0=64`; blockwise methods have exact `L0(z_C)=13`.

| Method | Intent AUC | Stability | Intent relation margin | Intent R@1 | Locale probe | Intent feature fraction | Locale feature fraction | `z_S` locale | `z_S` intent | Rec. FVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw `H` | .5824 | .0737 | -.4487 | .0492 | 1.0000 | .0521 | .9180 | — | — | 1.0000 |
| Global BatchTopK | .5349 ± .0016 | .0065 ± .0002 | -.5614 ± .0177 | .0346 ± .0065 | .9914 ± .0008 | .1493 ± .0052 | .7941 ± .0062 | — | — | .7593 ± .0007 |
| Blockwise reconstruction control | .5179 ± .0039 | .0075 ± .0019 | -.5677 ± .0354 | .0472 ± .0108 | .9882 ± .0045 | .1497 ± .0090 | .8016 ± .0085 | .9907 ± .0031 | .0823 ± .0056 | .7582 ± .0007 |
| **Reciprocal factor SAE (ours)** | **.5227 ± .0055** | **.0211 ± .0016** | **-.0879 ± .0163** | **.0511 ± .0031** | **.9792 ± .0005** | **.1792 ± .0083** | **.7604 ± .0048** | **.9937 ± .0015** | **.0581 ± .0102** | .7060 ± .0037 |

Raw `H` is a 768-active-coordinate dense reference. The learned-method comparison is capacity- and sparsity-matched.

## Controlled effect

Relative to the exact blockwise reconstruction control, reciprocal supervision:

- raises cross-locale feature stability from `.0075` to `.0211`, a `181.5%` relative increase;
- raises the intent relation margin by `.4797`;
- raises mean intent concept AUC by `.0048`;
- raises intent retrieval R@1 by `.0039`;
- lowers `z_C` locale-probe accuracy by `.0090`;
- raises the intent-oriented feature fraction by `.0294`;
- lowers the locale-oriented feature fraction by `.0411`;
- raises `z_S` locale accuracy by `.0030`;
- lowers `z_S` intent leakage by `.0242`;
- keeps reconstruction FVE within `6.9%` of the control.

The stability, locale leakage, intent-feature fraction, locale-feature fraction, `z_S` locale accuracy, and `z_S` intent leakage improvements occur in every paired seed. Relative to global BatchTopK, reciprocal supervision produces `3.2x` the cross-locale stability and a `.4735` larger relation margin at the same total activity.

## Interpretation

The Pythia experiment supports model-family transfer. Pythia's raw multilingual intent geometry is substantially weaker than Gemma's, yet the same reciprocal sparse objective still produces the most stable learned intent route and the cleanest opposing-route diagnostics among the matched sparse methods. No Pythia-specific loss, route allocation, sparsity budget, or evaluator was introduced.

This is the paper's T-SAE-aligned second-model result: the target is improved stable factor organization in an overcomplete sparse dictionary, not dense-model task accuracy.

## Reproduction

Download the public model once:

```powershell
hf download EleutherAI/pythia-160m --local-dir D:\data\pythia-160m
```

Extract frozen activations:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_pythia_massive_factor_sae.py --extract
```

Run the smoke check:

```powershell
uv run python code\run_pythia_massive_factor_sae.py --smoke
```

Train all nine checkpoints and evaluate them:

```powershell
uv run python code\run_pythia_massive_factor_sae.py --train
uv run python code\run_pythia_massive_factor_sae.py --evaluate
```

## Artifacts

- `Report/factor_sae_pythia160m_transfer_per_seed.csv`
- `Report/factor_sae_pythia160m_transfer_summary.csv`
- `Report/factor_sae_pythia160m_transfer.json`
- `checkpoint/pythia160m_factor_sae/` — nine full checkpoints, approximately 0.475 GiB.
- `archive/smoke/2026-09-02_pythia160m_transfer/` — one-step smoke checkpoints.
- `D:\data\InvariantConceptExtraction\pythia160m_massive_factor_sae` — frozen activation cache.

## Verification

- Ten focused architecture and relation tests pass.
- All nine checkpoints contain 30 epochs and the frozen manifest hash.
- All learned representations evaluate at exact total `L0=64`.
- Both blockwise representations evaluate at exact `L0(z_C)=13`.
- Paper claims use only the three-seed summary and paired controlled deltas above.

## Next step

Build the T-SAE-style qualitative stability figure: show individual intent features remaining active across locale changes and changing at controlled intent boundaries.
