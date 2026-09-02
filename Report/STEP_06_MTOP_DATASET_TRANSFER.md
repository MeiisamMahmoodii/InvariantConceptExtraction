# Step 6 — MTOP dataset transfer

**Status: complete and passed.**

## Scientific object

- `C = intent`.
- `S = language`.
- `z_C` is the intent-dominant sparse route.
- `z_S` is the language-dominant sparse route.
- For `z_C`, positives share intent and change language; opposing examples change intent while keeping the anchor language.
- `z_S` uses the reciprocal relation.
- MTOP has no required exact-translation identifier, so the exact-ID positive fraction is `0`.

## Frozen protocol

The transfer experiment changes the dataset, languages, and intent inventory while retaining the MASSIVE method and optimization choices.

- Frozen activations: Gemma 2 2B, 2,304 dimensions.
- Sparse model: direct fourfold BatchTopK SAE, 9,216 features.
- Route widths: `z_C = 2,765`, `z_S = 6,451`.
- Blockwise activity: `k_C = 13`, `k_S = 51`, total mean `L0 = 64`.
- Training languages: German, English, Spanish, and French.
- Held-out test languages: Hindi and Thai.
- Intent inventory: 50 intents.
- Representation-training rows: 10,213.
- Disjoint feature-selection rows: 800, exactly four per intent and seen language.
- Untouched test rows: 5,261.
- Seeds: `20260827`, `20260828`, `20260829`.
- Training: 30 epochs, batch size 128, AdamW, learning rate `1e-4`.

The manifest is `data/mtop_factor_sae_manifest.json`, with SHA-256 `7e3f0c13e42fa7016d0d2d678335eed7ae5cdb73549daa13d28d4cba7ded83e3`. Its audit confirms complete intent coverage, disjoint splits, and valid relations for all 5,261 test anchors.

## Evaluation

All sparse models are evaluated with deterministic BatchTopK batches, so every comparison has exactly `L0 = 64`; blockwise models have exactly `L0(z_C) = 13`. Intent concept AUC is the mean one-vs-rest AUC across the 50 intents. Because MTOP does not provide parallel Hindi/Thai test IDs, cross-language feature stability is the correlation between each active feature's 50-intent activation profile in Hindi and the corresponding profile in Thai. The same variance floor, activity threshold, probe split, and 10,000-resample bootstrap implementation are used for every method.

## Definitive results

Means and standard deviations are over three paired end-to-end seeds.

| Method | Intent balanced accuracy | Intent AUC | Cross-language stability | Intent relation margin | Language probe | Intent retrieval R@1 | `z_S` language | `z_S` intent | Rec. FVE | Total L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw `H` | .7315 | .7859 | .5443 | -.0724 | 1.0000 | .7998 | — | — | — | 2304 |
| Global BatchTopK | .5235 ± .0075 | .6529 ± .0038 | .3605 ± .0023 | .0341 ± .0024 | .9837 ± .0020 | .7010 ± .0062 | — | — | .1411 ± .0002 | 64 |
| Blockwise reconstruction control | .3533 ± .0200 | .5959 ± .0186 | .3701 ± .0119 | .0228 ± .0100 | .9416 ± .0050 | .4900 ± .0391 | .9795 ± .0027 | .5071 ± .0140 | .1414 ± .0007 | 64 |
| **Reciprocal factor SAE (ours)** | **.4820 ± .0200** | **.8707 ± .0017** | **.4978 ± .0153** | **.3441 ± .0105** | **.9278 ± .0034** | **.7809 ± .0166** | **.9857 ± .0035** | **.4639 ± .0062** | .1211 ± .0016 | 64 |

## Controlled comparison

Relative to the exact blockwise reconstruction control, reciprocal supervision:

- raises mean intent concept AUC by `.2748`;
- raises cross-language stability by `.1277`, a `34.5%` relative increase;
- raises the intent relation margin by `.3214`;
- raises intent retrieval R@1 by `.2910`;
- raises intent balanced accuracy by `.1287`;
- lowers `z_C` language-probe accuracy by `.0138`;
- raises `z_S` language accuracy by `.0062` and lowers `z_S` intent leakage by `.0432`;
- keeps exactly the same total and route-specific activity budgets.

Relative to global BatchTopK at the same total activity, the proposed model improves intent AUC by `.2178`, stability by `.1373`, relation margin by `.3101`, retrieval R@1 by `.0799`, and lowers language-probe accuracy by `.0559`.

Raw `H` is retained only as a dense reference. The proposed 64-active-feature code has higher intent concept AUC than raw `H` (`.8707` versus `.7859`) while learning an explicit reciprocal route structure.

## Interpretation

MTOP supports dataset transfer. With no parallel identifiers and two unseen test languages, the same controlled relations produce more stable intent features than both sparse controls. The opposing route also moves in the reciprocal direction: it improves language recovery while reducing intent leakage. This is the result needed for the paper's transfer claim.

The intent/locale orientation-count statistic is retained in the complete CSV but omitted from the MTOP headline table because it saturates for all sparse methods and therefore does not distinguish them. Reconstruction remains reported as a fidelity measure; the primary target is factor organization at fixed sparsity.

## Reproduction

Build or audit the frozen manifest:

```powershell
python code\evaluate_mtop_factor_sae.py --build-manifest
```

Train all nine checkpoints:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\run_mtop_factor_sae.py --train
```

Run the definitive evaluator:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run python code\evaluate_mtop_factor_sae.py --evaluate
```

Machine-readable results:

- `Report/factor_sae_step6_mtop_test_per_seed.csv`
- `Report/factor_sae_step6_mtop_test_summary.csv`
- `Report/factor_sae_step6_mtop_test.json`

Checkpoints are in `checkpoint/mtop_factor_sae/`. One-step smoke artifacts are in `archive/smoke/2026-09-02_step6_mtop/`. The earlier threshold-inference evaluation was not deleted; it is archived at `archive/results/2026-09-02_mtop_threshold_inference_provisional/` and is not used in the paper.

## Verification

- Nine full checkpoints are present: three methods times three seeds.
- Every checkpoint completed 30 epochs and records the frozen manifest hash.
- Final evaluation activity is exactly `64` for all learned methods and exactly `13` in `z_C` for both blockwise methods.
- The focused MTOP, protocol, and relation tests pass.

## Next staged experiment

The next replication step is model-family transfer on MASSIVE using the same direct reciprocal factor SAE. The MTOP result requires no method change.
