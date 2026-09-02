# Factor-Contrastive SAE reproducibility artifact

This artifact contains the frozen training/evaluation code, deterministic split manifests, protocol tests, and every machine-readable CSV used by the paper tables.

## Environment

```powershell
uv sync
uv run pytest -q
```

The exact dependency lock is `uv.lock`. Training requires one CUDA GPU. The reported runs used an NVIDIA RTX 3090 with 24 GB memory.

## External inputs

Large frozen activation arrays and model checkpoints are not duplicated in this compact artifact. The code expects:

- Gemma MASSIVE arrays and metadata under `data/massive_partition_artifacts/`;
- MTOP arrays and metadata under `data/mtop_intent_artifacts/`;
- the public MASSIVE and MTOP text datasets for re-extraction;
- a local `EleutherAI/pythia-160m` snapshot for the Pythia extraction command.

`data/canonical_evaluation_manifest.json` and `data/mtop_factor_sae_manifest.json` fix every training, validation, feature-selection, and test ID.

## Main commands

```powershell
uv run python code/run_massive_factor_sae.py --final
uv run python code/evaluate_massive_factor_sae.py --final
uv run python code/run_massive_factor_sae.py --final --width-multiplier 8 --experiment batchtopk --experiment block_control --experiment reciprocal
uv run python code/evaluate_massive_factor_sae.py --width8
uv run python code/run_mtop_factor_sae.py --train
uv run python code/evaluate_mtop_factor_sae.py --evaluate
uv run python code/run_pythia_massive_factor_sae.py --extract
uv run python code/run_pythia_massive_factor_sae.py --train
uv run python code/run_pythia_massive_factor_sae.py --evaluate
uv run python code/evaluate_feature_interpretability.py
```

## Results

The `Report/` directory includes validation selection, per-seed final results, mean/standard-deviation summaries, interpretability results, and the complete feature catalogue. These CSVs are the direct sources for the paper tables.
