# Factor-Contrastive Sparse Autoencoders

This repository contains the current direct sparse factor-SAE study. Frozen language-model activations are encoded into one overcomplete dictionary with two blockwise BatchTopK routes: an intent-dominant route and a locale-dominant route. Reciprocal controlled relations organize the routes while an ordinary decoder retains the reconstruction objective.

## Active project

- `SCIENTIFIC_OBJECT.md` — frozen scientific object, protocol, and headline evidence
- `code/` — the 13 training, evaluation, relation, and figure entry points used by the study
- `tests/` — five tests for the active relation sampler, evaluator, sparse loss, and MTOP protocol
- `data/` — MASSIVE and MTOP frozen inputs plus canonical manifests
- `checkpoint/` — Gemma, MTOP, and Pythia factor-SAE checkpoints
- `Report/` — current machine-readable results and step reports
- `paper/` — current manuscript, bibliography, source data, and four Diagram Design figures
- `output/pdf/InvariantConceptExtraction_factor_sae.pdf` — compiled paper

## Main commands

```powershell
uv run python code/run_massive_factor_sae.py --final
uv run python code/evaluate_massive_factor_sae.py --final
uv run python code/evaluate_massive_factor_sae.py --width8
uv run python code/run_mtop_factor_sae.py --train
uv run python code/evaluate_mtop_factor_sae.py --evaluate
uv run python code/run_pythia_massive_factor_sae.py --train
uv run python code/run_pythia_massive_factor_sae.py --evaluate
uv run python code/evaluate_feature_interpretability.py
```

Earlier experiments are preserved under `archive/`; none were deleted.
