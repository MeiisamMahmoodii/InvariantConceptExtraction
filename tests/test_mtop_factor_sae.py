import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import mtop_factor_evaluator as mtop


def test_manifest_is_balanced_and_nonparallel(tmp_path):
    manifest = mtop.build_manifest(tmp_path / "manifest.json")
    audit = mtop.audit_manifest(manifest)
    assert audit["training_rows"] == 10213
    assert audit["feature_selection_rows"] == 800
    assert audit["test_rows"] == 5261
    assert audit["intents"] == 50
    assert audit["relation_checks"] == {
        "zC_same_intent_fraction": 1.0,
        "zC_different_language_fraction": 1.0,
        "zC_different_id_fraction": 1.0,
        "zS_same_language_fraction": 1.0,
        "zS_different_intent_fraction": 1.0,
    }


def test_intent_profile_stability_needs_no_parallel_ids():
    metadata = pd.DataFrame({
        "id": [f"item-{row}" for row in range(12)],
        "intent": np.repeat([0, 1, 2], 4),
        "locale": ["hi", "hi", "th", "th"] * 3,
    })
    intent = metadata.intent.to_numpy(dtype=np.float64)
    values = np.stack((intent, intent * 2, metadata.locale.eq("hi").to_numpy()), axis=1)
    manifest = {
        "intents": [0, 1, 2],
        "locales": {"held_out": ["hi", "th"]},
        "constants": {"variance_floor": 1e-12, "bootstrap_resamples": 100},
        "seeds": {"bootstrap": 20260827},
    }
    result = mtop.intent_profile_stability(
        values, metadata, np.array([True, True, True]), manifest
    )
    assert result["stable_features"] == 2
    assert result["profile_intents"] == 3
    assert result["value"] == 1.0
