import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from intent_locale_relations import audit_sample, build_relation_index, relation_margin, sample_intent_relations


def test_massive_uses_equal_exact_and_nonexact_intent_positives():
    metadata = pd.DataFrame([
        {"id": item, "intent": intent, "locale": locale}
        for intent in (0, 1)
        for item in range(intent * 4, intent * 4 + 4)
        for locale in ("en", "fr")
    ])
    index = build_relation_index(metadata)
    anchors = np.arange(len(metadata))
    c_positive, s_positive = sample_intent_relations(
        anchors, index, np.random.default_rng(7), exact_id_positive_fraction=0.5
    )
    audit = audit_sample(anchors, c_positive, s_positive, index)
    assert audit == {
        "pairs": len(anchors),
        "zC_same_intent_fraction": 1.0,
        "zC_different_locale_fraction": 1.0,
        "zC_exact_id_fraction": 0.5,
        "zC_nonexact_different_id_fraction": 0.5,
        "zS_same_locale_fraction": 1.0,
        "zS_different_intent_fraction": 1.0,
    }


def test_mtop_uses_only_nonexact_intent_positives():
    metadata = pd.DataFrame([
        {"id": f"{locale}-{intent}-{item}", "intent": intent, "locale": locale}
        for locale in ("en", "fr") for intent in (0, 1) for item in range(2)
    ])
    index = build_relation_index(metadata)
    anchors = np.arange(len(metadata))
    c_positive, s_positive = sample_intent_relations(
        anchors, index, np.random.default_rng(8), exact_id_positive_fraction=0
    )
    audit = audit_sample(anchors, c_positive, s_positive, index)
    assert audit["zC_same_intent_fraction"] == 1
    assert audit["zC_different_locale_fraction"] == 1
    assert audit["zC_exact_id_fraction"] == 0
    assert audit["zC_nonexact_different_id_fraction"] == 1
    assert audit["zS_same_locale_fraction"] == 1
    assert audit["zS_different_intent_fraction"] == 1


def test_nonexact_positive_never_falls_back_to_the_anchor_id():
    metadata = pd.DataFrame([
        {"id": "only", "intent": 0, "locale": locale} for locale in ("en", "fr")
    ] + [
        {"id": "other", "intent": 1, "locale": locale} for locale in ("en", "fr")
    ])
    index = build_relation_index(metadata)
    with pytest.raises(ValueError, match="singleton intents"):
        sample_intent_relations([0], index, np.random.default_rng(9), 0)


def test_singleton_intent_is_exact_without_changing_global_mix():
    metadata = pd.DataFrame([
        {"id": item, "intent": intent, "locale": locale}
        for intent, items in ((0, ("only",)), (1, ("a", "b", "c")))
        for item in items for locale in ("en", "fr")
    ])
    index = build_relation_index(metadata)
    anchors = np.arange(len(metadata))
    positive, negative = sample_intent_relations(anchors, index, np.random.default_rng(4), 0.5)
    audit = audit_sample(anchors, positive, negative, index)
    assert audit["zC_exact_id_fraction"] == 0.5
    assert np.all(index[0][positive[:2]] == index[0][anchors[:2]])


def test_exact_parallel_grid_supports_single_id_per_intent_audit():
    metadata = pd.DataFrame([
        {"id": intent, "intent": intent, "locale": locale}
        for intent in (0, 1) for locale in ("en", "fr")
    ])
    values = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    result = relation_margin(
        values, metadata, samples=4, seed=10, exact_id_positive_fraction=1.0
    )
    assert result["same_intent_different_locale_cosine"] == 1
