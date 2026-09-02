import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import canonical_evaluator as evaluator


def tiny_manifest():
    test_ids = ["q0", "q1", "q2", "q3"]
    triples = []
    intents = {"q0": 0, "q1": 0, "q2": 1, "q3": 1}
    for index, identifier in enumerate(test_ids):
        other_same = "q1" if identifier == "q0" else "q0" if identifier == "q1" else "q3" if identifier == "q2" else "q2"
        other_intent = "q2" if intents[identifier] == 0 else "q0"
        for locale, opposite in (("ar", "zh"), ("zh", "ar")):
            triples.append([[identifier, locale], [other_same, opposite], [other_intent, locale]])
    return {
        "schema_version": 1,
        "dataset": "synthetic",
        "splits": {"training_ids": ["t0", "t1", "t2", "t3"], "validation_ids": ["v0", "v1", "v2", "v3"], "test_ids": test_ids},
        "locales": {"seen": ["en", "fr"], "held_out": ["ar", "zh"]},
        "intents": [0, 1],
        "probe_split": {"training_rows": [["t0", "en"], ["t1", "fr"], ["t2", "en"], ["t3", "fr"]], "examples_per_intent": 2, "evaluation_ids_ref": "test_ids"},
        "locale_probe_split": {"training_ids": ["q0", "q2"], "test_ids": ["q1", "q3"]},
        "feature_selection_split": {"ids_ref": "validation_ids", "locales": ["en", "fr"]},
        "intent_retrieval": {"query_locale": "ar", "bank_locale": "zh", "relevance": "same_intent"},
        "relation_split": {"exact_id_positive_fraction": 0.0, "triples": triples},
        "seeds": {"representation_training": [1], "manifest_split": 1, "probe": 1, "locale_probe": 1, "relation": 1, "feature_selection": 1, "bootstrap": 1},
        "constants": {"variance_floor": 1e-12, "activation_epsilon": 1e-8, "minimum_activity_rate": 1e-3, "orientation_ratio": 1.1, "bootstrap_resamples": 200, "reconstruction_space": "standardized", "probe": "test"},
    }


def test_manifest_relations_match_the_scientific_object():
    manifest = evaluator.load_manifest()
    metadata = pd.read_csv(ROOT / "data" / "massive_partition_artifacts" / "test_metadata.csv")
    lookup = {tuple(row): index for index, row in enumerate(metadata[["id", "locale"]].astype(str).values)}
    triples = manifest["relation_split"]["triples"]
    anchor = np.asarray([lookup[tuple(row[0])] for row in triples])
    positive = np.asarray([lookup[tuple(row[1])] for row in triples])
    negative = np.asarray([lookup[tuple(row[2])] for row in triples])
    assert np.mean(metadata.id.to_numpy()[anchor] == metadata.id.to_numpy()[positive]) == 0.5
    assert np.all(metadata.intent.to_numpy()[anchor] == metadata.intent.to_numpy()[positive])
    assert np.all(metadata.locale.to_numpy()[anchor] != metadata.locale.to_numpy()[positive])
    assert np.all(metadata.intent.to_numpy()[anchor] != metadata.intent.to_numpy()[negative])
    assert np.all(metadata.locale.to_numpy()[anchor] == metadata.locale.to_numpy()[negative])


def test_dense_retrieval_uses_intent_not_exact_item():
    manifest = tiny_manifest()
    metadata = pd.DataFrame(
        [(identifier, locale, 0 if identifier in ("q0", "q1") else 1) for identifier in manifest["splits"]["test_ids"] for locale in ("ar", "zh")],
        columns=["id", "locale", "intent"],
    )
    values = np.asarray([[1.0, 0.0] if intent == 0 else [0.0, 1.0] for intent in metadata.intent])
    result = evaluator.intent_retrieval(values, metadata, manifest)
    assert result["R@1"]["value"] == 1.0
    assert result["MRR"]["value"] == 1.0


def test_sparse_metrics_share_activity_variance_and_bootstrap_rules():
    manifest = tiny_manifest()
    train_meta = pd.DataFrame(
        [("t0", "en", 0), ("t1", "fr", 0), ("t2", "en", 1), ("t3", "fr", 1)],
        columns=["id", "locale", "intent"],
    )
    validation_meta = pd.DataFrame(
        [(identifier, locale, 0 if identifier in ("v0", "v1") else 1) for identifier in ("v0", "v1", "v2", "v3") for locale in ("en", "fr")],
        columns=["id", "locale", "intent"],
    )
    test_meta = pd.DataFrame(
        [(identifier, locale, 0 if identifier in ("q0", "q1") else 1) for identifier in ("q0", "q1", "q2", "q3") for locale in ("ar", "zh")],
        columns=["id", "locale", "intent"],
    )

    def code(metadata):
        return np.asarray([
            [intent == 0, intent == 1, locale in ("en", "ar"), locale in ("fr", "zh")]
            for locale, intent in zip(metadata.locale, metadata.intent)
        ], dtype=np.float32)

    train_code, validation_code, test_code = map(code, (train_meta, validation_meta, test_meta))
    result = evaluator.evaluate_sparse_code(
        train_code, validation_code, test_code, train_meta, validation_meta, test_meta,
        test_code, test_code, manifest,
    )
    assert result["intent_accuracy_from_sparse_code"] == 1.0
    assert result["mean_intent_concept_auc"]["value"] == 1.0
    assert result["cross_locale_feature_stability"]["value"] == 1.0
    assert result["reconstruction_mse"] == 0.0


def test_sparse_validation_uses_disjoint_semantic_ids():
    manifest = tiny_manifest()
    manifest["splits"]["validation_ids"] = [str(i) for i in range(8)]
    manifest["locales"]["seen"] = ["a", "b"]
    manifest["intents"] = [0, 1]
    rows = [(identifier, locale, identifier % 2) for identifier in range(8) for locale in ("a", "b")]
    metadata = pd.DataFrame(rows, columns=("id", "locale", "intent"))
    code = np.zeros((len(metadata), 4), np.float32)
    code[np.arange(len(code)), metadata.intent.to_numpy()] = 1
    code[:, 2] = (metadata.locale == "a").astype(float)
    code[:, 3] = (metadata.locale == "b").astype(float)
    result = evaluator.evaluate_sparse_validation(code, metadata, code, code, manifest)
    assert result["fit_semantic_ids"] == result["score_semantic_ids"] == 4
    assert result["locale_probe_fit_semantic_ids"] == result["locale_probe_score_semantic_ids"] == 4
    assert result["mean_intent_concept_auc"]["value"] == 1.0
    assert result["reconstruction_mse"] == 0.0
    assert result["mean_active_features_l0"] == 2.0


def test_k_sparse_probe_and_reconstruction_metrics():
    labels = np.asarray([0, 0, 1, 1])
    values = np.asarray([[2, 0], [1, 0], [0, 1], [0, 2]], dtype=np.float32)
    curve = evaluator.k_sparse_probe_curve(values, labels, values, labels, 1, ks=(1, "all"))
    reconstruction = evaluator.reconstruction_metrics(values, values)
    assert curve["all"]["balanced_accuracy"] == 1.0
    assert curve["1"]["selected_feature_count"] == 1
    assert reconstruction == {"mse": 0.0, "fraction_variance_explained": 1.0, "mean_cosine_similarity": 1.0}
