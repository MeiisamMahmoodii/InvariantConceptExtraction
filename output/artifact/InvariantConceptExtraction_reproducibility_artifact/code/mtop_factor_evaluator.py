"""Frozen MTOP manifest and evaluator for the direct factor SAE."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import canonical_evaluator as common
import intent_locale_relations as relations


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "mtop_intent_artifacts"
MANIFEST = ROOT / "data" / "mtop_factor_sae_manifest.json"
SEED = 20260827
SEEDS = (20260827, 20260828, 20260829)
SEEN = ("de", "en", "es", "fr")
HELD_OUT = ("hi", "th")


def _key(row):
    return str(row[0]), str(row[1])


def _lookup(metadata):
    keys = list(zip(metadata.id.astype(str), metadata.locale.astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError("metadata must contain unique (id, locale) rows")
    return {key: row for row, key in enumerate(keys)}


def rows(metadata, keys):
    lookup = _lookup(metadata)
    try:
        return np.asarray([lookup[_key(key)] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"manifest row is absent from metadata: {error.args[0]}") from error


def manifest_sha256(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "MTOP":
        raise ValueError("unsupported MTOP manifest")
    if tuple(manifest["locales"]["seen"]) != SEEN or tuple(manifest["locales"]["held_out"]) != HELD_OUT:
        raise ValueError("MTOP language split changed")
    if tuple(manifest["seeds"]["representation_training"]) != SEEDS:
        raise ValueError("MTOP training seeds changed")
    split_sets = {
        name: set(map(str, manifest["splits"][name]))
        for name in ("training_ids", "validation_ids", "test_ids")
    }
    if split_sets["training_ids"] & split_sets["validation_ids"]:
        raise ValueError("training and feature-selection IDs overlap")
    if manifest["relation_split"]["exact_id_positive_fraction"] != 0.0:
        raise ValueError("MTOP must not assume exact translation IDs")
    if len(manifest["intents"]) != 50:
        raise ValueError("MTOP transfer requires the selected 50 intents")
    return manifest


def load_manifest(path=MANIFEST):
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def build_manifest(path=MANIFEST):
    train = pd.read_csv(ART / "train_metadata.csv")
    test = pd.read_csv(ART / "test_metadata.csv")
    intents = sorted(train.intent.unique().tolist())
    rng = np.random.default_rng(SEED)

    validation_rows = []
    for intent in intents:
        for locale in SEEN:
            candidates = np.flatnonzero(
                (train.intent.to_numpy() == intent) & (train.locale.to_numpy() == locale)
            )
            validation_rows.extend(rng.choice(candidates, 4, replace=False).tolist())
    validation_rows = np.asarray(sorted(validation_rows), dtype=np.int64)
    validation_ids = set(train.id.astype(str).to_numpy()[validation_rows])
    training_rows = np.flatnonzero(~train.id.astype(str).isin(validation_ids).to_numpy())

    probe_rows = []
    for intent in intents:
        candidates = training_rows[train.intent.to_numpy()[training_rows] == intent]
        probe_rows.extend(
            train.iloc[rng.choice(candidates, 100, replace=False)][["id", "locale"]]
            .astype(str).values.tolist()
        )
    locale_rows = []
    for locale in SEEN:
        candidates = training_rows[train.locale.to_numpy()[training_rows] == locale]
        locale_rows.extend(
            train.iloc[rng.choice(candidates, 100, replace=False)][["id", "locale"]]
            .astype(str).values.tolist()
        )

    locale_probe_train, locale_probe_test = [], []
    for locale in HELD_OUT:
        identifiers = test.loc[test.locale == locale, "id"].astype(str).to_numpy()
        identifiers = identifiers[rng.permutation(len(identifiers))]
        midpoint = len(identifiers) // 2
        locale_probe_train.extend(identifiers[:midpoint].tolist())
        locale_probe_test.extend(identifiers[midpoint:].tolist())

    index = relations.build_relation_index(test[["id", "intent", "locale"]])
    anchors = np.arange(len(test))
    positives, negatives = relations.sample_intent_relations(anchors, index, rng, 0.0)
    keys = list(zip(test.id.astype(str), test.locale.astype(str)))
    triples = [
        [list(keys[anchor]), list(keys[positive]), list(keys[negative])]
        for anchor, positive, negative in zip(anchors, positives, negatives)
    ]

    manifest = {
        "schema_version": 1,
        "dataset": "MTOP",
        "scientific_object": {
            "C": "intent",
            "S": "language",
            "zC": "intent-dominant sparse route",
            "zS": "language-dominant sparse route",
        },
        "splits": {
            "training_ids": sorted(train.id.astype(str).to_numpy()[training_rows].tolist()),
            "validation_ids": sorted(validation_ids),
            "test_ids": sorted(test.id.astype(str).tolist()),
        },
        "locales": {"seen": list(SEEN), "held_out": list(HELD_OUT)},
        "intents": intents,
        "probe_split": {
            "training_rows": probe_rows,
            "examples_per_intent": 100,
            "locale_training_rows": locale_rows,
            "examples_per_locale": 100,
        },
        "locale_probe_split": {
            "training_ids": locale_probe_train,
            "test_ids": locale_probe_test,
        },
        "feature_selection_split": {
            "rows": train.iloc[validation_rows][["id", "locale"]].astype(str).values.tolist(),
            "examples_per_intent_locale": 4,
            "locales": list(SEEN),
        },
        "intent_retrieval": {
            "query_locale": HELD_OUT[0],
            "bank_locale": HELD_OUT[1],
            "relevance": "same_intent",
        },
        "relation_split": {
            "exact_id_positive_fraction": 0.0,
            "triples": triples,
        },
        "stability": {
            "definition": "per-feature Pearson correlation between held-out-language intent-mean profiles",
            "units": "active sparse features",
        },
        "seeds": {
            "representation_training": list(SEEDS),
            "manifest_split": SEED,
            "probe": SEED,
            "locale_probe": SEED,
            "relation": SEED,
            "feature_selection": SEED,
            "bootstrap": SEED,
        },
        "constants": {
            "variance_floor": 1e-12,
            "activation_epsilon": 1e-8,
            "minimum_activity_rate": 1e-3,
            "orientation_ratio": 1.1,
            "bootstrap_resamples": 10_000,
            "reconstruction_space": "featurewise-standardized input representation",
            "probe": "StandardScaler(with_mean=False) + SGDClassifier(loss=log_loss, alpha=1e-4, max_iter=1000, tol=1e-3)",
        },
    }
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_manifest(manifest=None):
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    train = pd.read_csv(ART / "train_metadata.csv")
    test = pd.read_csv(ART / "test_metadata.csv")
    training_ids = set(manifest["splits"]["training_ids"])
    validation_ids = set(manifest["splits"]["validation_ids"])
    if training_ids | validation_ids != set(train.id.astype(str)):
        raise ValueError("MTOP training and validation IDs do not partition the activation pool")
    if set(manifest["splits"]["test_ids"]) != set(test.id.astype(str)):
        raise ValueError("MTOP test IDs changed")
    feature_rows = rows(train, manifest["feature_selection_split"]["rows"])
    feature = train.iloc[feature_rows].groupby(["intent", "locale"]).size()
    if len(feature) != 50 * 4 or not feature.eq(4).all():
        raise ValueError("feature-selection split is not balanced by intent and language")
    probe_rows = rows(train, manifest["probe_split"]["training_rows"])
    if not train.iloc[probe_rows].groupby("intent").size().eq(100).all():
        raise ValueError("intent probe split is not balanced")
    triples = manifest["relation_split"]["triples"]
    anchors = rows(test, [triple[0] for triple in triples])
    positives = rows(test, [triple[1] for triple in triples])
    negatives = rows(test, [triple[2] for triple in triples])
    checks = {
        "zC_same_intent_fraction": float(np.mean(test.intent.to_numpy()[anchors] == test.intent.to_numpy()[positives])),
        "zC_different_language_fraction": float(np.mean(test.locale.to_numpy()[anchors] != test.locale.to_numpy()[positives])),
        "zC_different_id_fraction": float(np.mean(test.id.to_numpy()[anchors] != test.id.to_numpy()[positives])),
        "zS_same_language_fraction": float(np.mean(test.locale.to_numpy()[anchors] == test.locale.to_numpy()[negatives])),
        "zS_different_intent_fraction": float(np.mean(test.intent.to_numpy()[anchors] != test.intent.to_numpy()[negatives])),
    }
    if any(value != 1.0 for value in checks.values()):
        raise ValueError(f"MTOP relation audit failed: {checks}")
    return {
        "manifest_sha256": manifest_sha256(manifest),
        "training_rows": len(training_ids),
        "feature_selection_rows": len(validation_ids),
        "test_rows": len(manifest["splits"]["test_ids"]),
        "intents": len(manifest["intents"]),
        "seen_languages": list(SEEN),
        "held_out_languages": list(HELD_OUT),
        "relation_pairs": len(triples),
        "relation_checks": checks,
    }


def intent_profile_stability(test_code, test_meta, active, manifest):
    intents = manifest["intents"]
    profiles = []
    for locale in manifest["locales"]["held_out"]:
        profiles.append(np.stack([
            np.asarray(test_code[(test_meta.locale == locale) & (test_meta.intent == intent)]).mean(0)
            for intent in intents
        ]))
    left, right = (profile[:, active].astype(np.float64) for profile in profiles)
    floor = manifest["constants"]["variance_floor"]
    valid = (left.var(0) > floor) & (right.var(0) > floor)
    if not valid.any():
        raise ValueError("no active feature varies across intents in both held-out languages")
    left, right = left[:, valid], right[:, valid]
    correlations = (
        ((left - left.mean(0)) * (right - right.mean(0))).mean(0)
        / np.sqrt(left.var(0) * right.var(0))
    )
    return common._bootstrap_mean(correlations, manifest) | {
        "stable_features": int(valid.sum()),
        "profile_intents": len(intents),
    }


def evaluate_route(train_code, validation_code, test_code, train_meta, validation_meta, test_meta, manifest):
    probe, _, _ = common._intent_probe(
        train_code, test_code, train_meta, test_meta, manifest
    )
    _, _, active, _, orientation = common._feature_statistics(
        validation_code, validation_meta, manifest
    )
    epsilon = manifest["constants"]["activation_epsilon"]
    return {
        "intent_accuracy": probe["accuracy"],
        "intent_balanced_accuracy": probe["balanced_accuracy"],
        "intent_macro_f1": probe["macro_f1"],
        "intent_retrieval": common.intent_retrieval(test_code, test_meta, manifest),
        "intent_relation_margin": common.intent_relation_margin(test_code, test_meta, manifest),
        "locale_probe": common.locale_probe(test_code, test_meta, manifest),
        "mean_intent_concept_auc": common._intent_concept_auc(
            validation_code, validation_meta, test_code, test_meta, manifest
        ),
        "cross_language_feature_stability": intent_profile_stability(
            test_code, test_meta, active, manifest
        ),
        **orientation,
        "fraction_alive": float(active.mean()),
        "mean_active_features_l0": float(
            (np.abs(np.asarray(test_code)) > epsilon).sum(1).mean()
        ),
    }


def evaluate_sparse(
    train_code, validation_code, test_code,
    train_meta, validation_meta, test_meta,
    test_input, test_reconstruction, manifest,
):
    result = evaluate_route(
        train_code, validation_code, test_code,
        train_meta, validation_meta, test_meta, manifest,
    )
    reconstruction = common.reconstruction_metrics(test_input, test_reconstruction)
    return result | {
        "reconstruction_mse": reconstruction["mse"],
        "reconstruction_fve": reconstruction["fraction_variance_explained"],
        "reconstruction_cosine": reconstruction["mean_cosine_similarity"],
    }
