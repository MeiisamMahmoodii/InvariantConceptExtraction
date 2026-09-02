"""Canonical MASSIVE manifest and evaluator for every paper representation."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import intent_locale_relations as relations


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "canonical_evaluation_manifest.json"
SEED = 20260827
TRAINING_SEEDS = (20260827, 20260828, 20260829)
HELD_OUT_LOCALES = ("ar-SA", "zh-CN")
VALIDATION_LOCALES = ("en-US", "ja-JP")
STABLE_INTENTS = tuple(sorted(set(range(60)) - {29, 37}))
PROBE_K = (1, 5, 10, 20, "all")


def _key(row):
    return str(row[0]), str(row[1])


def _row_lookup(metadata):
    keys = list(zip(metadata.id.astype(str), metadata.locale.astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError("metadata must contain one row per (id, locale)")
    return {key: index for index, key in enumerate(keys)}


def _rows(metadata, keys):
    lookup = _row_lookup(metadata)
    try:
        return np.asarray([lookup[_key(key)] for key in keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"manifest row is absent from metadata: {error.args[0]}") from error


def manifest_sha256(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or not manifest.get("dataset"):
        raise ValueError("unsupported canonical evaluation manifest")
    splits = manifest["splits"]
    split_sets = {name: set(map(str, splits[name])) for name in ("training_ids", "validation_ids", "test_ids")}
    if any(split_sets[a] & split_sets[b] for a, b in (("training_ids", "validation_ids"), ("training_ids", "test_ids"), ("validation_ids", "test_ids"))):
        raise ValueError("training, validation, and test IDs must be disjoint")
    if set(manifest["locales"]["seen"]) & set(manifest["locales"]["held_out"]):
        raise ValueError("seen and held-out locales must be disjoint")
    if len(manifest["intents"]) < 2:
        raise ValueError("evaluation requires at least two intents")
    if manifest["dataset"] == "MASSIVE":
        if len(manifest["intents"]) != 58:
            raise ValueError("the MASSIVE canonical evaluator requires the shared 58 intents")
        if manifest["relation_split"]["exact_id_positive_fraction"] != 0.5:
            raise ValueError("MASSIVE relation evaluation must use 50% exact-ID positives")
        if tuple(manifest["seeds"]["representation_training"]) != TRAINING_SEEDS:
            raise ValueError("representation-training seeds changed")
        if tuple(manifest["feature_selection_split"]["stability_locales"]) != VALIDATION_LOCALES:
            raise ValueError("validation locales changed")
    return manifest


def build_manifest(path=MANIFEST):
    artifacts = ROOT / "data" / "massive_partition_artifacts"
    full_train = pd.read_csv(artifacts / "train_metadata.csv")
    test = pd.read_csv(artifacts / "test_metadata.csv")
    full_ids = np.sort(full_train.id.unique())
    rng = np.random.default_rng(SEED)
    validation_ids = sorted(map(str, rng.choice(full_ids, max(1, round(0.1 * len(full_ids))), replace=False)))
    training_ids = sorted(set(map(str, full_ids)) - set(validation_ids))
    test_ids = sorted(test.id.astype(str).unique().tolist())
    seen = sorted(full_train.locale.unique().tolist())
    train = full_train[full_train.id.astype(str).isin(training_ids)].reset_index(drop=True)
    validation = full_train[full_train.id.astype(str).isin(validation_ids)].reset_index(drop=True)
    probe_rows = []
    for intent in STABLE_INTENTS:
        candidates = np.flatnonzero(train.intent.to_numpy() == intent)
        chosen = rng.choice(candidates, 100, replace=False)
        probe_rows.extend(train.iloc[chosen][["id", "locale"]].astype(str).values.tolist())
    locale_probe_rows = []
    for locale in seen:
        candidates = np.flatnonzero(train.locale.to_numpy() == locale)
        chosen = rng.choice(candidates, 100, replace=False)
        locale_probe_rows.extend(train.iloc[chosen][["id", "locale"]].astype(str).values.tolist())
    feature_rows = []
    feature_examples_per_intent = 90
    for intent in STABLE_INTENTS:
        candidates = np.flatnonzero(validation.intent.to_numpy() == intent)
        chosen = rng.choice(candidates, feature_examples_per_intent, replace=False)
        feature_rows.extend(validation.iloc[chosen][["id", "locale"]].astype(str).values.tolist())

    shuffled_test_ids = rng.permutation(test_ids)
    midpoint = len(shuffled_test_ids) // 2
    index = relations.build_relation_index(test[["id", "locale", "intent"]])
    anchors = np.arange(len(test))
    positives, negatives = relations.sample_intent_relations(anchors, index, rng, 0.5)
    keys = list(zip(test.id.astype(str), test.locale.astype(str)))
    triples = [[list(keys[a]), list(keys[p]), list(keys[n])] for a, p, n in zip(anchors, positives, negatives)]

    manifest = {
        "schema_version": 1,
        "dataset": "MASSIVE",
        "scientific_object": {"C": "intent", "S": "locale/language", "zC": "intent-dominant route", "zS": "locale-dominant route"},
        "splits": {"training_ids": training_ids, "validation_ids": validation_ids, "test_ids": test_ids},
        "locales": {"seen": seen, "held_out": list(HELD_OUT_LOCALES)},
        "intents": list(STABLE_INTENTS),
        "probe_split": {
            "training_rows": probe_rows,
            "examples_per_intent": 100,
            "locale_training_rows": locale_probe_rows,
            "examples_per_locale": 100,
            "evaluation_ids_ref": "test_ids",
        },
        "locale_probe_split": {"training_ids": shuffled_test_ids[:midpoint].tolist(), "test_ids": shuffled_test_ids[midpoint:].tolist()},
        "feature_selection_split": {
            "ids_ref": "validation_ids",
            "rows": feature_rows,
            "examples_per_intent": feature_examples_per_intent,
            "locales": seen,
            "stability_locales": list(VALIDATION_LOCALES),
        },
        "intent_retrieval": {"query_locale": HELD_OUT_LOCALES[0], "bank_locale": HELD_OUT_LOCALES[1], "relevance": "same_intent"},
        "relation_split": {"exact_id_positive_fraction": 0.5, "triples": triples},
        "seeds": {
            "representation_training": list(TRAINING_SEEDS),
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
    train = pd.read_csv(ROOT / "data" / "massive_partition_artifacts" / "train_metadata.csv")
    test = pd.read_csv(ROOT / "data" / "massive_partition_artifacts" / "test_metadata.csv")
    observed = {name: set(map(str, manifest["splits"][name])) for name in ("training_ids", "validation_ids", "test_ids")}
    if observed["training_ids"] | observed["validation_ids"] != set(train.id.astype(str)):
        raise ValueError("manifest training and validation IDs do not partition the activation training set")
    if observed["test_ids"] != set(test.id.astype(str)):
        raise ValueError("manifest test IDs do not match the activation test set")
    probe_rows = _rows(train, manifest["probe_split"]["training_rows"])
    probe_counts = train.intent.to_numpy()[probe_rows]
    if any(np.sum(probe_counts == intent) != manifest["probe_split"]["examples_per_intent"] for intent in manifest["intents"]):
        raise ValueError("probe split is not balanced across intents")
    locale_rows = _rows(train, manifest["probe_split"]["locale_training_rows"])
    locale_counts = train.locale.to_numpy()[locale_rows]
    if any(np.sum(locale_counts == locale) != manifest["probe_split"]["examples_per_locale"] for locale in manifest["locales"]["seen"]):
        raise ValueError("probe split is not balanced across locales")
    validation = train[train.id.astype(str).isin(manifest["splits"]["validation_ids"])].reset_index(drop=True)
    feature_rows = _rows(validation, manifest["feature_selection_split"]["rows"])
    feature_counts = validation.intent.to_numpy()[feature_rows]
    if any(np.sum(feature_counts == intent) != manifest["feature_selection_split"]["examples_per_intent"] for intent in manifest["intents"]):
        raise ValueError("feature-selection split is not balanced across intents")
    triples = manifest["relation_split"]["triples"]
    anchor = _rows(test, [triple[0] for triple in triples])
    positive = _rows(test, [triple[1] for triple in triples])
    negative = _rows(test, [triple[2] for triple in triples])
    exact = test.id.to_numpy()[anchor] == test.id.to_numpy()[positive]
    checks = {
        "exact_id_positive_fraction": float(exact.mean()),
        "zC_same_intent_fraction": float(np.mean(test.intent.to_numpy()[anchor] == test.intent.to_numpy()[positive])),
        "zC_different_locale_fraction": float(np.mean(test.locale.to_numpy()[anchor] != test.locale.to_numpy()[positive])),
        "matched_negative_different_intent_fraction": float(np.mean(test.intent.to_numpy()[anchor] != test.intent.to_numpy()[negative])),
        "matched_negative_anchor_locale_fraction": float(np.mean(test.locale.to_numpy()[anchor] == test.locale.to_numpy()[negative])),
    }
    if checks != {
        "exact_id_positive_fraction": 0.5,
        "zC_same_intent_fraction": 1.0,
        "zC_different_locale_fraction": 1.0,
        "matched_negative_different_intent_fraction": 1.0,
        "matched_negative_anchor_locale_fraction": 1.0,
    }:
        raise ValueError(f"canonical relation audit failed: {checks}")
    return {
        "manifest_sha256": manifest_sha256(manifest),
        "training_ids": len(observed["training_ids"]),
        "validation_ids": len(observed["validation_ids"]),
        "test_ids": len(observed["test_ids"]),
        "seen_locales": len(manifest["locales"]["seen"]),
        "held_out_locales": manifest["locales"]["held_out"],
        "intents": len(manifest["intents"]),
        "probe_rows": len(probe_rows),
        "locale_probe_rows": len(locale_rows),
        "feature_selection_rows": len(feature_rows),
        "relation_pairs": len(triples),
        "relation_checks": checks,
    }


def _bootstrap_mean(values, manifest):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(manifest["seeds"]["bootstrap"])
    count = manifest["constants"]["bootstrap_resamples"]
    means = np.empty(count, dtype=np.float64)
    for start in range(0, count, 1000):
        size = min(1000, count - start)
        means[start:start + size] = values[rng.integers(0, len(values), (size, len(values)))].mean(1)
    return {"value": float(values.mean()), "bootstrap_95_ci": [float(x) for x in np.quantile(means, (0.025, 0.975))], "units": int(len(values))}


def _classifier(seed):
    return SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=seed)


def _probe(train_values, train_labels, test_values, test_labels, seed):
    sparse_input = sp.issparse(train_values) or sp.issparse(test_values)
    if not sparse_input:
        train_values, test_values = np.asarray(train_values), np.asarray(test_values)
        sparse_input = np.mean(train_values == 0) > 0.5 and np.mean(test_values == 0) > 0.5
    train_values = sp.csr_matrix(train_values) if sparse_input else np.asarray(train_values)
    test_values = sp.csr_matrix(test_values) if sparse_input else np.asarray(test_values)
    scaler = StandardScaler(with_mean=False).fit(train_values)
    model = _classifier(seed).fit(scaler.transform(train_values), train_labels)
    prediction = model.predict(scaler.transform(test_values))
    return {
        "accuracy": float(np.mean(prediction == test_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, prediction)),
        "macro_f1": float(f1_score(test_labels, prediction, average="macro")),
    }, prediction


def k_sparse_probe_curve(train_values, train_labels, test_values, test_labels, seed, ks=PROBE_K):
    """Fit the same probe after selecting the top k coordinates by training-only ANOVA."""
    train_values, test_values = np.asarray(train_values), np.asarray(test_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        scores = f_classif(train_values, np.asarray(train_labels))[0]
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    sparse_input = np.mean(train_values == 0) > 0.5
    result = {}
    for k in ks:
        selected = order if k == "all" else order[: min(int(k), len(order))]
        selected_train, selected_test = train_values[:, selected], test_values[:, selected]
        if sparse_input:
            selected_train, selected_test = sp.csr_matrix(selected_train), sp.csr_matrix(selected_test)
        metrics, _ = _probe(
            selected_train, train_labels, selected_test, test_labels, seed,
        )
        result[str(k)] = metrics | {"selected_feature_count": int(len(selected))}
    return result


def validation_semantic_rows(metadata, manifest):
    """Return disjoint validation fit/score rows, stratified by intent and frozen by ID."""
    rng = np.random.default_rng(manifest["seeds"]["feature_selection"])
    allowed = set(map(str, manifest["splits"]["validation_ids"]))
    semantic = metadata[metadata.id.astype(str).isin(allowed)].drop_duplicates("id")
    fit_ids, score_ids = set(), set()
    for intent in manifest["intents"]:
        intent_ids = semantic.loc[semantic.intent == intent, "id"].astype(str).to_numpy()
        if len(intent_ids) < 2:
            raise ValueError(f"intent {intent} has fewer than two validation semantic IDs")
        intent_ids = intent_ids[rng.permutation(len(intent_ids))]
        split = max(1, len(intent_ids) // 2)
        fit_ids.update(intent_ids[:split])
        score_ids.update(intent_ids[split:])
    ids = metadata.id.astype(str).to_numpy()
    locale_mask = metadata.locale.isin(manifest["feature_selection_split"]["stability_locales"]).to_numpy()
    return (
        np.flatnonzero(locale_mask & np.isin(ids, list(fit_ids))),
        np.flatnonzero(locale_mask & np.isin(ids, list(score_ids))),
        fit_ids,
        score_ids,
    )


def _orientation_on_rows(values, metadata, rows, manifest):
    values = np.asarray(values[rows], dtype=np.float64)
    intents = metadata.intent.to_numpy()[rows]
    locales = metadata.locale.to_numpy()[rows]
    constants = manifest["constants"]
    mean, variance = values.mean(0), values.var(0)
    scale = np.sqrt(np.maximum(variance, constants["variance_floor"]))
    active = (np.abs(values) > constants["activation_epsilon"]).mean(0) >= constants["minimum_activity_rate"]
    intent_score = np.stack([values[intents == label].mean(0) for label in manifest["intents"]]).max(0) - mean
    locale_score = np.stack([values[locales == label].mean(0) for label in np.unique(locales)]).max(0) - mean
    intent_score, locale_score = intent_score / scale, locale_score / scale
    ratio = constants["orientation_ratio"]
    if not active.any():
        raise ValueError("representation has no active features under the canonical definition")
    return active, {
        "active_features": int(active.sum()),
        "intent_oriented_feature_fraction": float(np.mean(intent_score[active] > ratio * locale_score[active])),
        "locale_oriented_feature_fraction": float(np.mean(locale_score[active] > ratio * intent_score[active])),
    }


def _validation_intent_auc(values, metadata, fit_rows, score_rows, manifest):
    fit, score = np.asarray(values[fit_rows], dtype=np.float64), np.asarray(values[score_rows], dtype=np.float64)
    fit_labels, score_labels = metadata.intent.to_numpy()[fit_rows], metadata.intent.to_numpy()[score_rows]
    mean, variance = fit.mean(0), fit.var(0)
    scale = np.sqrt(np.maximum(variance, manifest["constants"]["variance_floor"]))
    feature_scores = np.stack([(fit[fit_labels == intent].mean(0) - mean) / scale for intent in manifest["intents"]])
    features = feature_scores.argmax(1)
    aucs = np.asarray([
        roc_auc_score(score_labels == intent, score[:, feature])
        for intent, feature in zip(manifest["intents"], features)
    ])
    return _bootstrap_mean(aucs, manifest) | {
        "selected_features": {str(intent): int(feature) for intent, feature in zip(manifest["intents"], features)}
    }


def _validation_stability(values, metadata, score_ids, active, manifest):
    left_locale, right_locale = manifest["feature_selection_split"]["stability_locales"]
    allowed = set(map(str, score_ids))
    left = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == left_locale) if str(metadata.id.iloc[i]) in allowed}
    right = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == right_locale) if str(metadata.id.iloc[i]) in allowed}
    ids = sorted(set(left) & set(right))
    x = np.asarray(values[[left[identifier] for identifier in ids]][:, active], dtype=np.float64)
    y = np.asarray(values[[right[identifier] for identifier in ids]][:, active], dtype=np.float64)
    floor = manifest["constants"]["variance_floor"]
    valid = (x.var(0) > floor) & (y.var(0) > floor)
    if not valid.any():
        raise ValueError("no active feature has variance in both validation locales")
    x, y = x[:, valid], y[:, valid]
    correlations = ((x - x.mean(0)) * (y - y.mean(0))).mean(0) / np.sqrt(x.var(0) * y.var(0))
    return _bootstrap_mean(correlations, manifest) | {"stable_features": int(valid.sum()), "paired_semantic_ids": len(ids)}


def reconstruction_metrics(target, reconstruction):
    target, reconstruction = np.asarray(target, dtype=np.float64), np.asarray(reconstruction, dtype=np.float64)
    residual = target - reconstruction
    mse = float(np.mean(residual ** 2))
    denominator = float(np.sum((target - target.mean(0)) ** 2))
    cosine = np.sum(target * reconstruction, axis=1) / (
        np.linalg.norm(target, axis=1) * np.linalg.norm(reconstruction, axis=1)
    ).clip(1e-12)
    return {
        "mse": mse,
        "fraction_variance_explained": float(1 - np.sum(residual ** 2) / denominator),
        "mean_cosine_similarity": float(cosine.mean()),
    }


def evaluate_validation_representation(train_values, validation_values, train_meta, validation_meta, manifest=None):
    """Canonical one-seed selector used before the untouched MASSIVE test set."""
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    intent_rows = _rows(train_meta, manifest["probe_split"]["training_rows"])
    fit_rows, score_rows, fit_ids, score_ids = validation_semantic_rows(validation_meta, manifest)
    selected_rows = _rows(validation_meta, manifest["feature_selection_split"]["rows"])
    selected_ids = validation_meta.id.astype(str).to_numpy()[selected_rows]
    feature_fit_rows = selected_rows[np.isin(selected_ids, list(fit_ids))]
    feature_score_rows = selected_rows[np.isin(selected_ids, list(score_ids))]
    active, orientation = _orientation_on_rows(validation_values, validation_meta, feature_fit_rows, manifest)
    return {
        "intent_k_sparse_probe": k_sparse_probe_curve(
            train_values[intent_rows], train_meta.intent.to_numpy()[intent_rows],
            validation_values[score_rows], validation_meta.intent.to_numpy()[score_rows],
            manifest["seeds"]["probe"],
        ),
        "locale_k_sparse_probe": k_sparse_probe_curve(
            validation_values[fit_rows], validation_meta.locale.to_numpy()[fit_rows],
            validation_values[score_rows], validation_meta.locale.to_numpy()[score_rows],
            manifest["seeds"]["locale_probe"],
        ),
        "mean_intent_concept_auc": _validation_intent_auc(
            validation_values, validation_meta, feature_fit_rows, feature_score_rows, manifest
        ),
        "cross_locale_feature_stability": _validation_stability(
            validation_values, validation_meta, score_ids, active, manifest
        ),
        **orientation,
        "fit_semantic_ids": len(fit_ids),
        "score_semantic_ids": len(score_ids),
        "fraction_alive": float(active.mean()),
        "mean_active_features_l0": float((
            np.abs(np.asarray(validation_values)) > manifest["constants"]["activation_epsilon"]
        ).sum(1).mean()),
    }


def _intent_probe(train_values, test_values, train_meta, test_meta, manifest):
    train_rows = _rows(train_meta, manifest["probe_split"]["training_rows"])
    allowed = set(map(str, manifest["splits"]["test_ids"]))
    test_rows = np.flatnonzero(test_meta.id.astype(str).isin(allowed).to_numpy() & test_meta.locale.isin(manifest["locales"]["held_out"]).to_numpy())
    result, prediction = _probe(
        train_values[train_rows], train_meta.intent.to_numpy()[train_rows],
        test_values[test_rows], test_meta.intent.to_numpy()[test_rows], manifest["seeds"]["probe"],
    )
    return result, prediction, test_rows


def probe_training_rows(metadata, manifest=None):
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    return _rows(metadata, manifest["probe_split"]["training_rows"])


def intent_retrieval(values, metadata, manifest):
    protocol = manifest["intent_retrieval"]
    left = np.flatnonzero(metadata.locale.to_numpy() == protocol["query_locale"])
    right = np.flatnonzero(metadata.locale.to_numpy() == protocol["bank_locale"])
    unit = np.asarray(values, dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True).clip(1e-12)
    order = np.argsort(-(unit[left] @ unit[right].T), axis=1)
    intents = metadata.intent.to_numpy()
    ranks = np.asarray([np.flatnonzero(intents[right][ranked] == intents[query])[0] + 1 for query, ranked in zip(left, order)])
    return {"R@1": _bootstrap_mean(ranks == 1, manifest), "MRR": _bootstrap_mean(1 / ranks, manifest), "queries": int(len(ranks))}


def intent_relation_margin(values, metadata, manifest):
    triples = manifest["relation_split"]["triples"]
    rows = _rows(metadata, [triple[0] for triple in triples])
    positives = _rows(metadata, [triple[1] for triple in triples])
    negatives = _rows(metadata, [triple[2] for triple in triples])
    unit = np.asarray(values, dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True).clip(1e-12)
    positive = np.sum(unit[rows] * unit[positives], axis=1)
    negative = np.sum(unit[rows] * unit[negatives], axis=1)
    margin = positive - negative
    return {
        "same_intent_different_locale_cosine": float(positive.mean()),
        "different_intent_same_locale_cosine": float(negative.mean()),
        "intent_dominance_margin": _bootstrap_mean(margin, manifest),
        "positive_margin_fraction": _bootstrap_mean(margin > 0, manifest),
        "pairs": int(len(margin)),
    }


def locale_probe(values, metadata, manifest):
    split = manifest["locale_probe_split"]
    train_rows = np.flatnonzero(metadata.id.astype(str).isin(split["training_ids"]).to_numpy())
    test_rows = np.flatnonzero(metadata.id.astype(str).isin(split["test_ids"]).to_numpy())
    result, prediction = _probe(
        values[train_rows], metadata.locale.to_numpy()[train_rows],
        values[test_rows], metadata.locale.to_numpy()[test_rows], manifest["seeds"]["locale_probe"],
    )
    correct = prediction == metadata.locale.to_numpy()[test_rows]
    by_id = pd.DataFrame({"id": metadata.id.astype(str).to_numpy()[test_rows], "correct": correct}).groupby("id").correct.mean()
    return {"accuracy": _bootstrap_mean(by_id.to_numpy(), manifest), "train_semantic_ids": len(split["training_ids"]), "test_semantic_ids": len(split["test_ids"])}


def effective_rank(values, variance_floor):
    centered = np.asarray(values, dtype=np.float64) - np.asarray(values, dtype=np.float64).mean(0)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    spectrum = spectrum[spectrum > variance_floor]
    probability = spectrum / spectrum.sum()
    return float(np.exp(-np.sum(probability * np.log(probability))))


def evaluate_dense_routes(train_zc, test_zc, train_meta, test_meta, train_zs=None, test_zs=None, manifest=None):
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    zc_probe, _, _ = _intent_probe(train_zc, test_zc, train_meta, test_meta, manifest)
    result = {
        "manifest_sha256": manifest_sha256(manifest),
        "zC": {
            "intent_balanced_accuracy": zc_probe["balanced_accuracy"],
            "intent_macro_f1": zc_probe["macro_f1"],
            "intent_retrieval": intent_retrieval(test_zc, test_meta, manifest),
            "intent_relation_margin": intent_relation_margin(test_zc, test_meta, manifest),
            "locale_probe": locale_probe(test_zc, test_meta, manifest),
            "effective_rank": effective_rank(test_zc, manifest["constants"]["variance_floor"]),
        },
    }
    if train_zs is not None and test_zs is not None:
        zs_probe, _, _ = _intent_probe(train_zs, test_zs, train_meta, test_meta, manifest)
        result["zS"] = {
            "intent_probe_accuracy": zs_probe["accuracy"],
            "intent_probe_balanced_accuracy": zs_probe["balanced_accuracy"],
            "locale_probe": locale_probe(test_zs, test_meta, manifest),
            "effective_rank": effective_rank(test_zs, manifest["constants"]["variance_floor"]),
        }
    return result


def _feature_selection_rows(metadata, manifest):
    if "rows" in manifest["feature_selection_split"]:
        return _rows(metadata, manifest["feature_selection_split"]["rows"])
    allowed = set(map(str, manifest["splits"]["validation_ids"]))
    locales = manifest["feature_selection_split"].get("locales", manifest["locales"]["seen"])
    return np.flatnonzero(metadata.id.astype(str).isin(allowed).to_numpy() & metadata.locale.isin(locales).to_numpy())


def _feature_statistics(values, metadata, manifest):
    rows = _feature_selection_rows(metadata, manifest)
    values = np.asarray(values[rows], dtype=np.float64)
    intents, locales = metadata.intent.to_numpy()[rows], metadata.locale.to_numpy()[rows]
    constants = manifest["constants"]
    mean = values.mean(0)
    variance = values.var(0)
    scale = np.sqrt(np.maximum(variance, constants["variance_floor"]))
    active = (np.abs(values) > constants["activation_epsilon"]).mean(0) >= constants["minimum_activity_rate"]
    intent_score = np.stack([values[intents == label].mean(0) for label in manifest["intents"]]).max(0) - mean
    locale_score = np.stack([values[locales == label].mean(0) for label in manifest["locales"]["seen"]]).max(0) - mean
    intent_score, locale_score = intent_score / scale, locale_score / scale
    ratio = constants["orientation_ratio"]
    intent_oriented = active & (intent_score > ratio * locale_score)
    locale_oriented = active & (locale_score > ratio * intent_score)
    if not active.any():
        raise ValueError("sparse code has no active features under the canonical definition")
    return values, intents, active, intent_score, {
        "active_features": int(active.sum()),
        "intent_oriented_feature_fraction": float(intent_oriented.sum() / active.sum()),
        "locale_oriented_feature_fraction": float(locale_oriented.sum() / active.sum()),
    }


def _intent_concept_auc(validation_code, validation_meta, test_code, test_meta, manifest):
    rows = _feature_selection_rows(validation_meta, manifest)
    values = np.asarray(validation_code[rows], dtype=np.float64)
    labels = validation_meta.intent.to_numpy()[rows]
    mean, variance = values.mean(0), values.var(0)
    scale = np.sqrt(np.maximum(variance, manifest["constants"]["variance_floor"]))
    per_intent_score = np.stack([(values[labels == intent].mean(0) - mean) / scale for intent in manifest["intents"]])
    features = per_intent_score.argmax(1)
    test_labels = test_meta.intent.to_numpy()
    aucs = np.asarray([roc_auc_score(test_labels == intent, np.asarray(test_code)[:, feature]) for intent, feature in zip(manifest["intents"], features)])
    return _bootstrap_mean(aucs, manifest) | {"selected_features": {str(intent): int(feature) for intent, feature in zip(manifest["intents"], features)}}


def _cross_locale_stability(test_code, test_meta, active, manifest):
    left_locale, right_locale = manifest["locales"]["held_out"]
    left = {str(test_meta.id.iloc[i]): i for i in np.flatnonzero(test_meta.locale.to_numpy() == left_locale)}
    right = {str(test_meta.id.iloc[i]): i for i in np.flatnonzero(test_meta.locale.to_numpy() == right_locale)}
    ids = sorted(set(left) & set(right) & set(map(str, manifest["splits"]["test_ids"])))
    x = np.asarray(test_code[[left[identifier] for identifier in ids]][:, active], dtype=np.float64)
    y = np.asarray(test_code[[right[identifier] for identifier in ids]][:, active], dtype=np.float64)
    floor = manifest["constants"]["variance_floor"]
    valid = (x.var(0) > floor) & (y.var(0) > floor)
    if not valid.any():
        raise ValueError("no active sparse feature has variance in both held-out locales")
    x, y = x[:, valid], y[:, valid]
    correlations = ((x - x.mean(0)) * (y - y.mean(0))).mean(0) / np.sqrt(x.var(0) * y.var(0))
    return _bootstrap_mean(correlations, manifest) | {"stable_features": int(valid.sum()), "paired_test_ids": len(ids)}


def evaluate_sparse_code(train_code, validation_code, test_code, train_meta, validation_meta, test_meta, test_input_standardized, test_reconstruction_standardized, manifest=None):
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    probe, _, _ = _intent_probe(train_code, test_code, train_meta, test_meta, manifest)
    _, _, active, _, orientation = _feature_statistics(validation_code, validation_meta, manifest)
    epsilon = manifest["constants"]["activation_epsilon"]
    reconstruction = reconstruction_metrics(test_input_standardized, test_reconstruction_standardized)
    return {
        "manifest_sha256": manifest_sha256(manifest),
        "intent_accuracy_from_sparse_code": probe["accuracy"],
        "intent_balanced_accuracy_from_sparse_code": probe["balanced_accuracy"],
        "locale_probe_accuracy_from_sparse_code": locale_probe(test_code, test_meta, manifest)["accuracy"],
        "mean_intent_concept_auc": _intent_concept_auc(validation_code, validation_meta, test_code, test_meta, manifest),
        "cross_locale_feature_stability": _cross_locale_stability(test_code, test_meta, active, manifest),
        **orientation,
        "fraction_alive": float(active.mean()),
        "reconstruction_mse": reconstruction["mse"],
        "reconstruction_fve": reconstruction["fraction_variance_explained"],
        "reconstruction_cosine": reconstruction["mean_cosine_similarity"],
        "mean_active_features_l0": float((np.abs(np.asarray(test_code)) > epsilon).sum(1).mean()),
    }


def evaluate_sparse_validation(code, metadata, input_standardized, reconstruction_standardized, manifest=None):
    """Validation-only selector; its ID split is derived from the frozen feature-selection seed."""
    manifest = load_manifest() if manifest is None else validate_manifest(manifest)
    rng = np.random.default_rng(manifest["seeds"]["feature_selection"])
    allowed = set(map(str, manifest["splits"]["validation_ids"]))
    semantic = metadata[metadata.id.astype(str).isin(allowed)].drop_duplicates("id")
    fit_ids, score_ids = set(), set()
    for intent in manifest["intents"]:
        intent_ids = semantic.loc[semantic.intent == intent, "id"].astype(str).to_numpy()
        if len(intent_ids) < 2:
            raise ValueError(f"intent {intent} has fewer than two validation semantic IDs")
        intent_ids = intent_ids[rng.permutation(len(intent_ids))]
        split = max(1, len(intent_ids) // 2)
        fit_ids.update(intent_ids[:split]); score_ids.update(intent_ids[split:])
    ids = metadata.id.astype(str).to_numpy()
    locales = metadata.locale.to_numpy()
    seen = np.isin(locales, manifest["locales"]["seen"])
    fit_rows = np.flatnonzero(seen & np.isin(ids, list(fit_ids)))
    score_rows = np.flatnonzero(seen & np.isin(ids, list(score_ids)))
    values = np.asarray(code, dtype=np.float64)
    fit, score = values[fit_rows], values[score_rows]
    fit_labels, score_labels = metadata.intent.to_numpy()[fit_rows], metadata.intent.to_numpy()[score_rows]
    mean, variance = fit.mean(0), fit.var(0)
    scale = np.sqrt(np.maximum(variance, manifest["constants"]["variance_floor"]))
    feature_scores = np.stack([(fit[fit_labels == intent].mean(0) - mean) / scale for intent in manifest["intents"]])
    features = feature_scores.argmax(1)
    aucs = np.asarray([
        roc_auc_score(score_labels == intent, score[:, feature])
        for intent, feature in zip(manifest["intents"], features)
    ])
    locale_rng = np.random.default_rng(manifest["seeds"]["locale_probe"])
    locale_fit_ids = set(locale_rng.choice(sorted(fit_ids), min(100, len(fit_ids)), replace=False))
    locale_score_ids = set(locale_rng.choice(sorted(score_ids), min(100, len(score_ids)), replace=False))
    locale_fit = np.flatnonzero(seen & np.isin(ids, list(locale_fit_ids)))
    locale_score = np.flatnonzero(seen & np.isin(ids, list(locale_score_ids)))
    locale_result, _ = _probe(
        values[locale_fit], locales[locale_fit], values[locale_score], locales[locale_score],
        manifest["seeds"]["locale_probe"],
    )
    epsilon = manifest["constants"]["activation_epsilon"]
    return {
        "manifest_sha256": manifest_sha256(manifest),
        "fit_semantic_ids": len(fit_ids),
        "score_semantic_ids": len(score_ids),
        "locale_probe_fit_semantic_ids": len(locale_fit_ids),
        "locale_probe_score_semantic_ids": len(locale_score_ids),
        "mean_intent_concept_auc": _bootstrap_mean(aucs, manifest),
        "locale_probe_accuracy": locale_result["accuracy"],
        "reconstruction_mse": float(np.mean(
            (np.asarray(reconstruction_standardized) - np.asarray(input_standardized)) ** 2
        )),
        "mean_active_features_l0": float((np.abs(values) > epsilon).sum(1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build-manifest", "validate-manifest", "audit-manifest"))
    args = parser.parse_args()
    manifest = build_manifest() if args.command == "build-manifest" else load_manifest()
    if args.command == "audit-manifest":
        print(json.dumps(audit_manifest(manifest), indent=2))
        return
    print(json.dumps({
        "path": str(MANIFEST.relative_to(ROOT)),
        "sha256": manifest_sha256(manifest),
        "training_ids": len(manifest["splits"]["training_ids"]),
        "validation_ids": len(manifest["splits"]["validation_ids"]),
        "test_ids": len(manifest["splits"]["test_ids"]),
        "probe_rows": len(manifest["probe_split"]["training_rows"]),
        "relation_pairs": len(manifest["relation_split"]["triples"]),
    }, indent=2))


if __name__ == "__main__":
    main()
