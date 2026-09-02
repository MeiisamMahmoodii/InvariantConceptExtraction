"""Sample and audit the intent/locale relations used by every dataset."""

import numpy as np
import pandas as pd
import torch


def build_relation_index(metadata: pd.DataFrame):
    required = {"id", "intent", "locale"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata.intent.nunique() < 2 or metadata.locale.nunique() < 2:
        raise ValueError("relation sampling requires at least two intents and two locales")

    ids, _ = pd.factorize(metadata.id, sort=True)
    intents, _ = pd.factorize(metadata.intent, sort=True)
    locales, _ = pd.factorize(metadata.locale, sort=True)
    groups = np.empty((metadata.intent.nunique(), metadata.locale.nunique()), dtype=object)
    for (intent, locale), rows in metadata.groupby([intents, locales]).indices.items():
        groups[intent, locale] = np.asarray(rows, dtype=np.int64)
    if any(groups[index] is None or not len(groups[index]) for index in np.ndindex(groups.shape)):
        raise ValueError("every intent must have examples in every locale")

    if metadata.groupby("id").intent.nunique().gt(1).any():
        raise ValueError("each ID must map to one intent")
    counts = metadata.groupby(["id", "locale"]).size()
    complete_ids = len(counts) == metadata.id.nunique() * metadata.locale.nunique() and counts.eq(1).all()
    id_locale = None
    if complete_ids:
        id_locale = np.full((metadata.id.nunique(), metadata.locale.nunique()), -1, dtype=np.int64)
        id_locale[ids, locales] = np.arange(len(metadata))
    return ids, intents, locales, groups, id_locale


def sample_intent_relations(rows, index, rng, exact_id_positive_fraction=0.5):
    """Return zC positives and zS positives under the intent/locale contract.

    zC positive: same intent, different locale.
    zS positive: different intent, same locale; this is the zC matched negative.
    """
    if not 0 <= exact_id_positive_fraction <= 1:
        raise ValueError("exact_id_positive_fraction must be between 0 and 1")

    ids, intents, locales, groups, id_locale = index
    rows = np.asarray(rows, dtype=np.int64)
    anchor_ids, anchor_intents, anchor_locales = ids[rows], intents[rows], locales[rows]
    c_locales = (anchor_locales + rng.integers(1, groups.shape[1], len(rows))) % groups.shape[1]
    s_intents = (anchor_intents + rng.integers(1, groups.shape[0], len(rows))) % groups.shape[0]

    exact_count = round(len(rows) * exact_id_positive_fraction)
    nonexact_possible = np.asarray([
        np.any(ids[groups[intent, locale]] != identifier)
        for identifier, intent, locale in zip(anchor_ids, anchor_intents, c_locales)
    ])
    forced_exact = ~nonexact_possible
    if forced_exact.sum() > exact_count:
        raise ValueError("requested exact-ID fraction is too small for singleton intents")
    exact = forced_exact.copy()
    eligible = np.flatnonzero(nonexact_possible)
    exact[rng.permutation(eligible)[: exact_count - forced_exact.sum()]] = True
    if exact.any() and id_locale is None:
        raise ValueError("exact-ID positives require every ID to occur in every locale")

    c_positive = np.empty(len(rows), dtype=np.int64)
    if exact.any():
        c_positive[exact] = id_locale[anchor_ids[exact], c_locales[exact]]
    for i in np.flatnonzero(~exact):
        candidates = groups[anchor_intents[i], c_locales[i]]
        candidates = candidates[ids[candidates] != anchor_ids[i]]
        if not len(candidates):
            raise ValueError("non-exact zC positives require another ID with the same intent")
        c_positive[i] = rng.choice(candidates)

    s_positive = np.array([
        rng.choice(groups[s_intents[i], anchor_locales[i]]) for i in range(len(rows))
    ])
    return c_positive, s_positive


def audit_sample(rows, c_positive, s_positive, index):
    ids, intents, locales, _, _ = index
    rows = np.asarray(rows)
    return {
        "pairs": int(len(rows)),
        "zC_same_intent_fraction": float(np.mean(intents[rows] == intents[c_positive])),
        "zC_different_locale_fraction": float(np.mean(locales[rows] != locales[c_positive])),
        "zC_exact_id_fraction": float(np.mean(ids[rows] == ids[c_positive])),
        "zC_nonexact_different_id_fraction": float(np.mean(ids[rows] != ids[c_positive])),
        "zS_same_locale_fraction": float(np.mean(locales[rows] == locales[s_positive])),
        "zS_different_intent_fraction": float(np.mean(intents[rows] != intents[s_positive])),
    }


def false_negative_masks(intents, positive_intents, locales, positive_locales, device=None):
    """Mask in-batch false negatives for both factor routes."""
    diagonal = np.eye(len(intents), dtype=bool)
    masks = (
        (intents[:, None] != positive_intents[None, :]) | diagonal,
        (locales[:, None] != positive_locales[None, :]) | diagonal,
    )
    return tuple(torch.from_numpy(mask).to(device) for mask in masks)


def relation_margin(values, metadata, samples=10000, seed=20260827, exact_id_positive_fraction=0.0):
    index = build_relation_index(metadata)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(metadata), min(samples, len(metadata)), replace=False)
    c_positive, s_positive = sample_intent_relations(
        rows, index, rng, exact_id_positive_fraction
    )
    unit = values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-12)
    same_intent = np.sum(unit[rows] * unit[c_positive], axis=1)
    same_locale = np.sum(unit[rows] * unit[s_positive], axis=1)
    delta = same_intent - same_locale
    return {
        "same_intent_different_locale_cosine": float(same_intent.mean()),
        "different_intent_same_locale_cosine": float(same_locale.mean()),
        "intent_dominance_margin": float(delta.mean()),
        "positive_margin_fraction": float(np.mean(delta > 0)),
        "samples": int(len(rows)),
    }
