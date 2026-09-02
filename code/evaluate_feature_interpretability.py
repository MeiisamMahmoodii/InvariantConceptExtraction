"""Evaluate intent purity and readable examples for frozen factor-SAE features."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

import canonical_evaluator as evaluator
import build_factor_stability_figure as stability


ROOT = Path(__file__).resolve().parents[1]
TOP_K = 20
OUTPUT = ROOT / "Report" / "factor_sae_feature_interpretability.json"
PER_SEED = ROOT / "Report" / "factor_sae_feature_interpretability_per_seed.csv"
CATALOGUE = ROOT / "Report" / "factor_sae_feature_catalogue.csv"
EXAMPLES = ROOT / "paper" / "figure_data" / "figure4_feature_examples.json"
APPENDIX = ROOT / "paper" / "factor_sae_feature_catalogue_appendix.tex"


def intent_names():
    metadata = json.loads(pq.read_schema(ROOT / "data" / "massive_all_test.parquet").metadata[b"huggingface"])
    return metadata["info"]["features"]["intent"]["names"]


def locale_entropy(locales):
    counts = pd.Series(locales).value_counts(normalize=True).to_numpy()
    return float(-(counts * np.log2(counts)).sum()) if len(counts) > 1 else 0.0


def mean_jaccard(sets):
    values = [len(left & right) / len(left | right) for left, right in combinations(sets, 2) if left | right]
    return float(np.mean(values)) if values else 0.0


def summarize_seed(rows, eligible_intents):
    frame = pd.DataFrame(rows)
    eligible = frame[frame.intent_id.isin(eligible_intents)]
    return {
        "mean_top20_intent_purity": float(eligible.top20_intent_purity.mean()),
        "reliable_intent_coverage": float((eligible.top20_intent_purity >= .8).mean()),
        "mean_locale_entropy": float(eligible.top20_locale_entropy.mean()),
        "mean_selected_feature_stability": float(frame.cross_locale_stability.mean()),
        "unique_selected_feature_fraction": float(frame.feature_id.nunique() / len(frame)),
        "mean_top_id_overlap": mean_jaccard([set(ids.split("|")) for ids in frame.top_ids]),
        "eligible_intents": int(len(eligible)),
    }


def main():
    manifest = evaluator.load_manifest()
    validation, validation_meta, test, test_meta, exact = stability.backbone_data("Gemma 2 2B", manifest)
    locales = manifest["locales"]["held_out"]
    names = intent_names()
    semantic = test_meta.drop_duplicates("id").copy()
    semantic["id"] = semantic.id.astype(str)
    support = semantic.intent.value_counts()
    eligible_intents = {int(intent) for intent, count in support.items() if count >= TOP_K}
    lookup = {(str(row.id), row.locale): i for i, row in test_meta.iterrows()}
    ids = semantic.id.tolist()
    left = np.asarray([lookup[(identifier, locales[0])] for identifier in ids])
    right = np.asarray([lookup[(identifier, locales[1])] for identifier in ids])
    semantic_labels = semantic.intent.to_numpy()

    text = pd.read_parquet(ROOT / "data" / "massive_all_test.parquet", columns=["id", "locale", "utt"])
    text["id"] = text.id.astype(str)
    english = text[text.locale == "en-US"].drop_duplicates("id").set_index("id").utt.to_dict()
    heldout_text = text[text.locale.isin(locales)].set_index(["id", "locale"]).utt.to_dict()

    catalogue, seed_rows, qualitative_candidates = [], [], None
    epsilon = manifest["constants"]["activation_epsilon"]
    for method in stability.METHODS:
        for seed in evaluator.TRAINING_SEEDS:
            stability.SEED = seed
            validation_code, test_code = stability.code_for_method(
                "Gemma 2 2B", method, validation, test, exact, stability.torch.device("cuda" if stability.torch.cuda.is_available() else "cpu")
            )
            features, validation_aucs, validation_correlations, _, _ = stability.feature_selection(
                validation_code, validation_meta, manifest
            )
            intent_rows, top_sets = [], []
            for intent_index, intent in enumerate(manifest["intents"]):
                feature = int(features[intent_index])
                scores = test_code[:, feature]
                pair_scores = np.maximum(scores[left], scores[right])
                provider = np.where(scores[left] >= scores[right], locales[0], locales[1])
                order = np.argsort(-pair_scores)
                active_order = order[pair_scores[order] > epsilon][:TOP_K]
                top_ids = [ids[index] for index in active_order]
                top_sets.append(set(top_ids))
                purity = float(np.mean(semantic_labels[active_order] == intent)) if len(active_order) else 0.0
                correlation = (
                    np.corrcoef(scores[left], scores[right])[0, 1]
                    if scores[left].std() and scores[right].std() else 0.0
                )
                row = {
                    "method": method, "seed": seed, "intent_id": int(intent), "intent_name": names[intent],
                    "feature_id": feature, "test_intent_support": int(support.get(intent, 0)),
                    "active_test_ids": int((pair_scores > epsilon).sum()), "top_k_observed": int(len(active_order)),
                    "top20_intent_purity": purity, "top20_locale_entropy": locale_entropy(provider[active_order]),
                    "cross_locale_stability": float(correlation), "heldout_auc": float(roc_auc_score(semantic_labels == intent, pair_scores)),
                    "validation_auc": float(validation_aucs[intent_index]),
                    "validation_stability": float(validation_correlations[intent_index]) if np.isfinite(validation_correlations[intent_index]) else None,
                    "top_ids": "|".join(top_ids),
                }
                intent_rows.append(row)
                catalogue.append(row)
            summary = summarize_seed(intent_rows, eligible_intents)
            seed_rows.append({"method": method, "seed": seed, **summary})
            if method == "Reciprocal factor SAE" and seed == evaluator.TRAINING_SEEDS[0]:
                finite = np.isfinite(validation_correlations)
                enough_support = np.asarray([support.get(intent, 0) >= TOP_K for intent in manifest["intents"]])
                eligible = finite & enough_support
                rank = validation_aucs + validation_correlations
                chosen = np.flatnonzero(eligible)[np.argsort(-rank[eligible])[:3]]
                qualitative_candidates = []
                rows_by_intent = {row["intent_id"]: row for row in intent_rows}
                for index in chosen:
                    intent = int(manifest["intents"][index])
                    row = rows_by_intent[intent]
                    feature = row["feature_id"]
                    scores = test_code[:, feature]
                    pair_scores = np.maximum(scores[left], scores[right])
                    provider = np.where(scores[left] >= scores[right], locales[0], locales[1])
                    examples = []
                    for identifier in row["top_ids"].split("|")[:3]:
                        semantic_index = ids.index(identifier)
                        locale = provider[semantic_index]
                        examples.append({
                            "id": identifier, "locale": locale,
                            "activation": float(pair_scores[semantic_index]),
                            "english": english.get(identifier, ""),
                            "heldout": heldout_text.get((identifier, locale), ""),
                        })
                    qualitative_candidates.append({**row, "examples": examples})

    per_seed = pd.DataFrame(seed_rows)
    summary = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "seeds": len(group)}
        for column in per_seed.columns.difference(["method", "seed"]):
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        summary.append(row)
    report = {
        "status": "frozen-checkpoint feature interpretability evaluation",
        "selection": "features and qualitative intents selected on disjoint validation fit/score IDs only",
        "evaluation": "top-20 unique semantic IDs on held-out Arabic/Chinese; headline purity covers intents with at least 20 test IDs",
        "top_k": TOP_K, "eligible_intents": len(eligible_intents), "seeds": list(evaluator.TRAINING_SEEDS),
        "summary": summary,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    PER_SEED.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(PER_SEED, index=False)
    pd.DataFrame(catalogue).to_csv(CATALOGUE, index=False)
    appendix = pd.DataFrame(catalogue)
    appendix = appendix[(appendix.method == "Reciprocal factor SAE") & (appendix.seed == evaluator.TRAINING_SEEDS[0])]
    appendix = appendix[[
        "intent_name", "feature_id", "top20_intent_purity", "cross_locale_stability", "heldout_auc", "active_test_ids"
    ]].sort_values("intent_name")
    appendix.columns = ["Intent", "Feature", "Purity@20", "Stability", "AUC", "Active IDs"]
    APPENDIX.write_text(
        appendix.to_latex(index=False, longtable=True, escape=True, float_format="%.3f",
                          caption="Complete seed-20260827 intent-feature catalogue for the reciprocal factor SAE. Features are selected on validation data and evaluated on held-out Arabic/Chinese semantic IDs.",
                          label="tab:factor-feature-catalogue") + "\n",
        encoding="utf-8",
    )
    EXAMPLES.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES.write_text(json.dumps({"protocol": report, "features": qualitative_candidates}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
