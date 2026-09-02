"""Build the validation-selected T-SAE-style feature-stability figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import canonical_evaluator as evaluator
import evaluate_massive_factor_sae as gemma
import run_pythia_massive_factor_sae as pythia
from run_massive_sparse_partition_pilot import SparsePartition


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260827
METHODS = ("BatchTopK SAE", "Blockwise SAE control", "Reciprocal factor SAE")
SHORT = ("Global BatchTopK", "Blockwise control", "Ours")
OUT_DATA = ROOT / "paper" / "figure_data" / "figure3_factor_stability.json"
OUT_PNG = ROOT / "paper" / "Figures" / "figure3_factor_stability.png"
OUT_PDF = ROOT / "paper" / "Figures" / "figure3_factor_stability.pdf"


def representative_intent(ours_auc, ours_stability, control_auc, control_stability, support, minimum_support=6):
    """Select the largest joint validation gain over the exact blockwise control."""
    valid = (np.isfinite(ours_auc) & np.isfinite(ours_stability) & np.isfinite(control_auc)
             & np.isfinite(control_stability) & (support >= minimum_support))
    candidates = valid & (ours_auc >= .8) & (ours_auc > control_auc) & (ours_stability > control_stability)
    if not candidates.any():
        raise ValueError("no high-AUC validation feature improves jointly over the blockwise control")
    gain = (ours_auc - control_auc) + (ours_stability - control_stability)
    indices = np.flatnonzero(candidates)
    return int(indices[np.argmax(gain[indices])])


def model_from_checkpoint(saved, input_width, device):
    config = saved["config"]
    model = SparsePartition(
        config.get("activation", "batchtopk"), config["c_fraction"], config.get("bandwidth", .05),
        total_k=config["total_k"], c_k=config["c_k"], sparsifier=config["sparsifier"],
        sparse_width=config.get("sparse_width", 9216), input_width=config.get("input_width", input_width),
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def encode(model, raw, mean, std, device, batch_size, exact_batchtopk):
    model.train(exact_batchtopk)
    routes_c, routes_s = [], []
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            values = (np.asarray(raw[start:start + batch_size], np.float32) - mean) / std
            z_c, z_s, _, _ = model(torch.from_numpy(values).to(device))
            routes_c.append(z_c.cpu().numpy())
            routes_s.append(z_s.cpu().numpy())
    return np.concatenate(routes_c), np.concatenate(routes_s)


def feature_selection(values, metadata, manifest):
    fit_rows, score_rows, _, score_ids = evaluator.validation_semantic_rows(metadata, manifest)
    fit, score = values[fit_rows], values[score_rows]
    fit_labels = metadata.intent.to_numpy()[fit_rows]
    score_labels = metadata.intent.to_numpy()[score_rows]
    mean, variance = fit.mean(0), fit.var(0)
    scale = np.sqrt(np.maximum(variance, manifest["constants"]["variance_floor"]))
    feature_scores = np.stack([
        (fit[fit_labels == intent].mean(0) - mean) / scale for intent in manifest["intents"]
    ])
    features = feature_scores.argmax(1)
    aucs = np.asarray([
        roc_auc_score(score_labels == intent, score[:, feature])
        for intent, feature in zip(manifest["intents"], features)
    ])
    left_locale, right_locale = manifest["feature_selection_split"]["stability_locales"]
    allowed = set(map(str, score_ids))
    left = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == left_locale)
            if str(metadata.id.iloc[i]) in allowed}
    right = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == right_locale)
             if str(metadata.id.iloc[i]) in allowed}
    ids = sorted(set(left) & set(right))
    x, y = values[[left[i] for i in ids]], values[[right[i] for i in ids]]
    correlations = np.asarray([
        np.corrcoef(x[:, feature], y[:, feature])[0, 1] if x[:, feature].std() and y[:, feature].std() else np.nan
        for feature in features
    ])
    return features, aucs, correlations, mean, scale


def checkpoint(backbone, method):
    if backbone == "Gemma 2 2B":
        return gemma.load_checkpoint(gemma.final_checkpoint(method, SEED))
    return torch.load(pythia.checkpoint_path(pythia.METHODS[method], SEED), map_location="cpu", weights_only=True)


def backbone_data(backbone, manifest):
    if backbone == "Gemma 2 2B":
        _, _, validation, validation_meta, test, test_meta = gemma.selected_data(manifest, True)
        return validation, validation_meta, test, test_meta, False
    _, _, validation, validation_meta, test, test_meta = pythia.selected_data(manifest)
    return validation, validation_meta, test, test_meta, True


def code_for_method(backbone, method, validation, test, exact_batchtopk, device):
    saved = checkpoint(backbone, method)
    config = saved["config"]
    mean = np.asarray(saved.get("input_mean", config.get("input_mean")), np.float32)
    std = np.asarray(saved.get("input_std", config.get("input_std")), np.float32)
    model = model_from_checkpoint(saved, test.shape[1], device)
    validation_c, validation_s = encode(model, validation, mean, std, device, 128, exact_batchtopk)
    test_c, test_s = encode(model, test, mean, std, device, 128, exact_batchtopk)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if config["sparsifier"] == "global":
        return np.concatenate((validation_c, validation_s), 1), np.concatenate((test_c, test_s), 1)
    return validation_c, test_c


def sorted_ids(metadata, intent, count=6):
    ids = metadata.loc[metadata.intent == intent, "id"].astype(str).drop_duplicates().tolist()
    ids.sort(key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    return ids[:count]


def trace_matrix(values, metadata, feature, ids, locales, mean, scale):
    lookup = {(str(row.id), row.locale): i for i, row in metadata.iterrows()}
    rows = [[lookup[(identifier, locale)] for identifier in ids] for locale in locales]
    return (values[np.asarray(rows), feature] - mean[feature]) / scale[feature]


def heldout_metrics(values, metadata, feature, intent, locales):
    auc = roc_auc_score(metadata.intent.to_numpy() == intent, values[:, feature])
    left = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == locales[0])}
    right = {str(metadata.id.iloc[i]): i for i in np.flatnonzero(metadata.locale.to_numpy() == locales[1])}
    ids = sorted(set(left) & set(right))
    x = values[[left[i] for i in ids], feature]
    y = values[[right[i] for i in ids], feature]
    correlation = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
    return float(auc), float(correlation)


def aggregate_metrics():
    files = {
        "Gemma 2 2B": ROOT / "Report" / "factor_sae_step4_definitive_test_per_seed.csv",
        "Pythia-160M": ROOT / "Report" / "factor_sae_pythia160m_transfer_per_seed.csv",
    }
    result = {}
    for backbone, path in files.items():
        frame = pd.read_csv(path)
        rows = {}
        for method in METHODS:
            selected = frame[frame.method == method].copy()
            gap = selected.intent_feature_fraction - selected.locale_feature_fraction
            rows[method] = {
                "stability_mean": float(selected.cross_locale_stability.mean()),
                "stability_std": float(selected.cross_locale_stability.std(ddof=1)),
                "orientation_gap_mean": float(gap.mean()),
                "orientation_gap_std": float(gap.std(ddof=1)),
            }
        result[backbone] = rows
    return result


def main():
    manifest = evaluator.load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {
        "selection": "Features use disjoint validation fit/score activations. The representative intent maximizes the joint validation AUC-and-stability gain over the exact blockwise control among intents with ours AUC >= .8 and at least six display IDs. Test activations are never used for selection.",
        "seed": SEED,
        "held_out_locales": manifest["locales"]["held_out"],
        "qualitative": {},
    }
    panels = []
    backbone = "Gemma 2 2B"
    validation, validation_meta, test, test_meta, exact_batchtopk = backbone_data(backbone, manifest)
    codes, selections = {}, {}
    for method in METHODS:
        validation_code, test_code = code_for_method(backbone, method, validation, test, exact_batchtopk, device)
        codes[method] = (validation_code, test_code)
        selections[method] = feature_selection(validation_code, validation_meta, manifest)
    support_by_label = test_meta.drop_duplicates("id").intent.value_counts()
    support = np.asarray([support_by_label.get(intent, 0) for intent in manifest["intents"]])
    ours = selections["Reciprocal factor SAE"]
    control = selections["Blockwise SAE control"]
    intent_index = representative_intent(ours[1], ours[2], control[1], control[2], support)
    target_intent = manifest["intents"][intent_index]
    ours_feature = int(ours[0][intent_index])
    score_rows = evaluator.validation_semantic_rows(validation_meta, manifest)[1]
    score_labels = validation_meta.intent.to_numpy()[score_rows]
    score_values = codes["Reciprocal factor SAE"][0][score_rows, ours_feature]
    alternatives = [intent for intent in manifest["intents"] if intent != target_intent and support_by_label.get(intent, 0) >= 6]
    alternative_means = np.asarray([score_values[score_labels == intent].mean() for intent in alternatives])
    contrast_intent = alternatives[int(np.argsort(alternative_means)[len(alternatives) // 2])]
    display_ids = sorted_ids(test_meta, target_intent) + sorted_ids(test_meta, contrast_intent)
    qualitative = {
        "backbone": backbone, "representative_intent": int(target_intent),
        "contrast_intent": int(contrast_intent), "display_ids": display_ids, "methods": {},
    }
    for method in METHODS:
        validation_code, test_code = codes[method]
        features, aucs, correlations, mean, scale = selections[method]
        feature = int(features[intent_index])
        matrix = trace_matrix(test_code, test_meta, feature, display_ids, manifest["locales"]["held_out"], mean, scale)
        test_auc, test_correlation = heldout_metrics(test_code, test_meta, feature, target_intent, manifest["locales"]["held_out"])
        item = {
            "feature": feature, "validation_auc": float(aucs[intent_index]),
            "validation_stability": float(correlations[intent_index]), "heldout_auc": test_auc,
            "heldout_stability": test_correlation, "standardized_activations": matrix.tolist(),
        }
        qualitative["methods"][method] = item
        panels.append((matrix, item, target_intent, contrast_intent))
    output["qualitative"] = qualitative
    output["aggregate_three_seed"] = aggregate_metrics()

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), constrained_layout=True)
    image = None
    for axis, index in zip(axes[0], (0, 2, 1)):
        matrix, item, target, contrast = panels[index]
        short = SHORT[index]
        image = axis.imshow(np.clip(matrix, -1, 3), cmap="Blues", vmin=-1, vmax=3, aspect="auto")
        axis.axvline(5.5, color="#D97706", linewidth=1.5)
        axis.set_xticks(range(12), [f"T{i}" for i in range(1, 7)] + [f"O{i}" for i in range(1, 7)])
        axis.set_yticks((0, 1), [locale.split("-")[0] for locale in manifest["locales"]["held_out"]])
        axis.set_title(f"{short}\nfeature {item['feature']}  |  held-out r={item['heldout_stability']:.2f}, AUC={item['heldout_auc']:.2f}", fontsize=9)
        if axis.get_subplotspec().is_first_col():
            axis.set_ylabel("Gemma 2 2B\nheld-out locale")
        axis.set_xlabel(f"target intent {target}     |     comparison intent {contrast}", fontsize=8)
        axis.tick_params(length=0, labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes[0], shrink=.75, pad=.015)
    colorbar.set_label("feature activation (validation-standardized)")
    colors = ("#9CA3AF", "#D97706", "#2563EB")
    aggregate = output["aggregate_three_seed"]
    for axis, backbone_name in zip(axes[1, :2], ("Gemma 2 2B", "Pythia-160M")):
        means = [aggregate[backbone_name][method]["stability_mean"] for method in METHODS]
        errors = [aggregate[backbone_name][method]["stability_std"] for method in METHODS]
        bars = axis.bar(range(3), means, yerr=errors, color=colors, capsize=3)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_xticks(range(3), ("Global", "Block", "Ours"))
        axis.set_ylabel("mean feature stability")
        axis.set_title(f"{backbone_name}: all active features")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=.7)
        axis.set_axisbelow(True)
    gap_axis = axes[1, 2]
    x = np.arange(2)
    width = .23
    for offset, method, color, label in zip((-width, 0, width), METHODS, colors, ("Global", "Block", "Ours")):
        means = [aggregate[name][method]["orientation_gap_mean"] for name in ("Gemma 2 2B", "Pythia-160M")]
        errors = [aggregate[name][method]["orientation_gap_std"] for name in ("Gemma 2 2B", "Pythia-160M")]
        gap_axis.bar(x + offset, means, width, yerr=errors, color=color, capsize=2, label=label)
    gap_axis.axhline(0, color="#374151", linewidth=.8)
    gap_axis.set_xticks(x, ("Gemma", "Pythia"))
    gap_axis.set_ylabel("intent frac. - locale frac.")
    gap_axis.set_title("Sparse feature orientation")
    gap_axis.legend(frameon=False, fontsize=8)
    gap_axis.spines[["top", "right"]].set_visible(False)
    gap_axis.grid(axis="y", color="#E5E7EB", linewidth=.7)
    gap_axis.set_axisbelow(True)
    fig.suptitle("Controlled relations stabilize intent features across held-out locales", fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"data": str(OUT_DATA), "png": str(OUT_PNG), "pdf": str(OUT_PDF)}, indent=2))


if __name__ == "__main__":
    main()
