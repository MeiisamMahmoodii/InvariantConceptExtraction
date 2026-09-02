"""One-seed layer sensitivity for the direct Factor-Contrastive SAE.

The study compares the reciprocal model with its exact blockwise
reconstruction control at fixed Gemma hidden-state indices. Activations and
checkpoints live on drive D; only compact reports are written to the project.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import canonical_evaluator as evaluator
import evaluate_massive_factor_sae as canonical
import run_massive_sparse_partition_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\data\InvariantConceptExtraction\gemma_massive_layer_sweep_full")
WORK = Path(r"D:\data\InvariantConceptExtraction\factor_sae_layer_sensitivity")
REPORT = ROOT / "Report" / "factor_sae_layer_sensitivity.json"
CSV = ROOT / "Report" / "factor_sae_layer_sensitivity.csv"
LAYERS = (4, 8, 16, 24)
METHODS = ("Blockwise control", "Ours")
SEED = 20260827


def selected_rows(metadata, manifest):
    validation_ids = set(map(str, manifest["splits"]["validation_ids"]))
    training = np.flatnonzero(~metadata.id.astype(str).isin(validation_ids).to_numpy())
    probe_keys = {tuple(map(str, row)) for row in manifest["probe_split"]["training_rows"]}
    probe = np.asarray([
        index
        for index, row in enumerate(metadata[["id", "locale"]].astype(str).itertuples(index=False, name=None))
        if row in probe_keys
    ])
    feature_keys = {
        tuple(map(str, row)) for row in manifest["feature_selection_split"]["rows"]
    }
    stability_locales = set(manifest["feature_selection_split"]["stability_locales"])
    validation = np.asarray([
        index
        for index, row in enumerate(metadata[["id", "locale"]].astype(str).itertuples(index=False, name=None))
        if row in feature_keys or (row[0] in validation_ids and row[1] in stability_locales)
    ])
    assert metadata.iloc[training].id.nunique() == 10343
    assert metadata.iloc[probe].id.nunique() == len(set(metadata.iloc[probe].id))
    return training, probe, validation


def layer_stats(raw, rows, layer):
    path = WORK / f"layer{layer}_normalization.npz"
    if path.exists():
        saved = np.load(path)
        return saved["mean"], saved["std"]
    total = np.zeros(raw.shape[1], np.float64)
    squared = np.zeros(raw.shape[1], np.float64)
    for start in range(0, len(rows), 4096):
        values = np.asarray(raw[rows[start : start + 4096]], np.float64)
        total += values.sum(0)
        squared += np.square(values).sum(0)
    mean = total / len(rows)
    std = np.sqrt(np.maximum(squared / len(rows) - np.square(mean), 1e-12))
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    np.savez(path, mean=mean, std=std, training_rows=len(rows))
    return mean, std


def training_args(method, epochs):
    ours = method == "Ours"
    return SimpleNamespace(
        activation="batchtopk",
        objective="matched" if ours else "reconstruction",
        route_mode="reciprocal" if ours else "none",
        sparsifier="block",
        sae_objective="standard",
        triplet_margin=0.2,
        c_fraction=0.3,
        relation_sampler="intent_50_50",
        total_k=64,
        c_k=13,
        sparse_width=9216,
        seed=SEED,
        epochs=epochs,
        anchors_per_epoch=45968,
        batch_size=128,
        eval_batch_size=128,
        audit_rows=5000,
        calibration_rows=2048,
        lr=1e-4,
        threshold_lr=1e-3,
        threshold_scale=1.0,
        temperature=0.07,
        contrast_weight=1.0 if ours else 0.0,
        intent_adversary_weight=0.0,
        swap_weight=0.0,
        l0_weight=0.1,
        softplus_alpha=1e-3,
        sparsity_warmup_epochs=5,
        bandwidth=0.05,
        max_steps=0,
    )


def fresh_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = pilot.SparsePartition(
        "batchtopk", 0.3, 0.05, total_k=64, c_k=13,
        sparsifier="block", sparse_width=9216, input_width=2304,
    ).to(pilot.DEVICE)
    model.initialize_fresh()
    fingerprint = hashlib.sha256(
        model.encoder.weight[:2].detach().cpu().numpy().tobytes()
    ).hexdigest()
    return model, fingerprint


def train_or_load(layer, method, raw, metadata, training_rows, mean, std, epochs):
    slug = "ours" if method == "Ours" else "blockwise_control"
    directory = WORK / f"layer{layer}" / slug
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"seed{SEED}.pt"
    model, fingerprint = fresh_model()
    args = training_args(method, epochs)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=pilot.DEVICE, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        assert saved["initialization_fingerprint"] == fingerprint
        return model, args, checkpoint, saved.get("history", []), fingerprint
    view = pilot.RowView(raw, training_rows)
    training_meta = metadata.iloc[training_rows].reset_index(drop=True)
    history = pilot.train(model, view, training_meta, mean, std, args)
    calibration = np.random.default_rng(SEED).choice(
        len(training_meta), args.calibration_rows, replace=False
    )
    thresholds = pilot.calibrate_batchtopk_thresholds(
        model, view, mean, std, calibration, len(calibration)
    )
    model.set_batchtopk_thresholds(*thresholds)
    torch.save({
        "state_dict": model.state_dict(),
        "input_mean": mean,
        "input_std": std,
        "history": history,
        "initialization_fingerprint": fingerprint,
        "config": vars(args) | {
            "layer": layer,
            "initialization": "fresh paired initialization",
            "batchtopk_inference_thresholds": thresholds,
        },
    }, checkpoint)
    return model, args, checkpoint, history, fingerprint


def evaluate_run(
    layer, method, model, raw, raw_test, metadata, test_metadata,
    probe_rows, validation_rows, mean, std, manifest, checkpoint, fingerprint,
):
    slug = "ours" if method == "Ours" else "blockwise_control"
    detail_path = WORK / f"layer{layer}" / slug / "evaluation.json"
    if detail_path.exists():
        return json.loads(detail_path.read_text(encoding="utf-8"))["summary"]
    raw_probe = np.asarray(raw[probe_rows], np.float32)
    raw_validation = np.asarray(raw[validation_rows], np.float32)
    raw_test_array = np.asarray(raw_test, np.float32)
    probe_meta = metadata.iloc[probe_rows].reset_index(drop=True)
    validation_meta = metadata.iloc[validation_rows].reset_index(drop=True)
    probe_c, probe_s, _ = canonical.encode(model, raw_probe, mean, std, "cuda", 128)
    validation_c, validation_s, _ = canonical.encode(
        model, raw_validation, mean, std, "cuda", 128
    )
    test_c, test_s, reconstruction = canonical.encode(
        model, raw_test_array, mean, std, "cuda", 128
    )
    standardized_test = (raw_test_array - mean) / std
    result = {
        "zC": canonical.evaluate_test_route(
            probe_c, validation_c, test_c, probe_meta, validation_meta,
            test_metadata, standardized_test, reconstruction, manifest,
        ),
        "zS": canonical.evaluate_test_route(
            probe_s, validation_s, test_s, probe_meta, validation_meta,
            test_metadata, standardized_test, reconstruction, manifest,
        ),
    }
    total_code = np.concatenate((test_c, test_s), axis=1)
    epsilon = manifest["constants"]["activation_epsilon"]
    result |= {
        "checkpoint": str(checkpoint),
        "reconstruction": evaluator.reconstruction_metrics(standardized_test, reconstruction),
        "total_l0": float((np.abs(total_code) > epsilon).sum(1).mean()),
        "total_fraction_alive": float(
            ((np.abs(total_code) > epsilon).mean(0)
             >= manifest["constants"]["minimum_activity_rate"]).mean()
        ),
    }
    summary = canonical.test_summary_row(method, SEED, result) | {
        "layer": layer,
        "initialization": "fresh paired",
        "initialization_fingerprint": fingerprint,
    }
    detail_path.write_text(
        json.dumps({"summary": summary, "result": result}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def dry_run(layers):
    metadata = pd.read_csv(DATA / "train_metadata.csv")
    test_metadata = pd.read_csv(DATA / "test_metadata.csv")
    manifest = evaluator.load_manifest()
    training, probe, validation = selected_rows(metadata, manifest)
    for layer in layers:
        train = np.load(DATA / f"raw_train_layer{layer}.npy", mmap_mode="r")
        test = np.load(DATA / f"raw_test_layer{layer}.npy", mmap_mode="r")
        assert train.shape == (len(metadata), 2304)
        assert test.shape == (len(test_metadata), 2304)
    model, fingerprint = fresh_model()
    with torch.inference_mode():
        zc, zs, reconstruction, _ = model(torch.zeros(2, 2304, device=pilot.DEVICE))
    assert zc.shape == (2, 2765) and zs.shape == (2, 6451)
    assert reconstruction.shape == (2, 2304)
    print(json.dumps({
        "status": "ready",
        "layers": list(layers),
        "training_rows": len(training),
        "probe_rows": len(probe),
        "feature_selection_rows": len(validation),
        "test_rows": len(test_metadata),
        "initialization_fingerprint": fingerprint,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=LAYERS, action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    layers = tuple(args.layer or LAYERS)
    methods = tuple(args.method or METHODS)
    if args.dry_run:
        dry_run(layers)
        return
    assert torch.cuda.is_available(), "GPU required"
    WORK.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(DATA / "train_metadata.csv")
    test_metadata = pd.read_csv(DATA / "test_metadata.csv")
    manifest = evaluator.load_manifest()
    training_rows, probe_rows, validation_rows = selected_rows(metadata, manifest)
    rows = []
    for layer in layers:
        raw = np.load(DATA / f"raw_train_layer{layer}.npy", mmap_mode="r")
        raw_test = np.load(DATA / f"raw_test_layer{layer}.npy", mmap_mode="r")
        mean, std = layer_stats(raw, training_rows, layer)
        for method in methods:
            model, _, checkpoint, _, fingerprint = train_or_load(
                layer, method, raw, metadata, training_rows, mean, std, args.epochs
            )
            row = evaluate_run(
                layer, method, model, raw, raw_test, metadata, test_metadata,
                probe_rows, validation_rows, mean, std, manifest, checkpoint, fingerprint,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["layer", "method"])
    frame.to_csv(CSV, index=False)
    paired = []
    for layer, group in frame.groupby("layer"):
        if set(group.method) != set(METHODS):
            continue
        control = group.set_index("method").loc["Blockwise control"]
        ours = group.set_index("method").loc["Ours"]
        paired.append({
            "layer": int(layer),
            "intent_auc_delta": float(ours.intent_concept_auc - control.intent_concept_auc),
            "stability_delta": float(ours.cross_locale_stability - control.cross_locale_stability),
            "locale_probe_delta": float(ours.locale_probe - control.locale_probe),
            "fve_delta": float(ours.reconstruction_fve - control.reconstruction_fve),
        })
    report = {
        "status": "one-seed appendix layer sensitivity",
        "manifest_sha256": evaluator.manifest_sha256(manifest),
        "protocol": {
            "model": "Gemma 2 2B",
            "dataset": "MASSIVE",
            "hidden_state_indices": list(layers),
            "methods": list(methods),
            "seed": SEED,
            "paired_fresh_initialization": True,
            "same_dictionary_and_activity_budgets": True,
            "same_training_and_canonical_evaluator": True,
            "checkpoints": str(WORK),
        },
        "rows": frame.to_dict(orient="records"),
        "paired_differences": paired,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(REPORT), "csv": str(CSV)}, indent=2))


if __name__ == "__main__":
    main()
