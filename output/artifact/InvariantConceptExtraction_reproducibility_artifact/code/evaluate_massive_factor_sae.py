"""Evaluate the Step 2 factor-SAEs with the frozen canonical protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy._core.multiarray import _reconstruct

import canonical_evaluator as evaluator
from run_massive_sparse_partition_pilot import SparsePartition


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data" / "massive_partition_artifacts"
CHECKPOINT_DIR = ROOT / "checkpoint" / "sparse_partition_pilot"
OUTPUT = ROOT / "Report" / "factor_sae_step3_canonical_validation.json"
CSV = ROOT / "Report" / "factor_sae_step3_canonical_validation.csv"
FINAL_OUTPUT = ROOT / "Report" / "factor_sae_step4_definitive_test.json"
FINAL_PER_SEED = ROOT / "Report" / "factor_sae_step4_definitive_test_per_seed.csv"
FINAL_SUMMARY = ROOT / "Report" / "factor_sae_step4_definitive_test_summary.csv"
WIDTH8_OUTPUT = ROOT / "Report" / "factor_sae_step5_width8_test.json"
WIDTH8_PER_SEED = ROOT / "Report" / "factor_sae_step5_width8_test_per_seed.csv"
WIDTH8_SUMMARY = ROOT / "Report" / "factor_sae_step5_width8_test_summary.csv"
SEEDS = (20260827, 20260828, 20260829)
CHECKPOINTS = {
    "BatchTopK SAE": "batchtopk_reconstruction_c30_seed20260827_ck13_global_none_val10_step2_batchtopk.pt",
    "Blockwise SAE control": "batchtopk_reconstruction_c30_seed20260827_ck13_none_val10_step2_block_control.pt",
    "Matryoshka SAE": "batchtopk_reconstruction_c30_seed20260827_ck13_global_matryoshka_none_val10_step2_matryoshka.pt",
    "One-sided factor SAE": "batchtopk_matched_c30_seed20260827_ck13_c_only_val10_step2_one_sided.pt",
    "Reciprocal factor SAE": "batchtopk_matched_c30_seed20260827_ck13_val10_step2_reciprocal.pt",
    "Triplet m=0.1": "batchtopk_triplet_c30_seed20260827_ck13_m0p1_val10_step2_triplet_m0p1.pt",
    "Triplet m=0.2": "batchtopk_triplet_c30_seed20260827_ck13_m0p2_val10_step2_triplet_m0p2.pt",
    "Triplet m=0.4": "batchtopk_triplet_c30_seed20260827_ck13_m0p4_val10_step2_triplet_m0p4.pt",
}
FINAL_METHODS = (
    "BatchTopK SAE", "Blockwise SAE control", "Matryoshka SAE",
    "One-sided factor SAE", "Reciprocal factor SAE", "Triplet m=0.2",
)
WIDTH8_METHODS = ("BatchTopK SAE", "Blockwise SAE control", "Reciprocal factor SAE")
WIDTH8_CHECKPOINTS = {
    "BatchTopK SAE": "batchtopk_reconstruction_c30_seed{seed}_w18432_k128_ck26_global_none_val10_step5_width8_batchtopk.pt",
    "Blockwise SAE control": "batchtopk_reconstruction_c30_seed{seed}_w18432_k128_ck26_none_val10_step5_width8_block_control.pt",
    "Reciprocal factor SAE": "batchtopk_matched_c30_seed{seed}_w18432_k128_ck26_val10_step5_width8_reciprocal.pt",
}


def load_checkpoint(path):
    torch.serialization.add_safe_globals([_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType])
    return torch.load(path, map_location="cpu", weights_only=True)


def selected_data(manifest, include_test=False):
    metadata = pd.read_csv(ARTIFACTS / "train_metadata.csv")
    raw = np.load(ARTIFACTS / "raw_train_layer8.npy", mmap_mode="r")
    keys = {tuple(map(str, row)) for row in manifest["probe_split"]["training_rows"]}
    train_rows = np.asarray([
        i for i, row in enumerate(metadata[["id", "locale"]].astype(str).itertuples(index=False, name=None))
        if row in keys
    ])
    feature_keys = {tuple(map(str, row)) for row in manifest["feature_selection_split"]["rows"]}
    validation_ids = set(manifest["splits"]["validation_ids"])
    stability_locales = set(manifest["feature_selection_split"]["stability_locales"])
    validation_rows = np.asarray([
        i for i, row in enumerate(metadata[["id", "locale"]].astype(str).itertuples(index=False, name=None))
        if row in feature_keys or (row[0] in validation_ids and row[1] in stability_locales)
    ])
    selected = (
        np.asarray(raw[train_rows], np.float32), metadata.iloc[train_rows].reset_index(drop=True),
        np.asarray(raw[validation_rows], np.float32), metadata.iloc[validation_rows].reset_index(drop=True),
    )
    if not include_test:
        return selected
    test_meta = pd.read_csv(ARTIFACTS / "test_metadata.csv")
    raw_test = np.load(ARTIFACTS / "raw_test_layer8.npy", mmap_mode="r")
    return selected + (np.asarray(raw_test, np.float32), test_meta)


def encode(model, raw, mean, std, device, batch_size):
    routes_c, routes_s, reconstructions = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            x = torch.from_numpy((raw[start:start + batch_size] - mean) / std).to(device)
            zc, zs, reconstruction, _ = model(x)
            routes_c.append(zc.cpu().numpy())
            routes_s.append(zs.cpu().numpy())
            reconstructions.append(reconstruction.cpu().numpy())
    return np.concatenate(routes_c), np.concatenate(routes_s), np.concatenate(reconstructions)


def evaluate_route(train, validation, train_meta, validation_meta, manifest):
    return evaluator.evaluate_validation_representation(
        train, validation, train_meta, validation_meta, manifest
    )


def final_checkpoint(name, seed):
    filename = CHECKPOINTS[name]
    if seed != SEEDS[0]:
        filename = filename.replace(f"seed{SEEDS[0]}", f"seed{seed}").replace("step2_", "step4_final_")
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {path}")
    return path


def width8_checkpoint(name, seed):
    path = CHECKPOINT_DIR / WIDTH8_CHECKPOINTS[name].format(seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"missing 8x checkpoint: {path}")
    return path


def evaluate_test_route(train, validation, test, train_meta, validation_meta, test_meta, target, reconstruction, manifest):
    return evaluator.evaluate_sparse_code(
        train, validation, test, train_meta, validation_meta, test_meta,
        target, reconstruction, manifest,
    )


def test_summary_row(method, seed, result):
    primary = result.get("global", result.get("zC"))
    surface = result.get("zS")
    row = {
        "method": method,
        "seed": seed,
        "intent_balanced_accuracy": primary["intent_balanced_accuracy_from_sparse_code"],
        "intent_accuracy": primary["intent_accuracy_from_sparse_code"],
        "intent_concept_auc": primary["mean_intent_concept_auc"]["value"],
        "cross_locale_stability": primary["cross_locale_feature_stability"]["value"],
        "locale_probe": primary["locale_probe_accuracy_from_sparse_code"]["value"],
        "intent_feature_fraction": primary["intent_oriented_feature_fraction"],
        "locale_feature_fraction": primary["locale_oriented_feature_fraction"],
        "fraction_alive": primary["fraction_alive"],
        "primary_l0": primary["mean_active_features_l0"],
    }
    if surface:
        row |= {
            "zS_locale_probe": surface["locale_probe_accuracy_from_sparse_code"]["value"],
            "zS_intent_balanced_accuracy": surface["intent_balanced_accuracy_from_sparse_code"],
        }
    if "reconstruction" in result:
        row |= {
            "reconstruction_mse": result["reconstruction"]["mse"],
            "reconstruction_fve": result["reconstruction"]["fraction_variance_explained"],
            "reconstruction_cosine": result["reconstruction"]["mean_cosine_similarity"],
            "total_l0": result["total_l0"],
            "total_fraction_alive": result["total_fraction_alive"],
        }
    return row


def aggregate(rows):
    frame = pd.DataFrame(rows)
    metrics = frame.select_dtypes(include="number").columns.difference(["seed"])
    summary = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "seeds": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return frame, pd.DataFrame(summary)


def run_final(
    args, manifest, methods=FINAL_METHODS, checkpoint_for=final_checkpoint,
    outputs=(FINAL_OUTPUT, FINAL_PER_SEED, FINAL_SUMMARY),
    status="definitive untouched MASSIVE test evaluation",
):
    raw_train, train_meta, raw_validation, validation_meta, raw_test, test_meta = selected_data(manifest, True)
    first = load_checkpoint(checkpoint_for(methods[0], SEEDS[0]))
    mean, std = np.asarray(first["input_mean"], np.float32), np.asarray(first["input_std"], np.float32)
    standardized_train = (raw_train - mean) / std
    standardized_validation = (raw_validation - mean) / std
    standardized_test = (raw_test - mean) / std
    raw = evaluate_test_route(
        standardized_train, standardized_validation, standardized_test,
        train_meta, validation_meta, test_meta, standardized_test, standardized_test, manifest,
    )
    for key in tuple(raw):
        if key.startswith("reconstruction_"):
            raw.pop(key)
    results = {"Raw H": {"global": raw}}
    rows = [test_summary_row("Raw H", "reference", results["Raw H"])]
    selected_methods = tuple(args.method or methods)
    if any(name not in methods for name in selected_methods):
        raise ValueError(f"evaluation accepts only: {', '.join(methods)}")
    for name in selected_methods:
        results[name] = {}
        for seed in tuple(args.seed or SEEDS):
            checkpoint_path = checkpoint_for(name, seed)
            checkpoint = load_checkpoint(checkpoint_path)
            config = checkpoint["config"]
            model = SparsePartition(
                config["activation"], config["c_fraction"], config["bandwidth"],
                config["total_k"], config["c_k"], config["sparsifier"],
                config.get("sparse_width", 9216),
            ).to(args.device)
            model.load_state_dict(checkpoint["state_dict"])
            train_c, train_s, _ = encode(model, raw_train, mean, std, args.device, args.batch_size)
            validation_c, validation_s, _ = encode(model, raw_validation, mean, std, args.device, args.batch_size)
            test_c, test_s, reconstruction = encode(model, raw_test, mean, std, args.device, args.batch_size)
            if config["sparsifier"] == "global":
                train_code = np.concatenate((train_c, train_s), axis=1)
                validation_code = np.concatenate((validation_c, validation_s), axis=1)
                test_code = np.concatenate((test_c, test_s), axis=1)
                result = {"global": evaluate_test_route(
                    train_code, validation_code, test_code, train_meta, validation_meta,
                    test_meta, standardized_test, reconstruction, manifest,
                )}
            else:
                result = {
                    "zC": evaluate_test_route(
                        train_c, validation_c, test_c, train_meta, validation_meta,
                        test_meta, standardized_test, reconstruction, manifest,
                    ),
                    "zS": evaluate_test_route(
                        train_s, validation_s, test_s, train_meta, validation_meta,
                        test_meta, standardized_test, reconstruction, manifest,
                    ),
                }
            total_code = np.concatenate((test_c, test_s), axis=1)
            epsilon = manifest["constants"]["activation_epsilon"]
            result |= {
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "reconstruction": evaluator.reconstruction_metrics(standardized_test, reconstruction),
                "total_l0": float((np.abs(total_code) > epsilon).sum(1).mean()),
                "total_fraction_alive": float(((np.abs(total_code) > epsilon).mean(0) >= manifest["constants"]["minimum_activity_rate"]).mean()),
            }
            results[name][str(seed)] = result
            row = test_summary_row(name, seed, result)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, train_c, train_s, validation_c, validation_s, test_c, test_s, reconstruction, total_code
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    per_seed, summary = aggregate(rows)
    report = {
        "manifest_sha256": evaluator.manifest_sha256(manifest),
        "status": status,
        "test_locales": manifest["locales"]["held_out"],
        "methods": results,
        "summary": [
            {key: value for key, value in row.items() if pd.notna(value)}
            for row in summary.to_dict(orient="records")
        ],
    }
    output, per_seed_output, summary_output = outputs
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    per_seed.to_csv(per_seed_output, index=False)
    summary.to_csv(summary_output, index=False)
    print(json.dumps({
        "json": str(output.relative_to(ROOT)),
        "per_seed_csv": str(per_seed_output.relative_to(ROOT)),
        "summary_csv": str(summary_output.relative_to(ROOT)),
    }, indent=2))


def summary_row(method, result):
    primary = result.get("global", result.get("zC"))
    surface = result.get("zS")
    row = {
        "method": method,
        "intent_bacc_k1": primary["intent_k_sparse_probe"]["1"]["balanced_accuracy"],
        "intent_bacc_k5": primary["intent_k_sparse_probe"]["5"]["balanced_accuracy"],
        "intent_bacc_k10": primary["intent_k_sparse_probe"]["10"]["balanced_accuracy"],
        "intent_bacc_k20": primary["intent_k_sparse_probe"]["20"]["balanced_accuracy"],
        "intent_bacc_all": primary["intent_k_sparse_probe"]["all"]["balanced_accuracy"],
        "locale_accuracy_all": primary["locale_k_sparse_probe"]["all"]["accuracy"],
        "intent_concept_auc": primary["mean_intent_concept_auc"]["value"],
        "cross_locale_stability": primary["cross_locale_feature_stability"]["value"],
        "intent_feature_fraction": primary["intent_oriented_feature_fraction"],
        "locale_feature_fraction": primary["locale_oriented_feature_fraction"],
        "fraction_alive": primary["fraction_alive"],
        "l0": primary["mean_active_features_l0"],
    }
    if surface:
        row |= {
            "zS_locale_accuracy_all": surface["locale_k_sparse_probe"]["all"]["accuracy"],
            "zS_intent_bacc_all": surface["intent_k_sparse_probe"]["all"]["balanced_accuracy"],
        }
    if "reconstruction" in result:
        row |= {
            "reconstruction_mse": result["reconstruction"]["mse"],
            "reconstruction_fve": result["reconstruction"]["fraction_variance_explained"],
            "reconstruction_cosine": result["reconstruction"]["mean_cosine_similarity"],
            "total_l0": result["total_l0"],
            "total_fraction_alive": result["total_fraction_alive"],
        }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(CHECKPOINTS), action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--final", action="store_true", help="evaluate all frozen seeds on untouched test activations")
    parser.add_argument("--width8", action="store_true", help="evaluate the frozen 8x robustness checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest = evaluator.load_manifest()
    assert not (args.final and args.width8)
    if args.width8:
        run_final(
            args, manifest, WIDTH8_METHODS, width8_checkpoint,
            (WIDTH8_OUTPUT, WIDTH8_PER_SEED, WIDTH8_SUMMARY),
            "8x dictionary robustness on untouched MASSIVE test locales",
        )
        return
    if args.final:
        run_final(args, manifest)
        return
    raw_train, train_meta, raw_validation, validation_meta = selected_data(manifest)
    first = load_checkpoint(CHECKPOINT_DIR / next(iter(CHECKPOINTS.values())))
    mean, std = np.asarray(first["input_mean"], np.float32), np.asarray(first["input_std"], np.float32)
    standardized_train = (raw_train - mean) / std
    standardized_validation = (raw_validation - mean) / std
    results = {
        "Raw H": {
            "global": evaluate_route(
                standardized_train, standardized_validation, train_meta, validation_meta, manifest
            )
        }
    }
    selected = args.method or tuple(CHECKPOINTS)
    for name in selected:
        checkpoint_path = CHECKPOINT_DIR / CHECKPOINTS[name]
        checkpoint = load_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        model = SparsePartition(
            config["activation"], config["c_fraction"], config["bandwidth"],
            config["total_k"], config["c_k"], config["sparsifier"],
            config.get("sparse_width", 9216),
        ).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        train_c, train_s, _ = encode(model, raw_train, mean, std, args.device, args.batch_size)
        validation_c, validation_s, reconstruction = encode(
            model, raw_validation, mean, std, args.device, args.batch_size
        )
        if config["sparsifier"] == "global":
            train_code = np.concatenate((train_c, train_s), axis=1)
            validation_code = np.concatenate((validation_c, validation_s), axis=1)
            result = {"global": evaluate_route(train_code, validation_code, train_meta, validation_meta, manifest)}
        else:
            result = {
                "zC": evaluate_route(train_c, validation_c, train_meta, validation_meta, manifest),
                "zS": evaluate_route(train_s, validation_s, train_meta, validation_meta, manifest),
            }
        total_code = np.concatenate((validation_c, validation_s), axis=1)
        epsilon = manifest["constants"]["activation_epsilon"]
        result |= {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "reconstruction": evaluator.reconstruction_metrics(standardized_validation, reconstruction),
            "total_l0": float((np.abs(total_code) > epsilon).sum(1).mean()),
            "total_fraction_alive": float(((np.abs(total_code) > epsilon).mean(0) >= manifest["constants"]["minimum_activity_rate"]).mean()),
        }
        results[name] = result
        print(json.dumps(summary_row(name, result)), flush=True)
        del model, train_c, train_s, validation_c, validation_s, reconstruction, total_code
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "manifest_sha256": evaluator.manifest_sha256(manifest),
        "selection_status": "validation only; untouched MASSIVE test activations were not loaded",
        "probe_k": list(evaluator.PROBE_K),
        "methods": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([summary_row(name, result) for name, result in results.items()]).to_csv(CSV, index=False)
    print(json.dumps({"json": str(OUTPUT.relative_to(ROOT)), "csv": str(CSV.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
