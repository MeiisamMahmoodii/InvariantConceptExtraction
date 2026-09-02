"""Evaluate the three MTOP factor-SAE methods with one frozen protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mtop_factor_evaluator as evaluator
from run_massive_sparse_partition_pilot import SparsePartition


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "mtop_intent_artifacts"
CHECKPOINTS = ROOT / "checkpoint" / "mtop_factor_sae"
OUTPUT = ROOT / "Report" / "factor_sae_step6_mtop_test.json"
PER_SEED = ROOT / "Report" / "factor_sae_step6_mtop_test_per_seed.csv"
SUMMARY = ROOT / "Report" / "factor_sae_step6_mtop_test_summary.csv"
METHODS = {
    "BatchTopK SAE": "batchtopk",
    "Blockwise SAE control": "block_control",
    "Reciprocal factor SAE": "reciprocal",
}
SEEDS = evaluator.SEEDS
DEVICE = "cuda"


def checkpoint_path(experiment, seed):
    path = CHECKPOINTS / f"mtop_{experiment}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing MTOP checkpoint: {path}")
    return path


def selected_data(manifest):
    metadata = pd.read_csv(ART / "train_metadata.csv")
    raw = np.load(ART / "raw_train_layer8.npy", mmap_mode="r")
    probe_rows = evaluator.rows(metadata, manifest["probe_split"]["training_rows"])
    validation_rows = evaluator.rows(
        metadata, manifest["feature_selection_split"]["rows"]
    )
    test_meta = pd.read_csv(ART / "test_metadata.csv")
    test = np.load(ART / "raw_test_layer8.npy", mmap_mode="r")
    return (
        np.asarray(raw[probe_rows], np.float32),
        metadata.iloc[probe_rows].reset_index(drop=True),
        np.asarray(raw[validation_rows], np.float32),
        metadata.iloc[validation_rows].reset_index(drop=True),
        np.asarray(test, np.float32),
        test_meta,
    )


def model_from_checkpoint(saved):
    config = saved["config"]
    model = SparsePartition(
        "batchtopk", config["c_fraction"], 0.05,
        total_k=config["total_k"], c_k=config["c_k"],
        sparsifier=config["sparsifier"], sparse_width=config["sparse_width"],
    ).to(DEVICE)
    model.load_state_dict(saved["state_dict"])
    return model


def encode(model, values, mean, std, batch_size=128):
    c, s, reconstruction = [], [], []
    # BatchTopK is the locked architecture: each fixed evaluation batch keeps
    # exactly B*k activations, so all methods have the same average L0.
    model.train()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(
                (values[start:start + batch_size] - mean) / std
            ).to(DEVICE)
            zc, zs, decoded, _ = model(x)
            c.append(zc.cpu().numpy())
            s.append(zs.cpu().numpy())
            reconstruction.append(decoded.cpu().numpy())
    return np.concatenate(c), np.concatenate(s), np.concatenate(reconstruction)


def standardized(values, mean, std):
    return (np.asarray(values, np.float32) - mean) / std


def row(method, seed, primary, z_s=None, total_l0=None):
    return {
        "method": method,
        "seed": seed,
        "intent_accuracy": primary["intent_accuracy"],
        "intent_balanced_accuracy": primary["intent_balanced_accuracy"],
        "intent_macro_f1": primary["intent_macro_f1"],
        "intent_retrieval_r1": primary["intent_retrieval"]["R@1"]["value"],
        "intent_retrieval_mrr": primary["intent_retrieval"]["MRR"]["value"],
        "intent_relation_margin": primary["intent_relation_margin"]["intent_dominance_margin"]["value"],
        "locale_probe": primary["locale_probe"]["accuracy"]["value"],
        "intent_concept_auc": primary["mean_intent_concept_auc"]["value"],
        "cross_language_stability": primary["cross_language_feature_stability"]["value"],
        "intent_feature_fraction": primary["intent_oriented_feature_fraction"],
        "locale_feature_fraction": primary["locale_oriented_feature_fraction"],
        "fraction_alive": primary["fraction_alive"],
        "primary_l0": primary["mean_active_features_l0"],
        "total_l0": total_l0,
        "reconstruction_mse": primary.get("reconstruction_mse"),
        "reconstruction_fve": primary.get("reconstruction_fve"),
        "reconstruction_cosine": primary.get("reconstruction_cosine"),
        "zS_locale_probe": None if z_s is None else z_s["locale_probe"]["accuracy"]["value"],
        "zS_intent_balanced_accuracy": None if z_s is None else z_s["intent_balanced_accuracy"],
    }


def summarize(frame):
    numeric = [column for column in frame.columns if column not in ("method", "seed")]
    output = []
    for method, group in frame.groupby("method", sort=False):
        item = {"method": method, "seeds": len(group)}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            item[f"{column}_mean"] = float(values.mean()) if len(values) else None
            item[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(item)
    return pd.DataFrame(output)


def evaluate():
    if not torch.cuda.is_available():
        raise RuntimeError("MTOP evaluation requires CUDA")
    manifest = evaluator.load_manifest()
    manifest_audit = evaluator.audit_manifest(manifest)
    train_raw, train_meta, validation_raw, validation_meta, test_raw, test_meta = selected_data(manifest)
    report = {
        "status": "definitive MTOP direct factor-SAE transfer",
        "manifest": manifest_audit,
        "inference": "fixed deterministic batches with exact BatchTopK activity budget",
        "runs": [],
    }
    rows = []

    first = torch.load(
        checkpoint_path("batchtopk", SEEDS[0]), map_location="cpu", weights_only=True
    )
    mean = first["config"]["input_mean"].numpy()
    std = first["config"]["input_std"].numpy()
    raw_primary = evaluator.evaluate_route(
        standardized(train_raw, mean, std),
        standardized(validation_raw, mean, std),
        standardized(test_raw, mean, std),
        train_meta, validation_meta, test_meta, manifest,
    )
    report["raw_H"] = raw_primary
    rows.append(row("Raw H", "reference", raw_primary, total_l0=2304.0))

    for method, experiment in METHODS.items():
        for seed in SEEDS:
            path = checkpoint_path(experiment, seed)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = saved["config"]
            if config["manifest_sha256"] != evaluator.manifest_sha256(manifest):
                raise ValueError(f"checkpoint manifest mismatch: {path}")
            mean = config["input_mean"].numpy()
            std = config["input_std"].numpy()
            model = model_from_checkpoint(saved)
            train_c, train_s, _ = encode(model, train_raw, mean, std)
            validation_c, validation_s, _ = encode(model, validation_raw, mean, std)
            test_c, test_s, reconstruction = encode(model, test_raw, mean, std)
            total_l0 = float(
                ((test_c != 0).sum(1) + (test_s != 0).sum(1)).mean()
            )
            if experiment == "batchtopk":
                train_primary = np.concatenate((train_c, train_s), axis=1)
                validation_primary = np.concatenate((validation_c, validation_s), axis=1)
                test_primary = np.concatenate((test_c, test_s), axis=1)
                z_s_result = None
            else:
                train_primary, validation_primary, test_primary = train_c, validation_c, test_c
                z_s_result = evaluator.evaluate_route(
                    train_s, validation_s, test_s,
                    train_meta, validation_meta, test_meta, manifest,
                )
            primary = evaluator.evaluate_sparse(
                train_primary, validation_primary, test_primary,
                train_meta, validation_meta, test_meta,
                standardized(test_raw, mean, std), reconstruction, manifest,
            )
            result = {
                "method": method,
                "experiment": experiment,
                "seed": seed,
                "checkpoint": str(path.relative_to(ROOT)),
                "primary": primary,
                "zS": z_s_result,
                "total_l0": total_l0,
                "history": saved["history"],
            }
            report["runs"].append(result)
            rows.append(row(method, seed, primary, z_s_result, total_l0))
            OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del model, train_c, train_s, validation_c, validation_s, test_c, test_s
            torch.cuda.empty_cache()

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(PER_SEED, index=False)
    summary.to_csv(SUMMARY, index=False)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.build_manifest:
        manifest = evaluator.build_manifest()
        print(json.dumps(evaluator.audit_manifest(manifest), indent=2))
    if args.evaluate:
        evaluate()
    if not (args.build_manifest or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()
