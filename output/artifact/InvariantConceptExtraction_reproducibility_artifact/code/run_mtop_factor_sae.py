"""Train the locked direct factor SAE on MTOP."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import intent_locale_relations as relations
import mtop_factor_evaluator as evaluator
from run_massive_sparse_partition_pilot import SparsePartition, relation_loss


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "mtop_intent_artifacts"
CHECKPOINTS = ROOT / "checkpoint" / "mtop_factor_sae"
REPORTS = ROOT / "Report" / "mtop_factor_sae_runs"
SMOKE = ROOT / "archive" / "smoke" / "2026-09-02_step6_mtop"
SEEDS = evaluator.SEEDS
EXPERIMENTS = ("batchtopk", "block_control", "reciprocal")
DEVICE = "cuda"
EPOCHS = 30
BATCH_SIZE = 128
WIDTH = 9216
TOTAL_K = 64
C_K = 13
C_FRACTION = 0.30
TEMPERATURE = 0.07
LEARNING_RATE = 1e-4


def training_data(manifest):
    metadata = pd.read_csv(ART / "train_metadata.csv")
    raw = np.load(ART / "raw_train_layer8.npy", mmap_mode="r")
    locale_by_id = dict(zip(metadata.id.astype(str), metadata.locale.astype(str)))
    selected = evaluator.rows(
        metadata,
        [[identifier, locale_by_id[identifier]]
         for identifier in manifest["splits"]["training_ids"]],
    )
    metadata = metadata.iloc[selected].reset_index(drop=True)
    values = np.asarray(raw[selected], dtype=np.float32)
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    std = values.std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    values = (values - mean) / std
    return values, metadata, mean, std


def make_model(experiment):
    sparsifier = "global" if experiment == "batchtopk" else "block"
    model = SparsePartition(
        "batchtopk", C_FRACTION, 0.05,
        total_k=TOTAL_K, c_k=C_K,
        sparsifier=sparsifier, sparse_width=WIDTH,
    ).to(DEVICE)
    model.initialize_fresh()
    return model


def calibrate(model, values, seed):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(values), min(2048, len(values)), replace=False)
    model.eval()
    with torch.inference_mode():
        pre = model.encoder(torch.from_numpy(values[rows]).to(DEVICE))
        if model.sparsifier == "global":
            code = model.batch_topk(pre, model.total_k)
            threshold = float(code[code > 0].min().cpu())
            model.set_batchtopk_thresholds(threshold, threshold)
            return [threshold, threshold]
        c = model.batch_topk(pre[:, :model.c_width], model.c_k)
        s = model.batch_topk(pre[:, model.c_width:], model.s_k)
        thresholds = [float(c[c > 0].min().cpu()), float(s[s > 0].min().cpu())]
        model.set_batchtopk_thresholds(*thresholds)
        return thresholds


def relation_audit(metadata, seed):
    rng = np.random.default_rng(seed)
    index = relations.build_relation_index(metadata)
    anchors = rng.choice(len(metadata), min(10000, len(metadata)), replace=False)
    positive, negative = relations.sample_intent_relations(anchors, index, rng, 0.0)
    audit = relations.audit_sample(anchors, positive, negative, index)
    expected = {
        "zC_same_intent_fraction": 1.0,
        "zC_different_locale_fraction": 1.0,
        "zC_exact_id_fraction": 0.0,
        "zC_nonexact_different_id_fraction": 1.0,
        "zS_same_locale_fraction": 1.0,
        "zS_different_intent_fraction": 1.0,
    }
    if any(audit[key] != value for key, value in expected.items()):
        raise ValueError(f"MTOP training relation audit failed: {audit}")
    return audit


def train(experiment, seed, values, metadata, epochs, max_steps):
    torch.manual_seed(seed)
    model = make_model(experiment)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    order_rng = np.random.default_rng(seed)
    relation_rng = np.random.default_rng(seed + 1)
    relation_index = (
        relations.build_relation_index(metadata) if experiment == "reciprocal" else None
    )
    history = []
    for epoch in range(epochs):
        order = order_rng.permutation(len(metadata))
        totals = {"loss": 0.0, "reconstruction": 0.0, "zC_relation": 0.0, "zS_relation": 0.0, "l0": 0.0}
        steps = 0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch_rows = order[start:start + BATCH_SIZE]
            anchor = torch.from_numpy(values[batch_rows]).to(DEVICE)
            zc, zs, reconstruction, _ = model(anchor)
            reconstruction_loss = F.mse_loss(reconstruction, anchor)
            c_loss = reconstruction_loss.new_zeros(())
            s_loss = reconstruction_loss.new_zeros(())
            if relation_index is not None:
                c_positive, s_positive = relations.sample_intent_relations(
                    batch_rows, relation_index, relation_rng, 0.0
                )
                zcp, zsn, _ = model.encode(
                    torch.from_numpy(values[c_positive]).to(DEVICE)
                )
                zcn, zsp, _ = model.encode(
                    torch.from_numpy(values[s_positive]).to(DEVICE)
                )
                c_loss = relation_loss(
                    zc, zcp, zcn, "matched", TEMPERATURE
                )
                s_loss = relation_loss(
                    zs, zsp, zsn, "matched", TEMPERATURE
                )
            route_loss = (c_loss + s_loss) / 2
            loss = reconstruction_loss + route_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            l0 = ((zc != 0).sum(1) + (zs != 0).sum(1)).float().mean()
            for key, value in (
                ("loss", loss), ("reconstruction", reconstruction_loss),
                ("zC_relation", c_loss), ("zS_relation", s_loss), ("l0", l0),
            ):
                totals[key] += float(value.detach())
            steps += 1
            if max_steps and steps >= max_steps:
                break
        row = {key: value / steps for key, value in totals.items()} | {"epoch": epoch + 1}
        history.append(row)
        print(json.dumps({"experiment": experiment, "seed": seed, **row}), flush=True)
        if max_steps:
            break
    thresholds = calibrate(model, values, seed)
    return model, history, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment", choices=EXPERIMENTS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()
    if args.train and args.smoke:
        raise ValueError("choose full training or smoke, not both")
    if not torch.cuda.is_available():
        raise RuntimeError("MTOP factor-SAE training requires CUDA")

    manifest = evaluator.load_manifest()
    audit = evaluator.audit_manifest(manifest)
    values, metadata, mean, std = training_data(manifest)
    training_relation_audit = relation_audit(metadata, SEEDS[0])
    experiments = tuple(args.experiment or EXPERIMENTS)
    seeds = tuple(args.seed or ((SEEDS[0],) if args.smoke else SEEDS))
    protocol = {
        "dataset": "MTOP",
        "manifest_sha256": evaluator.manifest_sha256(manifest),
        "architecture": "2304 -> 9216 direct blockwise BatchTopK -> 2304",
        "route_widths": [2765, 6451],
        "training_activity": {"zC": 13, "zS": 51, "total": 64},
        "relation": "same-intent/different-language positive; different-intent/anchor-language opposing example",
        "exact_id_positive_fraction": 0.0,
        "epochs": 1 if args.smoke else EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "seeds": list(seeds),
        "experiments": list(experiments),
        "manifest_audit": audit,
        "training_relation_audit": training_relation_audit,
    }
    print(json.dumps(protocol, indent=2), flush=True)
    if not (args.train or args.smoke):
        return

    checkpoint_dir = SMOKE / "checkpoints" if args.smoke else CHECKPOINTS
    report_dir = SMOKE / "reports" if args.smoke else REPORTS
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        for seed in seeds:
            suffix = "_smoke" if args.smoke else ""
            checkpoint = checkpoint_dir / f"mtop_{experiment}_seed{seed}{suffix}.pt"
            if checkpoint.exists():
                print(json.dumps({"status": "already complete", "checkpoint": str(checkpoint)}), flush=True)
                continue
            model, history, thresholds = train(
                experiment, seed, values, metadata,
                1 if args.smoke else EPOCHS,
                1 if args.smoke else 0,
            )
            config = protocol | {
                "experiment": experiment,
                "seed": seed,
                "sparsifier": "global" if experiment == "batchtopk" else "block",
                "route_mode": "reciprocal" if experiment == "reciprocal" else "none",
                "input_mean": torch.from_numpy(mean),
                "input_std": torch.from_numpy(std),
                "batchtopk_inference_thresholds": thresholds,
                "sparse_width": WIDTH,
                "c_fraction": C_FRACTION,
                "total_k": TOTAL_K,
                "c_k": C_K,
            }
            torch.save(
                {"state_dict": model.state_dict(), "config": config, "history": history},
                checkpoint,
            )
            report = {
                "status": "smoke" if args.smoke else "complete",
                "checkpoint": str(checkpoint),
                "config": {key: value for key, value in config.items() if key not in ("input_mean", "input_std")},
                "history": history,
            }
            (report_dir / f"mtop_{experiment}_seed{seed}{suffix}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
