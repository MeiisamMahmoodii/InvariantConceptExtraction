"""Pythia-160M model-family transfer for the direct reciprocal factor SAE."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import canonical_evaluator as evaluator
import intent_locale_relations as relations
from run_massive_sparse_partition_pilot import SparsePartition, relation_loss


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"D:\data\pythia-160m")
DATA = Path(r"D:\data\InvariantConceptExtraction\pythia160m_massive_factor_sae")
GEMMA_ARTIFACTS = ROOT / "data" / "massive_partition_artifacts"
CHECKPOINTS = ROOT / "checkpoint" / "pythia160m_factor_sae"
SMOKE = ROOT / "archive" / "smoke" / "2026-09-02_pythia160m_transfer"
OUTPUT = ROOT / "Report" / "factor_sae_pythia160m_transfer.json"
PER_SEED = ROOT / "Report" / "factor_sae_pythia160m_transfer_per_seed.csv"
SUMMARY = ROOT / "Report" / "factor_sae_pythia160m_transfer_summary.csv"
SEEDS = evaluator.TRAINING_SEEDS
EXPERIMENTS = ("batchtopk", "block_control", "reciprocal")
METHODS = {
    "BatchTopK SAE": "batchtopk",
    "Blockwise SAE control": "block_control",
    "Reciprocal factor SAE": "reciprocal",
}
DEVICE = "cuda"
INPUT_WIDTH = 768
SPARSE_WIDTH = 9216
TOTAL_K = 64
C_K = 13
C_FRACTION = 0.30
EPOCHS = 30
ANCHORS_PER_EPOCH = 45968
BATCH_SIZE = 128
TEMPERATURE = 0.07
LEARNING_RATE = 1e-4


class RowView:
    def __init__(self, values, rows):
        self.values = values
        self.rows = np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, rows):
        return self.values[self.rows[rows]]


def aligned_texts(metadata, parquet):
    source = pd.read_parquet(parquet)[["id", "locale", "intent", "utt"]].copy()
    source["id"] = source.id.astype(str)
    source = source.rename(columns={"intent": "source_intent"})
    keys = metadata[["id", "locale", "intent"]].copy()
    keys["id"] = keys.id.astype(str)
    keys["_order"] = np.arange(len(keys))
    aligned = keys.merge(source, on=["id", "locale"], how="left", validate="one_to_one")
    aligned = aligned.sort_values("_order")
    if aligned.utt.isna().any() or not np.array_equal(
        aligned.intent.to_numpy(), aligned.source_intent.to_numpy()
    ):
        raise ValueError(f"MASSIVE text alignment failed for {parquet}")
    return aligned.utt.astype(str).tolist()


def extract_split(name, metadata, texts, batch_size):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"raw_{name}_layer8.npy"
    complete = DATA / f"raw_{name}_layer8.complete"
    progress = DATA / f"raw_{name}_layer8.progress.json"
    expected = (len(texts), INPUT_WIDTH)
    if complete.exists():
        values = np.load(path, mmap_mode="r")
        if values.shape != expected:
            raise ValueError(f"completed {name} activation shape is {values.shape}, expected {expected}")
        return

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).to(DEVICE).eval()
    if model.config.hidden_size != INPUT_WIDTH or len(model.layers) <= 7:
        raise ValueError("unexpected Pythia-160M architecture")
    if path.exists() and np.load(path, mmap_mode="r").shape != expected:
        raise ValueError(f"incomplete {name} activation has the wrong shape")
    output = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype="float32", shape=expected
    )
    start = json.loads(progress.read_text())["next"] if progress.exists() else 0
    captured = {}
    handle = model.layers[7].register_forward_hook(
        lambda _module, _inputs, value: captured.__setitem__(
            "value", value[0] if isinstance(value, tuple) else value
        )
    )
    with torch.inference_mode():
        for chunk_start in range(start, len(texts), 8192):
            chunk_end = min(chunk_start + 8192, len(texts))
            order = np.arange(chunk_start, chunk_end)
            order = order[np.argsort([len(texts[index]) for index in order])]
            for offset in range(0, len(order), batch_size):
                rows = order[offset:offset + batch_size]
                tokens = tokenizer(
                    [texts[index] for index in rows], padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                ).to(DEVICE)
                model(**tokens, use_cache=False)
                hidden = captured.pop("value")
                mask = tokens.attention_mask.unsqueeze(-1)
                output[rows] = (
                    (hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)
                ).float().cpu().numpy()
            output.flush()
            progress.write_text(json.dumps({"next": chunk_end}) + "\n")
            print(f"Pythia {name} activations {chunk_end}/{len(texts)}", flush=True)
    handle.remove()
    if not np.isfinite(np.asarray(output[: min(1024, len(output))])).all():
        raise ValueError(f"non-finite {name} activations")
    complete.write_text("complete\n")
    progress.unlink(missing_ok=True)
    metadata.to_csv(DATA / f"{name}_metadata.csv", index=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def extract(batch_size):
    if not MODEL.exists():
        raise FileNotFoundError(f"Pythia model is missing: {MODEL}")
    train_meta = pd.read_csv(GEMMA_ARTIFACTS / "train_metadata.csv")
    test_meta = pd.read_csv(GEMMA_ARTIFACTS / "test_metadata.csv")
    extract_split(
        "train", train_meta,
        aligned_texts(train_meta, ROOT / "data" / "massive_all_train.parquet"),
        batch_size,
    )
    extract_split(
        "test", test_meta,
        aligned_texts(test_meta, ROOT / "data" / "massive_all_test.parquet"),
        batch_size,
    )
    manifest = {
        "model": "EleutherAI/pythia-160m",
        "local_model": str(MODEL),
        "hidden_state_index": 8,
        "captured_module": "GPTNeoXModel.layers[7] output",
        "pooling": "masked mean over non-padding tokens",
        "max_length": 128,
        "input_width": INPUT_WIDTH,
        "train_shape": list(np.load(DATA / "raw_train_layer8.npy", mmap_mode="r").shape),
        "test_shape": list(np.load(DATA / "raw_test_layer8.npy", mmap_mode="r").shape),
        "training_performed": False,
    }
    (DATA / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def training_data(manifest):
    metadata = pd.read_csv(GEMMA_ARTIFACTS / "train_metadata.csv")
    raw = np.load(DATA / "raw_train_layer8.npy", mmap_mode="r")
    allowed = set(map(str, manifest["splits"]["training_ids"]))
    rows = np.flatnonzero(metadata.id.astype(str).isin(allowed).to_numpy())
    selected_meta = metadata.iloc[rows].reset_index(drop=True)
    return RowView(raw, rows), selected_meta


def moments(values):
    total = np.zeros(INPUT_WIDTH, np.float64)
    squared = np.zeros(INPUT_WIDTH, np.float64)
    for start in range(0, len(values), 4096):
        batch = np.asarray(values[start:start + 4096], np.float32)
        total += batch.sum(0)
        squared += np.square(batch, dtype=np.float64).sum(0)
    mean = total / len(values)
    std = np.sqrt(np.maximum(squared / len(values) - mean * mean, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def make_model(experiment):
    model = SparsePartition(
        "batchtopk", C_FRACTION, 0.05, total_k=TOTAL_K, c_k=C_K,
        sparsifier="global" if experiment == "batchtopk" else "block",
        sparse_width=SPARSE_WIDTH, input_width=INPUT_WIDTH,
    ).to(DEVICE)
    model.initialize_fresh()
    return model


def relation_audit(metadata, seed):
    rng = np.random.default_rng(seed)
    index = relations.build_relation_index(metadata)
    anchors = rng.choice(len(metadata), min(10000, len(metadata)), replace=False)
    positive, negative = relations.sample_intent_relations(anchors, index, rng, 0.5)
    audit = relations.audit_sample(anchors, positive, negative, index)
    expected = {
        "zC_same_intent_fraction": 1.0,
        "zC_different_locale_fraction": 1.0,
        "zC_exact_id_fraction": 0.5,
        "zC_nonexact_different_id_fraction": 0.5,
        "zS_same_locale_fraction": 1.0,
        "zS_different_intent_fraction": 1.0,
    }
    if any(audit[key] != value for key, value in expected.items()):
        raise ValueError(f"Pythia relation audit failed: {audit}")
    return audit


def normalized(values, rows, mean, std):
    batch = (np.asarray(values[rows], np.float32) - mean) / std
    return torch.from_numpy(batch).to(DEVICE)


def train_one(experiment, seed, raw, metadata, mean, std, epochs, max_steps):
    torch.manual_seed(seed)
    model = make_model(experiment)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed)
    relation_rng = np.random.default_rng(seed + 1)
    relation_index = relations.build_relation_index(metadata) if experiment == "reciprocal" else None
    history = []
    for epoch in range(epochs):
        order = order_rng.choice(
            len(metadata), min(ANCHORS_PER_EPOCH, len(metadata)), replace=False
        )
        totals = {"loss": 0.0, "reconstruction": 0.0, "zC_relation": 0.0, "zS_relation": 0.0, "l0": 0.0}
        steps = 0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            anchor = normalized(raw, rows, mean, std)
            zc, zs, reconstruction, _ = model(anchor)
            reconstruction_loss = F.mse_loss(reconstruction, anchor)
            c_loss = s_loss = reconstruction_loss.new_zeros(())
            if relation_index is not None:
                c_positive, s_positive = relations.sample_intent_relations(
                    rows, relation_index, relation_rng, 0.5
                )
                zcp, zsn, _ = model.encode(normalized(raw, c_positive, mean, std))
                zcn, zsp, _ = model.encode(normalized(raw, s_positive, mean, std))
                c_loss = relation_loss(zc, zcp, zcn, "matched", TEMPERATURE)
                s_loss = relation_loss(zs, zsp, zsn, "matched", TEMPERATURE)
            loss = reconstruction_loss + (c_loss + s_loss) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            values = {
                "loss": loss, "reconstruction": reconstruction_loss,
                "zC_relation": c_loss, "zS_relation": s_loss,
                "l0": ((zc != 0).sum(1) + (zs != 0).sum(1)).float().mean(),
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            steps += 1
            if max_steps and steps >= max_steps:
                break
        row = {key: value / steps for key, value in totals.items()} | {"epoch": epoch + 1}
        history.append(row)
        print(json.dumps({"experiment": experiment, "seed": seed, **row}), flush=True)
        if max_steps:
            break
    return model, history


def train(smoke=False, experiments=EXPERIMENTS, seeds=SEEDS):
    manifest = evaluator.load_manifest()
    raw, metadata = training_data(manifest)
    mean, std = moments(raw)
    audit = relation_audit(metadata, SEEDS[0])
    checkpoint_dir = SMOKE if smoke else CHECKPOINTS
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "MASSIVE",
        "model": "EleutherAI/pythia-160m",
        "hidden_state_index": 8,
        "manifest_sha256": evaluator.manifest_sha256(manifest),
        "architecture": "768 -> 9216 direct BatchTopK SAE -> 768",
        "route_widths": [round(SPARSE_WIDTH * C_FRACTION), SPARSE_WIDTH - round(SPARSE_WIDTH * C_FRACTION)],
        "training_activity": {"zC": C_K, "zS": TOTAL_K - C_K, "total": TOTAL_K},
        "epochs": 1 if smoke else EPOCHS,
        "anchors_per_epoch": ANCHORS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "exact_id_positive_fraction": 0.5,
        "relation_audit": audit,
    }
    print(json.dumps(protocol, indent=2), flush=True)
    for experiment in experiments:
        for seed in seeds:
            suffix = "_smoke" if smoke else ""
            checkpoint = checkpoint_dir / f"pythia160m_{experiment}_seed{seed}{suffix}.pt"
            if checkpoint.exists():
                print(json.dumps({"status": "already complete", "checkpoint": str(checkpoint)}), flush=True)
                continue
            model, history = train_one(
                experiment, seed, raw, metadata, mean, std,
                1 if smoke else EPOCHS, 1 if smoke else 0,
            )
            config = protocol | {
                "experiment": experiment,
                "seed": seed,
                "sparsifier": "global" if experiment == "batchtopk" else "block",
                "input_width": INPUT_WIDTH,
                "sparse_width": SPARSE_WIDTH,
                "c_fraction": C_FRACTION,
                "total_k": TOTAL_K,
                "c_k": C_K,
                "input_mean": torch.from_numpy(mean),
                "input_std": torch.from_numpy(std),
            }
            torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, checkpoint)
            del model
            torch.cuda.empty_cache()


def selected_data(manifest):
    metadata = pd.read_csv(GEMMA_ARTIFACTS / "train_metadata.csv")
    raw = np.load(DATA / "raw_train_layer8.npy", mmap_mode="r")
    lookup = {
        (str(identifier), str(locale)): index
        for index, (identifier, locale) in enumerate(metadata[["id", "locale"]].itertuples(index=False, name=None))
    }
    def rows(keys):
        return np.asarray([lookup[(str(identifier), str(locale))] for identifier, locale in keys])
    probe_rows = rows(manifest["probe_split"]["training_rows"])
    feature_keys = {tuple(map(str, key)) for key in manifest["feature_selection_split"]["rows"]}
    validation_ids = set(map(str, manifest["splits"]["validation_ids"]))
    stability_locales = set(manifest["feature_selection_split"]["stability_locales"])
    validation_rows = np.asarray([
        index for index, (identifier, locale) in enumerate(metadata[["id", "locale"]].astype(str).itertuples(index=False, name=None))
        if (identifier, locale) in feature_keys or (identifier in validation_ids and locale in stability_locales)
    ])
    test_meta = pd.read_csv(GEMMA_ARTIFACTS / "test_metadata.csv")
    test = np.load(DATA / "raw_test_layer8.npy", mmap_mode="r")
    return (
        np.asarray(raw[probe_rows], np.float32), metadata.iloc[probe_rows].reset_index(drop=True),
        np.asarray(raw[validation_rows], np.float32), metadata.iloc[validation_rows].reset_index(drop=True),
        np.asarray(test, np.float32), test_meta,
    )


def checkpoint_path(experiment, seed):
    path = CHECKPOINTS / f"pythia160m_{experiment}_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing Pythia checkpoint: {path}")
    return path


def encode(model, values, mean, std):
    c, s, reconstruction = [], [], []
    model.train()
    with torch.inference_mode():
        for start in range(0, len(values), BATCH_SIZE):
            x = torch.from_numpy((values[start:start + BATCH_SIZE] - mean) / std).to(DEVICE)
            zc, zs, decoded, _ = model(x)
            c.append(zc.cpu().numpy())
            s.append(zs.cpu().numpy())
            reconstruction.append(decoded.cpu().numpy())
    return np.concatenate(c), np.concatenate(s), np.concatenate(reconstruction)


def result_row(method, seed, primary, z_s=None, total_l0=None):
    return {
        "method": method,
        "seed": seed,
        "intent_balanced_accuracy": primary["intent_balanced_accuracy_from_sparse_code"],
        "intent_concept_auc": primary["mean_intent_concept_auc"]["value"],
        "cross_locale_stability": primary["cross_locale_feature_stability"]["value"],
        "intent_relation_margin": primary["intent_relation_margin"]["intent_dominance_margin"]["value"],
        "intent_retrieval_r1": primary["intent_retrieval"]["R@1"]["value"],
        "locale_probe": primary["locale_probe_accuracy_from_sparse_code"]["value"],
        "intent_feature_fraction": primary["intent_oriented_feature_fraction"],
        "locale_feature_fraction": primary["locale_oriented_feature_fraction"],
        "primary_l0": primary["mean_active_features_l0"],
        "total_l0": total_l0,
        "reconstruction_fve": primary["reconstruction_fve"],
        "zS_locale_probe": None if z_s is None else z_s["locale_probe_accuracy_from_sparse_code"]["value"],
        "zS_intent_balanced_accuracy": None if z_s is None else z_s["intent_balanced_accuracy_from_sparse_code"],
    }


def summarize(frame):
    output = []
    for method, group in frame.groupby("method", sort=False):
        item = {"method": method, "seeds": len(group)}
        for column in frame.columns.difference(["method", "seed"]):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            item[f"{column}_mean"] = float(values.mean()) if len(values) else None
            item[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return pd.DataFrame(output)


def evaluate():
    manifest = evaluator.load_manifest()
    audit = evaluator.audit_manifest(manifest)
    train_raw, train_meta, validation_raw, validation_meta, test_raw, test_meta = selected_data(manifest)
    first = torch.load(checkpoint_path("batchtopk", SEEDS[0]), map_location="cpu", weights_only=True)
    mean = first["config"]["input_mean"].numpy()
    std = first["config"]["input_std"].numpy()
    standardized = lambda values: (np.asarray(values, np.float32) - mean) / std
    raw_primary = evaluator.evaluate_sparse_code(
        standardized(train_raw), standardized(validation_raw), standardized(test_raw),
        train_meta, validation_meta, test_meta,
        standardized(test_raw), standardized(test_raw), manifest,
    )
    raw_primary["intent_relation_margin"] = evaluator.intent_relation_margin(
        standardized(test_raw), test_meta, manifest
    )
    raw_primary["intent_retrieval"] = evaluator.intent_retrieval(
        standardized(test_raw), test_meta, manifest
    )
    report = {"status": "Pythia-160M model-family transfer", "manifest": audit, "raw_H": raw_primary, "runs": []}
    rows = [result_row("Raw H", "reference", raw_primary, total_l0=float(INPUT_WIDTH))]
    for method, experiment in METHODS.items():
        for seed in SEEDS:
            path = checkpoint_path(experiment, seed)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = saved["config"]
            if config["manifest_sha256"] != evaluator.manifest_sha256(manifest):
                raise ValueError(f"manifest mismatch: {path}")
            mean = config["input_mean"].numpy()
            std = config["input_std"].numpy()
            model = SparsePartition(
                "batchtopk", config["c_fraction"], 0.05,
                total_k=config["total_k"], c_k=config["c_k"],
                sparsifier=config["sparsifier"], sparse_width=config["sparse_width"],
                input_width=config["input_width"],
            ).to(DEVICE)
            model.load_state_dict(saved["state_dict"])
            train_c, train_s, _ = encode(model, train_raw, mean, std)
            validation_c, validation_s, _ = encode(model, validation_raw, mean, std)
            test_c, test_s, reconstruction = encode(model, test_raw, mean, std)
            total_l0 = float(((test_c != 0).sum(1) + (test_s != 0).sum(1)).mean())
            if experiment == "batchtopk":
                train_primary = np.concatenate((train_c, train_s), 1)
                validation_primary = np.concatenate((validation_c, validation_s), 1)
                test_primary = np.concatenate((test_c, test_s), 1)
                z_s_result = None
            else:
                train_primary, validation_primary, test_primary = train_c, validation_c, test_c
                z_s_result = evaluator.evaluate_sparse_code(
                    train_s, validation_s, test_s, train_meta, validation_meta, test_meta,
                    standardized(test_raw), reconstruction, manifest,
                )
            primary = evaluator.evaluate_sparse_code(
                train_primary, validation_primary, test_primary,
                train_meta, validation_meta, test_meta,
                standardized(test_raw), reconstruction, manifest,
            )
            primary["intent_relation_margin"] = evaluator.intent_relation_margin(
                test_primary, test_meta, manifest
            )
            primary["intent_retrieval"] = evaluator.intent_retrieval(
                test_primary, test_meta, manifest
            )
            run = {
                "method": method, "experiment": experiment, "seed": seed,
                "checkpoint": str(path.relative_to(ROOT)), "primary": primary,
                "zS": z_s_result, "total_l0": total_l0, "history": saved["history"],
            }
            report["runs"].append(run)
            rows.append(result_row(method, seed, primary, z_s_result, total_l0))
            OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
            del model, train_c, train_s, validation_c, validation_s, test_c, test_s
            torch.cuda.empty_cache()
    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    per_seed.to_csv(PER_SEED, index=False)
    summary.to_csv(SUMMARY, index=False)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--experiment", choices=EXPERIMENTS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--extraction-batch-size", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Pythia transfer requires CUDA")
    if args.extract:
        extract(args.extraction_batch_size)
    if args.smoke or args.train:
        train(
            smoke=args.smoke,
            experiments=tuple(args.experiment or EXPERIMENTS),
            seeds=tuple(args.seed or ((SEEDS[0],) if args.smoke else SEEDS)),
        )
    if args.evaluate:
        evaluate()
    if not (args.extract or args.smoke or args.train or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()
