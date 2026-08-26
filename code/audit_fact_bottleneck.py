"""Four required frozen audits for the 256-D fact bottleneck."""

import csv
import json
import random
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ACT = Path(os.environ.get("ACTIVATION_DIR", ROOT / "data" / "activations"))
METADATA = ACT / "gemma2_2b_layer_sweep_metadata.csv"
RUN_NAME = os.environ.get("RUN_NAME", "fact_bottleneck")
EMBEDDINGS = Path(os.environ.get("EMBEDDINGS_PATH", ROOT / "checkpoint" / f"{RUN_NAME}_layer8_encoded_all.npy"))
REPORT = ROOT / "Report" / f"{RUN_NAME}_audit.json"
TRAIN_S = set(os.environ.get("TRAIN_FAMILIES", "declarative,question,paraphrase").split(","))
HELD_S, SEED, N = {"declarative", "question", "paraphrase", "formal", "structured", "indirect"} - TRAIN_S, 20260825, 10_000


def cosine_stats(x, pairs):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    values = np.sum(x[a] * x[b], 1) / (np.linalg.norm(x[a], axis=1) * np.linalg.norm(x[b], axis=1))
    return values, {"n": len(values), "mean": float(values.mean()), "std": float(values.std()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def retrieve(x, queries, bank):
    bank_x = x[[r["activation_row"] for r in bank]]; bank_x /= np.linalg.norm(bank_x, axis=1, keepdims=True); ranks = []
    for row in queries:
        q = x[row["activation_row"]] / np.linalg.norm(x[row["activation_row"]]); order = np.argsort(-(bank_x @ q))
        ranks.append(next(i + 1 for i, index in enumerate(order) if bank[index]["fact_id"] == row["fact_id"]))
    ranks = np.array(ranks)
    return {"queries": len(queries), "bank_rows": len(bank), "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)), "R@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1 / ranks))}


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for r in rows: r["activation_row"] = int(r["activation_row"])
    x = np.load(EMBEDDINGS); train = np.array([r["C_split"] == "C_train" for r in rows]); test = [r for r in rows if r["C_split"] == "C_test"]
    labels = np.array([r["S_family"] for r in rows]); scaler = StandardScaler().fit(x[train]); probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(x[train]), labels[train])
    test_indices = np.array([r["activation_row"] for r in test]); pred = probe.predict(scaler.transform(x[test_indices])); actual = labels[test_indices]
    by_fact, by_template = defaultdict(list), defaultdict(list)
    for r in test: by_fact[r["fact_id"]].append(r); by_template[(r["S_family"], r["S_variant"])].append(r)
    rng = random.Random(SEED); positive, negative = [], []
    while len(positive) < N:
        a, b = rng.sample(by_fact[rng.choice(list(by_fact))], 2)
        if a["S_family"] != b["S_family"]: positive.append((a["activation_row"], b["activation_row"]))
    while len(negative) < N:
        a, b = rng.sample(by_template[rng.choice(list(by_template))], 2)
        if a["fact_id"] != b["fact_id"]: negative.append((a["activation_row"], b["activation_row"]))
    pos_values, pos_stats = cosine_stats(x, positive); neg_values, neg_stats = cosine_stats(x, negative)
    test_x = x[test_indices]; centered = test_x - test_x.mean(0, keepdims=True); eig = np.linalg.svd(centered, compute_uv=False) ** 2
    pr = float(eig.sum() ** 2 / np.square(eig).sum()); erank = float(np.exp(-np.sum((eig / eig.sum()) * np.log(eig / eig.sum() + 1e-30))))
    norms = np.linalg.norm(test_x, axis=1)
    train_bank, held = [r for r in test if r["S_family"] in TRAIN_S], [r for r in test if r["S_family"] in HELD_S]
    report = {"encoder": f"{RUN_NAME}_layer8 frozen", "evaluation": "C_test only", "S_train_families": sorted(TRAIN_S), "held_out_families": sorted(HELD_S), "S_family_probe": {"overall_accuracy": float(np.mean(pred == actual)), "S_train_rows_accuracy": float(np.mean(pred[[r["S_family"] in TRAIN_S for r in test]] == actual[[r["S_family"] in TRAIN_S for r in test]])), "held_out_family_accuracy": float(np.mean(pred[[r["S_family"] in HELD_S for r in test]] == actual[[r["S_family"] in HELD_S for r in test]])), "chance": 1 / len(set(labels))}, "unseen_S_fact_retrieval": {"held_out_S_to_train_S": retrieve(x, held, train_bank), "train_S_to_held_out_S": retrieve(x, train_bank, held)}, "matched_pair_geometry": {"positive_same_fact_different_S": pos_stats, "negative_different_fact_same_template_variant": neg_stats, "positive_cosine_exceeds_negative_cosine": float(np.mean(pos_values > neg_values))}, "collapse": {"dimensions": int(test_x.shape[1]), "participation_ratio": pr, "entropy_effective_rank": erank, "norm": {"mean": float(norms.mean()), "std": float(norms.std()), "p05": float(np.quantile(norms, .05)), "median": float(np.median(norms)), "p95": float(np.quantile(norms, .95))}}, "new_training": "none", "sae_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
