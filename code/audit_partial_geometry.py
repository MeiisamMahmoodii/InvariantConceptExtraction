"""Collapse and pair-geometry audit for the frozen partial-C encoder."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "activations" / "gemma2_2b_layer_sweep_metadata.csv"
EMBEDDINGS = ROOT / "checkpoint" / "partial_contrastive_encoder_layer8_encoded_all.npy"
REPORT = ROOT / "Report" / "partial_contrastive_geometry_audit.json"
SEED, N = 20260825, 10_000


def stats(x, pairs):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    values = np.sum(x[a] * x[b], axis=1) / (np.linalg.norm(x[a], axis=1) * np.linalg.norm(x[b], axis=1))
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    rows = [r for r in rows if r["C_split"] == "C_test"]
    x = np.load(EMBEDDINGS); rng = random.Random(SEED)
    by_fact, by_relation, by_subject, by_domain = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_fact[r["fact_id"]].append(r); by_relation[r["C_relation"]].append(r)
        by_subject[(r["C_domain"], r["C_subject_id"])].append(r); by_domain[r["C_domain"]].append(r)
    relations, subjects, domains = list(by_relation), list(by_subject), list(by_domain)
    pairs = {"same_exact_fact_different_S": [], "same_relation_different_subject": [], "same_subject_different_relation": [], "same_domain_only": [], "no_shared_C_factor": []}
    while len(pairs["same_exact_fact_different_S"]) < N:
        a, b = rng.sample(by_fact[rng.choice(list(by_fact))], 2)
        if a["S_family"] != b["S_family"]: pairs["same_exact_fact_different_S"].append((a["activation_row"], b["activation_row"]))
    while len(pairs["same_relation_different_subject"]) < N:
        a, b = rng.sample(by_relation[rng.choice(relations)], 2)
        if a["C_subject_id"] != b["C_subject_id"]: pairs["same_relation_different_subject"].append((a["activation_row"], b["activation_row"]))
    while len(pairs["same_subject_different_relation"]) < N:
        a, b = rng.sample(by_subject[rng.choice(subjects)], 2)
        if a["C_relation"] != b["C_relation"]: pairs["same_subject_different_relation"].append((a["activation_row"], b["activation_row"]))
    while len(pairs["same_domain_only"]) < N:
        group = by_domain[rng.choice(domains)]; a, b = rng.sample(group, 2)
        if a["C_subject_id"] != b["C_subject_id"] and a["C_relation"] != b["C_relation"]: pairs["same_domain_only"].append((a["activation_row"], b["activation_row"]))
    while len(pairs["no_shared_C_factor"]) < N:
        a, b = rng.choice(by_domain[domains[0]]), rng.choice(by_domain[domains[1]])
        pairs["no_shared_C_factor"].append((a["activation_row"], b["activation_row"]))
    test_x = x[[r["activation_row"] for r in rows]]; centered = test_x - test_x.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False); values = singular ** 2
    pr = float(values.sum() ** 2 / np.square(values).sum())
    probability = values / values.sum(); entropy_rank = float(np.exp(-np.sum(probability * np.log(probability + 1e-30))))
    norms = np.linalg.norm(test_x, axis=1)
    report = {"encoder": "partial_contrastive_encoder_layer8 frozen", "evaluation": "C_test only", "collapse": {"dimensions": int(test_x.shape[1]), "participation_ratio": pr, "entropy_effective_rank": entropy_rank, "norm": {"mean": float(norms.mean()), "std": float(norms.std()), "min": float(norms.min()), "p05": float(np.quantile(norms, .05)), "median": float(np.median(norms)), "p95": float(np.quantile(norms, .95)), "max": float(norms.max())}, "random_unrelated_mean_cosine": stats(x, pairs["no_shared_C_factor"])["mean"]}, "cosine_distributions": {name: stats(x, value) for name, value in pairs.items()}, "new_training": "none", "sae_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
