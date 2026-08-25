"""Analyze frozen activations with pair distributions and diagnostic linear probes only."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations"
REPORT = ROOT / "Report" / "representation_diagnostic_report.json"
SEED = 20260825
N_PAIRS = 10_000


def cosine(matrix, pairs):
    left, right = np.array([pair[0] for pair in pairs]), np.array([pair[1] for pair in pairs])
    a, b = matrix[left], matrix[right]
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def summary(values):
    return {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)), "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)), "median": float(np.median(values)), "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95))}


def sample_pairs(rng, by_fact, by_relation, by_subject, by_domain):
    facts = list(by_fact)
    relations, subjects, domains = list(by_relation), list(by_subject), list(by_domain)
    pairs = {"A_same_fact_different_S": [], "B_different_fact_same_relation": [], "C_different_relation_same_subject": [], "D_different_domain": []}
    while len(pairs["A_same_fact_different_S"]) < N_PAIRS:
        group = by_fact[rng.choice(facts)]
        left, right = rng.sample(group, 2)
        if left["S_family"] != right["S_family"]:
            pairs["A_same_fact_different_S"].append((left["activation_row"], right["activation_row"]))
    while len(pairs["B_different_fact_same_relation"]) < N_PAIRS:
        group = by_relation[rng.choice(relations)]
        left, right = rng.sample(group, 2)
        if left["fact_id"] != right["fact_id"]:
            pairs["B_different_fact_same_relation"].append((left["activation_row"], right["activation_row"]))
    while len(pairs["C_different_relation_same_subject"]) < N_PAIRS:
        group = by_subject[rng.choice(subjects)]
        left, right = rng.sample(group, 2)
        if left["C_relation"] != right["C_relation"]:
            pairs["C_different_relation_same_subject"].append((left["activation_row"], right["activation_row"]))
    while len(pairs["D_different_domain"]) < N_PAIRS:
        left, right = rng.choice(by_domain[domains[0]]), rng.choice(by_domain[domains[1]])
        pairs["D_different_domain"].append((left["activation_row"], right["activation_row"]))
    return pairs


def fit_probe(matrix, labels, train):
    scaler = StandardScaler().fit(matrix[train])
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(matrix[train]), labels[train])
    return scaler, probe


def project_out(matrix, fitted_probes):
    weights = np.vstack([probe.coef_ / scaler.scale_ for scaler, probe in fitted_probes])
    basis, _ = np.linalg.qr(weights.T)
    rank = np.linalg.matrix_rank(weights)
    basis = basis[:, :rank]
    return matrix - (matrix @ basis) @ basis.T, int(rank)


def probe_accuracy(matrix, labels, train, test):
    scaler, probe = fit_probe(matrix, labels, train)
    return float(probe.score(scaler.transform(matrix[test]), labels[test]))


def main():
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["activation_row"] = int(row["activation_row"])
    by_fact, by_relation, by_subject, by_domain = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_fact[row["fact_id"]].append(row); by_relation[row["C_relation"]].append(row); by_subject[(row["C_domain"], row["C_subject_id"])].append(row); by_domain[row["C_domain"]].append(row)
    pairs = sample_pairs(random.Random(SEED), by_fact, by_relation, by_subject, by_domain)
    train = np.array([row["C_split"] == "C_train" for row in rows])
    test = np.array([row["C_split"] == "C_test" for row in rows])
    layers = {}
    for layer in (5, 8, 13, 21):
        matrix = np.load(ACT / f"gemma2_2b_layer{layer}_mean" / "activations.npy")
        labels_by_field = {field: np.array([row[field] for row in rows]) for field in ("S_family", "C_domain", "C_relation")}
        fitted = {field: fit_probe(matrix, labels, train) for field, labels in labels_by_field.items()}
        probes = {}
        for field in ("S_family", "C_domain", "C_relation"):
            scaler, probe = fitted[field]
            probes[field] = {"test_accuracy": float(probe.score(scaler.transform(matrix[test]), labels_by_field[field][test])), "chance": 1 / len(set(labels_by_field[field])), "train_rows": int(train.sum()), "test_rows": int(test.sum())}
        s_removed, s_rank = project_out(matrix, [fitted["S_family"]])
        c_removed, c_rank = project_out(matrix, [fitted["C_domain"], fitted["C_relation"]])
        after_s_removed = {field: probe_accuracy(s_removed, labels_by_field[field], train, test) for field in ("C_domain", "C_relation")}
        after_c_removed = probe_accuracy(c_removed, labels_by_field["S_family"], train, test)
        layers[str(layer)] = {"activation_shape": list(matrix.shape), "pair_similarity_distributions": {name: summary(cosine(matrix, values)) for name, values in pairs.items()}, "diagnostic_linear_probes": probes, "linear_subspace_removal": {"S_family_subspace_rank": s_rank, "C_subspace_rank": c_rank, "C_test_accuracy_after_removing_S_subspace": after_s_removed, "S_family_test_accuracy_after_removing_C_subspace": after_c_removed}}
    report = {"layers": layers, "split": "C_train versus C_test; subject-disjoint by construction", "training_performed": "diagnostic logistic-regression probes only; no InfoNCE or SAE"}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
