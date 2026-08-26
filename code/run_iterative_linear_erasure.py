"""Iteratively erase linear S/C probe subspaces from frozen activations; no encoder or SAE training."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations"
REPORT = ROOT / "Report" / "iterative_linear_erasure_report.json"
LAYERS = (5, 8, 13, 21)
SEED = 20260825
MAX_RANK = 64
NEAR_CHANCE_MARGIN = 0.02
MATERIAL_DROP = 0.005
MIN_ITERATIONS = 2
FIELDS = ("S_family", "C_domain", "C_relation")


def fit_probe(matrix, labels, train):
    scaler = StandardScaler().fit(matrix[train])
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(matrix[train]), labels[train])
    return scaler, probe


def accuracy(matrix, labels, train, test):
    scaler, probe = fit_probe(matrix, labels, train)
    return float(probe.score(scaler.transform(matrix[test]), labels[test]))


def learned_basis(matrix, labels, train):
    scaler, probe = fit_probe(matrix, labels, train)
    weights = probe.coef_ / scaler.scale_
    _, singular_values, right = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-6))
    return right[:rank].T, rank


def remove(matrix, basis):
    return matrix - (matrix @ basis) @ basis.T


def metrics(matrix, labels, train, test):
    return {field: accuracy(matrix, labels[field], train, test) for field in FIELDS}


def iterative_erasure(matrix, target, labels, train, test):
    residual = matrix.copy()
    history = [{"iteration": 0, "new_rank": 0, "cumulative_rank": 0, **metrics(residual, labels, train, test)}]
    chance = 1 / len(set(labels[target]))
    for iteration in range(1, MAX_RANK + 1):
        basis, new_rank = learned_basis(residual, labels[target], train)
        remaining = MAX_RANK - history[-1]["cumulative_rank"]
        if new_rank > remaining:
            basis, new_rank = basis[:, :remaining], remaining
        residual = remove(residual, basis)
        row = {"iteration": iteration, "new_rank": new_rank, "cumulative_rank": history[-1]["cumulative_rank"] + new_rank, **metrics(residual, labels, train, test)}
        history.append(row)
        drop = history[-2][target] - row[target]
        if row[target] <= chance + NEAR_CHANCE_MARGIN:
            stop = "near_chance"
            break
        if iteration >= MIN_ITERATIONS and drop < MATERIAL_DROP:
            stop = "no_material_decrease"
            break
        if row["cumulative_rank"] >= MAX_RANK:
            stop = "max_rank"
            break
    else:
        stop = "max_iterations"
    return {"target": target, "chance": chance, "stop": stop, "history": history}


def main():
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    labels = {field: np.array([row[field] for row in rows]) for field in FIELDS}
    train = np.array([row["C_split"] == "C_train" for row in rows])
    test = np.array([row["C_split"] == "C_test" for row in rows])
    layers = {}
    for layer in LAYERS:
        matrix = np.load(ACT / f"gemma2_2b_layer{layer}_mean" / "activations.npy")
        own = {}
        for target in FIELDS:
            basis, rank = learned_basis(matrix, labels[target], train)
            own[target] = {"rank": rank, "test_accuracy_before": accuracy(matrix, labels[target], train, test), "test_accuracy_after_own_subspace_removal": accuracy(remove(matrix, basis), labels[target], train, test)}
        layers[str(layer)] = {"own_subspace_removal": own, "iterative_S_family_erasure": iterative_erasure(matrix, "S_family", labels, train, test), "iterative_C_relation_erasure": iterative_erasure(matrix, "C_relation", labels, train, test)}
        print(f"completed_layer={layer}")
    report = {"configuration": {"layers": LAYERS, "max_cumulative_rank": MAX_RANK, "near_chance_margin": NEAR_CHANCE_MARGIN, "material_drop": MATERIAL_DROP, "min_iterations": MIN_ITERATIONS, "projection_learning_split": "C_train only", "evaluation_split": "C_test only"}, "layers": layers, "training_performed": "diagnostic logistic-regression probes only; no InfoNCE or SAE"}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
