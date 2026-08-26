"""Audit frozen contrastive embeddings on held-out subjects."""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "activations" / "gemma2_2b_layer_sweep_metadata.csv"
EMBEDDINGS = Path(os.environ.get("ENCODED_PATH", ROOT / "checkpoint" / "contrastive_encoder_layer8_encoded_all.npy"))
REPORT = Path(os.environ.get("AUDIT_REPORT_PATH", ROOT / "Report" / "contrastive_representation_diagnostic_report.json"))
SEED, N_PAIRS = 20260825, 10_000
FIELDS = ("S_family", "C_domain", "C_relation")


def probe_accuracy(x, y, train, test):
    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(x[train]), y[train])
    return float(model.score(scaler.transform(x[test]), y[test]))


def cosine_summary(x, pairs):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    values = np.sum(x[a] * x[b], axis=1) / (np.linalg.norm(x[a], axis=1) * np.linalg.norm(x[b], axis=1))
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def pairs_for_test(rows):
    rng, groups = random.Random(SEED), [defaultdict(list) for _ in range(4)]
    for row in rows:
        groups[0][row["fact_id"]].append(row)
        groups[1][row["C_relation"]].append(row)
        groups[2][(row["C_domain"], row["C_subject_id"])].append(row)
        groups[3][row["C_domain"]].append(row)
    out = {"A_same_fact_different_S": [], "B_different_fact_same_relation": [], "C_different_relation_same_subject": [], "D_different_domain": []}
    while len(out["A_same_fact_different_S"]) < N_PAIRS:
        a, b = rng.sample(groups[0][rng.choice(list(groups[0]))], 2)
        if a["S_family"] != b["S_family"]: out["A_same_fact_different_S"].append((a["activation_row"], b["activation_row"]))
    while len(out["B_different_fact_same_relation"]) < N_PAIRS:
        a, b = rng.sample(groups[1][rng.choice(list(groups[1]))], 2)
        if a["fact_id"] != b["fact_id"]: out["B_different_fact_same_relation"].append((a["activation_row"], b["activation_row"]))
    while len(out["C_different_relation_same_subject"]) < N_PAIRS:
        a, b = rng.sample(groups[2][rng.choice(list(groups[2]))], 2)
        if a["C_relation"] != b["C_relation"]: out["C_different_relation_same_subject"].append((a["activation_row"], b["activation_row"]))
    domains = list(groups[3])
    while len(out["D_different_domain"]) < N_PAIRS:
        a, b = rng.choice(groups[3][domains[0]]), rng.choice(groups[3][domains[1]])
        out["D_different_domain"].append((a["activation_row"], b["activation_row"]))
    return out


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    x = np.load(EMBEDDINGS)
    train = np.array([r["C_split"] == "C_train" for r in rows])
    test = np.array([r["C_split"] == "C_test" for r in rows])
    held_out_s = np.array([r["C_split"] == "C_test" and r["S_split"] == "S_test" for r in rows])
    s_train_s = np.array([r["C_split"] == "C_test" and r["S_split"] == "S_train" for r in rows])
    labels = {field: np.array([r[field] for r in rows]) for field in FIELDS}
    test_rows = [r for r in rows if r["C_split"] == "C_test"]
    report = {
        "encoder": f"{EMBEDDINGS}; frozen during audit",
        "diagnostic_probe_training": "C_train subjects; probes may use all five S labels, unlike the contrastive encoder",
        "evaluation": "C_test subjects", "held_out_surface_families": ["formal", "structured"],
        "diagnostic_linear_probes": {f: {"C_test_accuracy": probe_accuracy(x, labels[f], train, test), "chance": 1 / len(set(labels[f]))} for f in FIELDS},
        "S_family_accuracy_on_C_test_S_train_rows": probe_accuracy(x, labels["S_family"], train, s_train_s),
        "S_family_accuracy_on_C_test_held_out_S_rows": probe_accuracy(x, labels["S_family"], train, held_out_s),
        "pair_similarity_distributions_C_test": {name: cosine_summary(x, pairs) for name, pairs in pairs_for_test(test_rows).items()},
        "rows": {"C_train": int(train.sum()), "C_test": int(test.sum()), "C_test_S_train": int(s_train_s.sum()), "C_test_S_test": int(held_out_s.sum())},
        "sae_trained": False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
