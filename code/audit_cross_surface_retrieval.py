"""Frozen-encoder C-test cosine, confusion, and cross-surface retrieval audit."""

import csv
import json
import os
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "activations" / "gemma2_2b_layer_sweep_metadata.csv"
EMBEDDINGS = Path(os.environ.get("ENCODED_PATH", ROOT / "checkpoint" / "contrastive_encoder_layer8_encoded_all.npy"))
REPORT = Path(os.environ.get("AUDIT_REPORT_PATH", ROOT / "Report" / "cross_surface_retrieval_audit.json"))
TRAIN_S, HELD_S = {"declarative", "question", "paraphrase"}, {"formal", "structured"}
FAMILIES, SEED = ["declarative", "question", "paraphrase", "formal", "structured"], 20260825


def summary(x, pairs):
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    values = np.sum(x[a] * x[b], axis=1) / (np.linalg.norm(x[a], axis=1) * np.linalg.norm(x[b], axis=1))
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def retrieve(x, queries, bank):
    bank_x = x[[r["activation_row"] for r in bank]]
    bank_x /= np.linalg.norm(bank_x, axis=1, keepdims=True)
    ranks = []
    for row in queries:
        q = x[row["activation_row"]]; scores = bank_x @ (q / np.linalg.norm(q))
        order = np.argsort(-scores)
        ranks.append(next(i + 1 for i, index in enumerate(order) if bank[index]["fact_id"] == row["fact_id"]))
    ranks = np.array(ranks)
    return {"queries": len(queries), "bank_rows": len(bank), "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)), "R@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1 / ranks)), "rank_summary": {"median": float(np.median(ranks)), "p95": float(np.quantile(ranks, .95)), "max": int(ranks.max())}}


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    x = np.load(EMBEDDINGS); c_train = np.array([r["C_split"] == "C_train" for r in rows])
    test = [r for r in rows if r["C_split"] == "C_test"]
    facts = defaultdict(list)
    for row in test: facts[row["fact_id"]].append(row)
    pairs = {"train_S_train_S": [], "train_S_held_out_S": [], "held_out_S_held_out_S": []}
    for members in facts.values():
        train, held = [r for r in members if r["S_family"] in TRAIN_S], [r for r in members if r["S_family"] in HELD_S]
        pairs["train_S_train_S"] += [(a["activation_row"], b["activation_row"]) for a, b in combinations(train, 2)]
        pairs["train_S_held_out_S"] += [(a["activation_row"], b["activation_row"]) for a, b in product(train, held)]
        pairs["held_out_S_held_out_S"] += [(a["activation_row"], b["activation_row"]) for a, b in combinations(held, 2)]
    labels = np.array([r["S_family"] for r in rows]); scaler = StandardScaler().fit(x[c_train])
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(x[c_train]), labels[c_train])
    actual = np.array([r["S_family"] for r in test]); predicted = probe.predict(scaler.transform(x[[r["activation_row"] for r in test]]))
    train_bank, held_bank = [r for r in test if r["S_family"] in TRAIN_S], [r for r in test if r["S_family"] in HELD_S]
    report = {"encoder": f"{EMBEDDINGS}; frozen checkpoint only", "evaluation": "C_test subjects only", "S_train_families": sorted(TRAIN_S), "held_out_families": sorted(HELD_S), "same_fact_cosine_distributions": {name: summary(x, values) for name, values in pairs.items()}, "S_family_confusion_matrix": {"rows_actual": FAMILIES, "columns_predicted": FAMILIES, "counts": {a: {p: int(np.sum((actual == a) & (predicted == p))) for p in FAMILIES} for a in FAMILIES}}, "retrieval_held_out_S_to_train_S": retrieve(x, held_bank, train_bank), "retrieval_train_S_to_held_out_S": retrieve(x, train_bank, held_bank), "new_training": "none; diagnostic S-family probe only", "sae_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
