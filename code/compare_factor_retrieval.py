"""Frozen raw-versus-contrastive C-test factor retrieval."""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "activations" / "gemma2_2b_layer_sweep_metadata.csv"
RAW = ROOT / "data" / "activations" / "gemma2_2b_layer8_mean" / "activations.npy"
CONTRASTIVE = ROOT / "checkpoint" / "contrastive_encoder_layer8_encoded_all.npy"
REPORT = ROOT / "Report" / "raw_vs_contrastive_factor_retrieval.json"


def retrieve(x, rows, mode):
    x = x[[r["activation_row"] for r in rows]].copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    fact = np.array([r["fact_id"] for r in rows])
    family = np.array([r["S_family"] for r in rows])
    relation = np.array([r["C_relation"] for r in rows])
    subject = np.array([f'{r["C_domain"]}:{r["C_subject_id"]}' for r in rows])
    ranks = []
    for i, row in enumerate(rows):
        bank = (fact != fact[i]) & (family != family[i])
        if mode == "relation":
            relevant = bank & (relation == relation[i]) & (subject != subject[i])
        else:
            relevant = bank & (subject == subject[i]) & (relation != relation[i])
        order = np.argsort(-(x[bank] @ x[i]))
        relevant_ordered = relevant[bank][order]
        ranks.append(int(np.flatnonzero(relevant_ordered)[0]) + 1)
    ranks = np.array(ranks)
    return {"queries": len(rows), "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)), "MRR": float(np.mean(1 / ranks)), "rank_summary": {"median": float(np.median(ranks)), "p95": float(np.quantile(ranks, .95)), "max": int(ranks.max())}}


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    test = [row for row in rows if row["C_split"] == "C_test"]
    report = {"evaluation": "C_test only", "query_unit": "each C-test surface row", "bank_exclusions": ["same fact_id", "same S_family as query"], "factor_targets": {"relation": "same C_relation and different C_subject", "subject": "same C_subject and different C_relation"}, "results": {}}
    runs = {"raw_gemma_layer8": np.load(RAW), "contrastive_layer8": np.load(CONTRASTIVE)}
    if os.environ.get("PARTIAL_EMBEDDINGS_PATH"):
        runs["partial_contrastive_layer8"] = np.load(os.environ["PARTIAL_EMBEDDINGS_PATH"])
    for name, matrix in runs.items():
        report["results"][name] = {"same_relation_different_subject": retrieve(matrix, test, "relation"), "same_subject_different_relation": retrieve(matrix, test, "subject")}
    report["new_training"] = "none"; report["sae_trained"] = False
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
