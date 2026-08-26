"""Compare frozen raw layer-8 and contrastive C-test cross-surface retrieval."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "activations" / "gemma2_2b_layer_sweep_metadata.csv"
RAW = ROOT / "data" / "activations" / "gemma2_2b_layer8_mean" / "activations.npy"
CONTRASTIVE = ROOT / "checkpoint" / "contrastive_encoder_layer8_encoded_all.npy"
REPORT = ROOT / "Report" / "raw_vs_contrastive_cross_surface_retrieval.json"
TRAIN_S, HELD_S = {"declarative", "question", "paraphrase"}, {"formal", "structured"}


def retrieve(x, queries, bank):
    bank_x = x[[r["activation_row"] for r in bank]]
    bank_x /= np.linalg.norm(bank_x, axis=1, keepdims=True)
    ranks = []
    for row in queries:
        q = x[row["activation_row"]]; order = np.argsort(-(bank_x @ (q / np.linalg.norm(q))))
        ranks.append(next(i + 1 for i, index in enumerate(order) if bank[index]["fact_id"] == row["fact_id"]))
    ranks = np.array(ranks)
    return {"queries": len(queries), "bank_rows": len(bank), "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)), "R@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1 / ranks))}


def main():
    with METADATA.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    test = [r for r in rows if r["C_split"] == "C_test"]
    train_bank = [r for r in test if r["S_family"] in TRAIN_S]
    formal = [r for r in test if r["S_family"] == "formal"]
    structured = [r for r in test if r["S_family"] == "structured"]
    held_bank = formal + structured
    train_queries = train_bank
    runs = {"raw_gemma_layer8": np.load(RAW), "contrastive_layer8": np.load(CONTRASTIVE)}
    report = {"evaluation": "C_test subjects only", "bank_train_S_families": sorted(TRAIN_S), "held_out_families": sorted(HELD_S), "results": {}}
    for name, x in runs.items():
        report["results"][name] = {
            "formal_to_train_S": retrieve(x, formal, train_bank),
            "structured_to_train_S": retrieve(x, structured, train_bank),
            "all_held_out_S_to_train_S": retrieve(x, held_bank, train_bank),
            "train_S_to_all_held_out_S": retrieve(x, train_queries, held_bank),
        }
    report["new_training"] = "none"; report["sae_trained"] = False
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
