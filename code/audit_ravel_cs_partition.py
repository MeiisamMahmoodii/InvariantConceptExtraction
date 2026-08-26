"""Frozen RAVEL C/S partition diagnostics on unseen entities and test templates."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RAVEL_PARTITION_RUN", "ravel_cs_partition")
DATA = ROOT / "data" / os.environ.get("RAVEL_PARTITION_DIR", "ravel_partition_layer8")
CKPT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / f"{RUN}_audit.json"
SEED = 20260825


def rank(x):
    values = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2; p = values / values.sum()
    return {"participation_ratio": float(values.sum() ** 2 / (values ** 2).sum()), "entropy_effective_rank": float(np.exp(-(p * np.log(p + 1e-30)).sum()))}


def train_to_heldout_probe(x, rows, label):
    train = np.array([r["C_split"] == "train" and r["S_split"] in {"train", "val"} for r in rows])
    test = np.array([r["C_split"] == "test" and r["S_split"] == "test" for r in rows])
    y = np.array([r[label] for r in rows]); scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(x[train]), y[train])
    return {"accuracy": float(np.mean(model.predict(scaler.transform(x[test])) == y[test])), "chance": 1 / len(set(y[train])), "train_rows": int(train.sum()), "heldout_test_rows": int(test.sum())}


def heldout_template_probe(x, rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["C_split"] == "test" and row["S_split"] == "test": groups[row["template_id"]].append(index)
    rng = np.random.default_rng(SEED); train, test = [], []
    for indices in groups.values():
        if len(indices) < 4: continue
        indices = rng.permutation(indices); cut = max(1, int(.7 * len(indices))); train.extend(indices[:cut]); test.extend(indices[cut:])
    y = np.array([r["template_id"] for r in rows]); scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(x[train]), y[train])
    return {"accuracy": float(np.mean(model.predict(scaler.transform(x[test])) == y[test])), "chance": 1 / len(set(y[train])), "scope": "diagnostic probe fitted only within frozen C_test/test-template rows; it does not update the partition", "template_classes": len(set(y[train]))}


def retrieval(x, rows, target):
    indices = [i for i, r in enumerate(rows) if r["C_split"] == "test" and r["S_split"] == "test"]
    bank = torch.from_numpy(x[indices]).to("cuda"); ranks = []
    for start in range(0, len(indices), 128):
        query_indices = indices[start:start + 128]; scores = torch.from_numpy(x[query_indices]).to("cuda") @ bank.T
        for local, query_index in enumerate(query_indices):
            for bank_index, row_index in enumerate(indices):
                if row_index == query_index or (target == "template_id" and rows[row_index]["fact_id"] == rows[query_index]["fact_id"]): scores[local, bank_index] = -float("inf")
            order = scores[local].argsort(descending=True).cpu().numpy()
            ranks.append(next(position + 1 for position, bank_index in enumerate(order) if rows[indices[bank_index]][target] == rows[query_index][target]))
    ranks = np.array(ranks)
    return {"queries": len(indices), "target": target, "R@1": float(np.mean(ranks <= 1)), "R@5": float(np.mean(ranks <= 5)), "R@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1 / ranks))}


def main():
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    z_c, z_s = np.load(CKPT / f"{RUN}_c_all.npy"), np.load(CKPT / f"{RUN}_s_all.npy")
    heldout = np.array([r["C_split"] == "test" and r["S_split"] == "test" for r in rows])
    report = {"evaluation": "frozen partition; published C_test entities and published test templates only", "z_C": {"attribute_probe_train_to_heldout": train_to_heldout_probe(z_c, rows, "C_relation"), "heldout_template_identity_probe": heldout_template_probe(z_c, rows), "same_fact_cross_heldout_template_retrieval": retrieval(z_c, rows, "fact_id"), "effective_rank": rank(z_c[heldout])}, "z_S": {"attribute_probe_train_to_heldout": train_to_heldout_probe(z_s, rows, "C_relation"), "heldout_template_identity_probe": heldout_template_probe(z_s, rows), "same_template_different_fact_retrieval": retrieval(z_s, rows, "template_id"), "effective_rank": rank(z_s[heldout])}, "partition_retrained": False, "SAE_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
