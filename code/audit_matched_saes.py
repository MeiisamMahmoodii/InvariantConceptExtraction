"""Compare held-out reconstruction and C/S selectivity of matched SAEs."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations_three_domain_natural_rewrite"
CODES = ROOT / "checkpoint" / "fact_bottleneck_three_domain_natural_rewrite_balanced_layer8_encoded_all.npy"
OUT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / "matched_sae_audit.json"
FEATURES = ROOT / "data" / "sae_features"
K, MIN_ACTIVE, DEVICE = 50, 10, "cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self, width, features):
        super().__init__(); self.encoder = nn.Linear(width, features); self.decoder = nn.Linear(features, width, bias=False); self.bias = nn.Parameter(torch.zeros(width))
    def forward(self, x):
        z = F.relu(self.encoder(x)); return z, self.decoder(z) + self.bias


def purity(labels): return max(Counter(labels).values()) / len(labels)


def audit(name, matrix, rows, test_indices):
    saved = torch.load(OUT / f"matched_sae_{name}.pt", map_location=DEVICE, weights_only=False); config = saved["config"]
    sae = SAE(config["input_width"], config["feature_width"]).to(DEVICE); sae.load_state_dict(saved["state_dict"]); sae.eval()
    x = ((matrix[test_indices].astype(np.float32) - saved["input_mean"]) / saved["input_std"])
    chunks, mse = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(x[start:start + 256]).to(DEVICE); z, reconstruction = sae(batch); chunks.append(z.cpu().numpy()); mse.append(F.mse_loss(reconstruction, batch, reduction="sum").item())
    z = np.concatenate(chunks); selected_rows = [rows[index] for index in test_indices]; details = []
    for feature in range(z.shape[1]):
        active = np.flatnonzero(z[:, feature] > 0)
        if len(active) < MIN_ACTIVE: continue
        top = active[np.argsort(-z[active, feature])[:K]]
        relation = purity([selected_rows[i]["C_relation"] for i in top]); family = purity([selected_rows[i]["S_family"] for i in top]); domain = purity([selected_rows[i]["C_domain"] for i in top])
        details.append({"feature_id": feature, "active_examples": len(active), "top_k": len(top), "C_relation_topk_purity": relation, "C_domain_topk_purity": domain, "S_family_topk_purity": family, "top_relation": Counter(selected_rows[i]["C_relation"] for i in top).most_common(1)[0][0], "top_S_family": Counter(selected_rows[i]["S_family"] for i in top).most_common(1)[0][0]})
    FEATURES.mkdir(exist_ok=True)
    with (FEATURES / f"{name}_feature_selectivity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    rel = np.array([row["C_relation_topk_purity"] for row in details]); fam = np.array([row["S_family_topk_purity"] for row in details])
    return {"input_width": config["input_width"], "feature_width": config["feature_width"], "held_out_examples": len(x), "standardized_reconstruction_mse": float(sum(mse) / x.size), "mean_L0": float((z > 0).sum(1).mean()), "active_feature_count": len(details), "mean_top50_C_relation_purity": float(rel.mean()), "mean_top50_C_domain_purity": float(np.mean([row["C_domain_topk_purity"] for row in details])), "mean_top50_S_family_purity": float(fam.mean()), "C_relation_selective_features": int(np.sum((rel >= .8) & (fam <= .5))), "S_family_selective_features": int(np.sum((fam >= .8) & (rel <= .5))), "feature_details": str((FEATURES / f"{name}_feature_selectivity.csv").relative_to(ROOT))}


def main():
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    test_indices = np.array([index for index, row in enumerate(rows) if row["C_split"] == "C_test"])
    raw = np.load(ACT / "gemma2_2b_layer8_mean" / "activations.npy"); c_block = np.load(CODES)
    report = {"evaluation": "C_test subjects only; includes all six surface families", "top_activation_set_size": K, "minimum_feature_activations": MIN_ACTIVE, "feature_selectivity_rule": "C-relation or S-family majority purity among a feature's top active examples", "raw_gemma_layer8": audit("raw_gemma_layer8", raw, rows, test_indices), "c_bottleneck": audit("c_bottleneck", c_block, rows, test_indices), "new_training": "none"}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
