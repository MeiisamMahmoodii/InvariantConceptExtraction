"""Train matched L1 SAEs on raw layer-8 and frozen C-bottleneck representations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations_three_domain_natural_rewrite"
CODES = ROOT / "checkpoint" / "fact_bottleneck_three_domain_natural_rewrite_balanced_layer8_encoded_all.npy"
OUT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / "matched_sae_training_report.json"
SEED, EPOCHS, BATCH, EXPANSION, L1 = 20260825, 30, 256, 4, 1e-3
FAMILIES = {"declarative", "question", "paraphrase", "formal", "structured"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self, width):
        super().__init__(); self.encoder = nn.Linear(width, width * EXPANSION); self.decoder = nn.Linear(width * EXPANSION, width, bias=False); self.bias = nn.Parameter(torch.zeros(width))
    def forward(self, x):
        z = F.relu(self.encoder(x)); return z, self.decoder(z) + self.bias
    def normalize_decoder(self):
        with torch.no_grad(): self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


def selected_rows(rows):
    rng = np.random.default_rng(SEED); candidates = [row for row in rows if row["C_split"] == "C_train" and row["S_family"] in FAMILIES]
    subjects = defaultdict(set)
    for row in candidates: subjects[row["C_domain"]].add(row["C_subject_id"])
    limit = min(len(values) for values in subjects.values()); chosen = set()
    for domain, values in sorted(subjects.items()): chosen.update((domain, subject) for subject in rng.choice(sorted(values), size=limit, replace=False))
    return [row for row in candidates if (row["C_domain"], row["C_subject_id"]) in chosen], {domain: limit for domain in sorted(subjects)}


def train(name, matrix, indices):
    torch.manual_seed(SEED)
    raw = matrix[indices].astype(np.float32); mean, std = raw.mean(0), raw.std(0).clip(1e-6)
    x = torch.from_numpy((raw - mean) / std).to(DEVICE); sae = SAE(x.shape[1]).to(DEVICE); optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-4); history = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x), device=DEVICE); total_recon = total_l1 = 0.0; steps = 0; sae.train()
        for batch in order.split(BATCH):
            z, reconstruction = sae(x[batch]); recon = F.mse_loss(reconstruction, x[batch]); sparse = z.mean(); loss = recon + L1 * sparse
            optimizer.zero_grad(); loss.backward(); optimizer.step(); sae.normalize_decoder(); total_recon += recon.item(); total_l1 += sparse.item(); steps += 1
        history.append({"epoch": epoch + 1, "reconstruction_mse": total_recon / steps, "mean_activation": total_l1 / steps}); print(f"{name} epoch={epoch + 1}/{EPOCHS} mse={history[-1]['reconstruction_mse']:.5f} mean_activation={history[-1]['mean_activation']:.5f}")
    path = OUT / f"matched_sae_{name}.pt"
    torch.save({"state_dict": sae.state_dict(), "input_mean": mean, "input_std": std, "config": {"input_width": x.shape[1], "feature_width": x.shape[1] * EXPANSION, "expansion_factor": EXPANSION, "l1_coefficient": L1, "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED, "standardization": "C_train per-dimension"}, "history": history}, path)
    return {"checkpoint": str(path.relative_to(ROOT)), "input_width": x.shape[1], "feature_width": x.shape[1] * EXPANSION, "final_reconstruction_mse": history[-1]["reconstruction_mse"], "final_mean_activation": history[-1]["mean_activation"], "history": history}


def main():
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    train_rows, domain_subjects = selected_rows(rows); indices = np.array([row["activation_row"] for row in train_rows])
    raw = np.load(ACT / "gemma2_2b_layer8_mean" / "activations.npy"); c_block = np.load(CODES)
    report = {"device": DEVICE, "training_rows": len(train_rows), "training_facts": len({row["fact_id"] for row in train_rows}), "training_subjects_per_domain": domain_subjects, "surface_families": sorted(FAMILIES), "held_out_surface_family": "indirect", "same_rows_for_both_saes": True, "method": "ReLU L1 SAE with unit-norm decoder columns", "expansion_factor": EXPANSION, "l1_coefficient": L1, "epochs": EPOCHS, "raw_gemma_layer8": train("raw_gemma_layer8", raw, indices), "c_bottleneck": train("c_bottleneck", c_block, indices), "contrastive_retrained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps({key: value for key, value in report.items() if key not in {"raw_gemma_layer8", "c_bottleneck"}}, indent=2))


if __name__ == "__main__": main()
