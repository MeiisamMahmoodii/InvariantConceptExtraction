"""Train a 256-D C-dominant bottleneck with matched controlled pairs only."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ACT = Path(os.environ.get("ACTIVATION_DIR", ROOT / "data" / "activations"))
CKPT = ROOT / "checkpoint"
RUN_NAME = os.environ.get("RUN_NAME", "fact_bottleneck")
REPORT = ROOT / "Report" / f"{RUN_NAME}_training_report.json"
S_TRAIN = set(os.environ.get("TRAIN_FAMILIES", "declarative,question,paraphrase").split(","))
BALANCE_DOMAINS = os.environ.get("BALANCE_DOMAINS", "0") == "1"
ALL_FAMILIES = {"declarative", "question", "paraphrase", "formal", "structured", "indirect"}
SEED, EPOCHS, BATCH, DIM, TEMPERATURE, NEGATIVES = 20260825, 30, 256, 256, 0.07, 31
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Bottleneck(nn.Module):
    def __init__(self, input_dim):
        super().__init__(); self.linear = nn.Linear(input_dim, DIM)
    def forward(self, x): return F.normalize(self.linear(x), dim=-1)


def main():
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    train = [r for r in rows if r["C_split"] == "C_train" and r["S_family"] in S_TRAIN]
    domain_subject_counts = {}
    if BALANCE_DOMAINS:
        subjects_by_domain = defaultdict(set)
        for row in train:
            subjects_by_domain[row["C_domain"]].add(row["C_subject_id"])
        limit = min(len(subjects) for subjects in subjects_by_domain.values())
        selected = set()
        for domain, subjects in sorted(subjects_by_domain.items()):
            chosen = rng.choice(sorted(subjects), size=limit, replace=False)
            selected.update((domain, subject) for subject in chosen)
        train = [r for r in train if (r["C_domain"], r["C_subject_id"]) in selected]
        domain_subject_counts = {domain: limit for domain in sorted(subjects_by_domain)}
    by_fact, by_template = defaultdict(list), defaultdict(list)
    for row in train:
        by_fact[row["fact_id"]].append(row); by_template[(row["S_family"], row["S_variant"])].append(row)
    assert all(len(rows) > NEGATIVES for rows in by_template.values())
    x = torch.from_numpy(np.load(ACT / "gemma2_2b_layer8_mean" / "activations.npy")).to(DEVICE)
    model = Bottleneck(x.shape[1]).to(DEVICE); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(EPOCHS):
        total = 0.0; steps = 0; model.train()
        for indices in np.array_split(rng.permutation(len(train)), max(1, len(train) // BATCH)):
            anchors = [train[i] for i in indices]; positives, negatives = [], []
            for anchor in anchors:
                positives.append(rng.choice([r for r in by_fact[anchor["fact_id"]] if r["S_family"] != anchor["S_family"]]))
                pool = [r for r in by_template[(anchor["S_family"], anchor["S_variant"])] if r["fact_id"] != anchor["fact_id"]]
                negatives.append(rng.choice(pool, size=NEGATIVES, replace=False))
            a = torch.tensor([r["activation_row"] for r in anchors], device=DEVICE)
            p = torch.tensor([r["activation_row"] for r in positives], device=DEVICE)
            n = torch.tensor([[r["activation_row"] for r in group] for group in negatives], device=DEVICE)
            za, zp = model(x[a]), model(x[p]); zn = model(x[n].reshape(-1, x.shape[1])).reshape(len(a), NEGATIVES, DIM)
            logits = torch.cat([(za * zp).sum(-1, keepdim=True), torch.einsum("bd,bnd->bn", za, zn)], dim=1) / TEMPERATURE
            loss = F.cross_entropy(logits, torch.zeros(len(a), dtype=torch.long, device=DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item(); steps += 1
        history.append({"epoch": epoch + 1, "loss": total / steps}); print(f"epoch={epoch + 1}/{EPOCHS} loss={history[-1]['loss']:.4f}")
    CKPT.mkdir(exist_ok=True); model.eval()
    with torch.no_grad(): encoded = model(x).cpu().numpy()
    np.save(CKPT / f"{RUN_NAME}_layer8_encoded_all.npy", encoded)
    torch.save({"state_dict": model.state_dict(), "config": {"input_layer": 8, "output_dim": DIM, "positive": "same fact, different S_family", "negative": "different fact, same S_family and S_variant", "negative_count": NEGATIVES, "temperature": TEMPERATURE, "S_train_families": sorted(S_TRAIN), "epochs": EPOCHS, "seed": SEED}}, CKPT / f"{RUN_NAME}_layer8.pt")
    report = {"device": DEVICE, "training_rows": len(train), "training_facts": len(by_fact), "domain_balanced": BALANCE_DOMAINS, "training_subjects_per_domain": domain_subject_counts, "output_dim": DIM, "positive": "same fact, different S_family", "negative": "different fact, same S_family and S_variant", "negative_count": NEGATIVES, "S_train_families": sorted(S_TRAIN), "held_out_families": sorted(ALL_FAMILIES - S_TRAIN), "final_loss": history[-1]["loss"], "history": history, "sae_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__": main()
