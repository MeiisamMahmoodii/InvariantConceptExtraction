"""Train one hierarchy-aware contrastive encoder on frozen layer-8 activations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations"
CKPT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / "partial_contrastive_encoder_report.json"
LAYER, SEED, EPOCHS, BATCH, HIDDEN, OUT = 8, 20260825, 30, 256, 1024, 1024
LR, WEIGHT_DECAY, TEMPERATURE, WEAK_WEIGHT = 1e-3, 1e-4, 0.07, 0.5
S_TRAIN = {"declarative", "question", "paraphrase"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, HIDDEN), nn.GELU(), nn.Dropout(0.1), nn.Linear(HIDDEN, OUT))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def masked_infonce(anchor, candidate, anchor_meta, candidate_meta):
    logits = anchor @ candidate.T / TEMPERATURE
    domain_a = anchor_meta[:, 0:1]; domain_c = candidate_meta[:, 0].unsqueeze(0)
    allowed = domain_a != domain_c
    allowed.fill_diagonal_(True)  # diagonal is the selected strong/weak positive.
    logits = logits.masked_fill(~allowed, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(anchor), device=DEVICE))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    for row in rows: row["activation_row"] = int(row["activation_row"])
    train = [r for r in rows if r["C_split"] == "C_train" and r["S_family"] in S_TRAIN]
    assert all(r["S_split"] == "S_train" for r in train)
    by_fact = defaultdict(list); facts = {}
    for row in train:
        by_fact[row["fact_id"]].append(row)
        facts[row["fact_id"]] = (row["C_domain"], row["C_relation"], f'{row["C_domain"]}:{row["C_subject_id"]}')
    fact_ids = list(facts); relation_pool, subject_pool = defaultdict(list), defaultdict(list)
    for fact_id, (_, relation, subject) in facts.items(): relation_pool[relation].append(fact_id); subject_pool[subject].append(fact_id)
    for fact_id in fact_ids:
        assert len(relation_pool[facts[fact_id][1]]) > 1 and len(subject_pool[facts[fact_id][2]]) > 1
    matrix = np.load(ACT / f"gemma2_2b_layer{LAYER}_mean" / "activations.npy")
    x = torch.from_numpy(matrix).to(DEVICE)
    encoder = Encoder(matrix.shape[1]).to(DEVICE); optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train)); total = 0.0; steps = 0; encoder.train()
        for start in range(0, len(train), BATCH):
            anchors = [train[i] for i in order[start:start + BATCH]]
            strong, weak = [], []
            for anchor in anchors:
                options = [r for r in by_fact[anchor["fact_id"]] if r["S_family"] != anchor["S_family"]]
                strong.append(rng.choice(options))
                domain, relation, subject = facts[anchor["fact_id"]]
                pool = relation_pool[relation] if rng.integers(2) == 0 else subject_pool[subject]
                weak_fact = rng.choice([item for item in pool if item != anchor["fact_id"]])
                weak.append(rng.choice(by_fact[weak_fact]))
            a_idx = torch.tensor([r["activation_row"] for r in anchors], device=DEVICE)
            s_idx = torch.tensor([r["activation_row"] for r in strong], device=DEVICE)
            w_idx = torch.tensor([r["activation_row"] for r in weak], device=DEVICE)
            a_meta = torch.tensor([[0 if facts[r["fact_id"]][0] == "geography" else 1] for r in anchors], device=DEVICE)
            s_meta = torch.tensor([[0 if facts[r["fact_id"]][0] == "geography" else 1] for r in strong], device=DEVICE)
            w_meta = torch.tensor([[0 if facts[r["fact_id"]][0] == "geography" else 1] for r in weak], device=DEVICE)
            z_a, z_s, z_w = encoder(x[a_idx]), encoder(x[s_idx]), encoder(x[w_idx])
            loss = masked_infonce(z_a, z_s, a_meta, s_meta) + WEAK_WEIGHT * masked_infonce(z_a, z_w, a_meta, w_meta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); steps += 1
        history.append({"epoch": epoch + 1, "loss": total / steps})
        print(f"epoch={epoch + 1}/{EPOCHS} loss={history[-1]['loss']:.4f}")
    CKPT.mkdir(exist_ok=True); encoder.eval()
    with torch.no_grad(): encoded = encoder(x).cpu().numpy()
    np.save(CKPT / "partial_contrastive_encoder_layer8_encoded_all.npy", encoded)
    torch.save({"state_dict": encoder.state_dict(), "config": {"layer": LAYER, "strong_positive": "same fact, different S family", "weak_positive": ["same relation, different subject", "same subject, different relation"], "neutral": "same domain only", "negative": "different domain / no shared C factor", "weak_loss_weight": WEAK_WEIGHT, "epochs": EPOCHS, "batch_size": BATCH, "temperature": TEMPERATURE, "seed": SEED}}, CKPT / "partial_contrastive_encoder_layer8.pt")
    report = {"layer": LAYER, "device": DEVICE, "training_rows": len(train), "training_facts": len(fact_ids), "S_train_families": sorted(S_TRAIN), "held_out_families": ["formal", "structured"], "strong_positive": "same fact, different S family", "weak_positive": ["same relation, different subject", "same subject, different relation"], "neutral": "same domain only", "negative": "no shared C factor", "weak_loss_weight": WEAK_WEIGHT, "final_loss": history[-1]["loss"], "history": history, "sae_trained": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__": main()
