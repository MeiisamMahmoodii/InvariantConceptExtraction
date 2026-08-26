"""Train a contrastive encoder on frozen layer-8 Gemma-2-2B activations.

Train rows: C_split == C_train AND S_family in {declarative, question, paraphrase}.
Positives: same fact_id from a different S_train family than the anchor.
Negatives: in-batch other rows (standard InfoNCE).
C hierarchy (domain/relation) is preserved because negatives include same-domain
and same-relation rows; we never force-push them.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "data" / "activations"
CKPT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / "contrastive_encoder_report.json"

LAYER = 8
S_TRAIN_FAMILIES = ("declarative", "question", "paraphrase")
SEED = 20260825
HIDDEN = 1024
OUT_DIM = 1024
DROPOUT = 0.1
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def load_train_rows():
    with (ACT / "gemma2_2b_layer_sweep_metadata.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    train = [row for row in rows if row["C_split"] == "C_train" and row["S_family"] in S_TRAIN_FAMILIES]
    matrix = np.load(ACT / f"gemma2_2b_layer{LAYER}_mean" / "activations.npy")
    by_fact = {}
    for row in train:
        by_fact.setdefault(row["fact_id"], []).append((int(row["activation_row"]), row["S_family"]))
    return rows, train, matrix, by_fact


def build_pairs(train_rows, by_fact, rng):
    """Pairs: (anchor_idx, positive_idx) — same fact, different S_train family."""
    pairs = []
    for members in by_fact.values():
        n = len(members)
        for i in range(n):
            for j in range(n):
                if i != j and members[i][1] != members[j][1]:
                    pairs.append((members[i][0], members[j][0]))
    rng.shuffle(pairs)
    return pairs


def info_nce(z, targets, temperature):
    logits = z @ z.T / temperature
    n = z.size(0)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, float("-inf"))
    return F.cross_entropy(logits, targets)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    rows, train_rows, matrix, by_fact = load_train_rows()
    pairs = build_pairs(train_rows, by_fact, rng)
    train_idx = np.array([int(row["activation_row"]) for row in train_rows], dtype=np.int64)
    assert all(row["C_split"] == "C_train" and row["S_family"] in S_TRAIN_FAMILIES for row in train_rows)
    assert not set(train_idx) & {int(row["activation_row"]) for row in rows if row["S_family"] in {"formal", "structured"}}

    pair_anchor = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=DEVICE)
    pair_positive = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=DEVICE)
    x_all = torch.from_numpy(matrix).to(DEVICE)

    in_dim = matrix.shape[1]
    encoder = Encoder(in_dim, HIDDEN, OUT_DIM, DROPOUT).to(DEVICE)
    optim = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    for epoch in range(EPOCHS):
        encoder.train()
        perm = torch.randperm(len(pairs), device=DEVICE)
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(pairs), BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            anchors, positives = pair_anchor[batch_idx], pair_positive[batch_idx]
            z_anchor, z_positive = encoder(x_all[anchors]), encoder(x_all[positives])
            targets = torch.arange(len(batch_idx), device=DEVICE)
            logits = z_anchor @ z_positive.T / TEMPERATURE
            loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            steps += 1
        epoch_loss /= max(steps, 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch={epoch + 1}/{EPOCHS} loss={epoch_loss:.4f}")

    CKPT.mkdir(exist_ok=True)
    encoder.eval()
    with torch.no_grad():
        z_full = encoder(torch.from_numpy(matrix).to(DEVICE)).cpu().numpy()
    np.save(CKPT / f"contrastive_encoder_layer{LAYER}_encoded_all.npy", z_full)

    torch.save({
        "state_dict": encoder.state_dict(),
        "config": {"layer": LAYER, "hidden": HIDDEN, "out_dim": OUT_DIM, "dropout": DROPOUT,
                   "S_train_families": list(S_TRAIN_FAMILIES), "epochs": EPOCHS,
                   "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "temperature": TEMPERATURE, "seed": SEED},
    }, CKPT / f"contrastive_encoder_layer{LAYER}.pt")

    report = {
        "layer": LAYER,
        "input_dim": int(in_dim),
        "output_dim": OUT_DIM,
        "S_train_families": list(S_TRAIN_FAMILIES),
        "S_test_families_held_out": ["formal", "structured"],
        "training_rows": int(len(train_idx)),
        "training_pairs": int(len(pairs)),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "device": DEVICE,
        "final_loss": history[-1]["loss"],
        "loss_history": history,
        "negative_strategy": "in-batch InfoNCE; negatives include same-domain and same-relation rows so reusable C hierarchy is not destroyed",
        "sae_trained": False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
