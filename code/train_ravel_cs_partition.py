"""Train the existing non-adversarial 128+128 partition on frozen RAVEL activations."""

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
RUN = os.environ.get("RAVEL_PARTITION_RUN", "ravel_cs_partition")
DATA = ROOT / "data" / os.environ.get("RAVEL_PARTITION_DIR", "ravel_partition_layer8")
OUT = ROOT / "checkpoint"
REPORT = ROOT / "Report" / f"{RUN}_training_report.json"
SEED, EPOCHS, BATCH, DIM, TEMP = 20260825, 30, 256, 128, 0.07
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Partition(nn.Module):
    def __init__(self, width):
        super().__init__(); self.c = nn.Linear(width, DIM); self.s = nn.Linear(width, DIM)
    def forward(self, x):
        return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)


def pair_loss(anchor, positive, negative):
    logits = torch.stack(((anchor * positive).sum(-1), (anchor * negative).sum(-1)), 1) / TEMP
    return F.cross_entropy(logits, torch.zeros(len(anchor), dtype=torch.long, device=DEVICE))


def main():
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows): row["activation_row"] = index
    train = [row for row in rows if row["C_split"] == "train" and row["S_split"] in {"train", "val"}]
    by_fact, by_template = defaultdict(list), defaultdict(list)
    for row in train:
        by_fact[row["fact_id"]].append(row); by_template[row["template_id"]].append(row)
    assert all(len(group) >= 2 for group in by_fact.values())
    assert all(len(group) >= 2 for group in by_template.values())
    x = torch.from_numpy(np.load(DATA / "gemma2_2b_layer8_mean.npy")).to(DEVICE)
    model = Partition(x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(EPOCHS):
        totals = [0.0, 0.0]; steps = 0; model.train()
        for choices in np.array_split(rng.permutation(len(train)), max(1, len(train) // BATCH)):
            anchors = [train[index] for index in choices]
            c_pos = [rng.choice([row for row in by_fact[a["fact_id"]] if row["template_id"] != a["template_id"]]) for a in anchors]
            c_neg = [rng.choice([row for row in by_template[a["template_id"]] if row["fact_id"] != a["fact_id"]]) for a in anchors]
            index_tensor = lambda group: torch.tensor([row["activation_row"] for row in group], device=DEVICE)
            c_anchor, s_anchor = model(x[index_tensor(anchors)])
            c_positive, _ = model(x[index_tensor(c_pos)])
            c_negative, _ = model(x[index_tensor(c_neg)])
            _, s_positive = model(x[index_tensor(c_neg)])
            _, s_negative = model(x[index_tensor(c_pos)])
            c_loss, s_loss = pair_loss(c_anchor, c_positive, c_negative), pair_loss(s_anchor, s_positive, s_negative)
            optimizer.zero_grad(); ((c_loss + s_loss) / 2).backward(); optimizer.step()
            totals[0] += c_loss.item(); totals[1] += s_loss.item(); steps += 1
        row = {"epoch": epoch + 1, "C_triplet_loss": totals[0] / steps, "S_triplet_loss": totals[1] / steps}; history.append(row)
        print(f"epoch={epoch + 1}/{EPOCHS} C_loss={row['C_triplet_loss']:.4f} S_loss={row['S_triplet_loss']:.4f}")
    model.eval()
    with torch.no_grad(): z_c, z_s = model(x)
    np.save(OUT / f"{RUN}_c_all.npy", z_c.cpu().numpy())
    np.save(OUT / f"{RUN}_s_all.npy", z_s.cpu().numpy())
    torch.save({"state_dict": model.state_dict(), "config": {"input_width": x.shape[1], "C_dim": DIM, "S_dim": DIM, "temperature": TEMP, "epochs": EPOCHS, "seed": SEED, "C_positive": "same fact, distinct source training template", "C_negative": "different fact, same exact source training template", "S_positive": "different fact, same exact source training template", "S_negative": "same fact, distinct source training template"}, "history": history}, OUT / f"{RUN}_layer8.pt")
    report = {"device": DEVICE, "training_rows": len(train), "training_facts": len(by_fact), "training_templates": len(by_template), "architecture": "2304 -> z_C(128) + z_S(128)", "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)", "epochs": EPOCHS, "batch": BATCH, "published_test_templates_used_in_training": 0, "decoder": False, "reconstruction": False, "adversary": False, "SAE": False, "history": history}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__": main()
