"""Read-only extraction of additional genuine top activations for Figure 2."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy._core.multiarray import _reconstruct


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data" / "massive_partition_artifacts"
SAE_ARTIFACTS = ROOT / "data" / "massive_sae_artifacts"
CHECKPOINTS = ROOT / "checkpoint"
OUT = ROOT / "paper" / "figure_data" / "figure2_feature_traces.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SAE(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.e = nn.Linear(width, width * 4)
        self.d = nn.Linear(width * 4, width, bias=False)
        self.b = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        activations = F.relu(self.e(x))
        values, indices = torch.topk(activations, 64, dim=1)
        sparse = torch.zeros_like(activations).scatter(1, indices, values)
        return sparse, self.d(sparse) + self.b


def feature_values(representation, feature_id):
    array_path = ARTIFACTS / "raw_train_layer8.npy" if representation == "H" else SAE_ARTIFACTS / f"{representation}_train.npy"
    array = np.load(array_path, mmap_mode="r")
    checkpoint_name = {
        "H": "massive_topk_raw_k64.pt",
        "z_C": "massive_topk_z_C_k64.pt",
        "z_S": "massive_topk_z_S_k64.pt",
    }[representation]
    torch.serialization.add_safe_globals([_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType])
    state = torch.load(CHECKPOINTS / checkpoint_name, map_location=DEVICE, weights_only=True)
    model = SAE(array.shape[1]).to(DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()
    mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    result = []
    with torch.no_grad():
        for start in range(0, len(array), 512):
            batch = (np.asarray(array[start:start + 512], dtype=np.float32) - mean) / std
            result.append(model(torch.from_numpy(batch).to(DEVICE))[0][:, feature_id].cpu().numpy())
    return np.concatenate(result)


def main():
    metadata = pd.read_csv(ARTIFACTS / "train_metadata.csv")
    texts = pd.read_parquet(ROOT / "data" / "massive_all_train.parquet")
    lookup = {(str(row.id), row.locale): row.utt for _, row in texts.iterrows()}
    output = {}
    for representation, feature in (("H", 7124), ("z_C", 334), ("z_S", 491)):
        values = feature_values(representation, feature)
        rows = []
        for index in np.argsort(-values)[:8]:
            row = metadata.iloc[index]
            rows.append({
                "activation": round(float(values[index]), 6), "intent": int(row.intent), "locale": row.locale,
                "id": str(row.id), "utterance": lookup.get((str(row.id), row.locale), ""),
            })
        output[f"{representation}_{feature}"] = rows
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert all(len(rows) == 8 for rows in output.values())


if __name__ == "__main__":
    main()
