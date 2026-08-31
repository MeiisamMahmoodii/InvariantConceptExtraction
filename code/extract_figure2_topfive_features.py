"""Read frozen pooled SAEs and select high-confidence feature cards for Figure 2B."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy._core.multiarray import _reconstruct


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "massive_partition_artifacts"
SAE_ART = ROOT / "data" / "massive_sae_artifacts"
CKPT = ROOT / "checkpoint"
OUT = ROOT / "paper" / "figure_data" / "figure2_topfive_features.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260827


class SAE(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.e = nn.Linear(width, width * 4)
        self.d = nn.Linear(width * 4, width, bias=False)
        self.b = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        dense = F.relu(self.e(x))
        values, indices = torch.topk(dense, 64, dim=1)
        sparse = torch.zeros_like(dense).scatter(1, indices, values)
        return sparse


def activations(model, data, mean, std):
    output = []
    with torch.no_grad():
        for start in range(0, len(data), 512):
            x = (np.asarray(data[start:start + 512], dtype=np.float32) - mean) / std
            output.append(model(torch.from_numpy(x).to(DEVICE)).cpu().numpy())
    return np.concatenate(output)


def load_sae(representation, width):
    filename = {"H": "massive_topk_raw_k64.pt", "zC": "massive_topk_z_C_k64.pt", "zS": "massive_topk_z_S_k64.pt"}[representation]
    torch.serialization.add_safe_globals([_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType])
    state = torch.load(CKPT / filename, map_location=DEVICE, weights_only=True)
    model = SAE(width).to(DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, np.asarray(state["mean"]), np.asarray(state["std"])


def cards(representation, data, metadata, intent_count, language_count):
    model, mean, std = load_sae(representation, data.shape[1])
    values = activations(model, data, mean, std)
    mean_value, sd_value = values.mean(0), values.std(0).clip(1e-6)
    intent_means = np.stack([values[metadata.intent.to_numpy() == label].mean(0) for label in np.unique(metadata.intent)])
    locale_means = np.stack([values[metadata.locale.to_numpy() == label].mean(0) for label in np.unique(metadata.locale)])
    intent_score = (intent_means.max(0) - mean_value) / sd_value
    locale_score = (locale_means.max(0) - mean_value) / sd_value
    active = mean_value > 1e-6
    intent_ok = active & (intent_score > 1.1 * locale_score)
    locale_ok = active & (locale_score > 1.1 * intent_score)

    def make(feature, kind):
        top_rows = np.argsort(-values[:, feature])[:20]
        if kind == "intent":
            group = int(metadata.intent.iloc[top_rows].mode().iloc[0])
            diversity = metadata.locale.iloc[top_rows].nunique()
            label = f"intent #{group}"
            evidence = f"same intent across {diversity} locales"
            score = intent_score[feature]
        else:
            group = metadata.locale.iloc[top_rows].mode().iloc[0]
            diversity = metadata.intent.iloc[top_rows].nunique()
            label = group
            evidence = f"same locale across {diversity} intents"
            score = locale_score[feature]
        return {"feature_id": int(feature), "orientation": kind, "label": label,
                "evidence": evidence, "selectivity_z": round(float(score), 3)}

    top_intent = [make(feature, "intent") for feature in np.argsort(-intent_score * intent_ok)[:intent_count]]
    top_locale = [make(feature, "language") for feature in np.argsort(-locale_score * locale_ok)[:language_count]]
    return top_intent, top_locale


def main():
    metadata = pd.read_csv(ART / "train_metadata.csv")
    index = np.random.default_rng(SEED).choice(len(metadata), 20_000, replace=False)
    selected_metadata = metadata.iloc[index].reset_index(drop=True)
    raw = np.load(ART / "raw_train_layer8.npy", mmap_mode="r")[index]
    zc = np.load(SAE_ART / "z_C_train.npy", mmap_mode="r")[index]
    zs = np.load(SAE_ART / "z_S_train.npy", mmap_mode="r")[index]
    h_intent, h_locale = cards("H", raw, selected_metadata, 3, 2)
    zc_intent, _ = cards("zC", zc, selected_metadata, 5, 0)
    _, zs_locale = cards("zS", zs, selected_metadata, 0, 5)
    output = {"H": h_intent + h_locale, "zC": zc_intent, "zS": zs_locale}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    assert all(len(output[key]) == 5 for key in output)


if __name__ == "__main__":
    main()
