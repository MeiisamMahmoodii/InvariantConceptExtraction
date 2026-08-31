"""Train only a token-set Sinkhorn C-decoder from frozen MASSIVE z_C activations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ART, PART = ROOT / "data" / "massive_token_layer8_bridge_partition", ROOT / "checkpoint" / "massive_token_layer8_bridge_partition.pt"
CKPT, OUT = ROOT / "checkpoint" / "massive_token_c_sinkhorn_decoder.pt", ROOT / "Report" / "massive_token_c_sinkhorn_decoder.json"
DEVICE, ENGLISH, HOLD, SEED, EPOCHS, BATCH, EPSILON, ITERS = "cuda", "en-US", {"ar-SA", "zh-CN"}, 20260827, 5, 16, 5000., 30

class Partition(nn.Module):
    def __init__(self): super().__init__(); self.c, self.s = nn.Linear(2304, 128), nn.Linear(2304, 128)
    def forward(self, x): return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)

def sinkhorn(decoded, target, decoded_mask, target_mask):
    cost = (decoded[:, :, None] - target[:, None, :]).square().sum(-1); valid = decoded_mask[:, :, None] & target_mask[:, None, :]
    kernel = (-cost / EPSILON).exp() * valid; a = decoded_mask / decoded_mask.sum(1, keepdim=True); b = target_mask / target_mask.sum(1, keepdim=True); u, v = torch.ones_like(a), torch.ones_like(b)
    for _ in range(ITERS):
        u = a / (kernel @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = b / (kernel.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    loss = ((u[:, :, None] * kernel * v[:, None, :]) * cost).sum((1, 2)).mean(); assert torch.isfinite(loss), "non-finite Sinkhorn loss"; return loss

def main():
    assert torch.cuda.is_available(); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    raw, md = np.load(ART / "train_tokens.npy", mmap_mode="r"), pd.read_csv(ART / "train_metadata.csv"); assert not set(md.locale) & HOLD
    partition = Partition().to(DEVICE); partition.load_state_dict(torch.load(PART, map_location=DEVICE, weights_only=True)["state_dict"]); partition.eval()
    with torch.no_grad(): zc = np.concatenate([partition(torch.from_numpy(np.asarray(raw[i:i + 4096])).to(DEVICE))[0].cpu().numpy() for i in range(0, len(raw), 4096)])
    groups = {key: value.index.to_numpy() for key, value in md.groupby(["id", "locale"])}
    pairs = [(groups[(utterance_id, locale)], groups[(utterance_id, ENGLISH)]) for utterance_id in sorted(set(md.id)) for locale in sorted(set(md.loc[md.id == utterance_id, "locale"]) - {ENGLISH})]; assert pairs
    decoder, opt, history = nn.Linear(128, 2304).to(DEVICE), None, []
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        losses = []
        for selection in np.array_split(rng.permutation(len(pairs)), max(1, len(pairs) // BATCH)):
            batch = [pairs[i] for i in selection]; ns, nt = max(len(p[0]) for p in batch), max(len(p[1]) for p in batch)
            source, target = np.zeros((len(batch), ns, 128), np.float32), np.zeros((len(batch), nt, 2304), np.float32); source_mask, target_mask = np.zeros((len(batch), ns), bool), np.zeros((len(batch), nt), bool)
            for i, (source_index, target_index) in enumerate(batch): source[i, :len(source_index)], target[i, :len(target_index)], source_mask[i, :len(source_index)], target_mask[i, :len(target_index)] = zc[source_index], raw[target_index], True, True
            loss = sinkhorn(decoder(torch.from_numpy(source).to(DEVICE)), torch.from_numpy(target).to(DEVICE), torch.from_numpy(source_mask).to(DEVICE), torch.from_numpy(target_mask).to(DEVICE)); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        history.append(float(np.mean(losses))); print(f"epoch={epoch + 1}/{EPOCHS} sinkhorn_cost={history[-1]:.3f}")
    report = {"decoder": "Linear(128,2304)", "frozen": {"gemma": True, "token_partition": True}, "target": "all English blocks.8.hook_resid_post tokens for the same MASSIVE id", "loss": {"name": "uniform-marginal differentiable Sinkhorn OT", "cost": "squared Euclidean", "epsilon": EPSILON, "iterations": ITERS, "token_position_alignment": False}, "training_pairs": len(pairs), "heldout_locales": sorted(HOLD), "epochs": EPOCHS, "sinkhorn_cost_history": history, "SAE_trained": False, "gemma_scope_loaded": False}
    torch.save({"state_dict": decoder.state_dict(), "config": report}, CKPT); OUT.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
