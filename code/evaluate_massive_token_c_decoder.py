"""Held-out Arabic/Chinese token-set evaluation of the frozen Sinkhorn C-decoder."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ART, TEST, PART, DECODER = ROOT / "data" / "massive_token_layer8_bridge_partition", ROOT / "data" / "massive_all_test.parquet", ROOT / "checkpoint" / "massive_token_layer8_bridge_partition.pt", ROOT / "checkpoint" / "massive_token_c_sinkhorn_decoder.pt"
OUT, MODEL, HOOK, DEVICE, ENGLISH, HOLD = ROOT / "Report" / "massive_token_c_decoder_heldout_eval.json", "google/gemma-2-2b", "blocks.8.hook_resid_post", "cuda", "en-US", ("ar-SA", "zh-CN")

class Partition(nn.Module):
    def __init__(self): super().__init__(); self.c, self.s = nn.Linear(2304, 128), nn.Linear(2304, 128)
    def forward(self, x): return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)

def extract_english(ids):
    rows = pd.read_parquet(TEST); rows = rows[rows.id.isin(ids) & (rows.locale == ENGLISH)].sort_values("id").reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    hf = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE).eval(); bridge = TransformerBridge.boot_transformers(MODEL, hf_model=hf, tokenizer=tokenizer, device=DEVICE, dtype=torch.bfloat16).eval(); output, metadata = [], []
    for start in range(0, len(rows), 16):
        batch = rows.iloc[start:start + 16]; encoded = tokenizer(batch.utt.tolist(), padding=True, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        with torch.inference_mode(): _, cache = bridge.run_with_cache(encoded.input_ids, attention_mask=encoded.attention_mask, names_filter=HOOK, return_cache_object=False)
        h = cache[HOOK]; assert h.shape == (*encoded.input_ids.shape, 2304); keep = encoded.attention_mask.bool() & encoded.input_ids.ne(tokenizer.bos_token_id); output.append(h[keep].float().cpu().numpy()); example, position = np.where(keep.cpu().numpy()); metadata.append(pd.DataFrame({"id": batch.id.to_numpy()[example], "token_position": position}))
    del bridge, hf; torch.cuda.empty_cache(); return np.concatenate(output), pd.concat(metadata, ignore_index=True)

def encode(model, x):
    with torch.no_grad(): return np.concatenate([model(torch.from_numpy(x[i:i + 4096]).to(DEVICE))[0].cpu().numpy() for i in range(0, len(x), 4096)])

def score(query, target):
    q = query / np.linalg.norm(query, axis=1, keepdims=True); t = target / np.linalg.norm(target, axis=1, keepdims=True); return float((q @ t.T).max(1).mean())

def norms(x): return {"mean": float(np.linalg.norm(x, axis=1).mean()), "std": float(np.linalg.norm(x, axis=1).std())}

def main():
    raw, md = np.load(ART / "test_tokens.npy"), pd.read_csv(ART / "test_metadata.csv"); english, en_md = extract_english(set(md.id))
    partition = Partition().to(DEVICE); partition.load_state_dict(torch.load(PART, map_location=DEVICE, weights_only=True)["state_dict"]); partition.eval(); zc = encode(partition, raw)
    decoder = nn.Linear(128, 2304).to(DEVICE); decoder.load_state_dict(torch.load(DECODER, map_location=DEVICE, weights_only=True)["state_dict"]); decoder.eval()
    with torch.no_grad(): decoded = np.concatenate([decoder(torch.from_numpy(zc[i:i + 4096]).to(DEVICE)).cpu().numpy() for i in range(0, len(zc), 4096)])
    foreign_groups, english_groups = {key: value.index.to_numpy() for key, value in md.groupby(["id", "locale"])}, {key: value.index.to_numpy() for key, value in en_md.groupby("id")}
    rows = []
    for (utterance_id, locale), foreign_index in foreign_groups.items():
        target = english[english_groups[utterance_id]]; raw_set, decoded_set = raw[foreign_index], decoded[foreign_index]
        rows.append({"locale": locale, "raw": score(raw_set, target), "decoded": score(decoded_set, target), "reverse_coverage": score(target, decoded_set)})
    result = pd.DataFrame(rows); result["delta"] = result.decoded - result.raw
    def summarize(frame):
        locale_mask = np.isin(md.locale, frame.locale.unique())
        return {"raw_foreign_to_english": float(frame.raw.mean()), "decoded_to_english": float(frame.decoded.mean()), "improvement": float(frame.delta.mean()), "fraction_delta_positive": float((frame.delta > 0).mean()), "reverse_coverage": float(frame.reverse_coverage.mean()), "activation_norms": {"h": norms(raw[locale_mask]), "h_en": norms(english), "h_C": norms(decoded[locale_mask])}}
    sample = decoded[:min(1000, len(decoded))]; unit = sample / np.linalg.norm(sample, axis=1, keepdims=True); np.fill_diagonal(unit @ unit.T, 0.)
    report = {"combined": summarize(result), "Arabic": summarize(result[result.locale == HOLD[0]]), "Chinese": summarize(result[result.locale == HOLD[1]]), "sinkhorn_training_loss": json.loads((ROOT / "Report" / "massive_token_c_sinkhorn_decoder.json").read_text())["sinkhorn_cost_history"], "collapse_diagnostics": {"decoded_coordinate_std_mean": float(decoded.std(0).mean()), "decoded_mean_random_pair_cosine": float((unit @ unit.T).sum() / (len(unit) * (len(unit) - 1)))}, "token_position_alignment": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
