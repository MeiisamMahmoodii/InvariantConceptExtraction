"""Rebuild canonical held-out MASSIVE token sets and evaluate frozen C-decoder."""
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
DATA, ART = ROOT / "data" / "massive_all_test.parquet", ROOT / "data" / "massive_token_layer8_canonical_heldout"
PART, DECODER, OUT = ROOT / "checkpoint" / "massive_token_layer8_bridge_partition.pt", ROOT / "checkpoint" / "massive_token_c_sinkhorn_decoder.pt", ROOT / "Report" / "massive_token_c_decoder_canonical_heldout_eval.json"
MODEL, HOOK, DEVICE, ENGLISH, HOLD, STABLE = "google/gemma-2-2b", "blocks.8.hook_resid_post", "cuda", "en-US", ("ar-SA", "zh-CN"), set(range(60)) - {29, 37}

class Partition(nn.Module):
    def __init__(self): super().__init__(); self.c, self.s = nn.Linear(2304, 128), nn.Linear(2304, 128)
    def forward(self, x): return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)

def extract(bridge, tokenizer, rows, locale):
    vectors, metadata = [], []
    for start in range(0, len(rows), 16):
        batch = rows.iloc[start:start + 16]; encoded = tokenizer(batch.utt.tolist(), padding=True, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        with torch.inference_mode(): _, cache = bridge.run_with_cache(encoded.input_ids, attention_mask=encoded.attention_mask, names_filter=HOOK, return_cache_object=False)
        h = cache[HOOK]; assert h.shape == (*encoded.input_ids.shape, 2304); keep = encoded.attention_mask.bool() & encoded.input_ids.ne(tokenizer.bos_token_id); vectors.append(h[keep].float().cpu().numpy()); example, position = np.where(keep.cpu().numpy()); metadata.append(pd.DataFrame({"id": batch.id.astype(str).to_numpy()[example], "locale": locale, "token_position": position}))
        print(f"{locale}: {min(start + len(batch), len(rows))}/{len(rows)} prompts")
    x, md = np.concatenate(vectors), pd.concat(metadata, ignore_index=True); np.save(ART / f"{locale}_tokens.npy", x); md.to_csv(ART / f"{locale}_metadata.csv", index=False); return x, md

def encode(model, x):
    with torch.no_grad(): return np.concatenate([model(torch.from_numpy(x[i:i + 4096]).to(DEVICE))[0].cpu().numpy() for i in range(0, len(x), 4096)])

def score(query, target):
    q, t = query / np.linalg.norm(query, axis=1, keepdims=True), target / np.linalg.norm(target, axis=1, keepdims=True); return float((q @ t.T).max(1).mean())

def norm(x): return {"mean": float(np.linalg.norm(x, axis=1).mean()), "std": float(np.linalg.norm(x, axis=1).std())}

def main():
    assert torch.cuda.is_available(); ART.mkdir(exist_ok=True); data = pd.read_parquet(DATA); data.id = data.id.astype(str); data = data[data.intent.isin(STABLE) & data.locale.isin((*HOLD, ENGLISH))].sort_values(["locale", "id"]).reset_index(drop=True)
    ids = {locale: set(data.loc[data.locale == locale, "id"]) for locale in (*HOLD, ENGLISH)}; missing = {locale: sorted(ids[locale] - ids[ENGLISH]) for locale in HOLD}; duplicates = {locale: int(data.loc[data.locale == locale, "id"].duplicated().sum()) for locale in (*HOLD, ENGLISH)}; assert not any(missing.values()) and not any(duplicates.values())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    hf = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE).eval(); bridge = TransformerBridge.boot_transformers(MODEL, hf_model=hf, tokenizer=tokenizer, device=DEVICE, dtype=torch.bfloat16).eval(); assert HOOK in bridge.hook_dict
    raw, metadata = {}, {}
    for locale in (*HOLD, ENGLISH): raw[locale], metadata[locale] = extract(bridge, tokenizer, data[data.locale == locale].reset_index(drop=True), locale)
    del bridge, hf; torch.cuda.empty_cache()
    partition = Partition().to(DEVICE); partition.load_state_dict(torch.load(PART, map_location=DEVICE, weights_only=True)["state_dict"]); partition.eval()
    decoder = nn.Linear(128, 2304).to(DEVICE); decoder.load_state_dict(torch.load(DECODER, map_location=DEVICE, weights_only=True)["state_dict"]); decoder.eval()
    decoded = {}
    for locale in HOLD:
        zc = encode(partition, raw[locale])
        with torch.no_grad(): decoded[locale] = np.concatenate([decoder(torch.from_numpy(zc[i:i + 4096]).to(DEVICE)).cpu().numpy() for i in range(0, len(zc), 4096)])
    groups = {locale: {key: value.index.to_numpy() for key, value in metadata[locale].groupby("id")} for locale in (*HOLD, ENGLISH)}
    rows = []
    for locale in HOLD:
        for utterance_id in ids[locale]:
            foreign, target, predicted = raw[locale][groups[locale][utterance_id]], raw[ENGLISH][groups[ENGLISH][utterance_id]], decoded[locale][groups[locale][utterance_id]]
            rows.append({"locale": locale, "raw": score(foreign, target), "decoded": score(predicted, target), "coverage": score(target, predicted)})
    result = pd.DataFrame(rows); result["delta"] = result.decoded - result.raw
    def summary(frame):
        locales = frame.locale.unique(); foreign = np.concatenate([raw[l] for l in locales]); prediction = np.concatenate([decoded[l] for l in locales])
        return {"raw_foreign_to_english": float(frame.raw.mean()), "decoded_to_english": float(frame.decoded.mean()), "improvement": float(frame.delta.mean()), "fraction_delta_positive": float((frame.delta > 0).mean()), "reverse_coverage": float(frame.coverage.mean()), "activation_norms": {"h": norm(foreign), "h_en": norm(raw[ENGLISH]), "h_C": norm(prediction)}}
    sample = np.concatenate([decoded[l] for l in HOLD])[:1000]; unit = sample / np.linalg.norm(sample, axis=1, keepdims=True); cosine = unit @ unit.T; np.fill_diagonal(cosine, 0.)
    losses = json.loads((ROOT / "Report" / "massive_token_c_sinkhorn_decoder.json").read_text())["sinkhorn_cost_history"]
    report = {"id_verification": {"arabic_ids": len(ids[HOLD[0]]), "chinese_ids": len(ids[HOLD[1]]), "matching_english_ids": len(ids[HOLD[0]] & ids[HOLD[1]] & ids[ENGLISH]), "missing_ids": missing, "duplicate_ids": duplicates, "heldout_subset_of_english": True}, "combined": summary(result), "Arabic": summary(result[result.locale == HOLD[0]]), "Chinese": summary(result[result.locale == HOLD[1]]), "sinkhorn_training_loss": losses, "collapse_diagnostics": {"decoded_coordinate_std_mean": float(np.concatenate([decoded[l] for l in HOLD]).std(0).mean()), "decoded_mean_random_pair_cosine": float(cosine.sum() / (len(sample) * (len(sample) - 1)))}, "token_position_alignment": False, "partition_retrained": False, "decoder_retrained": False, "gemma_scope_loaded": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
