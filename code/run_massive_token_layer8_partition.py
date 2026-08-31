"""Token-instance MASSIVE C/S partition at blocks.8.hook_resid_post; no SAE/decoder."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TRAIN, TEST = ROOT / "data" / "massive_all_train.parquet", ROOT / "data" / "massive_all_test.parquet"
ART, CKPT, OUT = ROOT / "data" / "massive_token_layer8_bridge_partition", ROOT / "checkpoint" / "massive_token_layer8_bridge_partition.pt", ROOT / "Report" / "massive_token_layer8_bridge_partition.json"
SEED, EPOCHS, BATCH, TEMP, N_PER_INTENT = 20260827, 30, 256, .07, 2
DEVICE, HOLD, STABLE, MODEL, HOOK = "cuda", ["ar-SA", "zh-CN"], set(range(60)) - {29, 37}, "google/gemma-2-2b", "blocks.8.hook_resid_post"


class Partition(nn.Module):
    def __init__(self):
        super().__init__(); self.c, self.s = nn.Linear(2304, 128), nn.Linear(2304, 128)
    def forward(self, x): return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)


def choose(df):
    return df.groupby("intent", group_keys=False).apply(lambda x: x.sample(min(N_PER_INTENT, len(x)), random_state=SEED)).id


def extract(bridge, tokenizer, df, name):
    vectors, metadata = [], []
    for start in range(0, len(df), 16):
        batch = df.iloc[start:start + 16]; encoded = tokenizer(batch.utt.tolist(), padding=True, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        with torch.inference_mode(): _, cache = bridge.run_with_cache(encoded.input_ids, attention_mask=encoded.attention_mask, names_filter=HOOK, return_cache_object=False)
        h = cache[HOOK]; assert h.ndim == 3 and h.shape == (*encoded.input_ids.shape, 2304), h.shape
        keep = encoded.attention_mask.bool() & encoded.input_ids.ne(tokenizer.bos_token_id)
        vectors.append(h[keep].float().cpu().numpy()); example, position = np.where(keep.cpu().numpy())
        metadata.append(pd.DataFrame({"id": batch.id.to_numpy()[example], "locale": batch.locale.to_numpy()[example], "intent": batch.intent.to_numpy()[example], "token_position": position}))
        print(f"{name}: {min(start + len(batch), len(df))}/{len(df)} prompts")
    x, md = np.concatenate(vectors).astype("float32", copy=False), pd.concat(metadata, ignore_index=True)
    np.save(ART / f"{name}_tokens.npy", x); md.to_csv(ART / f"{name}_metadata.csv", index=False)
    return x, md


def triplet(a, p, n):
    return F.cross_entropy(torch.stack(((a * p).sum(1), (a * n).sum(1)), 1) / TEMP, torch.zeros(len(a), device=DEVICE, dtype=torch.long))


def encode(model, x):
    out = [[], []]
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            c, s = model(torch.from_numpy(x[start:start + 4096]).to(DEVICE)); out[0].append(c.cpu().numpy()); out[1].append(s.cpu().numpy())
    return tuple(np.concatenate(v) for v in out)


def probe(x, y, test_x, test_y):
    scale = StandardScaler().fit(x); clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=SEED).fit(scale.transform(x), y)
    return float((clf.predict(scale.transform(test_x)) == test_y).mean())


def retrieval(x, metadata):
    a, b = (metadata.locale.to_numpy() == locale for locale in HOLD); q, key = F.normalize(torch.from_numpy(x[a]), dim=1).numpy(), F.normalize(torch.from_numpy(x[b]), dim=1).numpy()
    candidate_ids, candidate_inverse = np.unique(metadata.id.to_numpy()[b], return_inverse=True); scores = q @ key.T
    by_id = np.stack([scores[:, candidate_inverse == i].max(1) for i in range(len(candidate_ids))], 1)  # max over candidate tokens: no position alignment.
    target = np.searchsorted(candidate_ids, metadata.id.to_numpy()[a]); ranks = np.argsort(-by_id, 1).argsort(1)[np.arange(len(q)), target] + 1
    return {"R@1": float((ranks == 1).mean()), "R@5": float((ranks <= 5).mean()), "MRR": float((1 / ranks).mean())}


def rank(x):
    values = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2; p = values / values.sum()
    return {"participation_ratio": float(values.sum() ** 2 / (values ** 2).sum()), "entropy_effective_rank": float(np.exp(-(p * np.log(p + 1e-30)).sum()))}


def collapse(x, rng):
    pairs = rng.integers(len(x), size=(min(10000, len(x)), 2)); return {"coordinate_std_mean": float(x.std(0).mean()), "mean_random_pair_cosine": float((x[pairs[:, 0]] * x[pairs[:, 1]]).sum(1).mean())}


def main():
    assert torch.cuda.is_available(); torch.manual_seed(SEED); rng = np.random.default_rng(SEED); ART.mkdir(exist_ok=True)
    train, test = pd.read_parquet(TRAIN), pd.read_parquet(TEST)
    train = train[train.id.isin(choose(train[(~train.locale.isin(HOLD)) & train.intent.isin(STABLE)].drop_duplicates("id"))) & ~train.locale.isin(HOLD)].sort_values(["id", "locale"]).reset_index(drop=True)
    test = test[test.id.isin(choose(test[test.locale.isin(HOLD) & test.intent.isin(STABLE)].drop_duplicates("id"))) & test.locale.isin(HOLD)].sort_values(["id", "locale"]).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE).eval()
    bridge = TransformerBridge.boot_transformers(MODEL, hf_model=hf_model, tokenizer=tokenizer, device=DEVICE, dtype=torch.bfloat16).eval(); assert HOOK in bridge.hook_dict
    raw, md = extract(bridge, tokenizer, train, "train"); raw_test, md_test = extract(bridge, tokenizer, test, "test"); del bridge, hf_model; torch.cuda.empty_cache()
    groups = {key: value.index.to_numpy() for key, value in md.groupby(["id", "locale"])}; ids, locales = md.id.to_numpy(), md.locale.to_numpy(); all_ids, all_locales = np.unique(ids), np.unique(locales)
    model, history = Partition().to(DEVICE), []
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        losses = []
        for batch in np.array_split(rng.permutation(len(raw)), max(1, len(raw) // BATCH)):
            bs_ids, bs_locales = ids[batch], locales[batch]
            same_id_other_locale = np.array([rng.choice(groups[(i, rng.choice(all_locales[all_locales != l]))]) for i, l in zip(bs_ids, bs_locales)])
            other_id_same_locale = np.array([rng.choice(groups[(rng.choice(all_ids[all_ids != i]), l)]) for i, l in zip(bs_ids, bs_locales)])
            a, p, n = (torch.from_numpy(raw[index]).to(DEVICE) for index in (batch, same_id_other_locale, other_id_same_locale)); zc, zs = model(a); zcp, zsn = model(p); zcn, zsp = model(n)
            lc, ls = triplet(zc, zcp, zcn), triplet(zs, zsp, zsn); opt.zero_grad(); ((lc + ls) / 2).backward(); opt.step(); losses.append((lc.item(), ls.item()))
        history.append({"epoch": epoch + 1, "C_triplet_loss": float(np.mean(losses, 0)[0]), "S_triplet_loss": float(np.mean(losses, 0)[1])}); print(history[-1])
    zc, zs, zctr, zstr = *encode(model, raw_test), *encode(model, raw)
    half = rng.permutation(len(raw_test))[:len(raw_test) // 2]; other = np.setdiff1d(np.arange(len(raw_test)), half)
    report = {"heldout_semantic_id_retrieval": {"raw": retrieval(raw_test, md_test), "z_C": retrieval(zc, md_test), "z_S": retrieval(zs, md_test)}, "heldout_locale_decodability": {"raw": probe(raw_test[half], md_test.locale.iloc[half], raw_test[other], md_test.locale.iloc[other]), "z_C": probe(zc[half], md_test.locale.iloc[half], zc[other], md_test.locale.iloc[other]), "z_S": probe(zs[half], md_test.locale.iloc[half], zs[other], md_test.locale.iloc[other])}, "effective_rank": {"z_C": rank(zc), "z_S": rank(zs)}, "training": {"losses": history, "collapse_diagnostics": {"z_C": collapse(zc, rng), "z_S": collapse(zs, rng)}, "train_tokens": len(raw), "heldout_tokens": len(raw_test), "token_matching": "independent token instances sampled by utterance id/locale; no position alignment", "decoder_trained": False, "gemma_scope_loaded": False}}
    torch.save({"state_dict": model.state_dict()}, CKPT); OUT.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
