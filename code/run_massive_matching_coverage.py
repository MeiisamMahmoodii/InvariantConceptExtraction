"""Matched-negative ablation and locale-coverage sweep on saved MASSIVE activations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "massive_partition_artifacts"
CKPT = ROOT / "checkpoint" / "massive_matching_coverage"
OUT = ROOT / "Report" / "massive_matching_coverage_audit.json"
DEVICE = "cuda"
SEED = 20260827
SEEDS = (20260828, 20260829, 20260830)
EPOCHS, SAE_EPOCHS, BATCH, TEMP = 30, 30, 256, 0.07
WIDTH, TOPK = 512, 64


class Partition(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Linear(2304, 128)
        self.s = nn.Linear(2304, 128)

    def forward(self, x):
        return F.normalize(self.c(x), dim=-1), F.normalize(self.s(x), dim=-1)


class TopKSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(128, WIDTH)
        self.decoder = nn.Linear(WIDTH, 128, bias=False)
        self.bias = nn.Parameter(torch.zeros(128))

    def forward(self, x):
        dense = F.relu(self.encoder(x))
        values, indices = torch.topk(dense, TOPK, dim=1)
        sparse = torch.zeros_like(dense).scatter(1, indices, values)
        return sparse, self.decoder(sparse) + self.bias

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


def contrastive(anchor, positive, negative):
    logits = torch.stack(((anchor * positive).sum(1), (anchor * negative).sum(1)), 1) / TEMP
    return F.cross_entropy(logits, torch.zeros(len(anchor), dtype=torch.long, device=DEVICE))


def effective_rank(x):
    values = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    probability = values / values.sum()
    return {
        "participation_ratio": float(values.sum() ** 2 / (values ** 2).sum()),
        "entropy_effective_rank": float(np.exp(-(probability * np.log(probability + 1e-30)).sum())),
    }


def retrieval(x, metadata):
    langs, ids = metadata.locale.to_numpy(), metadata.id.to_numpy()
    ar, zh = np.where(langs == "ar-SA")[0], np.where(langs == "zh-CN")[0]
    query = F.normalize(torch.from_numpy(x[ar]), dim=1).numpy()
    key = F.normalize(torch.from_numpy(x[zh]), dim=1).numpy()
    order = np.argsort(-(query @ key.T), 1)
    ranks = np.array([np.where(ids[zh][row] == ids[ar[i]])[0][0] + 1 for i, row in enumerate(order)])
    return {"R@1": float((ranks == 1).mean()), "R@5": float((ranks <= 5).mean()), "MRR": float((1 / ranks).mean())}


def locale_probe(x, labels, seed):
    order = np.random.default_rng(seed).permutation(len(x))
    left, right = order[: len(order) // 2], order[len(order) // 2 :]
    scaler = StandardScaler().fit(x[left])
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(scaler.transform(x[left]), labels[left])
    return float((model.predict(scaler.transform(x[right])) == labels[right]).mean())


def encode(model, raw, branch):
    output = np.empty((len(raw), 128), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(raw), 4096):
            output[start : start + 4096] = model(torch.from_numpy(np.asarray(raw[start : start + 4096])).to(DEVICE))[branch].cpu().numpy()
    return output


def encode_rows(model, raw, rows, branch):
    output = np.empty((len(rows), 128), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 4096):
            selected = rows[start : start + 4096]
            output[start : start + len(selected)] = model(torch.from_numpy(np.asarray(raw[selected])).to(DEVICE))[branch].cpu().numpy()
    return output


def matched_margin(model, raw, grid, locale_codes, seed, count=20000):
    rng = np.random.default_rng(seed + 99)
    n_ids, n_locales = len(grid), len(locale_codes)
    ids = rng.integers(n_ids, size=count)
    locale = rng.integers(n_locales, size=count)
    other_locale = (locale + rng.integers(1, n_locales, size=count)) % n_locales
    other_id = (ids + rng.integers(1, n_ids, size=count)) % n_ids
    index = grid[:, locale_codes]
    rows = (index[ids, locale], index[ids, other_locale], index[other_id, locale])
    with torch.no_grad():
        anchor = torch.from_numpy(np.asarray(raw[rows[0]])).to(DEVICE)
        positive = torch.from_numpy(np.asarray(raw[rows[1]])).to(DEVICE)
        competing = torch.from_numpy(np.asarray(raw[rows[2]])).to(DEVICE)
        zca, zsa = model(anchor)
        zcp, zsn = model(positive)
        zcn, zsp = model(competing)
        c_margin = (zca * zcp).sum(1) - (zca * zcn).sum(1)
        s_margin = (zsa * zsp).sum(1) - (zsa * zsn).sum(1)
    return {"zC": float(c_margin.mean()), "zS": float(s_margin.mean())}


def train_partition(raw, raw_test, metadata_test, grid, locale_codes, seed, negative_kind, tag):
    path = CKPT / f"partition_{tag}.pt"
    torch.manual_seed(seed)
    model = Partition().to(DEVICE)
    history = []
    if path.exists():
        saved = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        history = saved["history"]
    else:
        rng = np.random.default_rng(seed)
        selected_grid = grid[:, locale_codes]
        n_ids, n_locales = selected_grid.shape
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(EPOCHS):
            losses = []
            for positions in np.array_split(rng.permutation(n_ids * n_locales), max(1, n_ids * n_locales // BATCH)):
                ids, locale = positions // n_locales, positions % n_locales
                other_locale = (locale + rng.integers(1, n_locales, len(positions))) % n_locales
                other_id = (ids + rng.integers(1, n_ids, len(positions))) % n_ids
                anchor_rows = selected_grid[ids, locale]
                same_c = selected_grid[ids, other_locale]
                same_s = selected_grid[other_id, locale]
                anchor = torch.from_numpy(np.asarray(raw[anchor_rows])).to(DEVICE)
                c_positive = torch.from_numpy(np.asarray(raw[same_c])).to(DEVICE)
                s_positive = torch.from_numpy(np.asarray(raw[same_s])).to(DEVICE)
                zca, zsa = model(anchor)
                zcp, zsn = model(c_positive)
                zcn, zsp = model(s_positive)
                if negative_kind == "matched":
                    c_negative, s_negative = zcn, zsn
                else:
                    both_rows = selected_grid[other_id, other_locale]
                    both_c, both_s = model(torch.from_numpy(np.asarray(raw[both_rows])).to(DEVICE))
                    c_negative, s_negative = both_c, both_s
                loss_c = contrastive(zca, zcp, c_negative)
                loss_s = contrastive(zsa, zsp, s_negative)
                loss = (loss_c + loss_s) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append((loss_c.item(), loss_s.item()))
            mean = np.mean(losses, 0)
            history.append({"epoch": epoch + 1, "zC": float(mean[0]), "zS": float(mean[1])})
            print(f"{tag} epoch={epoch + 1}/{EPOCHS} zC={mean[0]:.4f} zS={mean[1]:.4f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "history": history}, path)
    zc_test, zs_test = encode(model, raw_test, 0), encode(model, raw_test, 1)
    labels = metadata_test.locale.to_numpy()
    return model, {
        "seed": seed,
        "locales": [str(x) for x in locale_codes],
        "negative": negative_kind,
        "zC": {"retrieval": retrieval(zc_test, metadata_test), "locale_probe": locale_probe(zc_test, labels, seed), "rank": effective_rank(zc_test)},
        "zS": {"retrieval": retrieval(zs_test, metadata_test), "locale_probe": locale_probe(zs_test, labels, seed), "rank": effective_rank(zs_test)},
        "matched_relation_margin": matched_margin(model, raw, grid, locale_codes, seed),
        "history": history,
    }, zc_test


def sae_activations(model, x, mean, std, batch=1024):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            tensor = torch.from_numpy(((np.asarray(x[start : start + batch], np.float32) - mean) / std)).to(DEVICE)
            output.append(model(tensor)[0].cpu().numpy())
    return np.concatenate(output)


def train_and_audit_sae(ztrain, ztest, train_labels, test_metadata, seed, tag):
    path = CKPT / f"sae_{tag}.pt"
    mean, std = ztrain.mean(0), ztrain.std(0).clip(1e-6)
    torch.manual_seed(seed)
    model = TopKSAE().to(DEVICE)
    history = []
    if path.exists():
        saved = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        history = saved["history"]
        mean, std = saved["mean"], saved["std"]
    else:
        rng = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(SAE_EPOCHS):
            losses = []
            for rows in np.array_split(rng.permutation(len(ztrain)), max(1, len(ztrain) // BATCH)):
                tensor = torch.from_numpy(((np.asarray(ztrain[rows], np.float32) - mean) / std)).to(DEVICE)
                _, reconstruction = model(tensor)
                loss = F.mse_loss(reconstruction, tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.normalize_decoder()
                losses.append(loss.item())
            history.append(float(np.mean(losses)))
            print(f"{tag} SAE epoch={epoch + 1}/{SAE_EPOCHS} mse={history[-1]:.5f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "history": history}, path)

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(ztrain), min(20000, len(ztrain)), replace=False)
    sampled = sae_activations(model, ztrain[sample], mean, std)
    sampled_labels = train_labels.iloc[sample]
    mu, sd = sampled.mean(0), sampled.std(0).clip(1e-6)
    intent_values = np.unique(sampled_labels.intent)
    locale_values = np.unique(sampled_labels.locale)
    intent_score = (np.stack([sampled[sampled_labels.intent == value].mean(0) for value in intent_values]).max(0) - mu) / sd
    locale_score = (np.stack([sampled[sampled_labels.locale == value].mean(0) for value in locale_values]).max(0) - mu) / sd
    active = mu > 1e-6
    intent = active & (intent_score > 1.1 * locale_score)
    language = active & (locale_score > 1.1 * intent_score)

    sums = np.zeros((len(intent_values), WIDTH), np.float64)
    total = np.zeros(WIDTH, np.float64)
    counts = np.bincount(np.searchsorted(intent_values, train_labels.intent), minlength=len(intent_values))
    for start in range(0, len(ztrain), 4096):
        stop = min(start + 4096, len(ztrain))
        activations = sae_activations(model, ztrain[start:stop], mean, std, batch=4096)
        codes = np.searchsorted(intent_values, train_labels.intent[start:stop])
        total += activations.sum(0)
        for code in np.unique(codes):
            sums[code] += activations[codes == code].sum(0)
    positive = sums / counts[:, None]
    negative = (total[None, :] - sums) / (len(ztrain) - counts)[:, None]
    selected = (positive - negative).argmax(1)

    test_acts = sae_activations(model, ztest, mean, std)
    ar = {test_metadata.id.iloc[i]: i for i in np.where(test_metadata.locale == "ar-SA")[0]}
    zh = {test_metadata.id.iloc[i]: i for i in np.where(test_metadata.locale == "zh-CN")[0]}
    pairs = [(ar[key], zh[key]) for key in ar.keys() & zh.keys()]
    left, right = [i for i, _ in pairs], [j for _, j in pairs]
    selected_stability = []
    for feature in selected:
        a, b = test_acts[left, feature], test_acts[right, feature]
        selected_stability.append(float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0)
    all_stability = []
    for feature in np.where(active)[0]:
        a, b = test_acts[left, feature], test_acts[right, feature]
        if a.std() and b.std():
            all_stability.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "active_features": int(active.sum()),
        "intent_oriented_fraction": float(intent.sum() / active.sum()),
        "language_oriented_fraction": float(language.sum() / active.sum()),
        "mixed_fraction": float((active & ~(intent | language)).sum() / active.sum()),
        "selected_feature_stability": float(np.mean(selected_stability)),
        "unfiltered_active_feature_stability": float(np.mean(all_stability)),
        "history": history,
    }


def setup():
    train = pd.read_csv(ART / "train_metadata.csv")
    test = pd.read_csv(ART / "test_metadata.csv")
    raw = np.load(ART / "raw_train_layer8.npy", mmap_mode="r")
    raw_test = np.asarray(np.load(ART / "raw_test_layer8.npy", mmap_mode="r"))
    id_code, ids = pd.factorize(train.id, sort=True)
    locale_code, locales = pd.factorize(train.locale, sort=True)
    grid = np.empty((len(ids), len(locales)), np.int64)
    grid[id_code, locale_code] = np.arange(len(train))
    assert grid.shape == (11492, 49) and np.unique(grid).size == len(train)
    return train, test, raw, raw_test, grid, np.asarray(locales)


def write_report(report):
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_negative(report, data):
    train, test, raw, raw_test, grid, locales = data
    model, result, zc_test = train_partition(raw, raw_test, test, grid, np.arange(len(locales)), SEED, "uncontrolled", "uncontrolled_k49_seed20260827")
    ztrain = encode(model, raw, 0)
    result["sae"] = train_and_audit_sae(ztrain, zc_test, train[["intent", "locale"]].reset_index(drop=True), test, SEED, "uncontrolled_k49_seed20260827")
    prior_partition = json.loads((ROOT / "Report" / "massive_partition_audit.json").read_text())
    prior_sae = json.loads((ROOT / "Report" / "massive_b2_infonce_sae_comparison.json").read_text())["branches"]["Matched_zC"]
    prior_selected = json.loads((ROOT / "Report" / "massive_sae_intent_monitoring_audit.json").read_text())["z_C"]["mean_arabic_chinese_feature_stability"]
    prior_sae["selected_feature_stability"] = prior_selected
    report["negative_ablation"] = {
        "controlled_matched": {
            "zC": {
                "retrieval": prior_partition["same_ID_cross_language_retrieval"]["z_C"],
                "locale_probe": prior_partition["heldout_language_probe_accuracy"]["z_C"],
                "rank": prior_partition["heldout_effective_rank"]["z_C"],
            },
            "sae": prior_sae,
        },
        "both_factors_change_negative": result,
        "interpretation": "Both variants use the same binary contrastive loss; only the negative relation changes.",
    }
    write_report(report)


def run_coverage(report, data):
    train, test, raw, raw_test, grid, locales = data
    coverage = {}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(locales))
        for count in (4, 8, 16, 32):
            selected = order[:count]
            model, result, zc_test = train_partition(raw, raw_test, test, grid, selected, seed, "matched", f"coverage_k{count}_seed{seed}")
            result["locale_names"] = locales[selected].tolist()
            if seed == SEEDS[0] and count in (4, 16):
                rows = grid[:, selected].reshape(-1)
                ztrain = encode_rows(model, raw, rows, 0)
                labels = train.iloc[rows][["intent", "locale"]].reset_index(drop=True)
                result["sae"] = train_and_audit_sae(ztrain, zc_test, labels, test, seed, f"coverage_k{count}_seed{seed}")
            coverage.setdefault(str(count), []).append(result)
            report["coverage_sweep"] = coverage
            write_report(report)
    robustness = json.loads((ROOT / "Report" / "massive_core_seed_robustness.json").read_text())
    coverage["49"] = [
        {
            "seed": row["seed"],
            "zC": {
                "retrieval": row["partition"]["zC"]["semantic_retrieval"],
                "locale_probe": row["partition"]["zC"]["locale_probe"],
                "rank": row["partition"]["zC"]["effective_rank"],
            },
            "zS": {
                "retrieval": row["partition"]["zS"]["semantic_retrieval"],
                "locale_probe": row["partition"]["zS"]["locale_probe"],
                "rank": row["partition"]["zS"]["effective_rank"],
            },
            "sae": {
                "active_features": row["sae"]["zC"]["active_features"],
                "intent_oriented_fraction": row["sae"]["zC"]["intent_oriented_fraction"],
                "language_oriented_fraction": row["sae"]["zC"]["language_oriented_fraction"],
                "unfiltered_active_feature_stability": row["sae"]["zC"]["heldout_arabic_chinese_stability"],
            },
            "source": "existing three-seed full-coverage audit",
        }
        for row in robustness["per_seed"]
    ]
    report["coverage_sweep"] = coverage
    write_report(report)


def smoke():
    grid = np.arange(30).reshape(5, 6)
    rng = np.random.default_rng(7)
    ids, locale = np.array([0, 1, 2]), np.array([0, 1, 2])
    other_locale = (locale + rng.integers(1, 6, len(ids))) % 6
    other_id = (ids + rng.integers(1, 5, len(ids))) % 5
    assert np.all(grid[ids, locale] != grid[ids, other_locale])
    assert np.all(grid[ids, locale] != grid[other_id, locale])
    assert np.all(ids != other_id) and np.all(locale != other_locale)
    print("smoke check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "negative", "coverage", "all"), default="all")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke()
        return
    assert torch.cuda.is_available()
    CKPT.mkdir(parents=True, exist_ok=True)
    report = json.loads(OUT.read_text()) if OUT.exists() else {"protocol": {"epochs": EPOCHS, "batch_size": BATCH, "temperature": TEMP, "heldout_locales": ["ar-SA", "zh-CN"]}}
    data = setup()
    if args.mode in ("negative", "all"):
        run_negative(report, data)
    if args.mode in ("coverage", "all"):
        run_coverage(report, data)


if __name__ == "__main__":
    main()
