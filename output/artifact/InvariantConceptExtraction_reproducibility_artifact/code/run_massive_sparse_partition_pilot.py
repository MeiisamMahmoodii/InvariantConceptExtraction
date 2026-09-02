"""Joint sparse MASSIVE z_C/z_S pilot: JumpReLU or blockwise Top-K."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy._core.multiarray import _reconstruct
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import intent_locale_relations as relations

try:
    from sae_lens.saes.jumprelu_sae import JumpReLU, Step
except ModuleNotFoundError:
    JumpReLU = Step = None


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "massive_partition_artifacts"
INIT = ROOT / "checkpoint" / "massive_topk_raw_k64.pt"
CHECKPOINTS = ROOT / "checkpoint" / "sparse_partition_pilot"
REPORTS = ROOT / "Report" / "sparse_partition_study" / "runs"
DEVICE = "cuda"
WIDTH = 2304
SPARSE_WIDTH = WIDTH * 4
TOTAL_K = 64
MATRYOSHKA_GROUPS = (
    SPARSE_WIDTH // 16,
    SPARSE_WIDTH // 16,
    SPARSE_WIDTH // 8,
    SPARSE_WIDTH // 4,
    SPARSE_WIDTH // 2,
)
HOLDOUT = ("ar-SA", "zh-CN")
EVAL_SEED = 20260827


class RowView:
    def __init__(self, values, rows):
        self.values = values
        self.rows = np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, rows):
        return self.values[self.rows[rows]]


def semantic_validation_split(metadata, fraction):
    if not fraction:
        return np.arange(len(metadata)), np.array([], dtype=metadata.id.dtype)
    ids = np.sort(metadata.id.unique())
    validation_ids = np.random.default_rng(EVAL_SEED).choice(
        ids, max(1, round(fraction * len(ids))), replace=False
    )
    return np.flatnonzero(~metadata.id.isin(validation_ids).to_numpy()), validation_ids


def locale_holdout_split(metadata, locales):
    heldout = metadata.locale.isin(locales).to_numpy()
    return np.flatnonzero(~heldout), np.flatnonzero(heldout)


def load_initial_checkpoint():
    torch.serialization.add_safe_globals(
        [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType]
    )
    return torch.load(INIT, map_location="cpu", weights_only=True)


class SparsePartition(nn.Module):
    def __init__(
        self, activation, c_fraction, bandwidth, total_k=TOTAL_K, c_k=None,
        sparsifier="block", sparse_width=SPARSE_WIDTH, input_width=WIDTH,
    ):
        super().__init__()
        if activation == "jumprelu" and JumpReLU is None:
            raise ModuleNotFoundError("JumpReLU experiments require sae_lens; BatchTopK does not")
        self.activation = activation
        self.input_width = input_width
        self.sparse_width = sparse_width
        self.c_width = round(sparse_width * c_fraction)
        self.s_width = sparse_width - self.c_width
        self.total_k = total_k
        self.c_k = round(total_k * c_fraction) if c_k is None else c_k
        self.s_k = total_k - self.c_k
        self.bandwidth = bandwidth
        self.sparsifier = sparsifier
        self.encoder = nn.Linear(input_width, sparse_width)
        self.decoder = nn.Linear(sparse_width, input_width, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(input_width))
        if activation == "jumprelu":
            self.threshold = nn.Parameter(torch.zeros(sparse_width))
        if activation == "batchtopk":
            self.register_buffer("c_threshold", torch.tensor(0.0))
            self.register_buffer("s_threshold", torch.tensor(0.0))
            self.register_buffer("global_threshold", torch.tensor(0.0))

    def load_initial(self, state):
        self.encoder.weight.data.copy_(state["e.weight"])
        self.encoder.bias.data.copy_(state["e.bias"])
        self.decoder.weight.data.copy_(state["d.weight"])
        self.output_bias.data.copy_(state["b"])

    def initialize_fresh(self):
        with torch.no_grad():
            self.encoder.bias.zero_()
            self.output_bias.zero_()
            self.decoder.weight.copy_(self.encoder.weight.T)
        self.normalize_decoder()

    def set_threshold(self, value):
        self.threshold.data.fill_(value)

    def set_batchtopk_thresholds(self, c_value, s_value):
        self.c_threshold.fill_(c_value)
        self.s_threshold.fill_(s_value)
        self.global_threshold.fill_(c_value)

    @staticmethod
    def batch_topk(values, k):
        flat = F.relu(values).flatten()
        kept, indices = torch.topk(flat, min(k * len(values), flat.numel()))
        return torch.zeros_like(flat).scatter(0, indices, kept).view_as(values)

    def encode(self, x):
        pre = self.encoder(x)
        if self.activation == "jumprelu":
            z = JumpReLU.apply(pre, self.threshold, self.bandwidth)
            return z[:, : self.c_width], z[:, self.c_width :], pre
        if self.activation in ("silu", "softplus"):
            z = F.silu(pre) if self.activation == "silu" else F.softplus(pre)
            return z[:, : self.c_width], z[:, self.c_width :], pre
        if self.activation == "batchtopk":
            if self.sparsifier == "global":
                if self.training:
                    z = self.batch_topk(pre, self.total_k)
                else:
                    z = F.relu(pre)
                    z = z * (z >= self.global_threshold)
                return z[:, : self.c_width], z[:, self.c_width :], pre
            c, s = pre[:, : self.c_width], pre[:, self.c_width :]
            if self.training:
                return self.batch_topk(c, self.c_k), self.batch_topk(s, self.s_k), pre
            c, s = F.relu(c), F.relu(s)
            return c * (c >= self.c_threshold), s * (s >= self.s_threshold), pre
        c, s = F.relu(pre[:, : self.c_width]), F.relu(pre[:, self.c_width :])
        cv, ci = torch.topk(c, self.c_k, dim=1)
        sv, si = torch.topk(s, self.s_k, dim=1)
        c = torch.zeros_like(c).scatter(1, ci, cv)
        s = torch.zeros_like(s).scatter(1, si, sv)
        return c, s, pre

    def forward(self, x):
        c, s, pre = self.encode(x)
        return c, s, self.decode(c, s), pre

    def decode(self, c, s):
        return self.decoder(torch.cat((c, s), dim=1)) + self.output_bias

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            )

    def matryoshka_loss(self, c, s, target):
        z = torch.cat((c, s), dim=1)
        losses = [F.mse_loss(self.output_bias.expand_as(target), target)]
        groups = (
            self.sparse_width // 16, self.sparse_width // 16,
            self.sparse_width // 8, self.sparse_width // 4,
            self.sparse_width // 2,
        )
        for end in np.cumsum(groups):
            reconstruction = F.linear(
                z[:, :end], self.decoder.weight[:, :end], self.output_bias
            )
            losses.append(F.mse_loss(reconstruction, target))
        return torch.stack(losses).mean()


def tensor(raw, rows, mean, std):
    values = (np.asarray(raw[rows], np.float32) - mean) / std
    return torch.from_numpy(values).to(DEVICE)


def calibrate_threshold(model, raw, mean, std, rows, batch_size):
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            pre = F.relu(model.encoder(tensor(raw, rows[start : start + batch_size], mean, std)))
            values.append(torch.topk(pre, model.total_k, dim=1).values[:, -1].cpu())
    return float(torch.cat(values).median().clamp_min(1e-4))


def calibrate_batchtopk_thresholds(model, raw, mean, std, rows, batch_size):
    c_values, s_values = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            pre = model.encoder(tensor(raw, rows[start : start + batch_size], mean, std))
            if model.sparsifier == "global":
                z = model.batch_topk(pre, model.total_k)
                c_values.append(z[z > 0].min().cpu())
                continue
            c = model.batch_topk(pre[:, : model.c_width], model.c_k)
            s = model.batch_topk(pre[:, model.c_width :], model.s_k)
            c_values.append(c[c > 0].min().cpu())
            s_values.append(s[s > 0].min().cpu())
    if model.sparsifier == "global":
        threshold = float(torch.stack(c_values).mean())
        return threshold, threshold
    return float(torch.stack(c_values).mean()), float(torch.stack(s_values).mean())


def paired_contrastive_loss(a, positive, temperature, mask=None, decoupled=False):
    a, positive = F.normalize(a, dim=1), F.normalize(positive, dim=1)
    logits = a @ positive.T / temperature
    target = torch.arange(len(a), device=a.device)
    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.inf)
    if not decoupled:
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    negatives = ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    if mask is not None:
        negatives &= mask
    assert torch.all(negatives.any(1)) and torch.all(negatives.any(0))
    forward = -logits.diag() + torch.logsumexp(logits.masked_fill(~negatives, -torch.inf), dim=1)
    backward = -logits.diag() + torch.logsumexp(logits.masked_fill(~negatives, -torch.inf), dim=0)
    return (forward.mean() + backward.mean()) / 2


def relation_loss(a, positive, negative, objective, temperature, mask=None, margin=0.2):
    if objective in ("symmetric_infonce", "symmetric_dcl", "masked_infonce", "masked_dcl"):
        return paired_contrastive_loss(
            a, positive, temperature,
            mask if objective.startswith("masked_") else None,
            decoupled=objective.endswith("_dcl"),
        )
    a, positive = F.normalize(a, dim=1), F.normalize(positive, dim=1)
    if objective == "infonce":
        target = torch.arange(len(a), device=a.device)
        return F.cross_entropy(a @ positive.T / temperature, target)
    negative = F.normalize(negative, dim=1)
    if objective == "triplet":
        positive_similarity = (a * positive).sum(1)
        negative_similarity = (a * negative).sum(1)
        return F.relu(margin + negative_similarity - positive_similarity).mean()
    logits = torch.stack(((a * positive).sum(1), (a * negative).sum(1)), dim=1)
    return F.cross_entropy(logits / temperature, torch.zeros(len(a), dtype=torch.long, device=a.device))


def false_negative_masks(ids, positive_ids, locales, positive_locales, device=DEVICE):
    diagonal = np.eye(len(ids), dtype=bool)
    return (
        torch.from_numpy((ids[:, None] != positive_ids[None, :]) | diagonal).to(device),
        torch.from_numpy((locales[:, None] != positive_locales[None, :]) | diagonal).to(device),
    )


def supervised_contrastive(values, labels, temperature):
    values = F.normalize(values, dim=1)
    logits = values @ values.T / temperature
    self_mask = ~torch.eye(len(values), dtype=torch.bool, device=values.device)
    positives = labels[:, None].eq(labels[None, :]) & self_mask
    assert torch.all(positives.any(1))
    logits = logits - logits.max(1, keepdim=True).values.detach()
    log_probability = logits - torch.logsumexp(
        logits.masked_fill(~self_mask, -torch.inf), dim=1, keepdim=True
    )
    return -(log_probability.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def train(model, raw, metadata, mean, std, args):
    order_rng = np.random.default_rng(args.seed)
    relation_rng = np.random.default_rng(args.seed + 1)
    id_codes, _ = pd.factorize(metadata.id, sort=True)
    locale_codes, _ = pd.factorize(metadata.locale, sort=True)
    intent_codes, _ = pd.factorize(metadata.intent, sort=True)
    intent_adversary = (
        nn.Linear(model.s_width, metadata.intent.nunique()).to(DEVICE)
        if args.intent_adversary_weight else None
    )
    relation_index = None
    grid = None
    locales = metadata.locale.nunique()
    if args.route_mode != "none" and args.relation_sampler == "intent_50_50":
        relation_index = relations.build_relation_index(metadata)
    elif args.route_mode != "none":
        grid = np.empty((metadata.id.nunique(), locales), np.int64)
        grid[id_codes, locale_codes] = np.arange(len(metadata))
        assert np.unique(grid).size == len(metadata)
    if model.activation == "jumprelu":
        ordinary = [parameter for name, parameter in model.named_parameters() if name != "threshold"]
        if intent_adversary is not None:
            ordinary.extend(intent_adversary.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": ordinary, "lr": args.lr}, {"params": [model.threshold], "lr": args.threshold_lr}],
            weight_decay=1e-4,
        )
    else:
        parameters = list(model.parameters())
        if intent_adversary is not None:
            parameters.extend(intent_adversary.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        order = order_rng.choice(
            len(metadata), min(args.anchors_per_epoch, len(metadata)), replace=False
        )
        totals = {key: 0.0 for key in ("loss", "reconstruction", "sae_objective", "swap", "c_relation", "s_relation", "intent_adversary", "l0")}
        steps = 0
        model.train()
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            ids, locales_here = id_codes[rows], locale_codes[rows]
            if args.route_mode == "none":
                c_positive = s_positive = None
            elif relation_index is None:
                c_positive = grid[
                    ids,
                    (locales_here + relation_rng.integers(1, locales, len(rows))) % locales,
                ]
                s_positive = grid[
                    (ids + relation_rng.integers(1, len(grid), len(rows))) % len(grid),
                    locales_here,
                ]
            else:
                c_positive, s_positive = relations.sample_intent_relations(
                    rows, relation_index, relation_rng, exact_id_positive_fraction=0.5
                )
            anchor = tensor(raw, rows, mean, std)
            zc, zs, reconstruction, pre = model(anchor)
            reconstruction_loss = F.mse_loss(reconstruction, anchor)
            sae_objective = (
                model.matryoshka_loss(zc, zs, anchor)
                if args.sae_objective == "matryoshka" else reconstruction_loss
            )
            swap_loss = reconstruction_loss.new_zeros(())
            c_loss = s_loss = reconstruction_loss.new_zeros(())
            if args.route_mode != "none":
                c_pos = tensor(raw, c_positive, mean, std)
                s_pos = tensor(raw, s_positive, mean, std)
                zcp, zsn, _ = model.encode(c_pos)
                zcn, zsp, _ = model.encode(s_pos)
            if args.route_mode != "none" and (args.swap_weight or args.objective == "supcon"):
                opposite = grid[id_codes[s_positive], locale_codes[c_positive]]
                other = tensor(raw, opposite, mean, std)
                zcb, zsb, _ = model.encode(other)
            if args.route_mode != "none" and args.objective == "supcon":
                combined_intents = torch.from_numpy(np.concatenate((
                    intent_codes[rows], intent_codes[c_positive],
                    intent_codes[s_positive], intent_codes[opposite],
                ))).to(DEVICE)
                combined_locales = torch.from_numpy(np.concatenate((
                    locale_codes[rows], locale_codes[c_positive],
                    locale_codes[s_positive], locale_codes[opposite],
                ))).to(DEVICE)
                c_loss = supervised_contrastive(
                    torch.cat((zc, zcp, zcn, zcb)), combined_intents, args.temperature
                )
                s_loss = supervised_contrastive(
                    torch.cat((zs, zsn, zsp, zsb)), combined_locales, args.temperature
                )
            elif args.route_mode != "none":
                masks = (None, None)
                if args.objective in ("masked_infonce", "masked_dcl"):
                    masks = false_negative_masks(
                        ids, id_codes[c_positive], locales_here, locale_codes[s_positive]
                    )
                c_loss = relation_loss(
                    zc, zcp, zcn, args.objective, args.temperature, masks[0],
                    args.triplet_margin,
                )
                if args.route_mode == "reciprocal":
                    s_loss = relation_loss(
                        zs, zsp, zsn, args.objective, args.temperature, masks[1],
                        args.triplet_margin,
                    )
            adversary_loss = reconstruction_loss.new_zeros(())
            if intent_adversary is not None:
                labels = torch.from_numpy(intent_codes[rows]).to(DEVICE)
                adversary_loss = F.cross_entropy(intent_adversary(ReverseGradient.apply(zs)), labels)
            if args.swap_weight:
                swap_loss = (
                    F.mse_loss(model.decode(zc, zsb), c_pos)
                    + F.mse_loss(model.decode(zcb, zs), s_pos)
                ) / 2
            l0 = ((zc != 0).sum(1) + (zs != 0).sum(1)).float().mean()
            if model.activation == "jumprelu":
                estimated_l0 = Step.apply(pre, model.threshold, model.bandwidth).sum(1).mean()
                sparsity_loss = args.l0_weight * estimated_l0 / model.total_k
            elif model.activation == "softplus":
                warmup = min((epoch + 1) / args.sparsity_warmup_epochs, 1.0)
                sparsity_loss = args.softplus_alpha * warmup * torch.cat((zc, zs), 1).sum(1).mean()
            else:
                sparsity_loss = reconstruction_loss.new_zeros(())
            route_loss = c_loss if args.route_mode == "c_only" else (c_loss + s_loss) / 2
            loss = (
                sae_objective + args.swap_weight * swap_loss
                + args.contrast_weight * route_loss
                + args.intent_adversary_weight * adversary_loss + sparsity_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model.activation == "jumprelu":
                with torch.no_grad():
                    model.threshold.clamp_(min=0)
            model.normalize_decoder()
            for key, value in (
                ("loss", loss), ("reconstruction", reconstruction_loss),
                ("sae_objective", sae_objective), ("swap", swap_loss),
                ("c_relation", c_loss), ("s_relation", s_loss),
                ("intent_adversary", adversary_loss), ("l0", l0),
            ):
                totals[key] += float(value.detach())
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        row = {key: value / steps for key, value in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if args.max_steps:
            break
    return history


def encode(model, raw, mean, std, batch_size):
    c, s = [], []
    squared_error = total_values = c_l0 = s_l0 = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            rows = slice(start, min(start + batch_size, len(raw)))
            x = tensor(raw, rows, mean, std)
            zc, zs, reconstruction, _ = model(x)
            c.append(zc.cpu().numpy())
            s.append(zs.cpu().numpy())
            squared_error += F.mse_loss(reconstruction, x, reduction="sum").item()
            total_values += x.numel()
            c_l0 += (zc != 0).sum().item()
            s_l0 += (zs != 0).sum().item()
    return np.concatenate(c), np.concatenate(s), {
        "standardized_reconstruction_mse": squared_error / total_values,
        "mean_l0_zC": c_l0 / len(raw),
        "mean_l0_zS": s_l0 / len(raw),
        "mean_l0_total": (c_l0 + s_l0) / len(raw),
    }


def retrieval(values, ids, locales, holdout=HOLDOUT):
    left, right = np.where(locales == holdout[0])[0], np.where(locales == holdout[1])[0]
    query = F.normalize(torch.from_numpy(values[left]), dim=1).numpy()
    key = F.normalize(torch.from_numpy(values[right]), dim=1).numpy()
    order = np.argsort(-(query @ key.T), axis=1)
    ranks = np.array([np.where(ids[right][row] == ids[left][i])[0][0] + 1 for i, row in enumerate(order)])
    return {"R@1": float((ranks == 1).mean()), "R@5": float((ranks <= 5).mean()), "MRR": float((1 / ranks).mean())}


def locale_probe(values, locales):
    order = np.random.default_rng(EVAL_SEED).permutation(len(values))
    left, right = np.array_split(order, 2)
    scaler = StandardScaler().fit(values[left])
    probe = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=EVAL_SEED)
    probe.fit(scaler.transform(values[left]), locales[left])
    return float((probe.predict(scaler.transform(values[right])) == locales[right]).mean())


def feature_summary(train_values, train_meta, test_values, test_meta, holdout=HOLDOUT):
    mean = train_values.mean(0)
    active = mean > 1e-6
    scale = train_values.std(0).clip(1e-6)
    intent = train_meta.intent.to_numpy()
    locale = train_meta.locale.to_numpy()
    intent_score = np.stack([train_values[intent == value].mean(0) for value in np.unique(intent)]).max(0) - mean
    locale_score = np.stack([train_values[locale == value].mean(0) for value in np.unique(locale)]).max(0) - mean
    intent_score, locale_score = intent_score / scale, locale_score / scale
    intent_features = active & (intent_score > 1.1 * locale_score)
    locale_features = active & (locale_score > 1.1 * intent_score)
    left = {str(test_meta.id.iloc[i]): i for i in np.where(test_meta.locale.to_numpy() == holdout[0])[0]}
    right = {str(test_meta.id.iloc[i]): i for i in np.where(test_meta.locale.to_numpy() == holdout[1])[0]}
    ids = sorted(left.keys() & right.keys())
    x = test_values[[left[key] for key in ids]][:, active].astype(np.float64)
    y = test_values[[right[key] for key in ids]][:, active].astype(np.float64)
    valid = (x.std(0) > 1e-6) & (y.std(0) > 1e-6)
    x, y = x[:, valid], y[:, valid]
    x = (x - x.mean(0)) / x.std(0).clip(1e-6)
    y = (y - y.mean(0)) / y.std(0).clip(1e-6)
    contributions = (x * y).mean(1)
    return {
        "active_features": int(active.sum()),
        "stable_active_features": int(valid.sum()),
        "intent_oriented_fraction": float(intent_features.sum() / active.sum()),
        "locale_oriented_fraction": float(locale_features.sum() / active.sum()),
        "cross_locale_stability": float(contributions.mean()),
    }, ids, contributions


def swap_reconstruction(model, raw, metadata, mean, std, batch_size):
    id_codes, _ = pd.factorize(metadata.id, sort=True)
    locale_codes, _ = pd.factorize(metadata.locale, sort=True)
    assert metadata.locale.nunique() == 2
    grid = np.empty((metadata.id.nunique(), 2), np.int64)
    grid[id_codes, locale_codes] = np.arange(len(metadata))
    first_ids = np.arange(len(grid))
    second_ids = np.roll(first_ids, 1)
    target_error = wrong_style_error = wrong_content_error = values = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(grid), batch_size):
            ia, ib = first_ids[start : start + batch_size], second_ids[start : start + batch_size]
            a = tensor(raw, grid[ia, 0], mean, std)
            b = tensor(raw, grid[ib, 1], mean, std)
            target_ab = tensor(raw, grid[ia, 1], mean, std)
            target_ba = tensor(raw, grid[ib, 0], mean, std)
            zca, zsa, _ = model.encode(a)
            zcb, zsb, _ = model.encode(b)
            swapped_ab, swapped_ba = model.decode(zca, zsb), model.decode(zcb, zsa)
            for output, target, wrong_style, wrong_content in (
                (swapped_ab, target_ab, a, b), (swapped_ba, target_ba, b, a)
            ):
                target_error += F.mse_loss(output, target, reduction="sum").item()
                wrong_style_error += F.mse_loss(output, wrong_style, reduction="sum").item()
                wrong_content_error += F.mse_loss(output, wrong_content, reduction="sum").item()
                values += output.numel()
    target_mse = target_error / values
    return {
        "target_mse": target_mse,
        "wrong_style_mse": wrong_style_error / values,
        "wrong_content_mse": wrong_content_error / values,
        "style_margin": wrong_style_error / values - target_mse,
        "content_margin": wrong_content_error / values - target_mse,
    }


def evaluate(model, raw_train, raw_test, train_meta, test_meta, mean, std, args, holdout=HOLDOUT):
    test_c, test_s, reconstruction = encode(model, raw_test, mean, std, args.eval_batch_size)
    sample_rows = np.random.default_rng(EVAL_SEED).choice(len(train_meta), min(args.audit_rows, len(train_meta)), replace=False)
    sample_c, sample_s, _ = encode(model, raw_train[sample_rows], mean, std, args.eval_batch_size)
    sampled_meta = train_meta.iloc[sample_rows].reset_index(drop=True)
    c_summary, ids, contributions = feature_summary(sample_c, sampled_meta, test_c, test_meta, holdout)
    s_summary, _, _ = feature_summary(sample_s, sampled_meta, test_s, test_meta, holdout)
    locales, semantic_ids = test_meta.locale.to_numpy(), test_meta.id.to_numpy()
    return {
        **reconstruction,
        "swap_reconstruction": swap_reconstruction(model, raw_test, test_meta, mean, std, args.eval_batch_size),
        "zC": {"retrieval": retrieval(test_c, semantic_ids, locales, holdout), "locale_probe": locale_probe(test_c, locales), **c_summary},
        "zS": {"retrieval": retrieval(test_s, semantic_ids, locales, holdout), "locale_probe": locale_probe(test_s, locales), **s_summary},
    }, ids, contributions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=("jumprelu", "topk", "batchtopk", "silu", "softplus"), required=True)
    parser.add_argument(
        "--objective",
        choices=("reconstruction", "matched", "triplet", "infonce", "symmetric_infonce", "symmetric_dcl", "masked_infonce", "masked_dcl", "supcon"),
        required=True,
    )
    parser.add_argument("--route-mode", choices=("none", "c_only", "reciprocal"), default="reciprocal")
    parser.add_argument("--sparsifier", choices=("block", "global"), default="block")
    parser.add_argument("--sae-objective", choices=("standard", "matryoshka"), default="standard")
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--initialization", choices=("pretrained", "fresh"), default="pretrained")
    parser.add_argument("--c-fraction", type=float, default=0.5)
    parser.add_argument(
        "--relation-sampler", choices=("exact_id", "intent_50_50"), default="exact_id"
    )
    parser.add_argument("--total-k", type=int, default=TOTAL_K)
    parser.add_argument("--c-k", type=int)
    parser.add_argument("--sparse-width", type=int, default=SPARSE_WIDTH)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--anchors-per-epoch", type=int, default=45968)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--audit-rows", type=int, default=5000)
    parser.add_argument("--calibration-rows", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold-lr", type=float, default=1e-3)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--intent-adversary-weight", type=float, default=0.0)
    parser.add_argument("--swap-weight", type=float, default=0.0)
    parser.add_argument("--l0-weight", type=float, default=0.1)
    parser.add_argument("--softplus-alpha", type=float, default=1e-3)
    parser.add_argument("--sparsity-warmup-epochs", type=int, default=5)
    parser.add_argument("--bandwidth", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--evaluation-split", choices=("test", "validation", "locale_holdout"), default="test")
    parser.add_argument("--evaluation-locales", nargs=2)
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert 0 < args.c_fraction < 1
    assert 1 < args.total_k < args.sparse_width
    assert args.sparse_width >= WIDTH
    assert args.c_k is None or (args.activation in ("topk", "batchtopk") and 0 < args.c_k < args.total_k)
    assert args.intent_adversary_weight >= 0
    assert not args.intent_adversary_weight or args.objective == "supcon"
    assert args.relation_sampler != "intent_50_50" or not args.swap_weight
    assert (args.route_mode == "none") == (args.contrast_weight == 0)
    assert (args.route_mode == "none") == (args.objective == "reconstruction")
    assert args.route_mode == "none" or args.sparsifier == "block"
    assert args.sae_objective != "matryoshka" or (
        args.route_mode == "none" and args.sparsifier == "global"
    )
    assert args.objective != "triplet" or args.triplet_margin > 0
    assert args.softplus_alpha >= 0 and args.sparsity_warmup_epochs > 0
    assert args.threshold_scale > 0
    assert not args.eval_only or args.load_checkpoint
    assert 0 <= args.validation_fraction < 1
    assert args.evaluation_split != "validation" or args.validation_fraction > 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    initial = load_initial_checkpoint()
    mean, std = np.asarray(initial["mean"]), np.asarray(initial["std"])
    all_train_meta = pd.read_csv(ART / "train_metadata.csv")
    all_raw_train = np.load(ART / "raw_train_layer8.npy", mmap_mode="r")
    holdout = tuple(args.evaluation_locales or ("en-US", "ja-JP"))
    if args.evaluation_split == "locale_holdout":
        train_rows, evaluation_rows = locale_holdout_split(all_train_meta, holdout)
        validation_ids = np.array([], dtype=all_train_meta.id.dtype)
    else:
        train_rows, validation_ids = semantic_validation_split(all_train_meta, args.validation_fraction)
    train_meta = all_train_meta.iloc[train_rows].reset_index(drop=True)
    raw_train = RowView(all_raw_train, train_rows)
    if args.evaluation_split == "validation":
        evaluation_rows = np.flatnonzero(
            all_train_meta.id.isin(validation_ids).to_numpy()
            & all_train_meta.locale.isin(holdout).to_numpy()
        )
        test_meta = all_train_meta.iloc[evaluation_rows].reset_index(drop=True)
        raw_test = RowView(all_raw_train, evaluation_rows)
    elif args.evaluation_split == "locale_holdout":
        test_meta = all_train_meta.iloc[evaluation_rows].reset_index(drop=True)
        raw_test = RowView(all_raw_train, evaluation_rows)
    else:
        test_meta = pd.read_csv(ART / "test_metadata.csv")
        raw_test = np.load(ART / "raw_test_layer8.npy", mmap_mode="r")
        holdout = tuple(args.evaluation_locales or HOLDOUT)
    assert set(test_meta.locale.unique()) == set(holdout)
    assert test_meta.groupby("locale").id.nunique().nunique() == 1
    model = SparsePartition(
        args.activation, args.c_fraction, args.bandwidth, args.total_k, args.c_k,
        args.sparsifier, args.sparse_width,
    ).to(DEVICE)
    if args.initialization == "pretrained":
        model.load_initial(initial["state_dict"])
    else:
        model.initialize_fresh()
    if args.load_checkpoint:
        loaded = torch.load(ROOT / args.load_checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(loaded["state_dict"])
    threshold = None
    if args.activation == "jumprelu":
        rows = np.random.default_rng(args.seed).choice(len(train_meta), args.calibration_rows, replace=False)
        threshold = args.threshold_scale * calibrate_threshold(model, raw_train, mean, std, rows, args.eval_batch_size)
        model.set_threshold(threshold)
    with torch.inference_mode():
        check = tensor(raw_train, np.arange(2), mean, std)
        c, s, reconstruction, _ = model(check)
        assert c.shape == (2, model.c_width) and s.shape == (2, model.s_width)
        assert reconstruction.shape == check.shape
        assert torch.isfinite(c).all() and torch.isfinite(s).all()
        assert (c != 0).sum() + (s != 0).sum() > 0
        if args.activation == "topk":
            assert torch.all((c != 0).sum(1) <= model.c_k) and torch.all((s != 0).sum(1) <= model.s_k)
    history = [] if args.eval_only else train(model, raw_train, train_meta, mean, std, args)
    batchtopk_thresholds = None
    if args.activation == "batchtopk" and not args.eval_only:
        rows = np.random.default_rng(args.seed).choice(len(train_meta), args.calibration_rows, replace=False)
        batchtopk_thresholds = calibrate_batchtopk_thresholds(
            model, raw_train, mean, std, rows, len(rows)
        )
        model.set_batchtopk_thresholds(*batchtopk_thresholds)
    tag = f"{args.activation}_{args.objective}_c{round(100 * args.c_fraction)}_seed{args.seed}"
    if args.sparse_width != SPARSE_WIDTH:
        tag += f"_w{args.sparse_width}"
    if args.total_k != TOTAL_K:
        tag += f"_k{args.total_k}"
    if args.c_k is not None:
        tag += f"_ck{args.c_k}"
    if args.sparsifier == "global":
        tag += "_global"
    if args.sae_objective == "matryoshka":
        tag += "_matryoshka"
    if args.route_mode != "reciprocal":
        tag += f"_{args.route_mode}"
    if args.objective == "triplet":
        tag += f"_m{args.triplet_margin:g}".replace(".", "p")
    if args.swap_weight:
        tag += f"_swap{args.swap_weight:g}".replace(".", "p")
    if args.l0_weight != 0.1:
        tag += f"_l0{args.l0_weight:g}".replace(".", "p")
    if args.threshold_scale != 1:
        tag += f"_ts{args.threshold_scale:g}".replace(".", "p")
    if args.validation_fraction:
        tag += f"_val{100 * args.validation_fraction:g}".replace(".", "p")
    if args.evaluation_split == "locale_holdout":
        tag += "_lh" + "_".join(locale.split("-")[0] for locale in holdout)
    if args.intent_adversary_weight:
        tag += f"_adv{args.intent_adversary_weight:g}".replace(".", "p")
    if args.label:
        tag += f"_{args.label}"
    if args.max_steps:
        tag += "_smoke"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / args.load_checkpoint if args.eval_only else CHECKPOINTS / f"{tag}.pt"
    if not args.eval_only:
        torch.save({
            "state_dict": model.state_dict(), "input_mean": mean, "input_std": std,
            "config": vars(args) | {"input_width": WIDTH,
            "c_width": model.c_width, "s_width": model.s_width, "initial_threshold": threshold,
            "batchtopk_inference_thresholds": batchtopk_thresholds,
            "threshold_parameterization": "direct, matching installed SAE Lens 6.49+",
            "initialization": args.initialization,
            "normalization_source": str(INIT.relative_to(ROOT))}, "history": history,
        }, checkpoint)
    report = {
        "status": "smoke" if args.max_steps else "pilot",
        "tag": tag,
        "protocol": vars(args) | {"input_width": WIDTH,
        "c_width": model.c_width, "s_width": model.s_width, "c_k": model.c_k,
        "s_k": model.s_k, "initial_threshold": threshold,
        "training_semantic_ids": int(train_meta.id.nunique()),
        "validation_semantic_ids": int(len(validation_ids)),
        "evaluation_locales": list(holdout),
        "batchtopk_inference_thresholds": batchtopk_thresholds,
        "threshold_parameterization": "direct, matching installed SAE Lens 6.49+",
        "initialization": args.initialization,
        "normalization_source": str(INIT.relative_to(ROOT)),
        "relation_source": (
            "none; reconstruction-only SAE control" if args.route_mode == "none"
            else "intent-level controlled relation on z_C only" if args.route_mode == "c_only"
            else "intent-level controlled reciprocal relations on z_C and z_S"
        )},
        "history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    if args.route_mode != "none" and args.relation_sampler == "intent_50_50":
        audit_index = relations.build_relation_index(train_meta)
        audit_rows = np.random.default_rng(args.seed).choice(
            len(train_meta), min(10000, len(train_meta)), replace=False
        )
        audit_c, audit_s = relations.sample_intent_relations(
            audit_rows, audit_index, np.random.default_rng(args.seed), 0.5
        )
        report["relation_audit"] = relations.audit_sample(
            audit_rows, audit_c, audit_s, audit_index
        )
    if not args.skip_eval:
        report["evaluation"], ids, contributions = evaluate(
            model, raw_train, raw_test, train_meta, test_meta, mean, std, args, holdout
        )
        csv_path = REPORTS / f"{tag}_zC_per_id.csv"
        pd.DataFrame({"id": ids, "stability_contribution": contributions}).to_csv(csv_path, index=False)
        report["per_id_stability"] = str(csv_path.relative_to(ROOT))
    report_path = REPORTS / f"{tag}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in ("history",)}, indent=2))


if __name__ == "__main__":
    main()
