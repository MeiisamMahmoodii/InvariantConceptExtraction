import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_massive_sparse_partition_pilot import (
    MATRYOSHKA_GROUPS,
    SparsePartition,
    relation_loss,
)


def test_global_batchtopk_and_matryoshka_protocol():
    model = SparsePartition("batchtopk", 0.30, 0.05, total_k=4, c_k=1, sparsifier="global")
    model.train()
    x = torch.randn(3, 2304)
    c, s, reconstruction, _ = model(x)
    assert (c != 0).sum() + (s != 0).sum() == 12
    assert reconstruction.shape == x.shape
    assert sum(MATRYOSHKA_GROUPS) == 9216
    assert model.matryoshka_loss(c, s, x).ndim == 0


def test_fresh_initialization_ties_encoder_and_decoder():
    model = SparsePartition("batchtopk", 0.30, 0.05, sparsifier="block")
    model.initialize_fresh()
    expected = model.encoder.weight.T
    expected = expected / expected.norm(dim=0, keepdim=True)
    assert torch.allclose(model.decoder.weight, expected)
    assert torch.count_nonzero(model.encoder.bias) == 0
    assert torch.count_nonzero(model.output_bias) == 0


def test_sparse_width_and_batch_budget_are_configurable():
    model = SparsePartition(
        "batchtopk", 0.30, 0.05, total_k=8, c_k=2,
        sparsifier="block", sparse_width=32,
    )
    model.train()
    c, s, reconstruction, _ = model(torch.randn(3, 2304))
    assert c.shape == (3, 10) and s.shape == (3, 22)
    assert (c != 0).sum() == 6 and (s != 0).sum() == 18
    assert reconstruction.shape == (3, 2304)


def test_input_width_is_configurable_for_model_transfer():
    model = SparsePartition(
        "batchtopk", 0.30, 0.05, total_k=8, c_k=2,
        sparsifier="block", sparse_width=32, input_width=12,
    )
    model.train()
    c, s, reconstruction, _ = model(torch.randn(3, 12))
    assert c.shape == (3, 10) and s.shape == (3, 22)
    assert reconstruction.shape == (3, 12)


def test_representative_intent_requires_stability_and_support():
    from build_factor_stability_figure import representative_intent

    ours_auc = np.array([.99, .80, .90, .85])
    ours_stability = np.array([-.5, .1, .8, .7])
    control_auc = np.array([.98, .1, .6, .7])
    control_stability = np.array([-.6, 0, .2, .65])
    support = np.array([10, 2, 10, 10])
    assert representative_intent(
        ours_auc, ours_stability, control_auc, control_stability, support
    ) == 2


def test_feature_interpretability_helpers():
    from evaluate_feature_interpretability import locale_entropy, mean_jaccard

    assert locale_entropy(["ar", "zh", "ar", "zh"]) == 1.0
    assert locale_entropy(["ar", "ar"]) == 0.0
    assert mean_jaccard([{"a", "b"}, {"b", "c"}]) == 1 / 3


def test_triplet_loss_enforces_the_requested_margin():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    easy_negative = torch.tensor([[0.0, 1.0]])
    hard_negative = torch.tensor([[1.0, 0.0]])
    assert relation_loss(anchor, positive, easy_negative, "triplet", 0.07, margin=0.2) == 0
    assert torch.isclose(
        relation_loss(anchor, positive, hard_negative, "triplet", 0.07, margin=0.2),
        torch.tensor(0.2),
    )
