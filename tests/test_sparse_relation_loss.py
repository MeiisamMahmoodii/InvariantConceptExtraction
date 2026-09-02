import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run_massive_sparse_partition_pilot import (
    RowView, false_negative_masks, locale_holdout_split, paired_contrastive_loss,
    semantic_validation_split,
)


def test_paired_contrastive_losses_are_symmetric_and_mask_false_negatives():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    positive = a.clone()
    mask = torch.tensor([
        [True, False, True],
        [False, True, True],
        [True, True, True],
    ])
    ordinary = paired_contrastive_loss(a, positive, 0.1)
    masked = paired_contrastive_loss(a, positive, 0.1, mask)
    reverse = paired_contrastive_loss(positive, a, 0.1, mask.T)
    dcl = paired_contrastive_loss(a, positive, 0.1, mask, decoupled=True)
    unmasked_dcl = paired_contrastive_loss(a, positive, 0.1, decoupled=True)
    assert masked < ordinary
    assert torch.allclose(masked, reverse)
    assert torch.isfinite(dcl)
    assert torch.isfinite(unmasked_dcl)


def test_semantic_validation_split_and_row_view_keep_groups_disjoint():
    metadata = pd.DataFrame({"id": np.repeat(np.arange(10), 2), "locale": ["a", "b"] * 10})
    train_rows, validation_ids = semantic_validation_split(metadata, 0.2)
    assert len(validation_ids) == 2
    assert not set(metadata.id.iloc[train_rows]) & set(validation_ids)
    view = RowView(np.arange(40).reshape(20, 2), train_rows)
    assert np.array_equal(view[:2], np.arange(40).reshape(20, 2)[train_rows[:2]])
    seen, heldout = locale_holdout_split(metadata, ("b",))
    assert set(metadata.locale.iloc[seen]) == {"a"}
    assert set(metadata.locale.iloc[heldout]) == {"b"}


def test_false_negative_masks_keep_pairs_and_exclude_equivalents():
    c_mask, s_mask = false_negative_masks(
        np.array([0, 0, 1]), np.array([0, 0, 1]),
        np.array([0, 1, 1]), np.array([0, 1, 1]), device="cpu",
    )
    assert c_mask.diag().all() and s_mask.diag().all()
    assert not c_mask[0, 1] and not c_mask[1, 0]
    assert not s_mask[1, 2] and not s_mask[2, 1]
