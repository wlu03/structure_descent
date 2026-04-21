"""Tests for :mod:`src.eval.metrics` (redesign.md §13).

Covers: Top-1 / Top-5 accuracy, MRR (1/(rank+1)), NLL (natural log),
AIC (``2k + 2 n_train * NLL``), BIC (``k log n_train + 2 n_train * NLL``),
the aggregator :class:`EvalMetrics`, and numpy-array acceptance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.eval.metrics import (
    EvalMetrics,
    aic,
    bic,
    compute_all,
    mrr,
    nll,
    topk_accuracy,
)


# ---------------------------------------------------------------------------
# Top-k
# ---------------------------------------------------------------------------


def test_top1_trivial_correct() -> None:
    """argmax == c* for every row → top-1 == 1.0."""
    N, J = 6, 10
    torch.manual_seed(0)
    logits = torch.randn(N, J)
    c_star = logits.argmax(dim=-1)
    assert topk_accuracy(logits, c_star, k=1) == pytest.approx(1.0)


def test_top1_trivial_wrong() -> None:
    """c* at the strictly lowest logit for every row → top-1 == 0.0."""
    N, J = 6, 10
    torch.manual_seed(1)
    logits = torch.randn(N, J)
    c_star = logits.argmin(dim=-1)
    assert topk_accuracy(logits, c_star, k=1) == pytest.approx(0.0)


def test_top5_strictly_ge_top1() -> None:
    """Top-5 ≥ Top-1 always (for any k1 ≤ k2, top-k2 ≥ top-k1)."""
    N, J = 32, 10
    torch.manual_seed(2)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    t1 = topk_accuracy(logits, c_star, k=1)
    t5 = topk_accuracy(logits, c_star, k=5)
    assert t5 >= t1


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_single_top1_hit() -> None:
    """All predictions correct → MRR == 1.0 (rank=0 everywhere)."""
    N, J = 8, 10
    torch.manual_seed(3)
    logits = torch.randn(N, J)
    c_star = logits.argmax(dim=-1)
    assert mrr(logits, c_star) == pytest.approx(1.0)


def test_mrr_all_last() -> None:
    """c* always the strictly worst-ranked → MRR == 1/J."""
    N, J = 8, 10
    torch.manual_seed(4)
    logits = torch.randn(N, J)
    c_star = logits.argmin(dim=-1)
    assert mrr(logits, c_star) == pytest.approx(1.0 / J)


def test_mrr_rank_convention_is_one_over_rank_plus_one() -> None:
    """Pin the 1/(rank+1) convention: rank=1 → 0.5, rank=2 → 1/3."""
    # Two rows: c*=1 is second-best, c*=2 is third-best.
    logits = torch.tensor(
        [
            [0.0, 2.0, 3.0, 1.0],  # sorted desc: [2(3.0), 1(2.0), 3(1.0), 0(0)], c*=1 → rank 1
            [0.0, 2.0, 1.0, 3.0],  # sorted desc: [3(3.0), 1(2.0), 2(1.0), 0(0)], c*=2 → rank 2
        ]
    )
    c_star = torch.tensor([1, 2])
    expected = 0.5 * (1.0 / 2.0 + 1.0 / 3.0)
    assert mrr(logits, c_star) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------


def test_nll_matches_torch_cross_entropy() -> None:
    """NLL equals F.cross_entropy(..., reduction='mean')."""
    N, J = 16, 10
    torch.manual_seed(5)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    expected = F.cross_entropy(logits, c_star, reduction="mean").item()
    assert nll(logits, c_star) == pytest.approx(expected, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------------------------


def test_aic_formula() -> None:
    """AIC(nll=1, k=100, n_train=1000) = 2·100 + 2·1000·1 = 2200."""
    assert aic(nll_val=1.0, k=100, n_train=1000) == pytest.approx(2200.0)


def test_bic_formula() -> None:
    """BIC(nll=1, k=100, n_train=1000) = 100·ln(1000) + 2·1000·1."""
    expected = 100.0 * math.log(1000.0) + 2000.0
    assert bic(nll_val=1.0, k=100, n_train=1000) == pytest.approx(expected)
    # Sanity: ln(1000) ≈ 6.9078 → ≈ 2690.78
    assert expected == pytest.approx(100.0 * 6.9077553 + 2000.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_compute_all_populates() -> None:
    """All six metric fields are set and ``to_dict`` exposes them all."""
    N, J = 32, 10
    torch.manual_seed(6)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    em = compute_all(logits, c_star, n_params=544_779, n_train=10_000)

    assert isinstance(em, EvalMetrics)

    # Populated (non-None) and of the right Python types.
    for field in ("top1", "top5", "mrr_val", "nll_val", "aic_val", "bic_val"):
        assert isinstance(getattr(em, field), float), field
    for field in ("n_params", "n_train"):
        assert isinstance(getattr(em, field), int), field

    d = em.to_dict()
    assert set(d.keys()) == {
        "top1", "top5", "mrr_val", "nll_val",
        "aic_val", "bic_val", "n_params", "n_train",
    }

    # Cross-check via individual functions.
    assert d["top1"] == pytest.approx(topk_accuracy(logits, c_star, k=1))
    assert d["top5"] == pytest.approx(topk_accuracy(logits, c_star, k=5))
    assert d["mrr_val"] == pytest.approx(mrr(logits, c_star))
    assert d["nll_val"] == pytest.approx(nll(logits, c_star))
    assert d["aic_val"] == pytest.approx(
        aic(d["nll_val"], k=544_779, n_train=10_000)
    )
    assert d["bic_val"] == pytest.approx(
        bic(d["nll_val"], k=544_779, n_train=10_000)
    )


def test_accepts_numpy_array() -> None:
    """Passing np.ndarray inputs yields the same numbers as torch tensors."""
    N, J = 12, 10
    torch.manual_seed(7)
    logits_t = torch.randn(N, J)
    c_star_t = torch.randint(0, J, (N,))

    logits_np = logits_t.numpy().astype(np.float32)
    c_star_np = c_star_t.numpy().astype(np.int64)

    assert topk_accuracy(logits_np, c_star_np, k=1) == pytest.approx(
        topk_accuracy(logits_t, c_star_t, k=1)
    )
    assert topk_accuracy(logits_np, c_star_np, k=5) == pytest.approx(
        topk_accuracy(logits_t, c_star_t, k=5)
    )
    assert mrr(logits_np, c_star_np) == pytest.approx(mrr(logits_t, c_star_t))
    assert nll(logits_np, c_star_np) == pytest.approx(
        nll(logits_t, c_star_t), rel=1e-5, abs=1e-6
    )

    em_np = compute_all(logits_np, c_star_np, n_params=1000, n_train=500)
    em_t = compute_all(logits_t, c_star_t, n_params=1000, n_train=500)
    assert em_np.to_dict().keys() == em_t.to_dict().keys()
    assert em_np.nll_val == pytest.approx(em_t.nll_val, rel=1e-5, abs=1e-6)


def test_k_param_propagates_to_aic_bic() -> None:
    """compute_all(n_params=544_779, ...) → aic/bic reflect that k."""
    N, J = 8, 10
    torch.manual_seed(8)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))

    k_default = 544_779
    n_train = 12_345
    em = compute_all(logits, c_star, n_params=k_default, n_train=n_train)

    assert em.n_params == k_default
    assert em.n_train == n_train
    assert em.aic_val == pytest.approx(
        2.0 * k_default + 2.0 * n_train * em.nll_val
    )
    assert em.bic_val == pytest.approx(
        k_default * math.log(n_train) + 2.0 * n_train * em.nll_val
    )

    # Changing k must change AIC by 2·Δk and BIC by Δk·ln(n_train).
    em2 = compute_all(logits, c_star, n_params=k_default + 100, n_train=n_train)
    assert em2.aic_val - em.aic_val == pytest.approx(200.0)
    assert em2.bic_val - em.bic_val == pytest.approx(100.0 * math.log(n_train))
