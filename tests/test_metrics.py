"""Tests for :mod:`src.eval.metrics` (redesign.md §13).

Covers: Top-1 / Top-5 accuracy, MRR (1/(rank+1)), NLL (natural log),
AIC (``2k + 2 n_train * NLL``), BIC (``k log n_train + 2 n_train * NLL``),
McFadden's pseudo-R² (``1 - NLL/log(J)``), top-1 ECE, the aggregator
:class:`EvalMetrics`, and numpy-array acceptance.
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
    brier,
    compute_all,
    ece,
    mcfadden_pseudo_r2,
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


def test_top3_monotone_between_top1_and_top5() -> None:
    """Top-1 ≤ Top-3 ≤ Top-5 by definition of top-k."""
    N, J = 64, 10
    torch.manual_seed(11)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    t1 = topk_accuracy(logits, c_star, k=1)
    t3 = topk_accuracy(logits, c_star, k=3)
    t5 = topk_accuracy(logits, c_star, k=5)
    assert t1 <= t3 <= t5


def test_top3_hand_calc() -> None:
    """Hand-built logits — c* is rank 0/2/3/9 across 4 events; top-3 = 3/4."""
    # Event 0: c* at top → in top-3.       (rank 0)
    # Event 1: c* at rank 2 → in top-3.    (rank 2)
    # Event 2: c* at rank 3 → NOT in top-3.(rank 3)
    # Event 3: c* at rank 9 → NOT in top-3.(rank 9)
    # Expected top-3 = 2/4 = 0.5.
    logits = torch.tensor(
        [
            [9, 1, 2, 3, 4, 5, 6, 7, 8, 0],   # argmax=0; c*=0 (rank 0)
            [9, 8, 7, 1, 2, 3, 4, 5, 6, 0],   # ranking: 0,1,2 → c*=2 (rank 2)
            [9, 8, 7, 6, 1, 2, 3, 4, 5, 0],   # top-3 = {0,1,2}; c*=3 (rank 3)
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],   # top-3 = {0,1,2}; c*=9 (rank 9)
        ],
        dtype=torch.float32,
    )
    c_star = torch.tensor([0, 2, 3, 9])
    assert topk_accuracy(logits, c_star, k=3) == pytest.approx(0.5)


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
# Brier (multi-class MSE on softmax probabilities)
# ---------------------------------------------------------------------------


def test_brier_perfect_prediction_is_zero() -> None:
    """All mass on c* every event → Brier == 0 (lower bound)."""
    # Build logits where the c*-th column is +inf-ish — softmax becomes
    # one-hot at c*, identical to the label, so (p - y)² == 0 everywhere.
    N, J = 8, 5
    torch.manual_seed(20)
    c_star = torch.randint(0, J, (N,))
    logits = torch.full((N, J), -100.0)
    logits.scatter_(1, c_star.unsqueeze(-1), 100.0)
    assert brier(logits, c_star) == pytest.approx(0.0, abs=1e-6)


def test_brier_uniform_prediction() -> None:
    """Uniform softmax (all-equal logits) → Brier == 1 - 1/J.

    Closed form: with p_j = 1/J for every j, the per-event sum
    Σ_j (p_j - y_j)² = (J-1) · (1/J)² + (1 - 1/J)² = 1 - 1/J.
    """
    N, J = 16, 4
    logits = torch.zeros(N, J)
    c_star = torch.randint(0, J, (N,))
    expected = 1.0 - 1.0 / J  # = 0.75 at J=4
    assert brier(logits, c_star) == pytest.approx(expected, rel=1e-6)


def test_brier_all_mass_on_one_wrong_class() -> None:
    """All mass on the same wrong class every event → Brier == 2 (upper bound)."""
    # Put +inf-ish weight on column 0 for every event; pick c* = J-1 so it
    # never coincides. Per-event sum = (1-0)² + (0-1)² = 2.
    N, J = 6, 5
    logits = torch.full((N, J), -100.0)
    logits[:, 0] = 100.0
    c_star = torch.full((N,), J - 1, dtype=torch.int64)
    assert brier(logits, c_star) == pytest.approx(2.0, abs=1e-6)


def test_brier_matches_manual_computation() -> None:
    """Cross-check against an explicit (softmax → one-hot → MSE) loop."""
    N, J = 10, 6
    torch.manual_seed(21)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    probs = F.softmax(logits, dim=-1).to(torch.float64)
    one_hot = F.one_hot(c_star, num_classes=J).to(torch.float64)
    expected = (probs - one_hot).pow(2).sum(dim=-1).mean().item()
    assert brier(logits, c_star) == pytest.approx(expected, rel=1e-9, abs=1e-9)


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
    """All metric fields are set and ``to_dict`` exposes them all."""
    N, J = 32, 10
    torch.manual_seed(6)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    em = compute_all(logits, c_star, n_params=544_779, n_train=10_000)

    assert isinstance(em, EvalMetrics)

    # Populated (non-None) and of the right Python types.
    for field in (
        "top1", "top3", "top5", "mrr_val", "nll_val", "brier_val",
        "aic_val", "bic_val", "pseudo_r2", "ece_val",
    ):
        assert isinstance(getattr(em, field), float), field
    for field in ("n_params", "n_train"):
        assert isinstance(getattr(em, field), int), field

    d = em.to_dict()
    assert set(d.keys()) == {
        "top1", "top3", "top5", "mrr_val", "nll_val", "brier_val",
        "aic_val", "bic_val", "pseudo_r2", "ece_val",
        "n_params", "n_train",
    }

    # Cross-check via individual functions.
    assert d["top1"] == pytest.approx(topk_accuracy(logits, c_star, k=1))
    assert d["top3"] == pytest.approx(topk_accuracy(logits, c_star, k=3))
    assert d["top5"] == pytest.approx(topk_accuracy(logits, c_star, k=5))
    assert d["mrr_val"] == pytest.approx(mrr(logits, c_star))
    assert d["nll_val"] == pytest.approx(nll(logits, c_star))
    assert d["brier_val"] == pytest.approx(brier(logits, c_star))
    assert d["aic_val"] == pytest.approx(
        aic(d["nll_val"], k=544_779, n_train=10_000)
    )
    assert d["bic_val"] == pytest.approx(
        bic(d["nll_val"], k=544_779, n_train=10_000)
    )
    assert d["pseudo_r2"] == pytest.approx(
        mcfadden_pseudo_r2(d["nll_val"], J=J)
    )
    assert d["ece_val"] == pytest.approx(ece(logits, c_star))


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


# ---------------------------------------------------------------------------
# McFadden pseudo-R²
# ---------------------------------------------------------------------------


def test_pseudo_r2_uniform_is_zero() -> None:
    """NLL == log(J) (uniform prediction) → pseudo-R² == 0."""
    for J in (2, 5, 10, 100):
        assert mcfadden_pseudo_r2(math.log(J), J=J) == pytest.approx(0.0)


def test_pseudo_r2_perfect_is_one() -> None:
    """NLL == 0 (perfect prediction) → pseudo-R² == 1."""
    for J in (2, 5, 10, 100):
        assert mcfadden_pseudo_r2(0.0, J=J) == pytest.approx(1.0)


def test_pseudo_r2_intermediate_hand_calc() -> None:
    """Hand-computed intermediate value: J=10, NLL=1.5 → 1 - 1.5/ln(10)."""
    J = 10
    nll_v = 1.5
    expected = 1.0 - 1.5 / math.log(10.0)
    assert mcfadden_pseudo_r2(nll_v, J=J) == pytest.approx(expected)
    # Spot check the magnitude: ln(10) ≈ 2.3026 → 1 - 0.6514 ≈ 0.3486.
    assert 0.34 < mcfadden_pseudo_r2(nll_v, J=J) < 0.36


def test_pseudo_r2_can_be_negative_when_worse_than_uniform() -> None:
    """NLL > log(J) → pseudo-R² < 0 (model is worse than uniform)."""
    J = 10
    # A model producing ~2x uniform NLL.
    assert mcfadden_pseudo_r2(2.0 * math.log(J), J=J) == pytest.approx(-1.0)
    assert mcfadden_pseudo_r2(5.0, J=J) < 0.0


def test_pseudo_r2_upper_bound_one() -> None:
    """NLL is non-negative → pseudo-R² ≤ 1 always."""
    for J in (2, 5, 10, 100):
        for nll_v in (0.0, 0.1, math.log(J), 2.0 * math.log(J)):
            assert mcfadden_pseudo_r2(nll_v, J=J) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def _logits_for_confidence(conf: float, J: int, n: int) -> torch.Tensor:
    """Build an (n, J) logits matrix whose softmax max-prob is exactly ``conf``
    (the non-argmax classes share the remaining mass uniformly). Used to
    construct ECE synthetic data without having to invert softmax twice."""
    assert 1.0 / J <= conf <= 1.0 - 1e-9
    # softmax(a, 0, 0, ...) = [e^a, 1, 1, ...] / (e^a + J - 1).
    # Solve for a: e^a / (e^a + J - 1) = conf
    # → e^a (1 - conf) = conf (J - 1)
    # → a = log(conf (J-1) / (1 - conf)).
    a = math.log(conf * (J - 1) / (1.0 - conf))
    row = torch.zeros(J)
    row[0] = a
    return row.unsqueeze(0).repeat(n, 1)


def test_ece_perfectly_calibrated() -> None:
    """100 events at 0.8 confidence with 80 correct → ECE == 0."""
    J = 5
    conf = 0.8
    n = 100
    logits = _logits_for_confidence(conf, J=J, n=n)
    # Class 0 is the top-1 for every row (see helper). Mark exactly 80
    # correct by setting c*=0 on 80 rows and c*=1 on the rest.
    c_star = torch.zeros(n, dtype=torch.int64)
    c_star[80:] = 1
    # Sanity: pred is 0 for every row → 80 correct → accuracy 0.8.
    assert ece(logits, c_star) == pytest.approx(0.0, abs=1e-6)


def test_ece_systematically_overconfident() -> None:
    """100 events at 0.9 confidence but only 50 correct → ECE ≈ 0.4."""
    J = 5
    conf = 0.9
    n = 100
    logits = _logits_for_confidence(conf, J=J, n=n)
    c_star = torch.zeros(n, dtype=torch.int64)
    c_star[50:] = 1  # 50 correct, 50 wrong
    # All rows land in the same bin, so ECE == |avg_conf - acc| = |0.9 - 0.5|.
    assert ece(logits, c_star) == pytest.approx(0.4, abs=1e-5)


def test_ece_empty_input_returns_zero() -> None:
    """N == 0 → ECE == 0.0 (no divide-by-zero)."""
    logits = torch.empty((0, 5))
    c_star = torch.empty((0,), dtype=torch.int64)
    assert ece(logits, c_star) == 0.0


def test_ece_single_event_does_not_crash() -> None:
    """N == 1 is a valid input; the metric is just |conf - correct|."""
    J = 4
    logits = _logits_for_confidence(0.7, J=J, n=1)
    # Correct prediction → ECE == |0.7 - 1.0| = 0.3.
    assert ece(logits, torch.tensor([0])) == pytest.approx(0.3, abs=1e-6)
    # Wrong prediction → ECE == |0.7 - 0.0| = 0.7.
    assert ece(logits, torch.tensor([1])) == pytest.approx(0.7, abs=1e-6)


def test_ece_skips_empty_bins() -> None:
    """Bins with no events must not contribute (and must not divide by 0)."""
    # All events at confidence 0.8 → only bin 12 of 15 is populated
    # (0.8 * 15 = 12; int() truncates; clamp to 14 is a no-op). All 14
    # other bins are empty; ECE must still evaluate cleanly.
    J = 5
    n = 20
    logits = _logits_for_confidence(0.8, J=J, n=n)
    c_star = torch.zeros(n, dtype=torch.int64)  # all correct
    # |0.8 - 1.0| = 0.2, single populated bin → ECE = 0.2.
    val = ece(logits, c_star, n_bins=15)
    assert val == pytest.approx(0.2, abs=1e-6)


def test_ece_bounded_in_unit_interval() -> None:
    """ECE is a weighted mean of |conf - acc| with both in [0,1] → ECE ∈ [0,1]."""
    N, J = 64, 8
    torch.manual_seed(9)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    v = ece(logits, c_star)
    assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# dtype / container tolerance for the new metrics
# ---------------------------------------------------------------------------


def test_ece_numpy_matches_torch() -> None:
    """np.ndarray inputs give the same ECE as torch tensors."""
    N, J = 32, 6
    torch.manual_seed(10)
    logits_t = torch.randn(N, J)
    c_star_t = torch.randint(0, J, (N,))

    logits_np = logits_t.numpy().astype(np.float32)
    c_star_np = c_star_t.numpy().astype(np.int64)

    assert ece(logits_np, c_star_np) == pytest.approx(
        ece(logits_t, c_star_t), rel=1e-5, abs=1e-6
    )


def test_pseudo_r2_is_pure_transform_of_nll() -> None:
    """pseudo_r2 depends only on the scalar NLL and J — feeding the same
    NLL as np.float32, Python float, or torch scalar gives identical output."""
    J = 10
    nll_v = 1.234
    a = mcfadden_pseudo_r2(nll_v, J=J)
    b = mcfadden_pseudo_r2(float(np.float32(nll_v)), J=J)
    c = mcfadden_pseudo_r2(float(torch.tensor(nll_v)), J=J)
    assert a == pytest.approx(b, rel=1e-6, abs=1e-7)
    assert a == pytest.approx(c, rel=1e-6, abs=1e-7)


def test_compute_all_pseudo_r2_and_ece_match_standalone() -> None:
    """compute_all wires pseudo_r2 and ece the same as their standalone calls."""
    N, J = 24, 7
    torch.manual_seed(11)
    logits = torch.randn(N, J)
    c_star = torch.randint(0, J, (N,))
    em = compute_all(logits, c_star, n_params=100, n_train=1000)
    assert em.pseudo_r2 == pytest.approx(mcfadden_pseudo_r2(em.nll_val, J=J))
    assert em.ece_val == pytest.approx(ece(logits, c_star))
