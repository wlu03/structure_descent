"""Tests for src/eval/strata.py (redesign.md §13, §12.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.eval.strata import (
    _default_metrics,
    activity_tertile_breakdown,
    category_breakdown,
    dominant_attribute,
    dominant_attribute_breakdown,
    repeat_novel_breakdown,
    stratify_by_key,
    time_of_day_breakdown,
)
from src.model.po_leu import POLEUIntermediates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(N: int, J: int = 10, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, J, generator=g)
    c_star = torch.randint(0, J, (N,), generator=g, dtype=torch.int64)
    return logits, c_star


def _make_intermediates(
    N: int,
    *,
    J: int = 10,
    K: int = 3,
    M: int = 5,
    seed: int = 0,
) -> POLEUIntermediates:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, J, K, M, generator=g)
    w_raw = torch.randn(N, M, generator=g)
    w = torch.softmax(w_raw, dim=-1)
    U = torch.zeros(N, J, K)
    S = torch.full((N, J, K), 1.0 / K)
    V = torch.zeros(N, J)
    return POLEUIntermediates(A=A, w=w, U=U, S=S, V=V)


# ---------------------------------------------------------------------------
# Generic stratifier
# ---------------------------------------------------------------------------


def test_stratify_by_key_sums_to_N():
    """Sum of "n" across groups equals N."""
    N, J = 50, 10
    logits, c_star = _make_logits(N, J)
    group_key = np.array([i % 4 for i in range(N)])
    out = stratify_by_key(logits, c_star, group_key)
    assert sum(g["n"] for g in out.values()) == N


def test_stratify_by_key_custom_fn():
    """Injected compute_metrics_fn is respected."""
    N, J = 20, 10
    logits, c_star = _make_logits(N, J)
    group_key = np.array([i % 2 for i in range(N)])

    def fake(lg, cs):
        return {"marker": float(lg.shape[0])}

    out = stratify_by_key(logits, c_star, group_key, compute_metrics_fn=fake)
    assert set(out.keys()) == {0, 1}
    for v in out.values():
        assert v["marker"] == float(v["n"])


# ---------------------------------------------------------------------------
# Default metrics sanity
# ---------------------------------------------------------------------------


def test_default_metrics_keys_and_ranges():
    logits, c_star = _make_logits(30, 10)
    m = _default_metrics(logits, c_star)
    assert set(m.keys()) == {"top1", "top5", "mrr", "nll"}
    assert 0.0 <= m["top1"] <= 1.0
    assert 0.0 <= m["top5"] <= 1.0
    assert 0.0 < m["mrr"] <= 1.0
    assert m["nll"] >= 0.0


def test_default_metrics_perfect_prediction():
    """If chosen alternative has the max logit for every event, top1=1."""
    N, J = 8, 10
    logits = torch.zeros(N, J)
    c_star = torch.arange(N) % J
    for i in range(N):
        logits[i, c_star[i]] = 100.0
    m = _default_metrics(logits, c_star)
    assert m["top1"] == pytest.approx(1.0)
    assert m["top5"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def test_category_breakdown_two_categories():
    N, J = 20, 10
    logits, c_star = _make_logits(N, J)
    category = np.array(["books"] * 10 + ["food"] * 10)
    out = category_breakdown(logits, c_star, category)
    assert set(out.keys()) == {"books", "food"}
    assert out["books"]["n"] == 10
    assert out["food"]["n"] == 10


def test_repeat_novel_breakdown_labels():
    N, J = 12, 10
    logits, c_star = _make_logits(N, J)
    is_novel = np.array([True, False] * (N // 2))
    out = repeat_novel_breakdown(logits, c_star, is_novel)
    assert set(out.keys()) == {"repeat", "novel"}
    assert out["novel"]["n"] == 6
    assert out["repeat"]["n"] == 6


def test_repeat_novel_breakdown_all_novel_omits_repeat():
    N, J = 8, 10
    logits, c_star = _make_logits(N, J)
    is_novel = np.ones(N, dtype=bool)
    out = repeat_novel_breakdown(logits, c_star, is_novel)
    assert set(out.keys()) == {"novel"}


def test_activity_tertile_buckets_are_equal_size_within_1():
    N, J = 30, 10
    logits, c_star = _make_logits(N, J)
    # Distinct, well-spread values → quantile split gives roughly equal tertiles.
    activity = np.linspace(0.0, 5.0, N)
    out = activity_tertile_breakdown(logits, c_star, activity)
    sizes = [out[k]["n"] for k in ("low", "mid", "high") if k in out]
    # All three buckets should be present at N=30 with distinct values.
    assert set(out.keys()) == {"low", "mid", "high"}
    assert max(sizes) - min(sizes) <= 1


def test_time_of_day_all_four_labels():
    N, J = 24, 10
    logits, c_star = _make_logits(N, J)
    hour = np.arange(24)  # one event per hour
    out = time_of_day_breakdown(logits, c_star, hour)
    assert set(out.keys()) == {"morning", "afternoon", "evening", "night"}
    # 6 hours each per spec boundaries.
    assert out["night"]["n"] == 6       # 0..5
    assert out["morning"]["n"] == 6     # 6..11
    assert out["afternoon"]["n"] == 6   # 12..17
    assert out["evening"]["n"] == 6     # 18..23


def test_time_of_day_rejects_out_of_range():
    logits, c_star = _make_logits(4, 10)
    with pytest.raises(ValueError):
        time_of_day_breakdown(logits, c_star, np.array([0, 5, 24, 7]))


# ---------------------------------------------------------------------------
# §12.3 dominant attribute
# ---------------------------------------------------------------------------


def test_dominant_attribute_shape():
    N, J, K, M = 16, 10, 3, 5
    inter = _make_intermediates(N, J=J, K=K, M=M)
    c_star = torch.randint(0, J, (N,), dtype=torch.int64)
    m_star = dominant_attribute(inter, c_star)
    assert m_star.shape == (N,)
    assert m_star.dtype == torch.int64
    assert int(m_star.min()) >= 0
    assert int(m_star.max()) < M


def test_dominant_attribute_logic():
    """Hand-crafted A zero everywhere except one m → dominant_attribute returns that m."""
    N, J, K, M = 5, 4, 3, 6
    for target_m in range(M):
        A = torch.zeros(N, J, K, M)
        A[..., target_m] = 1.0                     # single attribute carries all mass
        w = torch.full((N, M), 1.0 / M)             # uniform weights → argmax is target_m
        inter = POLEUIntermediates(
            A=A, w=w,
            U=torch.zeros(N, J, K),
            S=torch.full((N, J, K), 1.0 / K),
            V=torch.zeros(N, J),
        )
        c_star = torch.randint(0, J, (N,), dtype=torch.int64)
        m_star = dominant_attribute(inter, c_star)
        assert torch.all(m_star == target_m), f"failed for target_m={target_m}"


def test_dominant_attribute_uses_absolute_value():
    """Negative u_m with large magnitude must still register as dominant."""
    N, J, K, M = 3, 4, 3, 5
    A = torch.zeros(N, J, K, M)
    # Chosen-alternative slot gets a huge negative value on attribute 2.
    c_star = torch.tensor([0, 0, 0], dtype=torch.int64)
    A[:, 0, :, 2] = -10.0
    # Small positive noise on another attribute — must lose to |-10|.
    A[:, 0, :, 0] = 0.5
    w = torch.full((N, M), 1.0 / M)
    inter = POLEUIntermediates(
        A=A, w=w,
        U=torch.zeros(N, J, K),
        S=torch.full((N, J, K), 1.0 / K),
        V=torch.zeros(N, J),
    )
    m_star = dominant_attribute(inter, c_star)
    assert torch.all(m_star == 2)


def test_dominant_attribute_breakdown_returns_dict_keyed_by_int():
    N, J, K, M = 24, 10, 3, 5
    inter = _make_intermediates(N, J=J, K=K, M=M, seed=7)
    logits, c_star = _make_logits(N, J, seed=7)
    out = dominant_attribute_breakdown(logits, c_star, inter)
    # Every key is a plain int in [0, M).
    for k in out:
        assert isinstance(k, int)
        assert 0 <= k < M
    # Counts sum to N.
    assert sum(v["n"] for v in out.values()) == N


# ---------------------------------------------------------------------------
# Empty-group omission
# ---------------------------------------------------------------------------


def test_empty_group_omitted():
    """A group value that doesn't appear in the data must not show up in the output."""
    N, J = 10, 10
    logits, c_star = _make_logits(N, J)
    # Only "a" appears; "b" / "c" never do.
    group_key = np.array(["a"] * N)
    out = stratify_by_key(logits, c_star, group_key)
    assert set(out.keys()) == {"a"}
    assert "b" not in out and "c" not in out
