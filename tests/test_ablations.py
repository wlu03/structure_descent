"""Tests for :mod:`src.model.ablations` (redesign.md §11 A7, A8).

All tests use the session-scoped ``synthetic_batch`` fixture
(B=4, J=10, K=3, d_e=768, p=26) from ``tests/conftest.py``.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

from src.model.ablations import (
    DEFAULT_HIDDEN,
    DEFAULT_P,
    ConcatIntermediates,
    ConcatUtility,
    FiLMIntermediates,
    FiLMUtility,
)
from src.model.attribute_heads import AttributeHeadStack


# ---------------------------------------------------------------------------
# Forward shapes
# ---------------------------------------------------------------------------


def test_concat_forward_shapes(synthetic_batch) -> None:
    """ConcatUtility default: logits (B, J); intermediates U/S (B, J, K),
    V (B, J); no A or w."""
    model = ConcatUtility()
    logits, inter = model(synthetic_batch.z_d, synthetic_batch.E)
    assert logits.shape == (synthetic_batch.B, synthetic_batch.J)
    assert isinstance(inter, ConcatIntermediates)
    assert inter.U.shape == (
        synthetic_batch.B,
        synthetic_batch.J,
        synthetic_batch.K,
    )
    assert inter.S.shape == (
        synthetic_batch.B,
        synthetic_batch.J,
        synthetic_batch.K,
    )
    assert inter.V.shape == (synthetic_batch.B, synthetic_batch.J)
    # Central A7 claim: no attribute decomposition is exposed.
    assert not hasattr(inter, "A")
    assert not hasattr(inter, "w")


def test_film_forward_shapes(synthetic_batch) -> None:
    """FiLMUtility default: logits (B, J); intermediates U/S (B, J, K),
    V (B, J); theta_d present."""
    model = FiLMUtility()
    logits, inter = model(synthetic_batch.z_d, synthetic_batch.E)
    assert logits.shape == (synthetic_batch.B, synthetic_batch.J)
    assert isinstance(inter, FiLMIntermediates)
    assert inter.U.shape == (
        synthetic_batch.B,
        synthetic_batch.J,
        synthetic_batch.K,
    )
    assert inter.S.shape == (
        synthetic_batch.B,
        synthetic_batch.J,
        synthetic_batch.K,
    )
    assert inter.V.shape == (synthetic_batch.B, synthetic_batch.J)
    gamma, beta = inter.theta_d
    assert gamma.shape == (synthetic_batch.B, DEFAULT_HIDDEN)
    assert beta.shape == (synthetic_batch.B, DEFAULT_HIDDEN)


# ---------------------------------------------------------------------------
# Backward finite
# ---------------------------------------------------------------------------


def test_concat_backward_finite(synthetic_batch) -> None:
    """Backward of logits.sum() gives finite grads on every parameter."""
    model = ConcatUtility()
    logits, _ = model(synthetic_batch.z_d, synthetic_batch.E)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_film_backward_finite(synthetic_batch) -> None:
    """Backward of logits.sum() gives finite grads on every parameter."""
    model = FiLMUtility()
    logits, _ = model(synthetic_batch.z_d, synthetic_batch.E)
    logits.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Structural ablation invariants
# ---------------------------------------------------------------------------


def test_concat_has_no_attribute_decomposition() -> None:
    """A7: no AttributeHeadStack anywhere in the submodule tree."""
    model = ConcatUtility()
    for _, sub in model.named_modules():
        assert not isinstance(sub, AttributeHeadStack)
    # And no attribute called "weight_net" or "heads" (defensive).
    names = {name for name, _ in model.named_modules()}
    assert "heads" not in names
    assert "weight_net" not in names


def test_film_has_modulator_linear() -> None:
    """A8: a Linear(p, 2*hidden) modulator is present, per the spec."""
    model = FiLMUtility()
    assert isinstance(model.modulator, nn.Linear)
    assert model.modulator.in_features == DEFAULT_P
    assert model.modulator.out_features == 2 * DEFAULT_HIDDEN


def test_film_initial_gamma_near_one(synthetic_batch) -> None:
    """Before any training step, γ lands within ±2.0 of 1.0 for any z_d.

    Xavier-uniform init gives small raw outputs; the +1 offset parks γ
    near 1, i.e. near-identity conditioning. Tolerance is wide (±2) on
    purpose: the contract is "stable start", not "exactly 1".
    """
    model = FiLMUtility()
    with torch.no_grad():
        _, inter = model(synthetic_batch.z_d, synthetic_batch.E)
    gamma, _ = inter.theta_d
    assert torch.all(torch.abs(gamma - 1.0) <= 2.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_both_ablations_deterministic_under_seed(synthetic_batch) -> None:
    """Same manual seed -> identical params and identical outputs for both."""
    torch.manual_seed(1234)
    concat_a = ConcatUtility()
    film_a = FiLMUtility()
    out_ca, _ = concat_a(synthetic_batch.z_d, synthetic_batch.E)
    out_fa, _ = film_a(synthetic_batch.z_d, synthetic_batch.E)

    torch.manual_seed(1234)
    concat_b = ConcatUtility()
    film_b = FiLMUtility()
    out_cb, _ = concat_b(synthetic_batch.z_d, synthetic_batch.E)
    out_fb, _ = film_b(synthetic_batch.z_d, synthetic_batch.E)

    for pa, pb in zip(concat_a.parameters(), concat_b.parameters()):
        assert torch.equal(pa, pb)
    for pa, pb in zip(film_a.parameters(), film_b.parameters()):
        assert torch.equal(pa, pb)
    assert torch.equal(out_ca, out_cb)
    assert torch.equal(out_fa, out_fb)


# ---------------------------------------------------------------------------
# CE-loss integration (sibling agent owns po_leu.py; skip if not present)
# ---------------------------------------------------------------------------


def test_ce_loss_integration(synthetic_batch) -> None:
    """Run cross_entropy_loss from po_leu.py against both ablations' logits.

    The ``po_leu`` sibling module owns the canonical loss (§9.1). If it's
    not yet committed, the test skips so this suite does not gate on
    cross-wave ordering.
    """
    try:
        po_leu = importlib.import_module("src.model.po_leu")
    except ImportError:
        pytest.skip("src.model.po_leu not yet available")
    if not hasattr(po_leu, "cross_entropy_loss"):
        pytest.skip("po_leu.cross_entropy_loss not yet exposed")

    cross_entropy_loss = po_leu.cross_entropy_loss

    concat = ConcatUtility()
    film = FiLMUtility()

    logits_c, _ = concat(synthetic_batch.z_d, synthetic_batch.E)
    logits_f, _ = film(synthetic_batch.z_d, synthetic_batch.E)

    loss_c = cross_entropy_loss(logits_c, synthetic_batch.c_star)
    loss_f = cross_entropy_loss(logits_f, synthetic_batch.c_star)

    assert torch.isfinite(loss_c).all()
    assert torch.isfinite(loss_f).all()


# ---------------------------------------------------------------------------
# uniform_salience flag wiring (A6 composed with A7 / A8)
# ---------------------------------------------------------------------------


def test_uniform_salience_flag_wires_through(synthetic_batch) -> None:
    """Both ablations with uniform_salience=True produce S == 1/K everywhere."""
    concat = ConcatUtility(uniform_salience=True)
    film = FiLMUtility(uniform_salience=True)

    _, inter_c = concat(synthetic_batch.z_d, synthetic_batch.E)
    _, inter_f = film(synthetic_batch.z_d, synthetic_batch.E)

    expected = torch.full_like(inter_c.S, 1.0 / synthetic_batch.K)
    assert torch.allclose(inter_c.S, expected, atol=1e-6)
    assert torch.allclose(inter_f.S, expected, atol=1e-6)
