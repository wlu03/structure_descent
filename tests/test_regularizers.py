"""Tests for src/train/regularizers.py (redesign.md §9.2)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from src.model.po_leu import POLEU
from src.model.weight_net import WeightNet
from src.train.regularizers import (
    DEFAULT_LAMBDA_DIVERSITY,
    DEFAULT_LAMBDA_MONOTONICITY,
    DEFAULT_LAMBDA_SALIENCE_ENTROPY,
    DEFAULT_LAMBDA_WEIGHT_L2,
    RegularizerConfig,
    combined_regularizer,
    outcome_diversity,
    price_monotonicity,
    salience_entropy,
    weight_net_l2,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# weight_net_l2
# ---------------------------------------------------------------------------

def test_weight_net_l2_sum_of_squares() -> None:
    """Manual sum-of-squares matches the function on a small WeightNet."""
    torch.manual_seed(42)
    wn = WeightNet(p=8, M=4, hidden=6)

    # Manually compute Σ ||fc.weight||_F^2 over both Linears.
    expected = (wn.fc1.weight ** 2).sum() + (wn.fc2.weight ** 2).sum()
    got = weight_net_l2(wn)

    assert got.dim() == 0
    assert torch.allclose(got, expected, atol=1e-7)


def test_weight_net_l2_excludes_biases() -> None:
    """Biases do not contribute even when explicitly non-zero."""
    wn = WeightNet(p=4, M=3, hidden=5)
    # Zero all weights; set biases to arbitrary values. L2 must be 0.
    with torch.no_grad():
        wn.fc1.weight.zero_()
        wn.fc2.weight.zero_()
        wn.fc1.bias.fill_(2.5)
        wn.fc2.bias.fill_(-1.3)

    got = weight_net_l2(wn)
    assert got.dim() == 0
    assert torch.allclose(got, torch.zeros(()), atol=1e-10)

    # Now set one weight entry to 3.0, check it picks up exactly 9.0.
    with torch.no_grad():
        wn.fc1.weight[0, 0] = 3.0
    got = weight_net_l2(wn)
    assert torch.allclose(got, torch.tensor(9.0), atol=1e-6)


# ---------------------------------------------------------------------------
# salience_entropy
# ---------------------------------------------------------------------------

def test_salience_entropy_uniform_is_max() -> None:
    """S = 1/K uniformly → H = log(K)."""
    B, J, K = 2, 5, 4
    S = torch.full((B, J, K), 1.0 / K)
    H = salience_entropy(S)
    assert H.dim() == 0
    assert torch.allclose(H, torch.tensor(math.log(K)), atol=1e-6)


def test_salience_entropy_onehot_is_zero() -> None:
    """S = one-hot → H = 0."""
    B, J, K = 3, 4, 5
    S = torch.zeros(B, J, K)
    # For each (b, j), put the mass on a deterministic k.
    for b in range(B):
        for j in range(J):
            S[b, j, (b + j) % K] = 1.0
    H = salience_entropy(S)
    assert H.dim() == 0
    # Analytical zero; the eps stabilizer contributes only at s_k = 0
    # entries which are multiplied by 0 (0 · log(eps) = 0).
    assert torch.allclose(H, torch.zeros(()), atol=1e-6)


def test_salience_entropy_nonneg() -> None:
    """Random softmax inputs give non-negative entropy."""
    torch.manual_seed(0)
    raw = torch.randn(4, 6, 5)
    S = torch.softmax(raw, dim=-1)
    H = salience_entropy(S)
    assert H.item() >= 0.0


# ---------------------------------------------------------------------------
# outcome_diversity
# ---------------------------------------------------------------------------

def test_outcome_diversity_identical_outcomes_is_one() -> None:
    """E with K identical (unit-norm) outcomes per alt → diversity = 1."""
    B, J, K, d_e = 2, 3, 4, 16
    # One random direction per (b, j); replicate K times.
    base = torch.randn(B, J, 1, d_e)
    base = base / base.norm(dim=-1, keepdim=True)
    E = base.expand(B, J, K, d_e).contiguous()

    d = outcome_diversity(E)
    assert d.dim() == 0
    assert torch.allclose(d, torch.tensor(1.0), atol=1e-6)


def test_outcome_diversity_orthogonal_outcomes_is_zero() -> None:
    """K=2 with orthogonal unit rows → diversity = 0."""
    B, J, K, d_e = 2, 3, 2, 4
    E = torch.zeros(B, J, K, d_e)
    # k=0 lies along basis[0]; k=1 along basis[1]. Both unit-norm, orthogonal.
    E[:, :, 0, 0] = 1.0
    E[:, :, 1, 1] = 1.0

    d = outcome_diversity(E)
    assert d.dim() == 0
    assert torch.allclose(d, torch.zeros(()), atol=1e-6)


# ---------------------------------------------------------------------------
# price_monotonicity
# ---------------------------------------------------------------------------

def test_price_monotonicity_returns_scalar(synthetic_batch) -> None:
    """Random E, prices → finite 0-dim scalar."""
    torch.manual_seed(0)
    model = POLEU()
    B, J = synthetic_batch.B, synthetic_batch.J
    prices = torch.rand(B, J) * 100.0
    val = price_monotonicity(model.heads, synthetic_batch.E, prices)
    assert val.dim() == 0
    assert torch.isfinite(val).item()


def test_price_monotonicity_nonneg(synthetic_batch) -> None:
    """Squared-ReLU surrogate is non-negative."""
    torch.manual_seed(1)
    model = POLEU()
    B, J = synthetic_batch.B, synthetic_batch.J
    prices = torch.rand(B, J) * 50.0 + 1.0
    val = price_monotonicity(model.heads, synthetic_batch.E, prices)
    assert val.item() >= 0.0


# ---------------------------------------------------------------------------
# combined_regularizer + RegularizerConfig
# ---------------------------------------------------------------------------

def _instrumented_forward(model: POLEU, z_d: torch.Tensor, E: torch.Tensor,
                          S_override: torch.Tensor):
    """Run POLEU.forward and patch intermediates.S with ``S_override``.

    The model's actual forward output is ignored for S; we build a
    POLEUIntermediates with the user-supplied salience so tests can
    pin H(S) to a specific value.
    """
    logits, inter = model(z_d, E)
    # Rebuild intermediates with the pinned S.
    from src.model.po_leu import POLEUIntermediates
    new_inter = POLEUIntermediates(A=inter.A, w=inter.w, U=inter.U,
                                   S=S_override, V=inter.V)
    return logits, new_inter


def test_combined_sign_on_entropy(synthetic_batch) -> None:
    """Higher H(S) → lower combined_regularizer (entropy is subtracted)."""
    torch.manual_seed(0)
    model = POLEU()
    z_d = synthetic_batch.z_d
    E = synthetic_batch.E
    B, J, K = synthetic_batch.B, synthetic_batch.J, synthetic_batch.K

    cfg = RegularizerConfig()  # defaults

    # Low-entropy S (near one-hot on k=0).
    S_low = torch.zeros(B, J, K)
    S_low[..., 0] = 1.0
    _, inter_low = _instrumented_forward(model, z_d, E, S_low)
    val_low = combined_regularizer(model, inter_low, E, prices=None, cfg=cfg)

    # High-entropy S (uniform).
    S_high = torch.full((B, J, K), 1.0 / K)
    _, inter_high = _instrumented_forward(model, z_d, E, S_high)
    val_high = combined_regularizer(model, inter_high, E, prices=None, cfg=cfg)

    # Higher entropy must yield a strictly smaller combined regularizer
    # (the minus sign on the entropy term is the whole point).
    assert val_high.item() < val_low.item()


def test_regularizer_config_from_default() -> None:
    """from_default loads configs/default.yaml and matches Appendix B."""
    cfg = RegularizerConfig.from_default()

    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        raw = yaml.safe_load(fh)
    reg = raw["regularizers"]

    # Values must match the YAML (Appendix B source of truth).
    assert cfg.weight_l2 == pytest.approx(float(reg["weight_l2"]))
    assert cfg.salience_entropy == pytest.approx(float(reg["salience_entropy"]))
    assert cfg.diversity == pytest.approx(float(reg["diversity"]))
    assert cfg.monotonicity_enabled is bool(reg["monotonicity"]["enabled"])
    assert cfg.monotonicity == pytest.approx(float(reg["monotonicity"]["lambda"]))

    # And those YAML values must match the Appendix B defaults baked into
    # the module (this guards against drift in either direction).
    assert cfg.weight_l2 == pytest.approx(DEFAULT_LAMBDA_WEIGHT_L2)
    assert cfg.salience_entropy == pytest.approx(DEFAULT_LAMBDA_SALIENCE_ENTROPY)
    assert cfg.monotonicity == pytest.approx(DEFAULT_LAMBDA_MONOTONICITY)
    assert cfg.diversity == pytest.approx(DEFAULT_LAMBDA_DIVERSITY)


def test_combined_finite_gradient(synthetic_batch) -> None:
    """Backward of combined_regularizer yields finite grads on model params."""
    torch.manual_seed(0)
    model = POLEU()
    z_d = synthetic_batch.z_d
    E = synthetic_batch.E
    B, J = synthetic_batch.B, synthetic_batch.J

    prices = torch.rand(B, J) * 20.0 + 1.0
    # Enable monotonicity too so all four terms contribute to the graph.
    cfg = RegularizerConfig(
        weight_l2=1e-3,
        salience_entropy=1e-2,
        monotonicity_enabled=True,
        monotonicity=1e-2,
        diversity=1e-2,
    )

    _, intermediates = model(z_d, E)
    total = combined_regularizer(model, intermediates, E, prices=prices, cfg=cfg)

    # Clear any stale grads, then backprop.
    model.zero_grad(set_to_none=True)
    total.backward()

    # At least some parameters must have grads, and all grads that exist
    # must be finite.
    any_grad = False
    for param in model.parameters():
        if param.grad is not None:
            any_grad = True
            assert torch.isfinite(param.grad).all().item(), (
                "Non-finite gradient encountered in combined regularizer."
            )
    assert any_grad, "No parameter received a gradient from combined regularizer."
