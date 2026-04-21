"""Tests for :mod:`src.model.salience_net` (redesign.md §7).

All tests use the session-scoped ``synthetic_batch`` fixture
(B=4, J=10, K=3, d_e=768, p=26) from ``tests/conftest.py``.
"""

from __future__ import annotations

import torch

from src.model.salience_net import (
    DEFAULT_D_E,
    DEFAULT_HIDDEN,
    DEFAULT_P,
    EXPECTED_PARAM_COUNT_DEFAULT,
    SalienceNet,
    UniformSalience,
)


# ---------------------------------------------------------------------------
# SalienceNet: structural invariants
# ---------------------------------------------------------------------------


def test_param_count_default() -> None:
    """§7.1 corrected: 794*64 + 64 + 64*1 + 1 = 50_945 trainable parameters.

    The spec's printed 50_881 drops the fc1 bias term; §0 mandates biases
    on every Linear, so the orchestrator override fixes the constant to
    50_945. See NOTES.md "per-head / salience parameter-count reconciliation".
    """
    net = SalienceNet()
    assert net.num_params() == EXPECTED_PARAM_COUNT_DEFAULT == 50_945


def test_xavier_init_and_zero_bias() -> None:
    """Biases start at zero (§0); weights are not all zero (Xavier-uniform)."""
    net = SalienceNet()
    assert torch.all(net.fc1.bias == 0.0)
    assert torch.all(net.fc2.bias == 0.0)
    assert not torch.all(net.fc1.weight == 0.0)
    assert not torch.all(net.fc2.weight == 0.0)

    # Xavier-uniform bound: U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).
    a1 = (6.0 / (net.fc1.in_features + net.fc1.out_features)) ** 0.5
    a2 = (6.0 / (net.fc2.in_features + net.fc2.out_features)) ** 0.5
    assert net.fc1.weight.abs().max().item() <= a1 + 1e-6
    assert net.fc2.weight.abs().max().item() <= a2 + 1e-6


# ---------------------------------------------------------------------------
# SalienceNet: forward-pass shape + softmax properties
# ---------------------------------------------------------------------------


def test_output_shape(synthetic_batch) -> None:
    """(B, J, K, d_e) + (B, p) -> (B, J, K)."""
    net = SalienceNet()
    out = net(synthetic_batch.E, synthetic_batch.z_d)
    assert out.shape == (synthetic_batch.B, synthetic_batch.J, synthetic_batch.K)


def test_softmax_over_k(synthetic_batch) -> None:
    """Each (b, j) row sums to 1.0 along the K axis (§7.4, atol 1e-6)."""
    net = SalienceNet()
    out = net(synthetic_batch.E, synthetic_batch.z_d)
    sums = out.sum(dim=-1)  # (B, J)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_softmax_over_k_nonneg(synthetic_batch) -> None:
    """Softmax output is strictly nonnegative."""
    net = SalienceNet()
    out = net(synthetic_batch.E, synthetic_batch.z_d)
    assert torch.all(out >= 0.0)


# ---------------------------------------------------------------------------
# UniformSalience (ablation A6)
# ---------------------------------------------------------------------------


def test_uniform_salience_constant(synthetic_batch) -> None:
    """Every entry is exactly 1/K (atol 1e-6)."""
    net = UniformSalience()
    out = net(synthetic_batch.E, synthetic_batch.z_d)
    assert out.shape == (synthetic_batch.B, synthetic_batch.J, synthetic_batch.K)
    expected = torch.full_like(out, 1.0 / synthetic_batch.K)
    assert torch.allclose(out, expected, atol=1e-6)
    # Rows sum to 1.0 too (§7.4).
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_uniform_salience_zero_params() -> None:
    """Ablation A6 has no trainable parameters."""
    assert UniformSalience().num_params() == 0


# ---------------------------------------------------------------------------
# Determinism + autograd
# ---------------------------------------------------------------------------


def test_deterministic_under_seed(synthetic_batch) -> None:
    """Same seed + same input -> bitwise-identical output."""
    torch.manual_seed(1234)
    net_a = SalienceNet()
    out_a = net_a(synthetic_batch.E, synthetic_batch.z_d)

    torch.manual_seed(1234)
    net_b = SalienceNet()
    out_b = net_b(synthetic_batch.E, synthetic_batch.z_d)

    # Equal params.
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb)
    # Equal output (exact equality, not just close).
    assert torch.equal(out_a, out_b)


def test_gradient_finite_e_and_z(synthetic_batch) -> None:
    """Backward of out.sum() gives finite grads on params AND inputs."""
    net = SalienceNet()
    E = synthetic_batch.E.detach().clone().requires_grad_(True)
    z_d = synthetic_batch.z_d.detach().clone().requires_grad_(True)

    out = net(E, z_d)
    loss = out.sum()
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"

    assert E.grad is not None
    assert torch.isfinite(E.grad).all()
    assert z_d.grad is not None
    assert torch.isfinite(z_d.grad).all()


# ---------------------------------------------------------------------------
# Broadcast semantics: no cross-batch leakage
# ---------------------------------------------------------------------------


def test_z_d_broadcast_correct(synthetic_batch) -> None:
    """Changing z_d[0] affects output[0, :, :] only; output[1:] is unchanged."""
    net = SalienceNet()
    E = synthetic_batch.E
    z_d = synthetic_batch.z_d.clone()

    with torch.no_grad():
        base = net(E, z_d)

        z_d_mod = z_d.clone()
        z_d_mod[0] = z_d_mod[0] + 3.7  # perturb just row 0
        modded = net(E, z_d_mod)

    # Row 0 must have changed somewhere.
    assert not torch.allclose(base[0], modded[0], atol=1e-6)
    # Rows 1..B-1 must be bitwise identical (same params, same inputs).
    assert torch.equal(base[1:], modded[1:])


# ---------------------------------------------------------------------------
# Sanity: module-level constants match the spec
# ---------------------------------------------------------------------------


def test_module_constants() -> None:
    assert DEFAULT_D_E == 768
    assert DEFAULT_P == 26
    assert DEFAULT_HIDDEN == 64
    assert EXPECTED_PARAM_COUNT_DEFAULT == 794 * 64 + 64 + 64 * 1 + 1
