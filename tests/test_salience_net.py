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
    """Group-2: 802*64 + 64 + 64*1 + 1 + 1*8 = 51_465 trainable parameters.

    The +520 over the original 50_945 comes from (a) widening fc1's
    in_dim by d_cat=8 → +512 weights, (b) adding a 1*8 nn.Embedding
    for the category lookup → +8. With n_categories=1 the embedding
    behaves as a single-bucket bias, preserving legacy semantics.
    """
    net = SalienceNet()
    assert net.num_params() == EXPECTED_PARAM_COUNT_DEFAULT == 51_465


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
    # Group-2 in_dim = d_e + p + d_cat = 802; +1*8 for the category emb.
    assert EXPECTED_PARAM_COUNT_DEFAULT == 802 * 64 + 64 + 64 * 1 + 1 + 1 * 8



# ---------------------------------------------------------------------------
# Group-2 category injection
# ---------------------------------------------------------------------------


def test_salience_with_category(synthetic_batch) -> None:
    """SalienceNet accepts (B,) int64 c and produces a (B,J,K) softmaxed output.

    Same shape contract as the c=None path, but the category embedding
    contributes a per-event additive feature in the MLP input.
    """
    sb = synthetic_batch
    net = SalienceNet(n_categories=4, d_cat=8)
    c = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = net(sb.E, sb.z_d, c)
    assert out.shape == (sb.B, sb.J, sb.K)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_salience_category_changes_output(synthetic_batch) -> None:
    """Two events with different c yield different S given the same (E, z_d).

    Drives the embedding lookup parameters; if the embedding had no
    effect, this test would silently pass on a c-blind net.
    """
    sb = synthetic_batch
    torch.manual_seed(0)
    net = SalienceNet(n_categories=4, d_cat=8)
    # Use the same first event's E and z_d for two synthetic batch items.
    E_pair = sb.E[:1].repeat(2, 1, 1, 1)
    z_pair = sb.z_d[:1].repeat(2, 1)

    c0 = torch.tensor([0, 0], dtype=torch.long)
    c1 = torch.tensor([0, 3], dtype=torch.long)
    out_a = net(E_pair, z_pair, c0)
    out_b = net(E_pair, z_pair, c1)
    # Row 0 (c=0 in both) must be identical.
    assert torch.equal(out_a[0], out_b[0])
    # Row 1 (c=0 vs c=3) must differ — embedding contributes.
    assert not torch.allclose(out_a[1], out_b[1], atol=1e-7)


def test_salience_category_grad_flows(synthetic_batch) -> None:
    """Gradient propagates back to cat_emb when c is supplied."""
    sb = synthetic_batch
    net = SalienceNet(n_categories=4, d_cat=8)
    c = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = net(sb.E, sb.z_d, c)
    loss = out.sum()
    loss.backward()
    assert net.cat_emb.weight.grad is not None
    assert torch.isfinite(net.cat_emb.weight.grad).all()
    assert net.cat_emb.weight.grad.norm().item() > 0.0


def test_salience_bad_c_dtype_raises(synthetic_batch) -> None:
    """Non-int64 c is rejected with a clear ValueError."""
    sb = synthetic_batch
    net = SalienceNet(n_categories=4)
    bad = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    try:
        net(sb.E, sb.z_d, bad)
    except ValueError as exc:
        assert "int64" in str(exc) or "torch.long" in str(exc)
    else:
        raise AssertionError("expected ValueError on non-int64 c")


def test_salience_bad_c_shape_raises(synthetic_batch) -> None:
    """Wrong-shape c is rejected."""
    sb = synthetic_batch
    net = SalienceNet(n_categories=4)
    bad_dim = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    try:
        net(sb.E, sb.z_d, bad_dim)
    except ValueError as exc:
        assert "1-D" in str(exc)
    else:
        raise AssertionError("expected ValueError on 2-D c")
    bad_b = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    try:
        net(sb.E, sb.z_d, bad_b)
    except ValueError as exc:
        assert "Batch mismatch" in str(exc) or "B=" in str(exc)
    else:
        raise AssertionError("expected ValueError on wrong B")


def test_salience_c_none_uses_zero_bucket(synthetic_batch) -> None:
    """c=None is equivalent to c=zeros(B,)."""
    sb = synthetic_batch
    torch.manual_seed(7)
    net = SalienceNet(n_categories=3)
    c_zero = torch.zeros(sb.B, dtype=torch.long)
    out_none = net(sb.E, sb.z_d, None)
    out_zero = net(sb.E, sb.z_d, c_zero)
    assert torch.equal(out_none, out_zero)
