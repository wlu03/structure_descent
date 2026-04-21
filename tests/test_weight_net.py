"""Tests for src/model/weight_net.py (redesign.md §6)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.weight_net import (
    DEFAULT_HIDDEN,
    DEFAULT_M,
    DEFAULT_P,
    EXPECTED_PARAM_COUNT_DEFAULT,
    WeightNet,
)


def test_param_count_default() -> None:
    """§6.1: 26*32 + 32 + 32*5 + 5 = 1,029."""
    net = WeightNet()
    assert net.num_params() == EXPECTED_PARAM_COUNT_DEFAULT == 1_029


def test_output_shape(synthetic_batch) -> None:
    net = WeightNet()
    w = net(synthetic_batch.z_d)
    assert w.shape == (synthetic_batch.B, DEFAULT_M)


def test_softmax_sums_to_one(synthetic_batch) -> None:
    net = WeightNet(normalization="softmax")
    w = net(synthetic_batch.z_d)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_softmax_nonneg(synthetic_batch) -> None:
    net = WeightNet(normalization="softmax")
    w = net(synthetic_batch.z_d)
    assert (w >= 0).all()


def test_softplus_normalization_sums_to_one(synthetic_batch) -> None:
    net = WeightNet(normalization="softplus")
    w = net(synthetic_batch.z_d)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_softplus_nonneg(synthetic_batch) -> None:
    net = WeightNet(normalization="softplus")
    w = net(synthetic_batch.z_d)
    assert (w >= 0).all()


def test_invalid_normalization_raises() -> None:
    with pytest.raises(ValueError):
        WeightNet(normalization="sigmoid")


def test_deterministic_under_seed(synthetic_batch) -> None:
    torch.manual_seed(1234)
    net_a = WeightNet()
    out_a = net_a(synthetic_batch.z_d)

    torch.manual_seed(1234)
    net_b = WeightNet()
    out_b = net_b(synthetic_batch.z_d)

    assert torch.equal(out_a, out_b)


def test_xavier_init_and_zero_bias() -> None:
    net = WeightNet()
    found_linear = False
    for module in net.modules():
        if isinstance(module, nn.Linear):
            found_linear = True
            assert torch.all(module.bias == 0)
    assert found_linear, "expected at least one nn.Linear in WeightNet"


def test_gradient_finite(synthetic_batch) -> None:
    net = WeightNet()
    w = net(synthetic_batch.z_d)
    # Scalar proxy loss — sum over all entries.
    loss = w.sum()
    loss.backward()
    for param in net.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_person_dependence() -> None:
    """Forward requires z_d; calling with no args must raise."""
    net = WeightNet()
    with pytest.raises(TypeError):
        net()  # type: ignore[call-arg]


def test_defaults_match_spec() -> None:
    """Guardrail: the §6.1 defaults cannot drift."""
    assert DEFAULT_P == 26
    assert DEFAULT_M == 5
    assert DEFAULT_HIDDEN == 32
