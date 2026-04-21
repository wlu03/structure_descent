"""Tests for src/model/attribute_heads.py (redesign.md §5, §9.4 step 2)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.attribute_heads import (
    DEFAULT_D_E,
    DEFAULT_HIDDEN,
    DEFAULT_M,
    EXPECTED_PARAM_COUNT_PER_HEAD,
    EXPECTED_PARAM_COUNT_STACK_M5,
    AttributeHead,
    AttributeHeadStack,
)


def test_single_head_param_count() -> None:
    head = AttributeHead(DEFAULT_D_E, DEFAULT_HIDDEN)
    assert head.num_params() == EXPECTED_PARAM_COUNT_PER_HEAD == 98_561


def test_stack_param_count_default() -> None:
    stack = AttributeHeadStack()
    assert stack.num_params() == EXPECTED_PARAM_COUNT_STACK_M5 == 492_805
    # Sanity: stack == M * per-head (heads are independent, no shared params).
    assert stack.num_params() == DEFAULT_M * EXPECTED_PARAM_COUNT_PER_HEAD


def test_stack_shape_contract(synthetic_batch) -> None:
    E = synthetic_batch.E
    assert E.shape == (4, 10, 3, 768)
    stack = AttributeHeadStack()
    out = stack(E)
    assert out.shape == (4, 10, 3, 5)
    assert out.dtype == E.dtype


def test_xavier_uniform_init_and_zero_bias() -> None:
    torch.manual_seed(0)
    stack = AttributeHeadStack()
    for mod in stack.modules():
        if isinstance(mod, nn.Linear):
            assert torch.all(mod.bias == 0), "all linear biases must be zero at init"
            w = mod.weight.detach()
            # Xavier-uniform weights have mean ~0 and a modest, non-zero std.
            # The fc2 weight is a (1, hidden) slab, so the sample mean is noisier
            # than fc1's (hidden, d_e); keep the bound loose enough for both.
            assert abs(w.mean().item()) < 0.1
            std = w.std().item()
            assert 0.0 < std < 1.0


def test_deterministic_under_seed(synthetic_batch) -> None:
    E = synthetic_batch.E
    torch.manual_seed(1234)
    s1 = AttributeHeadStack()
    out1 = s1(E)
    torch.manual_seed(1234)
    s2 = AttributeHeadStack()
    out2 = s2(E)
    assert torch.allclose(out1, out2)
    # Parameter tensors should also match element-for-element.
    for p1, p2 in zip(s1.parameters(), s2.parameters()):
        assert torch.allclose(p1, p2)


def test_heads_independent(synthetic_batch) -> None:
    torch.manual_seed(7)
    stack = AttributeHeadStack()
    E = synthetic_batch.E

    with torch.no_grad():
        baseline = stack(E).clone()

    # Perturb only head 0's parameters.
    with torch.no_grad():
        for p in stack.heads[0].parameters():
            p.add_(torch.randn_like(p) * 0.5)
        perturbed = stack(E)

    # Head 0 output (last-axis slice 0) should change.
    assert not torch.allclose(baseline[..., 0], perturbed[..., 0])
    # All other heads' outputs must be bitwise identical.
    for m in range(1, stack.M):
        assert torch.equal(baseline[..., m], perturbed[..., m]), (
            f"head {m} output changed when only head 0 was perturbed"
        )


def test_gradient_finite(synthetic_batch) -> None:
    torch.manual_seed(0)
    stack = AttributeHeadStack()
    E = synthetic_batch.E.clone().requires_grad_(False)
    out = stack(E)
    loss = out.sum()
    loss.backward()
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_person_independence(synthetic_batch) -> None:
    """Default module is person-independent (§5.3): forward takes only E."""
    stack = AttributeHeadStack()
    E = synthetic_batch.E
    z_d = synthetic_batch.z_d  # (B, p)
    # Calling with the embedding alone must succeed.
    _ = stack(E)
    # Passing z_d as an extra positional argument must fail — the contract
    # is "only the embedding tensor."
    with pytest.raises(TypeError):
        stack(E, z_d)


def test_dropout_absent() -> None:
    stack = AttributeHeadStack()
    for mod in stack.modules():
        assert not isinstance(mod, nn.Dropout), "dropout not allowed in attribute heads"
        assert not isinstance(mod, nn.LayerNorm), "layernorm not allowed in attribute heads"
        assert not isinstance(mod, nn.BatchNorm1d), "batchnorm not allowed in attribute heads"
        assert not isinstance(mod, nn.BatchNorm2d), "batchnorm not allowed in attribute heads"
        assert not isinstance(mod, nn.BatchNorm3d), "batchnorm not allowed in attribute heads"
