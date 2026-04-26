"""Attribute heads for PO-LEU (redesign.md §5).

Implements ``M`` independent small MLPs ``u_m : R^{d_e} -> R`` that score
each outcome embedding on attribute ``m``. The heads are person-independent
by default (§5.3): the forward pass consumes only the embedding tensor.

Shapes follow redesign.md §9.4 step 2: given ``E`` of shape
``(B, J, K, d_e)``, :class:`AttributeHeadStack` returns ``(B, J, K, M)``.
"""

from __future__ import annotations

import torch
from torch import nn


# Module-level constants (redesign.md §5.1).
DEFAULT_M: int = 5
DEFAULT_D_E: int = 768
DEFAULT_HIDDEN: int = 128

# 768*128 + 128 + 128*1 + 1 = 98_561 (both Linears carry biases per §0
# "zero init for biases"; orchestrator override supersedes §5.1's stated 98_433,
# which drops the fc1 bias term — see NOTES.md "per-head / salience
# parameter-count reconciliation").
EXPECTED_PARAM_COUNT_PER_HEAD: int = 98_561
# 5 * 98_561 = 492_805
EXPECTED_PARAM_COUNT_STACK_M5: int = 492_805


def _xavier_init_linear(layer: nn.Linear) -> None:
    """Xavier-uniform on weight, zero on bias (redesign.md §0)."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def _kaiming_relu_init_linear(layer: nn.Linear) -> None:
    """Kaiming-uniform (fan_in, ReLU gain) on weight, zero on bias.

    Better matched than Xavier for the ReLU non-linearity in fc1 — Xavier
    targets a symmetric activation (tanh) and under-scales for ReLU,
    leaving early-epoch activations small enough that one head can lose
    the routing race in the weight net and never recover (the dead-m0
    pattern observed at n=10). Kaiming+ReLU restores symmetric variance
    propagation across all M heads at init.
    """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)


class AttributeHead(nn.Module):
    """A single attribute head ``u_m : R^{d_e} -> R`` (§5.1).

    Layers (exactly, per §5.1):
        Linear(d_e -> hidden) -> ReLU -> Linear(hidden -> 1).

    No final activation. No dropout / layernorm / residuals. Biases start
    at zero. The first layer (``fc1``) uses Kaiming-uniform init to
    match the ReLU non-linearity that follows; the output layer
    (``fc2``) keeps Xavier — its output is consumed linearly via the
    weight net, so symmetric init is the right default.
    """

    def __init__(self, d_e: int = DEFAULT_D_E, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.d_e = int(d_e)
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(self.d_e, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()
        _kaiming_relu_init_linear(self.fc1)
        _xavier_init_linear(self.fc2)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Score embeddings on this attribute.

        Shape contract:
            e: (..., d_e)
            out: (..., 1)
        """
        return self.fc2(self.act(self.fc1(e)))

    def num_params(self) -> int:
        """Total trainable parameter count of this head (should be 98_561)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttributeHeadStack(nn.Module):
    """Stack of ``M`` independent :class:`AttributeHead` modules (§5.1, §5.4).

    Shared across people (§5.3): ``forward`` takes only the embedding tensor.
    Person-dependent heads are a Wave-5 ablation (A5 in §11) and are NOT
    supported here.
    """

    def __init__(
        self,
        M: int = DEFAULT_M,
        d_e: int = DEFAULT_D_E,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.M = int(M)
        self.d_e = int(d_e)
        self.hidden = int(hidden)
        # §5.1: "M independent small MLPs" — ModuleList, not one big Linear.
        self.heads = nn.ModuleList(
            [AttributeHead(d_e=self.d_e, hidden=self.hidden) for _ in range(self.M)]
        )

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """Apply each head to the embedding tensor.

        Shape contract:
            E: (..., d_e)     # typically (B, J, K, d_e)
            out: (..., M)     # typically (B, J, K, M)
        """
        # Each head emits (..., 1); concatenate along the last axis to get (..., M).
        per_head = [head(E) for head in self.heads]
        return torch.cat(per_head, dim=-1)

    def num_params(self) -> int:
        """Total trainable parameter count of the stack (should be 492_805 for M=5)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
