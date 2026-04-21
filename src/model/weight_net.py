"""Weight network w_m(z_d) — redesign.md §6.

A small MLP R^p -> R^M followed by a softmax (default) or softplus-based
normalization over M. Output rows sum to 1 along the attribute axis and are
interpreted as the "share of attentional budget" person d allocates to each
attribute m (§6.2).

Default config (redesign.md §6.1):
    Linear(p=26 -> hidden=32) -> ReLU -> Linear(hidden=32 -> M=5) -> Softmax(-1)
    param count = 26*32 + 32 + 32*5 + 5 = 1,029.

The ``normalization`` switch exposes the §11 A4 ablation (softplus-normalized
weights) without a second module.
"""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_P = 26
DEFAULT_M = 5
DEFAULT_HIDDEN = 32
# 26*32 + 32 + 32*5 + 5 (§6.1).
EXPECTED_PARAM_COUNT_DEFAULT = 1_029

_ALLOWED_NORMALIZATIONS = ("softmax", "softplus")


class WeightNet(nn.Module):
    """Per-person attribute weight network (§6).

    Parameters
    ----------
    p:
        Effective dimensionality of ``z_d`` (redesign §2.1 binding decision:
        p = 26).
    M:
        Number of attribute heads (redesign §5.2 default: 5).
    hidden:
        Hidden layer width (redesign §6.1 default: 32).
    normalization:
        ``"softmax"`` (default, §6.1) or ``"softplus"`` (§6.2 / A4 ablation).
        Both normalize along the last axis so rows sum to 1.
    """

    def __init__(
        self,
        p: int = DEFAULT_P,
        M: int = DEFAULT_M,
        hidden: int = DEFAULT_HIDDEN,
        normalization: str = "softmax",
    ) -> None:
        super().__init__()

        if normalization not in _ALLOWED_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_ALLOWED_NORMALIZATIONS}; "
                f"got {normalization!r}"
            )

        self.p = p
        self.M = M
        self.hidden = hidden
        self.normalization = normalization

        self.fc1 = nn.Linear(p, hidden)
        self.fc2 = nn.Linear(hidden, M)
        self.relu = nn.ReLU()

        # Softmax module is only used in the softmax branch; kept as a module
        # attribute so the architecture mirrors the §6.1 table exactly.
        self.softmax = nn.Softmax(dim=-1)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # Xavier-uniform linear weights, zero biases (§0 conventions).
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, z_d: torch.Tensor) -> torch.Tensor:
        """Compute per-person attribute weights.

        Parameters
        ----------
        z_d:
            Person feature tensor, shape ``(B, p)``.

        Returns
        -------
        torch.Tensor
            Attribute weight tensor of shape ``(B, M)``. Rows sum to 1 over
            the attribute axis.
        """
        raw = self.fc2(self.relu(self.fc1(z_d)))  # (B, M)

        if self.normalization == "softmax":
            return self.softmax(raw)

        # "softplus" — §6.2 ablation: softplus(raw) row-normalized.
        sp = torch.nn.functional.softplus(raw)
        denom = sp.sum(dim=-1, keepdim=True)
        return sp / denom

    def num_params(self) -> int:
        """Number of trainable parameters (redesign §6.1 target: 1,029)."""
        return sum(int(param.numel()) for param in self.parameters() if param.requires_grad)
