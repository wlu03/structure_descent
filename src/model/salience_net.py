"""Salience network for PO-LEU (redesign.md §7).

Implements ``s_k^{(j)}(e_k, z_d)``: a small MLP that consumes the
concatenation of an outcome embedding ``e_k`` with the person feature
vector ``z_d`` and returns a scalar per outcome, then softmaxes across
the ``K`` outcomes within each alternative ``j``.

Shapes follow redesign.md §9.4 step 5: given ``E`` of shape
``(B, J, K, d_e)`` and ``z_d`` of shape ``(B, p)``, :class:`SalienceNet`
returns ``(B, J, K)``, with rows summing to 1 along the ``K`` axis for
each ``(b, j)``.

This file also exposes :class:`UniformSalience`, the ablation A6
variant (§7.3, §11) that returns ``1/K`` per entry with zero trainable
parameters.
"""

from __future__ import annotations

import torch
from torch import nn


# Module-level constants (redesign.md §7.1 + Appendix B).
DEFAULT_D_E: int = 768
DEFAULT_P: int = 26
DEFAULT_HIDDEN: int = 64

# §7.1 (orchestrator-corrected): 794*64 + 64 + 64*1 + 1 = 50_945.
# Spec §7.1 printed 50_881, which drops the fc1 bias term; per §0 biases
# are present on every Linear. The orchestrator decision records the
# two-Linear-with-biases architecture as authoritative — see NOTES.md
# "per-head / salience parameter-count reconciliation".
EXPECTED_PARAM_COUNT_DEFAULT: int = 50_945


def _xavier_init_linear(layer: nn.Linear) -> None:
    """Xavier-uniform on weight, zero on bias (redesign.md §0)."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class SalienceNet(nn.Module):
    """Salience MLP ``s_k^{(j)}(e_k, z_d)`` (§7.1).

    Layers (exactly, per §7.1):
        Linear(d_e + p -> hidden) -> ReLU -> Linear(hidden -> 1)
        then softmax over ``K`` within each ``(b, j)``.

    The softmax is applied AFTER the MLP produces a scalar per outcome
    (not before): raw scores are computed per ``(b, j, k)`` and then
    normalized across ``k`` within each alternative.

    No dropout / layernorm / residuals. Biases start at zero; weights
    are Xavier-uniform. Deterministic given a fixed seed.
    """

    def __init__(
        self,
        d_e: int = DEFAULT_D_E,
        p: int = DEFAULT_P,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.d_e = int(d_e)
        self.p = int(p)
        self.hidden = int(hidden)
        self.in_dim = self.d_e + self.p  # 794 by default
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()
        _xavier_init_linear(self.fc1)
        _xavier_init_linear(self.fc2)

    def forward(self, E: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        """Compute salience ``s_k^{(j)}`` per outcome, softmaxed over ``K``.

        Shape contract:
            E:   (B, J, K, d_e)
            z_d: (B, p)
            out: (B, J, K), each row along K sums to 1.0

        Steps (§7.1, §9.4 step 5):
            (a) Broadcast ``z_d`` from (B, p) to (B, J, K, p).
            (b) Concatenate with ``E`` along the last dim -> (B, J, K, d_e+p).
            (c) Apply the MLP -> (B, J, K, 1).
            (d) Squeeze the trailing singleton -> (B, J, K).
            (e) Softmax over K within each (B, J).
        """
        if E.dim() != 4:
            raise ValueError(
                f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
            )
        if z_d.dim() != 2:
            raise ValueError(
                f"z_d must be 2-D (B, p); got shape {tuple(z_d.shape)}."
            )
        B, J, K, d_e = E.shape
        if d_e != self.d_e:
            raise ValueError(
                f"E last dim ({d_e}) != configured d_e ({self.d_e})."
            )
        if z_d.shape[0] != B:
            raise ValueError(
                f"Batch mismatch: E has B={B} but z_d has B={z_d.shape[0]}."
            )
        if z_d.shape[1] != self.p:
            raise ValueError(
                f"z_d last dim ({z_d.shape[1]}) != configured p ({self.p})."
            )

        # (a) broadcast z_d to (B, J, K, p) without copying where possible.
        z_bcast = z_d[:, None, None, :].expand(-1, J, K, -1)

        # (b) concat along feature axis -> (B, J, K, d_e + p).
        x = torch.cat([E, z_bcast], dim=-1)

        # (c) MLP -> (B, J, K, 1).
        raw = self.fc2(self.act(self.fc1(x)))

        # (d) squeeze trailing scalar dim -> (B, J, K).
        raw = raw.squeeze(-1)

        # (e) softmax over K within each (b, j).
        return torch.softmax(raw, dim=-1)

    def num_params(self) -> int:
        """Total trainable parameter count (50_945 at defaults; §7.1 corrected)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UniformSalience(nn.Module):
    """Ablation A6: uniform salience ``s_k = 1/K`` (§7.3, §11).

    Matches the :class:`SalienceNet` call signature and output shape so
    the two modules are one-line swaps in the end-to-end forward pass
    (§9.4 step 5). No trainable parameters.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, E: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        """Return a uniform ``(B, J, K)`` tensor of ``1/K`` per entry.

        Shape contract:
            E:   (B, J, K, d_e)
            z_d: (B, p)      (unused; accepted for API parity)
            out: (B, J, K), every entry exactly 1/K.
        """
        if E.dim() != 4:
            raise ValueError(
                f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
            )
        B, J, K, _ = E.shape
        # dtype/device follow E so downstream elementwise ops just work.
        return torch.full(
            (B, J, K),
            fill_value=1.0 / float(K),
            dtype=E.dtype,
            device=E.device,
        )

    def num_params(self) -> int:
        """Always 0: this module has no trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
