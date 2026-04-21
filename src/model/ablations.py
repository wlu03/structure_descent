"""Ablation utility models for PO-LEU (redesign.md §11 rows A7, A8).

Both modules implement an alternative to the attribute decomposition
(``u_m``, ``w_m``) used by PO-LEU: they collapse the decomposition into
a single utility function ``U_k`` computed directly from the outcome
embedding ``e_k`` and person vector ``z_d``. The salience + softmax
choice-probability layers (§7, §8) are left intact so these drop in at
the training-loop level as one-line swaps for :class:`POLEU`.

Two variants, matching the rows of the §11 ablation table:

* :class:`ConcatUtility` — **A7**: ``U_k = h_ψ([e_k; z_d])`` implemented
  as a two-layer MLP over the concatenation. Per §11, A7 is the
  "central ablation" — if it matches A0 on NLL, the paper's
  interpretability claim is the only remaining differentiator.
* :class:`FiLMUtility` — **A8**: ``U_k = h_ψ(e_k; θ_d)`` with
  ``θ_d = g(z_d)`` injected via FiLM-style affine conditioning on the
  hidden activations of the utility MLP.

Both forward passes return ``(logits, intermediates)`` where
``intermediates`` is a per-ablation dataclass exposing the tensors an
interpretability report can read (``U``, ``S``, ``V``; and additionally
``theta_d`` for FiLM). Neither ablation exposes ``A`` (per-head scores)
or ``w`` (weights) — those are the decomposition A7/A8 drop.

The cross-entropy loss for these ablations is the same as for PO-LEU;
callers should import ``cross_entropy_loss`` from
``src.model.po_leu`` directly (documented below; not re-exported here).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.model.salience_net import SalienceNet, UniformSalience


# Module-level constants (redesign.md §11 A7/A8).
#
# Both hidden sizes mirror the attribute-heads default (§5.1 hidden=128)
# so A7/A8 have comparable per-outcome capacity to A0's M=5 stack of
# width-128 heads. Parameter counts are NOT pinned by §11 — A7/A8 are
# structural ablations, not capacity-matched controls.
DEFAULT_HIDDEN: int = 128
DEFAULT_SALIENCE_HIDDEN: int = 64
DEFAULT_D_E: int = 768
DEFAULT_P: int = 26
DEFAULT_TEMPERATURE: float = 1.0


def _xavier_init_linear(layer: nn.Linear) -> None:
    """Xavier-uniform weights, zero biases (redesign.md §0)."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def _build_salience(
    *,
    d_e: int,
    p: int,
    hidden: int,
    uniform: bool,
) -> nn.Module:
    """Return a salience module — shared with :class:`POLEU` via reuse of
    :class:`SalienceNet` / :class:`UniformSalience` (§11 A6 flag).

    No re-implementation: both modules live in ``src.model.salience_net``
    and carry their own param-count / init contracts.
    """
    if uniform:
        return UniformSalience()
    return SalienceNet(d_e=d_e, p=p, hidden=hidden)


# ---------------------------------------------------------------------------
# A7 — concatenation utility
# ---------------------------------------------------------------------------


@dataclass
class ConcatIntermediates:
    """Tensors exposed by :class:`ConcatUtility` for interpretability.

    Intentionally missing ``A`` and ``w`` — A7's central point is that
    there is no attribute decomposition.

    Shape contract:
        U: (B, J, K)  -- per-outcome scalar utility from the MLP
        S: (B, J, K)  -- salience (softmax over K)
        V: (B, J)     -- alternative value Σ_k S_k U_k
    """

    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor


class ConcatUtility(nn.Module):
    """Ablation A7: single MLP on ``[e_k; z_d]`` replaces the decomposition.

    Architecture (mirroring §5.1's per-head two-Linear MLP at a comparable
    hidden width, per §11 "fit similar" hypothesis):

        Linear(d_e + p -> hidden) -> ReLU -> Linear(hidden -> 1)

    Salience and softmax-choice layers are unchanged (§7, §8). Temperature
    is a non-trainable hyperparameter (§8.2).
    """

    def __init__(
        self,
        *,
        d_e: int = DEFAULT_D_E,
        p: int = DEFAULT_P,
        hidden: int = DEFAULT_HIDDEN,
        salience_hidden: int = DEFAULT_SALIENCE_HIDDEN,
        uniform_salience: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.d_e = int(d_e)
        self.p = int(p)
        self.hidden = int(hidden)
        self.salience_hidden = int(salience_hidden)
        self.uniform_salience = bool(uniform_salience)
        # Temperature is not a Parameter (§8.2): store as a buffer so it
        # travels with the module on .to()/.state_dict() without being
        # picked up by optimizers.
        self.register_buffer(
            "temperature", torch.tensor(float(temperature), dtype=torch.float32)
        )

        # MLP over the concatenation.
        self.fc1 = nn.Linear(self.d_e + self.p, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()
        _xavier_init_linear(self.fc1)
        _xavier_init_linear(self.fc2)

        # Salience — reused, not reimplemented.
        self.salience = _build_salience(
            d_e=self.d_e,
            p=self.p,
            hidden=self.salience_hidden,
            uniform=self.uniform_salience,
        )

    def forward(
        self, z_d: torch.Tensor, E: torch.Tensor
    ) -> tuple[torch.Tensor, ConcatIntermediates]:
        """End-to-end forward pass for A7.

        Shape contract:
            z_d: (B, p)
            E:   (B, J, K, d_e)
            logits: (B, J) = V / τ
            intermediates.U: (B, J, K)
            intermediates.S: (B, J, K)
            intermediates.V: (B, J)

        Steps:
            (a) Broadcast ``z_d`` to (B, J, K, p) and concat with ``E``.
            (b) MLP -> (B, J, K, 1) -> squeeze -> (B, J, K) = ``U``.
            (c) ``S`` = salience(E, z_d), softmaxed over K (§7).
            (d) ``V`` = (S * U).sum(-1) (§8.1).
            (e) ``logits`` = V / τ (§8.2).
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

        # (a) broadcast + concat.
        z_bcast = z_d[:, None, None, :].expand(-1, J, K, -1)
        x = torch.cat([E, z_bcast], dim=-1)  # (B, J, K, d_e + p)

        # (b) MLP -> per-outcome scalar utility.
        U = self.fc2(self.act(self.fc1(x))).squeeze(-1)  # (B, J, K)

        # (c) salience (§7).
        S = self.salience(E, z_d)  # (B, J, K)

        # (d) alternative value (§8.1).
        V = (S * U).sum(dim=-1)  # (B, J)

        # (e) logits (§8.2).
        logits = V / self.temperature

        return logits, ConcatIntermediates(U=U, S=S, V=V)

    def num_params(self) -> int:
        """Total trainable parameter count (not pinned by §11)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# A8 — FiLM utility
# ---------------------------------------------------------------------------


@dataclass
class FiLMIntermediates:
    """Tensors exposed by :class:`FiLMUtility` for interpretability.

    Shape contract:
        theta_d: tuple (γ, β), each (B, hidden) — FiLM modulation
                 produced from z_d. Exposes the person-dependent part
                 of the utility function for §12 introspection.
        U: (B, J, K)
        S: (B, J, K)
        V: (B, J)
    """

    theta_d: tuple[torch.Tensor, torch.Tensor]
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor


class FiLMUtility(nn.Module):
    """Ablation A8: FiLM-style person conditioning on a shared utility MLP.

    ``U_k = h_ψ(e_k; θ_d)`` with ``θ_d = g(z_d) = (γ, β)``, both of shape
    ``(B, hidden)``. The utility backbone is:

        h = ReLU(fc1(e_k))          -- shared across people
        h' = γ * h + β              -- per-person affine (FiLM)
        U_k = fc2(h')               -- scalar

    γ is initialized to ``≈ 1`` via a ``+1.0`` offset on the modulator
    output, so at step 0 the backbone approximates a vanilla MLP and
    training starts from a stable, near-identity conditioning point.

    Salience and softmax-choice layers are unchanged (§7, §8).
    Temperature is non-trainable (§8.2).
    """

    def __init__(
        self,
        *,
        d_e: int = DEFAULT_D_E,
        p: int = DEFAULT_P,
        hidden: int = DEFAULT_HIDDEN,
        salience_hidden: int = DEFAULT_SALIENCE_HIDDEN,
        uniform_salience: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()
        self.d_e = int(d_e)
        self.p = int(p)
        self.hidden = int(hidden)
        self.salience_hidden = int(salience_hidden)
        self.uniform_salience = bool(uniform_salience)
        self.register_buffer(
            "temperature", torch.tensor(float(temperature), dtype=torch.float32)
        )

        # FiLM modulator: z_d -> (γ, β). One Linear produces 2*hidden
        # outputs; we split in the forward pass. The +1 offset on γ lives
        # in forward(), not in the bias, so `named_modules()` still
        # recognises a plain Linear(p, 2*hidden).
        self.modulator = nn.Linear(self.p, 2 * self.hidden)

        # Utility backbone.
        self.backbone_fc1 = nn.Linear(self.d_e, self.hidden)
        self.backbone_fc2 = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()

        _xavier_init_linear(self.modulator)
        _xavier_init_linear(self.backbone_fc1)
        _xavier_init_linear(self.backbone_fc2)

        # Salience — reused, not reimplemented.
        self.salience = _build_salience(
            d_e=self.d_e,
            p=self.p,
            hidden=self.salience_hidden,
            uniform=self.uniform_salience,
        )

    def _compute_theta(
        self, z_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce FiLM parameters (γ, β) from ``z_d``.

        Shape contract:
            z_d: (B, p)
            returns (γ, β), each (B, hidden).

        γ is raw_γ + 1 so that Xavier-uniform init (small raw_γ) gives
        γ ≈ 1 at step 0 — identity conditioning.
        """
        raw = self.modulator(z_d)  # (B, 2*hidden)
        gamma_raw = raw[:, : self.hidden]
        beta = raw[:, self.hidden :]
        gamma = gamma_raw + 1.0
        return gamma, beta

    def forward(
        self, z_d: torch.Tensor, E: torch.Tensor
    ) -> tuple[torch.Tensor, FiLMIntermediates]:
        """End-to-end forward pass for A8.

        Shape contract:
            z_d: (B, p)
            E:   (B, J, K, d_e)
            logits: (B, J) = V / τ
            intermediates.theta_d: (γ, β), each (B, hidden)
            intermediates.U: (B, J, K)
            intermediates.S: (B, J, K)
            intermediates.V: (B, J)

        Steps:
            (a) (γ, β) = modulator(z_d).
            (b) h = ReLU(fc1(E))                      -- (B, J, K, hidden)
            (c) h = γ[:, None, None, :] * h + β[:, None, None, :]
            (d) U = fc2(h).squeeze(-1)                -- (B, J, K)
            (e) S = salience(E, z_d); V = Σ_k S_k U_k; logits = V / τ.
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

        # (a) FiLM parameters from z_d.
        gamma, beta = self._compute_theta(z_d)  # each (B, hidden)

        # (b) shared backbone hidden activations.
        h = self.act(self.backbone_fc1(E))  # (B, J, K, hidden)

        # (c) FiLM affine per person; broadcast (B, hidden) across (J, K).
        h = gamma[:, None, None, :] * h + beta[:, None, None, :]

        # (d) scalar utility per outcome.
        U = self.backbone_fc2(h).squeeze(-1)  # (B, J, K)

        # (e) salience + value + logits (§7, §8).
        S = self.salience(E, z_d)
        V = (S * U).sum(dim=-1)
        logits = V / self.temperature

        return logits, FiLMIntermediates(theta_d=(gamma, beta), U=U, S=S, V=V)

    def num_params(self) -> int:
        """Total trainable parameter count (not pinned by §11)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_SALIENCE_HIDDEN",
    "DEFAULT_D_E",
    "DEFAULT_P",
    "DEFAULT_TEMPERATURE",
    "ConcatIntermediates",
    "ConcatUtility",
    "FiLMIntermediates",
    "FiLMUtility",
]
