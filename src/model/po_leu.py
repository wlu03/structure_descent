"""End-to-end default PO-LEU model (redesign.md §8, §9.4, Appendix A).

Composes :class:`AttributeHeadStack`, :class:`WeightNet`, and
:class:`SalienceNet` (or :class:`UniformSalience` for ablation A6) into a
single ``nn.Module`` whose :meth:`POLEU.forward` returns a ``(logits,
POLEUIntermediates)`` tuple — the intermediates dataclass carries the
tensors needed by the §12 interpretability protocol (``A, w, U, S, V``).

The forward sequence follows Appendix A pseudocode literally; we do not
refactor into a fancier assembly.

Default parameter count (heads + weight net + salience net):
``492,805 + 1,029 + 50,945 = 544,779``, matching the orchestrator-level
reconciliation recorded in ``NOTES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.model.attribute_heads import AttributeHeadStack
from src.model.salience_net import SalienceNet, UniformSalience
from src.model.weight_net import WeightNet


# Default trainable parameter count at spec defaults (M=5, p=26, d_e=768,
# attribute_hidden=128, weight_hidden=32, salience_hidden=64):
#   heads      = 492_805
#   weight_net =   1_029
#   salience   =  50_945
#   total      = 544_779
# See NOTES.md "per-head / salience parameter-count reconciliation".
EXPECTED_PARAM_COUNT_DEFAULT: int = 544_779


@dataclass
class POLEUIntermediates:
    """Tensors emitted by :meth:`POLEU.forward` for §12 interpretability.

    Attributes
    ----------
    A:
        Attribute scores, shape ``(B, J, K, M)``.
    w:
        Person weights (softmax/softplus-normalized over ``M``), ``(B, M)``.
    U:
        Outcome utility ``U = (A * w).sum(-1)``, shape ``(B, J, K)``.
    S:
        Salience softmaxed over ``K``, shape ``(B, J, K)``.
    V:
        Alternative value ``V = (S * U).sum(-1)``, shape ``(B, J)``.
    """

    A: torch.Tensor
    w: torch.Tensor
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return intermediates as ``{"A","w","U","S","V"}`` dict."""
        return {"A": self.A, "w": self.w, "U": self.U, "S": self.S, "V": self.V}


class POLEU(nn.Module):
    """End-to-end PO-LEU model (§8, §9.4, Appendix A).

    Composes the three small nets and returns logits + intermediates.

    Parameters
    ----------
    M:
        Number of attribute heads (§5.2, default 5).
    K:
        Outcomes per alternative (§3, default 3). Stored for introspection;
        inferred from ``E`` at call time.
    J:
        Alternatives per decision (default 10). Stored for introspection;
        inferred from ``E`` at call time.
    d_e:
        Embedding dim (§4.1, default 768).
    p:
        Effective person-feature dim (§2.1 reconciliation, default 26).
    attribute_hidden, weight_hidden, salience_hidden:
        Hidden widths (Appendix B: 128, 32, 64).
    weight_normalization:
        ``"softmax"`` (default, §6.1) or ``"softplus"`` (A4 ablation).
    uniform_salience:
        If ``True`` the salience module is :class:`UniformSalience`
        (ablation A6, §7.3, §11) and contributes zero parameters.
    temperature:
        Softmax temperature ``τ`` from §8.2. Stored as a plain float and
        **not** a trainable parameter (§9.5).
    """

    def __init__(
        self,
        *,
        M: int = 5,
        K: int = 3,
        J: int = 10,
        d_e: int = 768,
        p: int = 26,
        attribute_hidden: int = 128,
        weight_hidden: int = 32,
        salience_hidden: int = 64,
        weight_normalization: str = "softmax",
        uniform_salience: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.M = int(M)
        self.K = int(K)
        self.J = int(J)
        self.d_e = int(d_e)
        self.p = int(p)
        self.attribute_hidden = int(attribute_hidden)
        self.weight_hidden = int(weight_hidden)
        self.salience_hidden = int(salience_hidden)
        self.weight_normalization = weight_normalization
        self.uniform_salience = bool(uniform_salience)

        # Temperature is a plain float, not a tensor/parameter/buffer
        # (§8.2, §9.5). A learned τ is explicitly called out as an ablation
        # and is not implemented here.
        self.temperature = float(temperature)

        self.heads = AttributeHeadStack(
            M=self.M,
            d_e=self.d_e,
            hidden=self.attribute_hidden,
        )
        self.weight_net = WeightNet(
            p=self.p,
            M=self.M,
            hidden=self.weight_hidden,
            normalization=self.weight_normalization,
        )
        if self.uniform_salience:
            self.salience: nn.Module = UniformSalience()
        else:
            self.salience = SalienceNet(
                d_e=self.d_e,
                p=self.p,
                hidden=self.salience_hidden,
            )

    def num_params(self) -> int:
        """Total trainable parameter count across all three submodules.

        Equals ``heads + weight_net + salience``; when
        ``uniform_salience=True`` the salience contribution is zero, so
        the total is ``heads + weight_net`` only.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        z_d: torch.Tensor,
        E: torch.Tensor,
    ) -> tuple[torch.Tensor, POLEUIntermediates]:
        """Run the Appendix A forward sequence.

        Shape contract
        --------------
        Inputs:
            z_d: (B, p)
            E:   (B, J, K, d_e)     (L2-normalized; we do not re-normalize)

        Returns:
            logits: (B, J)
            POLEUIntermediates with
                A: (B, J, K, M)
                w: (B, M)
                U: (B, J, K)
                S: (B, J, K)
                V: (B, J)
        """
        # (a) Attribute scores — step 2 of §9.4 / Appendix A step 1.
        A = self.heads(E)                                  # (B, J, K, M)

        # (b) Person weights — §9.4 step 3 / Appendix A step 2.
        w = self.weight_net(z_d)                           # (B, M)

        # (c) Outcome utility — §9.4 step 4 / Appendix A step 3.
        U = (A * w[:, None, None, :]).sum(dim=-1)          # (B, J, K)

        # (d) Salience — §9.4 step 5 / Appendix A step 4. SalienceNet
        # already applies the softmax-over-K internally (see §7.1).
        S = self.salience(E, z_d)                          # (B, J, K)

        # (e) Alternative value — §8.1 / §9.4 step 6 / Appendix A step 5.
        V = (S * U).sum(dim=-1)                            # (B, J)

        # (f) Temperature-scaled logits — §8.2 / §9.4 step 7.
        logits = V / self.temperature                      # (B, J)

        return logits, POLEUIntermediates(A=A, w=w, U=U, S=S, V=V)


def choice_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the alternative axis (§8.2).

    Shape contract:
        logits: (B, J)
        out:    (B, J), rows sum to 1.0 over J.

    Pure helper — no mutation of ``logits``.
    """
    return torch.softmax(logits, dim=-1)


def cross_entropy_loss(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    omega: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-event cross-entropy with optional importance weights (§9.1).

    Computes per-event NLL
        ``ℓ_t = -log softmax(logits_t)[c*_t]``
    and aggregates:
        - ``omega is None``  → plain mean over the batch.
        - ``omega`` given    → ``sum(omega * ℓ) / sum(omega)`` (§9.1).

    Shape contract:
        logits: (B, J)
        c_star: (B,) int64, entries in ``{0, ..., J-1}``
        omega:  None or (B,) float
        out:    scalar tensor.

    Invariant:
        ``cross_entropy_loss(logits, c*)`` equals
        ``cross_entropy_loss(logits, c*, torch.ones(B))``.
    """
    # Per-event CE without reduction — matches -log softmax(logits)[c*].
    per_event = torch.nn.functional.cross_entropy(
        logits, c_star, reduction="none"
    )  # (B,)

    if omega is None:
        return per_event.mean()

    # Weighted mean; guard division by the sum of weights as in §9.1.
    omega = omega.to(per_event.dtype)
    weighted_sum = (omega * per_event).sum()
    denom = omega.sum()
    return weighted_sum / denom
