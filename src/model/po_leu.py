"""End-to-end default PO-LEU model (redesign.md §8, §9.4, Appendix A).

Composes :class:`AttributeHeadStack`, :class:`WeightNet`, and
:class:`SalienceNet` (or :class:`UniformSalience` for ablation A6) into a
single ``nn.Module`` whose :meth:`POLEU.forward` returns a ``(logits,
POLEUIntermediates)`` tuple — the intermediates dataclass carries the
tensors needed by the §12 interpretability protocol (``A, w, U, S, V``).

The forward sequence follows Appendix A pseudocode literally; we do not
refactor into a fancier assembly.

Strategy B — Sifringer feature-partition tabular residual
---------------------------------------------------------
Optional linear branch (off by default) that gives PO-LEU direct access to
numeric tabular features (``price``, ``log1p_price``, ``price_rank``) which
are otherwise invisible to its frozen sentence-transformer encoder. Adds a
single learned coefficient vector ``beta_tab`` (shape ``(F,)``); forward
becomes::

    V_total[b, j] = V[b, j] + Σ_f β_f · x_tab_std[b, j, f]
    logits        = V_total / τ

where ``x_tab_std`` is the column-wise z-score of ``x_tab`` against
train-set statistics held as buffers on the module. ``β`` is initialised
to zero, so a freshly constructed model with the residual enabled produces
**bit-identical** logits to baseline PO-LEU until training starts.

Backward compatibility: ``forward(z_d, E)`` (no ``x_tab``) and
``forward(z_d, E, x_tab=None)`` reproduce baseline behaviour exactly. The
residual is fully gated by ``tabular_residual_enabled=True`` *and* a
non-``None`` ``x_tab`` argument; either condition false skips the branch.

No L2 penalty is applied to ``β`` (mirrors ``src/baselines/duet_ga.py``
which only regularises the neural branch). Keeping ``β`` un-shrunk
preserves its statistical interpretability — by deliberate design the
linear coefficients of the tabular features can be read directly off the
trained model and reported alongside per-attribute weights.

Default parameter count (heads + weight net + salience net):
``492,805 + 1,029 + 50,945 = 544,779``, matching the orchestrator-level
reconciliation recorded in ``NOTES.md``. Enabling the tabular residual
adds ``F`` parameters (3 with the default feature list) on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from src.model.attribute_heads import AttributeHeadStack
from src.model.salience_net import SalienceNet, UniformSalience
from src.model.weight_net import WeightNet


# Default feature list for the Strategy B residual. ``popularity_rank`` is
# deliberately EXCLUDED: PO-LEU's encoder text already renders a coarse
# popularity band, so adding it numerically creates an identifiability
# fight between the linear branch and the frozen-encoder neural branch.
# The Sifringer L-MNL principle is "linear branch sees what the neural
# branch can't" — the neural branch can't see numeric price, so price
# (raw + monotone transforms) is the right partition.
DEFAULT_TABULAR_FEATURES: tuple[str, ...] = ("price", "log1p_price", "price_rank")


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
    V_residual:
        Tabular linear-branch contribution ``(x_tab_std @ β)`` of shape
        ``(B, J)`` when the Strategy B residual is active, otherwise
        ``None``. Allows downstream interpretability code to attribute
        score mass to the linear vs. neural branches.
    V_total:
        Effective per-alternative value used to form logits. When the
        residual is inactive ``V_total`` is identical to ``V``; when
        active it equals ``V + V_residual``. Always populated for
        symmetry with the actual logit pipeline.
    """

    A: torch.Tensor
    w: torch.Tensor
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor
    V_residual: Optional[torch.Tensor] = None
    V_total: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return intermediates as a tensor dict.

        Always contains ``{"A","w","U","S","V"}`` for backward
        compatibility. ``V_residual`` and ``V_total`` are included only
        when populated (i.e., when the Strategy B residual fired);
        downstream readers that only know about the original five keys
        keep working unchanged.
        """
        out: dict[str, torch.Tensor] = {
            "A": self.A,
            "w": self.w,
            "U": self.U,
            "S": self.S,
            "V": self.V,
        }
        if self.V_residual is not None:
            out["V_residual"] = self.V_residual
        if self.V_total is not None:
            out["V_total"] = self.V_total
        return out


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
    tabular_residual_enabled:
        Strategy B switch. When ``True`` the model allocates a learnable
        coefficient vector ``beta_tab`` of shape ``(F,)`` initialised to
        zero, plus matching standardisation buffers ``x_tab_mean`` and
        ``x_tab_std`` (also zero-init: an unfit module standardises
        identity-style ``(x - 0) / 1``). The residual fires only when
        ``forward`` is called with a non-``None`` ``x_tab``. With
        ``beta_tab`` at zero the logits are bit-identical to baseline
        PO-LEU regardless of ``x_tab`` content — making the change a
        proper residual rather than a re-initialisation. Default
        ``False`` keeps the public surface identical for unmodified
        callers.
    tabular_features:
        Names of the per-alternative tabular features the residual
        consumes, used only as a self-describing label and to size
        ``beta_tab`` / the standardisation buffers when
        ``tabular_residual_enabled=True``. Defaults to
        :data:`DEFAULT_TABULAR_FEATURES` =
        ``("price", "log1p_price", "price_rank")``. Ignored when the
        residual is disabled.
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
        tabular_residual_enabled: bool = False,
        tabular_features: Sequence[str] = DEFAULT_TABULAR_FEATURES,
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

        # ----- Strategy B tabular residual ----------------------------------
        # NB: ``beta_tab`` is a leaf nn.Parameter so it gets picked up by
        # ``model.parameters()`` and the existing Adam optimiser without any
        # special wiring. It is **not** part of ``model.weight_net``, so the
        # §9.2 ``weight_net_l2`` regularizer (which walks ``weight_net.named_modules``)
        # does not see it — exactly the "no L2 on β" property called for by
        # the Sifringer/DUET design (see ``src/baselines/duet_ga.py`` lines
        # 286-303). A future contributor adding a model-wide L2 must pattern-
        # exclude ``beta_tab`` to preserve this; see :meth:`tabular_residual_param_names`.
        self.tabular_residual_enabled = bool(tabular_residual_enabled)
        self.tabular_features: tuple[str, ...] = tuple(tabular_features)
        if self.tabular_residual_enabled:
            n_tab = len(self.tabular_features)
            if n_tab == 0:
                raise ValueError(
                    "tabular_residual_enabled=True requires at least one "
                    "feature name in tabular_features; got an empty list."
                )
            self.beta_tab = nn.Parameter(torch.zeros(n_tab))
            # Buffers persist across `to(device)` and state_dict round-trips.
            # Mean=0, std=1 is the identity transform, so an unfit module
            # passes raw ``x_tab`` through; calling
            # :meth:`set_tabular_feature_stats` later replaces these with
            # train-set statistics for proper standardisation.
            self.register_buffer("x_tab_mean", torch.zeros(n_tab))
            self.register_buffer("x_tab_std", torch.ones(n_tab))
        else:
            # Keep attribute names predictable even when the residual is
            # off so introspection / state_dict comparisons across
            # configs do not have to special-case missing keys. Allocate
            # zero-element tensors that are cheap and signal "unused".
            self.register_parameter("beta_tab", None)
            self.register_buffer("x_tab_mean", None)
            self.register_buffer("x_tab_std", None)

    # --- helpers --------------------------------------------------------
    def set_tabular_feature_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> None:
        """Install column-wise standardisation statistics for ``x_tab``.

        Call once after assembling the train batch (see
        ``src/data/batching.py``) with the per-feature train-set mean
        and std. The standardisation makes the components of ``β``
        directly comparable in scale — a column that varies by 10x in
        natural units and one that varies by 0.1x will both contribute
        on a unit-variance scale after standardisation, so ``|β_f|``
        ranks features by their *standardised* importance, mirroring
        how :func:`src.baselines.data_adapter._build_feature_matrix`
        treats raw vs. log-price.

        ``std`` entries below ``eps`` are clamped up to ``eps`` to keep
        the standardisation finite for degenerate (constant) columns;
        such columns then contribute approximately zero to logits
        regardless of ``β`` since the standardised value is also zero,
        which is the desired no-op behavior.
        """
        if not self.tabular_residual_enabled:
            raise RuntimeError(
                "set_tabular_feature_stats called but tabular_residual is "
                "disabled on this POLEU instance. Construct with "
                "tabular_residual_enabled=True first."
            )
        n_tab = len(self.tabular_features)
        if mean.shape != (n_tab,) or std.shape != (n_tab,):
            raise ValueError(
                f"mean/std must have shape ({n_tab},); got "
                f"mean={tuple(mean.shape)}, std={tuple(std.shape)}."
            )
        # Convert to the same dtype/device as the existing buffer.
        target = self.x_tab_mean
        self.x_tab_mean = mean.detach().to(dtype=target.dtype, device=target.device)
        std_clamped = torch.clamp(std.detach(), min=eps).to(
            dtype=target.dtype, device=target.device
        )
        self.x_tab_std = std_clamped

    def tabular_residual_param_names(self) -> tuple[str, ...]:
        """Return parameter names that must be excluded from any L2 / weight-decay
        scheme to preserve β's interpretability (no shrinkage).

        Empty tuple when the residual is disabled.
        """
        return ("beta_tab",) if self.tabular_residual_enabled else ()

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
        x_tab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, POLEUIntermediates]:
        """Run the Appendix A forward sequence, optionally with the
        Strategy B tabular residual.

        Shape contract
        --------------
        Inputs:
            z_d:   (B, p)
            E:     (B, J, K, d_e)     (L2-normalized; we do not re-normalize)
            x_tab: (B, J, F)          per-alternative tabular features.
                   Optional; required only when
                   ``tabular_residual_enabled=True``. ``F`` must equal
                   ``len(self.tabular_features)``. When ``None`` (or when
                   the residual is disabled at construction time), the
                   linear branch is skipped and the model produces
                   bit-identical logits to baseline PO-LEU.

        Returns:
            logits: (B, J)
            POLEUIntermediates with
                A:          (B, J, K, M)
                w:          (B, M)
                U:          (B, J, K)
                S:          (B, J, K)
                V:          (B, J)
                V_residual: (B, J) when residual fired, else None
                V_total:    (B, J) when residual fired, else None

        Errors
        ------
        Raises ``ValueError`` when ``x_tab`` is supplied but the residual
        is disabled, or when its shape doesn't match
        ``(B, J, len(tabular_features))``.
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

        # ---- Strategy B residual --------------------------------------
        # The residual fires iff (1) the model was constructed with
        # ``tabular_residual_enabled=True`` AND (2) the caller actually
        # threaded ``x_tab`` through. The "not enabled but x_tab passed"
        # case is treated as a programmer error rather than silently
        # ignored: it almost always indicates a config drift between the
        # data assembler (which rendered features) and the model
        # constructor (which forgot to allocate β).
        V_residual: Optional[torch.Tensor] = None
        V_total: Optional[torch.Tensor] = None
        if x_tab is not None:
            if not self.tabular_residual_enabled:
                raise ValueError(
                    "x_tab was supplied but tabular_residual is disabled "
                    "on this POLEU instance. Either construct with "
                    "tabular_residual_enabled=True or pass x_tab=None."
                )
            B, J = V.shape
            n_tab = len(self.tabular_features)
            if x_tab.dim() != 3 or x_tab.shape != (B, J, n_tab):
                raise ValueError(
                    f"x_tab shape {tuple(x_tab.shape)} does not match "
                    f"(B={B}, J={J}, F={n_tab}) where F is "
                    f"len(tabular_features)={self.tabular_features!r}."
                )
            # Standardise per-feature using the registered (possibly
            # train-fit) statistics. Buffers broadcast over (B, J).
            x_std = (x_tab - self.x_tab_mean) / self.x_tab_std  # (B, J, F)
            V_residual = (x_std * self.beta_tab).sum(dim=-1)    # (B, J)
            V_total = V + V_residual
            logits = V_total / self.temperature                 # (B, J)
        else:
            # (f) Temperature-scaled logits — §8.2 / §9.4 step 7.
            logits = V / self.temperature                       # (B, J)

        return logits, POLEUIntermediates(
            A=A, w=w, U=U, S=S, V=V, V_residual=V_residual, V_total=V_total
        )


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
