"""Loss-augmenting regularizers for PO-LEU training (redesign.md §9.2).

Four pure, composable, differentiable terms — each takes tensors that the
training loop already has (typically :class:`POLEUIntermediates` from
``POLEU.forward`` plus auxiliary inputs) and returns a 0-dim scalar.

| Term                      | Sign in total   | Purpose                                   |
|---------------------------|-----------------|-------------------------------------------|
| Weight-net L2             | +λ_w · L2       | Shrink weight-net parameters (§9.2).      |
| Salience entropy          | **−**λ_H · H(S) | Encourage spread over outcomes (§9.2).    |
| Price monotonicity        | +λ_mono · M     | Financial head non-increasing in price.   |
| Outcome diversity         | +λ_div · D      | Penalize near-duplicate outcomes (§9.2).  |

Entropy sign convention: :func:`salience_entropy` returns the **positive**
entropy ``H(S) ≥ 0``. The combined regularizer subtracts ``λ_H · H(S)`` so
the total loss decreases as ``H`` grows — this is the §9.2 "minimize
negative entropy, i.e. encourage spread" contract.

Price monotonicity is implemented at the **alternative level**: embeddings
``E`` are fixed (frozen encoder output), so we cannot differentiate the
financial head ``u_0`` with respect to the scalar price. Instead, within
each batch we mean-pool ``u_0`` over ``K`` outcomes to get one financial
score per alternative, sort alternatives by price within the batch, and
penalize the squared positive part of consecutive ``Δu_fin / Δprice``.
This is the alt-level surrogate recorded in NOTES.md.

All functions are pure (no side effects, no state mutation), deterministic,
and return 0-dim tensors that backprop cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from src.model.po_leu import POLEU, POLEUIntermediates


# ---------------------------------------------------------------------------
# §9.2 defaults — also pinned in configs/default.yaml (Appendix B).
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_WEIGHT_L2: float = 1e-4
DEFAULT_LAMBDA_SALIENCE_ENTROPY: float = 1e-3
DEFAULT_LAMBDA_MONOTONICITY: float = 1e-3
DEFAULT_LAMBDA_DIVERSITY: float = 1e-4
DEFAULT_MONOTONICITY_ENABLED: bool = False

# Default financial-head index (§5.2 ordering: financial is attribute 0).
DEFAULT_FINANCIAL_HEAD_IDX: int = 0


# ---------------------------------------------------------------------------
# 1. Weight-net L2 (§9.2 row 1).
# ---------------------------------------------------------------------------

def weight_net_l2(weight_net_module: nn.Module) -> torch.Tensor:
    """Sum of squared ``nn.Linear.weight`` tensors in ``weight_net_module``.

    Parameters
    ----------
    weight_net_module:
        The weight network (``model.weight_net`` for a :class:`POLEU`).
        May be any ``nn.Module``; we walk ``named_modules()`` and sum the
        squared weight matrix of every ``nn.Linear`` found.

    Returns
    -------
    torch.Tensor
        0-dim scalar ``Σ_{L ∈ Linears} ||L.weight||_F^2``.

    Notes
    -----
    §9.2 lists the regularizer as ``||φ_w||_2^2`` and motivates it as a
    weight-parameter shrink. Biases are **excluded** (spec says "weight
    parameters"; we take this literally because biases are initialized to
    zero and their L2 tends to be absorbed by downstream layers). Only
    ``nn.Linear.weight`` tensors contribute. The softmax normalization
    module has no trainable weights, so it contributes zero.
    """
    total: Optional[torch.Tensor] = None
    for _, sub in weight_net_module.named_modules():
        if isinstance(sub, nn.Linear):
            term = (sub.weight ** 2).sum()
            total = term if total is None else total + term

    if total is None:
        # No linear layers found — return a genuine 0 scalar on CPU so the
        # caller can still multiply by λ without dtype gymnastics.
        return torch.zeros((), dtype=torch.get_default_dtype())
    return total


# ---------------------------------------------------------------------------
# 2. Salience entropy (§9.2 row 2).
# ---------------------------------------------------------------------------

def salience_entropy(S: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Mean per-(b,j) Shannon entropy of the salience softmax over ``K``.

    ``H(S) = mean_{b,j} [ -Σ_k s_k log(s_k + eps) ]``.

    Parameters
    ----------
    S:
        Salience tensor of shape ``(B, J, K)``. Rows sum to 1 along ``K``.
    eps:
        Small additive stabilizer inside the ``log`` so entries ``s_k = 0``
        do not produce ``-inf``.

    Returns
    -------
    torch.Tensor
        0-dim scalar, non-negative, bounded above by ``log(K)``.

    Sign convention
    ---------------
    Returns the **positive** entropy ``H(S) ≥ 0``. The training loop should
    *subtract* ``λ_H · H(S)`` from the loss to implement the §9.2
    "minimize negative entropy, i.e. encourage spread" prescription. This
    is what :func:`combined_regularizer` does; ad-hoc callers must match.
    """
    if S.dim() != 3:
        raise ValueError(
            f"S must be 3-D (B, J, K); got shape {tuple(S.shape)}."
        )
    # -Σ_k s_k log(s_k + eps)  per (b, j); mean over (b, j).
    log_term = torch.log(S + eps)
    per_bj = -(S * log_term).sum(dim=-1)          # (B, J)
    return per_bj.mean()


# ---------------------------------------------------------------------------
# 3. Price monotonicity (§9.2 row 3, alt-level surrogate).
# ---------------------------------------------------------------------------

def price_monotonicity(
    heads: nn.Module,
    E: torch.Tensor,
    prices: torch.Tensor,
    *,
    financial_head_idx: int = DEFAULT_FINANCIAL_HEAD_IDX,
) -> torch.Tensor:
    """Alt-level price-monotonicity penalty for the financial head.

    §9.2 asks for ``mean[max(0, -∂_{p_j} u_m(e_k))]^2``. Because ``E`` is a
    frozen-encoder output and carries no gradient path to the scalar
    ``price``, we implement the alt-level surrogate:

    1. For each ``(b, j)``, score every outcome embedding with the
       financial head and mean-pool over ``K`` to get one financial
       utility per alternative: ``u_fin(b, j) ∈ R``.
    2. Within each batch element ``b``, sort the ``J`` alternatives by
       ascending price.
    3. Form consecutive finite-difference slopes
       ``Δu / Δp = (u_fin[j+1] - u_fin[j]) / (p[j+1] - p[j])``.
    4. Penalize the mean of ``ReLU(Δu / Δp) ** 2``. A negative or zero
       slope (finance utility non-increasing in price) contributes
       nothing; a positive slope is quadratically penalized.

    Ties in price contribute zero (``Δp = 0`` pairs are masked out).

    Parameters
    ----------
    heads:
        Module exposing ``heads.heads[financial_head_idx]`` — e.g., an
        :class:`AttributeHeadStack` (``heads.heads`` is a ``ModuleList``).
    E:
        Embedding tensor ``(B, J, K, d_e)``.
    prices:
        Per-alternative prices ``(B, J)`` float tensor.
    financial_head_idx:
        Index of the financial head within ``heads.heads``. Default 0
        (§5.2 ordering).

    Returns
    -------
    torch.Tensor
        0-dim, non-negative scalar.
    """
    if E.dim() != 4:
        raise ValueError(
            f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
        )
    if prices.dim() != 2:
        raise ValueError(
            f"prices must be 2-D (B, J); got shape {tuple(prices.shape)}."
        )
    B, J, K, _ = E.shape
    if prices.shape[0] != B or prices.shape[1] != J:
        raise ValueError(
            f"prices shape {tuple(prices.shape)} must be (B={B}, J={J})."
        )

    fin_head = heads.heads[financial_head_idx]

    # Step 1: score every outcome with the financial head and mean-pool over K.
    # head output is (B, J, K, 1); squeeze last dim -> (B, J, K); mean -> (B, J).
    u_fin_per_k = fin_head(E).squeeze(-1)          # (B, J, K)
    u_fin = u_fin_per_k.mean(dim=-1)               # (B, J)

    # Step 2: sort alternatives by price within each batch.
    prices_f = prices.to(u_fin.dtype)
    sort_vals, sort_idx = torch.sort(prices_f, dim=-1)      # both (B, J)
    u_sorted = torch.gather(u_fin, dim=-1, index=sort_idx)  # (B, J)

    # Step 3: consecutive finite differences.
    du = u_sorted[:, 1:] - u_sorted[:, :-1]                 # (B, J-1)
    dp = sort_vals[:, 1:] - sort_vals[:, :-1]               # (B, J-1)

    # Mask out Δp == 0 pairs (price ties carry no monotonicity info).
    valid = dp > 0
    # Avoid division by zero with a safe denominator; masked contributions
    # are zeroed out afterwards.
    dp_safe = torch.where(valid, dp, torch.ones_like(dp))
    slope = du / dp_safe                                    # (B, J-1)
    penalty = torch.clamp(slope, min=0.0) ** 2              # (B, J-1)
    penalty = torch.where(valid, penalty, torch.zeros_like(penalty))

    n_valid = valid.sum()
    if n_valid.item() == 0:
        # All prices tied within every batch element — no monotonicity
        # signal available. Return 0 as a trivially-differentiable scalar
        # tied to E so backward still works.
        return (E * 0.0).sum()

    return penalty.sum() / n_valid.to(penalty.dtype)


# ---------------------------------------------------------------------------
# 4. Outcome diversity (§9.2 row 4).
# ---------------------------------------------------------------------------

def outcome_diversity(E: torch.Tensor) -> torch.Tensor:
    """Mean max-pairwise-cosine between the ``K`` outcomes of each alt.

    For each ``(b, j)``, compute ``max_{k ≠ k'} cos(e_k, e_{k'})`` and mean
    over ``(b, j)``. Assumes rows of ``E`` are already L2-normalized (§4.2
    step 4); cosine similarity then reduces to an inner product.

    Parameters
    ----------
    E:
        Embedding tensor ``(B, J, K, d_e)``, L2-normalized along ``d_e``.

    Returns
    -------
    torch.Tensor
        0-dim scalar in approximately ``[0, 1]``.

    Edge cases
    ----------
    * ``K == 1``: no pairs exist; returns a trivially-differentiable 0.
    """
    if E.dim() != 4:
        raise ValueError(
            f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
        )
    B, J, K, _ = E.shape

    if K < 2:
        # No off-diagonal pairs; diversity trivially zero.
        return (E * 0.0).sum()

    # Gram matrix per (b, j): (B, J, K, K). Cosine == dot product when rows
    # are unit-norm.
    G = torch.einsum("bjkd,bjld->bjkl", E, E)              # (B, J, K, K)

    # Mask the diagonal (k == k'); use -inf so max never picks it.
    diag_mask = torch.eye(K, dtype=torch.bool, device=E.device).view(1, 1, K, K)
    neg_inf = torch.finfo(G.dtype).min
    G_masked = G.masked_fill(diag_mask, neg_inf)

    # Max over (k, k') excluding self; first collapse k' then k.
    max_per_k = G_masked.max(dim=-1).values                # (B, J, K)
    max_per_bj = max_per_k.max(dim=-1).values              # (B, J)
    return max_per_bj.mean()


# ---------------------------------------------------------------------------
# 5. RegularizerConfig (§9.2 λ table + configs/default.yaml Appendix B).
# ---------------------------------------------------------------------------

# Resolve configs/default.yaml relative to the repo root so that
# ``from_default`` keeps working whichever cwd pytest runs from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


@dataclass
class RegularizerConfig:
    """Regularizer λ coefficients (§9.2, Appendix B).

    Attributes
    ----------
    weight_l2:
        λ on :func:`weight_net_l2` (§9.2 row 1).
    salience_entropy:
        λ on :func:`salience_entropy`. Applied with a **minus** sign by
        :func:`combined_regularizer` so entropy is *maximized* in the
        loss (§9.2).
    monotonicity_enabled:
        Gate on the optional price-monotonicity term (§9.2 row 3).
    monotonicity:
        λ on :func:`price_monotonicity`. Ignored if
        ``monotonicity_enabled`` is ``False``.
    diversity:
        λ on :func:`outcome_diversity` (§9.2 row 4).
    """

    weight_l2: float = DEFAULT_LAMBDA_WEIGHT_L2
    salience_entropy: float = DEFAULT_LAMBDA_SALIENCE_ENTROPY
    monotonicity_enabled: bool = DEFAULT_MONOTONICITY_ENABLED
    monotonicity: float = DEFAULT_LAMBDA_MONOTONICITY
    diversity: float = DEFAULT_LAMBDA_DIVERSITY

    @classmethod
    def from_default(cls) -> "RegularizerConfig":
        """Load Appendix B defaults from ``configs/default.yaml``.

        The YAML block ``regularizers:`` must supply
        ``weight_l2``, ``salience_entropy``, ``diversity`` (scalars) and
        ``monotonicity: {enabled: bool, lambda: float, ...}``.

        Returns
        -------
        RegularizerConfig
            Instance with fields taken from the YAML file; ``from_default``
            applies no silent fallbacks — missing keys raise ``KeyError``.
        """
        import yaml

        with open(_DEFAULT_CONFIG_PATH, "r") as fh:
            cfg = yaml.safe_load(fh)

        reg = cfg["regularizers"]
        mono_block = reg["monotonicity"]
        return cls(
            weight_l2=float(reg["weight_l2"]),
            salience_entropy=float(reg["salience_entropy"]),
            monotonicity_enabled=bool(mono_block["enabled"]),
            monotonicity=float(mono_block["lambda"]),
            diversity=float(reg["diversity"]),
        )


# ---------------------------------------------------------------------------
# 6. Combined regularizer (§9.2 total).
# ---------------------------------------------------------------------------

def combined_regularizer(
    model: POLEU,
    intermediates: POLEUIntermediates,
    E: torch.Tensor,
    prices: Optional[torch.Tensor] = None,
    cfg: Optional[RegularizerConfig] = None,
) -> torch.Tensor:
    """Sum the four §9.2 regularizers, signed correctly.

    ``total = λ_w · weight_net_l2
            − λ_H · salience_entropy
            + (λ_mono · price_monotonicity   if enabled and prices is not None else 0)
            + λ_div · outcome_diversity``

    The **minus sign** on the entropy term implements §9.2 "minimize
    negative entropy, i.e. encourage spread": we want a larger ``H(S)`` to
    *decrease* the loss, so ``H`` enters with a negative coefficient.

    Parameters
    ----------
    model:
        :class:`POLEU` instance. Only ``model.weight_net`` and
        ``model.heads`` are consulted.
    intermediates:
        Output of ``model.forward``. Only ``intermediates.S`` is used
        directly; the other tensors are available to callers that want
        to fold in extra diagnostics but are not consumed here.
    E:
        Outcome embedding tensor ``(B, J, K, d_e)`` — same ``E`` passed to
        the forward pass. Used by the diversity and (optional)
        monotonicity terms.
    prices:
        Optional ``(B, J)`` price tensor. Required for the monotonicity
        term; if ``None`` (or ``cfg.monotonicity_enabled`` is ``False``)
        the term is silently skipped.
    cfg:
        Regularizer coefficients. Defaults to
        :meth:`RegularizerConfig.from_default` (Appendix B) when ``None``.

    Returns
    -------
    torch.Tensor
        0-dim scalar to add to the data loss.
    """
    if cfg is None:
        cfg = RegularizerConfig.from_default()

    wl2 = weight_net_l2(model.weight_net)
    ent = salience_entropy(intermediates.S)
    div = outcome_diversity(E)

    total = cfg.weight_l2 * wl2 - cfg.salience_entropy * ent + cfg.diversity * div

    if cfg.monotonicity_enabled and prices is not None:
        mono = price_monotonicity(model.heads, E, prices)
        total = total + cfg.monotonicity * mono

    return total
