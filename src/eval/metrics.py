"""Evaluation metrics for PO-LEU (redesign.md §13).

Pure, stateless functions computing the six numbers reported on the held-out
test split: Top-1 accuracy, Top-5 accuracy, MRR, NLL, AIC, BIC. All functions
accept either a ``torch.Tensor`` or a ``numpy.ndarray`` for the logits and
``c*`` labels — inputs are converted with ``torch.as_tensor`` internally.

Conventions (§13 + orchestrator notes)
--------------------------------------
- **Natural log everywhere.** NLL is the mean over N of ``-log P(c*)`` using
  ``torch.nn.functional.log_softmax`` (natural log). BIC uses ``math.log``
  on ``n_train`` (also natural log). AIC and BIC therefore use the same
  log base as NLL.
- **MRR convention.** ``rank`` is 0-indexed (top-ranked item has ``rank=0``)
  and the reciprocal rank is ``1 / (rank + 1)``. Matches §13 bullet
  "MRR: 1 / (rank + 1)".
- **Top-k tie breaking.** We defer to ``torch.topk``'s default behaviour,
  which resolves ties by preferring the lower index. This is stable enough
  for reporting; tests pin the behaviour.
- **AIC formula (§13).** ``AIC = 2k + 2 n_train * NLL``. The caller decides
  which NLL (train or test) to pass; §13 uses test-NLL.
- **BIC formula (§13).** ``BIC = k * log(n_train) + 2 n_train * NLL``.

No file I/O, no RNG, no printing. Stratified breakdowns are handled by the
sibling ``src/eval/strata.py`` module (not this file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


# Type alias for the two input forms we accept.
ArrayLike = Union[torch.Tensor, np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_logits(logits: ArrayLike) -> torch.Tensor:
    """Coerce ``logits`` to a 2-D float torch tensor of shape ``(N, J)``."""
    t = torch.as_tensor(logits)
    if t.dim() != 2:
        raise ValueError(
            f"logits must be 2-D (N, J); got shape {tuple(t.shape)}."
        )
    # NLL and softmax need floating dtype.
    if not torch.is_floating_point(t):
        t = t.to(torch.float32)
    return t


def _as_labels(c_star: ArrayLike, n: int, num_classes: int) -> torch.Tensor:
    """Coerce ``c_star`` to an int64 tensor of shape ``(N,)`` and range-check."""
    t = torch.as_tensor(c_star)
    if t.dim() != 1:
        raise ValueError(
            f"c_star must be 1-D (N,); got shape {tuple(t.shape)}."
        )
    if t.shape[0] != n:
        raise ValueError(
            f"c_star length ({t.shape[0]}) does not match logits N ({n})."
        )
    t = t.to(torch.int64)
    if num_classes > 0 and t.numel() > 0:
        lo, hi = int(t.min().item()), int(t.max().item())
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"c_star contains out-of-range labels "
                f"[{lo}, {hi}] for J={num_classes}."
            )
    return t


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def topk_accuracy(
    logits: ArrayLike,
    c_star: ArrayLike,
    k: int = 1,
) -> float:
    """Fraction of rows where ``c*`` is among the top-``k`` logits.

    Shape contract:
        logits: (N, J)
        c_star: (N,)
        out:    Python float.

    ``k`` is clamped to ``min(k, J)``. Ties are broken by
    ``torch.topk`` (default: lower index first).
    """
    t = _as_logits(logits)
    N, J = t.shape
    y = _as_labels(c_star, n=N, num_classes=J)

    if N == 0:
        return 0.0

    k = max(1, min(int(k), J))
    # top-k indices along the J axis, (N, k).
    topk_idx = torch.topk(t, k=k, dim=-1).indices
    # compare against c_star broadcast to (N, 1).
    hits = (topk_idx == y.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().item())


def mrr(logits: ArrayLike, c_star: ArrayLike) -> float:
    """Mean Reciprocal Rank using the ``1 / (rank + 1)`` convention (§13).

    Shape contract:
        logits: (N, J)
        c_star: (N,)
        out:    Python float.

    ``rank`` is 0-indexed — the top-ranked alternative has ``rank=0`` and
    reciprocal rank ``1.0``; the second gets ``0.5``; etc. Computed as
    ``1 + (# logits strictly greater than the logit at c*) +
    0.5 * (# ties other than c* itself)`` to keep ties unbiased (an
    all-equal row contributes ``2 / (J + 1)`` rather than ``1`` or ``1/J``).
    """
    t = _as_logits(logits)
    N, J = t.shape
    y = _as_labels(c_star, n=N, num_classes=J)

    if N == 0:
        return 0.0

    # gather the logit at the true class, shape (N, 1).
    true_logit = t.gather(dim=-1, index=y.unsqueeze(-1))

    # strictly greater + ties are both relative to the true logit.
    strictly_greater = (t > true_logit).sum(dim=-1).to(torch.float64)
    ties = (t == true_logit).sum(dim=-1).to(torch.float64) - 1.0  # exclude self
    # 0-indexed expected rank under a random tie break.
    rank = strictly_greater + 0.5 * ties
    reciprocal = 1.0 / (rank + 1.0)
    return float(reciprocal.mean().item())


def nll(logits: ArrayLike, c_star: ArrayLike) -> float:
    """Mean per-event negative log-likelihood, natural log.

    Shape contract:
        logits: (N, J)
        c_star: (N,)
        out:    Python float.

    Equivalent to ``torch.nn.functional.cross_entropy(logits, c*,
    reduction='mean')`` — softmax over ``J`` followed by ``-log P(c*)``
    averaged over ``N``.
    """
    t = _as_logits(logits)
    N, J = t.shape
    y = _as_labels(c_star, n=N, num_classes=J)

    if N == 0:
        return 0.0

    return float(F.cross_entropy(t, y, reduction="mean").item())


def aic(nll_val: float, k: int, n_train: int) -> float:
    """AIC from a pre-computed NLL: ``2k + 2 * n_train * nll_val`` (§13).

    ``nll_val`` is the mean per-event NLL (natural log); §13 plugs in the
    test-split value, but this function is purely the formula so the caller
    is free to use train-NLL for diagnostics.
    """
    return 2.0 * float(k) + 2.0 * float(n_train) * float(nll_val)


def bic(nll_val: float, k: int, n_train: int) -> float:
    """BIC from a pre-computed NLL: ``k * log(n_train) + 2 n_train * nll_val``.

    ``log`` is the natural log (``math.log``), matching the NLL/AIC
    convention. As with :func:`aic`, the caller decides which NLL to pass.
    """
    return float(k) * math.log(float(n_train)) + 2.0 * float(n_train) * float(nll_val)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """All six §13 numbers plus the ``k`` / ``n_train`` used for AIC/BIC.

    Attributes
    ----------
    top1, top5:
        Top-k accuracies in ``[0, 1]``.
    mrr_val:
        Mean Reciprocal Rank, ``1 / (rank + 1)`` convention.
    nll_val:
        Mean per-event NLL (natural log).
    aic_val, bic_val:
        ``2k + 2 n_train * NLL`` and ``k log n_train + 2 n_train * NLL``.
    n_params:
        Trainable parameter count ``k`` (e.g., :meth:`POLEU.num_params`).
    n_train:
        Training-set size used for the AIC/BIC penalty base.
    """

    top1: float
    top5: float
    mrr_val: float
    nll_val: float
    aic_val: float
    bic_val: float
    n_params: int
    n_train: int

    def to_dict(self) -> dict:
        """Plain-dict view for logging / JSON serialization."""
        return asdict(self)


def compute_all(
    logits: ArrayLike,
    c_star: ArrayLike,
    *,
    n_params: int,
    n_train: int,
) -> EvalMetrics:
    """Compute all six §13 numbers in one pass.

    Shape contract:
        logits:    (N, J)
        c_star:    (N,)
        n_params:  int, trainable parameter count ``k`` (§13 uses 544,779
                   for the default config).
        n_train:   int, size of the training split.
        out:       :class:`EvalMetrics`.

    Top-k list is fixed to ``[1, 5]`` per Appendix B / §13.
    """
    t = _as_logits(logits)
    N, J = t.shape
    y = _as_labels(c_star, n=N, num_classes=J)

    top1 = topk_accuracy(t, y, k=1)
    top5 = topk_accuracy(t, y, k=5)
    mrr_v = mrr(t, y)
    nll_v = nll(t, y)
    aic_v = aic(nll_v, k=n_params, n_train=n_train)
    bic_v = bic(nll_v, k=n_params, n_train=n_train)

    return EvalMetrics(
        top1=top1,
        top5=top5,
        mrr_val=mrr_v,
        nll_val=nll_v,
        aic_val=aic_v,
        bic_val=bic_v,
        n_params=int(n_params),
        n_train=int(n_train),
    )
