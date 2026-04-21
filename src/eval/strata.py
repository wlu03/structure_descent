"""Stratified evaluation breakdowns (redesign.md §13, §12.3).

Implements the §13 "stratified breakdowns: by category, repeat/novel,
activity tertile, time-of-day, and dominant attribute" over PO-LEU logits,
plus the §12.3 dominant-attribute computation
``m* = argmax_m w_m · |u_m(e_{c*})|``.

Design notes
------------
* Pure functions; every stratifier takes ``(logits, c_star, group_key)``-
  shape inputs and returns a ``{group_value: {metric: value, ..., "n": n}}``
  dict. This lets callers consume any stratifier uniformly.
* No file I/O; deterministic.
* Empty groups (zero events) are **omitted** from the output (not present
  with NaNs).
* Default metrics computed: ``top1``, ``top5``, ``mrr``, ``nll`` (no
  AIC/BIC — those need ``n_train`` + ``k`` which are out of scope here).
  If ``src/eval/metrics.py`` exists and exposes ``compute_all``, callers
  can pass it in as ``compute_metrics_fn`` — the default otherwise is
  this module's own :func:`_default_metrics`.
* §12.3 aggregation across the K outcomes: we use the **mean over K** of
  ``|u_m(e_k)|`` for the chosen alternative (the spec writes
  ``u_m(e_{c*})`` for a single embedding; PO-LEU carries K outcomes per
  alternative, so we reduce across K to land on a per-(event, attribute)
  scalar). See NOTES.md.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.eval.metrics import mrr as _mrr
from src.eval.metrics import nll as _nll
from src.eval.metrics import topk_accuracy
from src.model.po_leu import POLEUIntermediates


# ---------------------------------------------------------------------------
# Default metric function (used when the caller does not pass one).
# ---------------------------------------------------------------------------


def _default_metrics(
    logits: torch.Tensor,
    c_star: torch.Tensor,
) -> dict[str, float]:
    """Compute ``top1``, ``top5``, ``mrr``, ``nll`` for a group of events.

    Shape contract:
        logits: (n, J)   — per-event scores over J alternatives.
        c_star: (n,) int64 in [0, J).

    Returns a dict with keys ``"top1", "top5", "mrr", "nll"``. Delegates
    to sibling :mod:`src.eval.metrics` so the strata module stays in
    lockstep with the §13 metric definitions (natural-log NLL,
    ``1 / (rank + 1)`` MRR, ``torch.topk`` tie-break). AIC/BIC are
    deliberately **not** included — those need ``n_train`` + ``k`` and
    are a full-report concern, not a strata concern.
    """
    return {
        "top1": topk_accuracy(logits, c_star, k=1),
        "top5": topk_accuracy(logits, c_star, k=5),
        "mrr": _mrr(logits, c_star),
        "nll": _nll(logits, c_star),
    }


# ---------------------------------------------------------------------------
# Generic stratifier.
# ---------------------------------------------------------------------------


def stratify_by_key(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    group_key: np.ndarray,
    *,
    compute_metrics_fn: Callable[[torch.Tensor, torch.Tensor], dict[str, float]]
    | None = None,
) -> dict:
    """Group events by ``group_key`` and report metrics per group.

    Shape contract:
        logits:     (N, J)
        c_star:     (N,) int64, entries in [0, J)
        group_key:  (N,) numpy array of any hashable dtype.

    Returns
    -------
    dict
        ``{group_value: {"top1": ..., "top5": ..., "mrr": ..., "nll": ...,
        "n": n_in_group}}``. Missing groups (zero events) are omitted.

    Notes
    -----
    * The generic stratifier is the workhorse; the wrappers below just
      pre-compute ``group_key`` from a semantic feature (category, hour,
      activity, dominant-attribute index, ...).
    * ``compute_metrics_fn`` defaults to :func:`_default_metrics`. Tests
      can inject a dummy fn to assert the grouping logic in isolation.
    """
    if compute_metrics_fn is None:
        compute_metrics_fn = _default_metrics

    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (N, J); got shape {tuple(logits.shape)}")
    N = logits.shape[0]
    if c_star.shape[0] != N:
        raise ValueError(
            f"c_star length {c_star.shape[0]} != logits N {N}"
        )
    group_key = np.asarray(group_key)
    if group_key.shape[0] != N:
        raise ValueError(
            f"group_key length {group_key.shape[0]} != logits N {N}"
        )

    out: dict = {}
    # Preserve first-appearance order for determinism with np.unique.
    unique_vals, first_idx = np.unique(group_key, return_index=True)
    order = np.argsort(first_idx)
    for v in unique_vals[order]:
        mask = group_key == v
        n = int(mask.sum())
        if n == 0:
            continue  # defensive; np.unique guarantees n>=1, but keep it.
        idx = torch.from_numpy(np.asarray(np.where(mask)[0], dtype=np.int64))
        group_logits = logits.index_select(0, idx)
        group_c_star = c_star.index_select(0, idx)
        metrics = dict(compute_metrics_fn(group_logits, group_c_star))
        metrics["n"] = n
        # Cast numpy scalar keys to plain Python so dict keys are
        # predictable (int/str/bool), not np.int64 / np.str_.
        key = v.item() if hasattr(v, "item") else v
        out[key] = metrics
    return out


# ---------------------------------------------------------------------------
# Stratifier wrappers.
# ---------------------------------------------------------------------------


def category_breakdown(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    category: np.ndarray,
) -> dict:
    """Stratify by event category (§13).

    Shape contract:
        logits:   (N, J)
        c_star:   (N,)
        category: (N,) — hashable entries (str or int).
    """
    return stratify_by_key(logits, c_star, np.asarray(category))


def repeat_novel_breakdown(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    is_novel: np.ndarray,
) -> dict:
    """Stratify by repeat (``is_novel == False``) vs novel (``True``).

    Shape contract:
        logits:   (N, J)
        c_star:   (N,)
        is_novel: (N,) bool.

    Missing groups are omitted: e.g. an all-novel batch returns only the
    ``"novel"`` key.
    """
    is_novel = np.asarray(is_novel).astype(bool)
    labels = np.where(is_novel, "novel", "repeat")
    return stratify_by_key(logits, c_star, labels)


def activity_tertile_breakdown(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    customer_activity: np.ndarray,
) -> dict:
    """Split by customer-activity tertile (low / mid / high) (§13).

    ``customer_activity`` is a continuous per-event measure (e.g.
    ``log1p(n_events)``). Tertile boundaries are the 1/3 and 2/3
    ``np.quantile`` values; ties fall into the lower bucket because
    ``np.digitize`` uses ``right=False`` (default). Labels are
    ``"low" | "mid" | "high"``.

    Shape contract:
        logits:             (N, J)
        c_star:              (N,)
        customer_activity:  (N,) float-like.
    """
    customer_activity = np.asarray(customer_activity, dtype=float)
    # Quantile-based edges → roughly equal-size tertiles. ``np.digitize``
    # buckets with right=False so ``customer_activity <= q1/3 → low``, etc.
    q_lo, q_hi = np.quantile(customer_activity, [1.0 / 3.0, 2.0 / 3.0])
    bin_idx = np.digitize(customer_activity, [q_lo, q_hi], right=False)
    # bin_idx in {0,1,2}. ``right=False`` puts values exactly on the
    # boundary into the lower bin, which matches numpy's default quantile
    # conventions and stays deterministic under ties.
    labels = np.array(["low", "mid", "high"])[bin_idx]
    return stratify_by_key(logits, c_star, labels)


# Time-of-day bucket boundaries per the task spec.
#   morning   : 6  <= hour < 12
#   afternoon : 12 <= hour < 18
#   evening   : 18 <= hour < 24
#   night     : 0  <= hour < 6
_TIME_OF_DAY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 6),
    ("morning", 6, 12),
    ("afternoon", 12, 18),
    ("evening", 18, 24),
)


def time_of_day_breakdown(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    hour: np.ndarray,
) -> dict:
    """Stratify by time-of-day bucket (§13).

    Buckets (half-open on the right, matching §13):
        night: 0-6, morning: 6-12, afternoon: 12-18, evening: 18-24.

    Shape contract:
        logits: (N, J)
        c_star: (N,)
        hour:   (N,) int in [0, 24).
    """
    hour = np.asarray(hour, dtype=int)
    if (hour < 0).any() or (hour >= 24).any():
        raise ValueError("hour entries must lie in [0, 24)")

    labels = np.empty(hour.shape, dtype=object)
    for name, lo, hi in _TIME_OF_DAY_BUCKETS:
        mask = (hour >= lo) & (hour < hi)
        labels[mask] = name
    return stratify_by_key(logits, c_star, labels)


# ---------------------------------------------------------------------------
# §12.3 dominant-attribute computation and breakdown.
# ---------------------------------------------------------------------------


def dominant_attribute(
    intermediates: POLEUIntermediates,
    c_star: torch.Tensor,
) -> torch.Tensor:
    """Per-event dominant attribute ``m* = argmax_m w_m · |u_m(e_{c*})|``.

    PO-LEU carries ``K`` outcomes per alternative; the spec notation
    ``u_m(e_{c*})`` writes a single embedding. We aggregate across the K
    outcomes by taking the **mean** of ``|u_m(e_k^{(c*)})|``; this
    preserves the "which attribute drives this alternative's score"
    reading without mixing in salience (which belongs to a separate
    interpretability axis).

    Shape contract:
        intermediates.A: (N, J, K, M)
        intermediates.w: (N, M)
        c_star:          (N,) int64 in [0, J).

    Returns
    -------
    torch.Tensor
        Shape ``(N,)``, dtype ``int64``, entries in ``[0, M)``.
    """
    A = intermediates.A
    w = intermediates.w
    if A.dim() != 4:
        raise ValueError(f"intermediates.A must be 4-D; got {tuple(A.shape)}")
    if w.dim() != 2:
        raise ValueError(f"intermediates.w must be 2-D; got {tuple(w.shape)}")
    N, J, K, M = A.shape
    if w.shape != (N, M):
        raise ValueError(
            f"intermediates.w shape {tuple(w.shape)} inconsistent with A (N,M)=({N},{M})"
        )
    if c_star.shape != (N,):
        raise ValueError(f"c_star must be shape ({N},); got {tuple(c_star.shape)}")

    # Gather |u_m(e_k)| for the chosen alternative of each event.
    c_idx = c_star.to(torch.int64).view(N, 1, 1, 1).expand(N, 1, K, M)
    A_chosen = A.gather(1, c_idx).squeeze(1).abs()           # (N, K, M)
    abs_u_chosen_mean = A_chosen.mean(dim=1)                 # (N, M)  mean over K

    score = w * abs_u_chosen_mean                             # (N, M)
    m_star = score.argmax(dim=1).to(torch.int64)              # (N,)
    return m_star


def dominant_attribute_breakdown(
    logits: torch.Tensor,
    c_star: torch.Tensor,
    intermediates: POLEUIntermediates,
) -> dict:
    """Stratify events by §12.3 dominant attribute index.

    Keys are plain Python ``int`` attribute indices in ``[0, M)`` —
    ``stratify_by_key`` casts the ``np.int64`` group key via ``.item()``.

    Shape contract:
        logits:        (N, J)
        c_star:         (N,)
        intermediates:  POLEUIntermediates from the same forward pass.
    """
    m_star = dominant_attribute(intermediates, c_star).detach().cpu().numpy()
    return stratify_by_key(logits, c_star, m_star)
