"""Unified evaluation harness for PO-LEU-comparable baselines.

This harness is a thin wrapper around :mod:`src.eval.metrics` that
guarantees baselines and PO-LEU are scored with **identical** formulas
and tie-breaking:

* top-1 / top-5 use :func:`src.eval.metrics.topk_accuracy`
  (``torch.topk`` tie-break: lower index wins).
* MRR uses :func:`src.eval.metrics.mrr` with the unbiased
  ``1 + strictly_greater + 0.5 * ties`` rank definition.
* NLL uses :func:`src.eval.metrics.nll` (natural-log cross-entropy).
* AIC / BIC use :func:`src.eval.metrics.aic` / :func:`src.eval.metrics.bic`.

Breakdowns (per-category, repeat-vs-novel) are computed locally here
because :mod:`src.eval.metrics` is intentionally stratification-free.

Every baseline exposes :meth:`FittedBaseline.score_events` returning a
list of per-event utility vectors; this harness stacks them into the
``(N, J)`` tensor that :mod:`src.eval.metrics` consumes, so all methods
are scored apples-to-apples.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from src.eval.metrics import (
    aic as _aic,
    bic as _bic,
    ece as _ece,
    mcfadden_pseudo_r2 as _pseudo_r2,
    mrr as _mrr,
    nll as _nll,
    topk_accuracy,
)

from .base import BaselineEventBatch, BaselineReport, FittedBaseline


def _stack_scores(
    scores_list: list[np.ndarray], *, context: str
) -> np.ndarray:
    """Stack a ragged list of per-event score vectors into an ``(N, J)``
    float32 array, requiring a uniform ``J``.

    Raises ``ValueError`` with a helpful message if the choice-set size
    varies across events — every current baseline + PO-LEU assumes
    uniform J per batch.
    """
    if not scores_list:
        return np.empty((0, 0), dtype=np.float32)
    first_J = int(np.asarray(scores_list[0]).shape[0])
    if first_J == 0:
        raise ValueError(f"{context}: first event has a 0-length score vector")
    out = np.empty((len(scores_list), first_J), dtype=np.float32)
    for i, s in enumerate(scores_list):
        arr = np.asarray(s, dtype=np.float32).ravel()
        if arr.shape[0] != first_J:
            raise ValueError(
                f"{context}: score vector at event {i} has length "
                f"{arr.shape[0]}, expected {first_J} (baselines require "
                "uniform choice-set size per batch)."
            )
        out[i] = arr
    return out


def evaluate_baseline(
    fitted: FittedBaseline,
    batch: BaselineEventBatch,
    train_n_events: Optional[int] = None,
    fit_time_seconds: float = 0.0,
) -> BaselineReport:
    """Score a fitted baseline on a held-out batch and return a
    :class:`BaselineReport` with the unified metric panel.

    Parameters
    ----------
    fitted
        A baseline implementing the :class:`FittedBaseline` protocol:
        ``score_events(batch)``, ``n_params``, ``description``.
    batch
        The held-out split (typically test).
    train_n_events
        Size of the training split that produced ``fitted`` — used for
        the AIC / BIC information-criterion penalty base. Falls back to
        ``len(batch)`` if ``None``, with a note that the resulting
        AIC / BIC is then descriptive-only (the formulas assume the
        training size).
    fit_time_seconds
        Wall-clock seconds spent fitting (propagated into the report).

    Returns
    -------
    BaselineReport
        Standardized report with ``top1``, ``top5``, ``mrr``,
        ``test_nll``, ``aic``, ``bic``, ``n_events`` + breakdowns.
    """
    t0 = time.perf_counter()
    scores_list = fitted.score_events(batch)
    scoring_time = time.perf_counter() - t0

    n = len(scores_list)
    if n == 0:
        raise ValueError("evaluate_baseline received an empty batch")
    if n != batch.n_events:
        raise ValueError(
            f"score_events returned {n} entries but batch has "
            f"{batch.n_events} events"
        )

    logits = _stack_scores(scores_list, context="evaluate_baseline")
    c_star = np.asarray(batch.chosen_indices, dtype=np.int64)

    # --- core metrics (delegated to src.eval.metrics for parity) -------
    top1 = topk_accuracy(logits, c_star, k=1)
    top5 = topk_accuracy(logits, c_star, k=5)
    mrr_val = _mrr(logits, c_star)
    nll_val = _nll(logits, c_star)

    n_for_ic = train_n_events if train_n_events is not None else n
    k = max(int(fitted.n_params), 1)
    aic_val = _aic(nll_val, k=k, n_train=n_for_ic)
    bic_val = _bic(nll_val, k=k, n_train=n_for_ic)
    pseudo_r2_val = _pseudo_r2(nll_val, J=int(logits.shape[1]))
    ece_val = _ece(logits, c_star)

    # --- breakdowns (stratification not in src.eval.metrics) ----------
    per_cat: Dict[str, Dict[str, int]] = {}
    per_bucket: Dict[str, Dict[str, int]] = {
        "repeat": {"t1": 0, "n": 0},
        "novel": {"t1": 0, "n": 0},
    }
    # Per-event top-1 correctness, computed with the same tie-break as
    # ``topk_accuracy`` so breakdowns sum to the aggregate top-1.
    top1_idx = np.argmax(logits, axis=-1)  # ties -> lowest index (argmax)
    # Note: np.argmax ties lowest-index, torch.topk ties lowest-index
    # too — both match; aggregate top1 reported above uses torch.topk.
    correct_mask = (top1_idx == c_star)

    for i, (cat, meta) in enumerate(zip(batch.categories, batch.metadata)):
        correct = int(correct_mask[i])
        cat_bucket = per_cat.setdefault(cat, {"t1": 0, "n": 0})
        cat_bucket["t1"] += correct
        cat_bucket["n"] += 1
        key = "repeat" if meta.get("is_repeat", False) else "novel"
        per_bucket[key]["t1"] += correct
        per_bucket[key]["n"] += 1

    metrics = {
        "top1": float(top1),
        "top5": float(top5),
        "mrr": float(mrr_val),
        "test_nll": float(nll_val),
        "pseudo_r2": float(pseudo_r2_val),
        "aic": float(aic_val),
        "bic": float(bic_val),
        "n_events": int(n),
        "ece": float(ece_val),
    }

    per_cat_rows = [
        {"category": c, "top1": d["t1"] / d["n"], "n_events": d["n"]}
        for c, d in sorted(per_cat.items(), key=lambda kv: -kv[1]["n"])
    ]
    per_rep_rows = [
        {"type": k_, "top1": v["t1"] / max(v["n"], 1), "n_events": v["n"]}
        for k_, v in per_bucket.items()
    ]

    try:
        import pandas as pd

        per_cat_df = pd.DataFrame(per_cat_rows)
        per_rep_df = pd.DataFrame(per_rep_rows)
    except ImportError:
        per_cat_df = per_cat_rows
        per_rep_df = per_rep_rows

    return BaselineReport(
        name=fitted.name,
        n_params=int(fitted.n_params),
        metrics=metrics,
        per_category=per_cat_df,
        per_repeat_novel=per_rep_df,
        fit_time_seconds=float(fit_time_seconds),
        extra={"scoring_time_seconds": scoring_time},
    )
