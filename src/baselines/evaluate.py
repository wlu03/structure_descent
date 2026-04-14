"""
Shared evaluation harness for the baseline suite.

Every baseline, regardless of internal representation, is evaluated by
calling fitted.score_events(batch) and converting the returned per-event
utility vectors into:
  - top-1, top-5 accuracy
  - Mean reciprocal rank
  - Per-event negative log-likelihood (via softmax over scores)
  - AIC / BIC using fitted.n_params and training size
  - Breakdowns by category and repeat/novel

This guarantees apples-to-apples comparison: every baseline is scored on
the same metric panel, and the only thing that differs is how it builds
its score_events callable.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from scipy.special import softmax

from .base import BaselineEventBatch, BaselineReport, FittedBaseline


def evaluate_baseline(
    fitted: FittedBaseline,
    batch: BaselineEventBatch,
    train_n_events: Optional[int] = None,
    fit_time_seconds: float = 0.0,
) -> BaselineReport:
    """
    Score a fitted baseline on a held-out batch and return a BaselineReport.

    Parameters
    ----------
    fitted : FittedBaseline
        Must implement score_events(batch), n_params, description.
    batch : BaselineEventBatch
        The held-out split (usually test).
    train_n_events : optional int
        Training set size used for AIC/BIC. Falls back to len(batch) if None.
        For information-criterion correctness you should pass the actual
        training size.
    fit_time_seconds : float
        Wall-clock seconds spent fitting (propagated into the report).

    Returns
    -------
    BaselineReport
    """
    t0 = time.perf_counter()
    scores_list = fitted.score_events(batch)
    scoring_time = time.perf_counter() - t0

    n = len(scores_list)
    if n == 0:
        raise ValueError("evaluate_baseline received an empty batch")
    if n != batch.n_events:
        raise ValueError(
            f"score_events returned {n} entries but batch has {batch.n_events} events"
        )

    top1 = 0
    top5 = 0
    rr_list: list[float] = []
    total_nll = 0.0

    per_cat: Dict[str, Dict[str, int]] = {}
    per_bucket: Dict[str, Dict[str, int]] = {
        "repeat": {"t1": 0, "n": 0},
        "novel": {"t1": 0, "n": 0},
    }

    for scores, chosen, cat, meta in zip(
        scores_list,
        batch.chosen_indices,
        batch.categories,
        batch.metadata,
    ):
        scores = np.asarray(scores, dtype=float).ravel()
        if scores.shape[0] == 0:
            raise ValueError("score_events returned a 0-length score vector")
        probs = softmax(scores)
        rank = int(np.sum(scores > scores[chosen]))
        correct = int(rank == 0)
        top1 += correct
        top5 += int(rank < 5)
        rr_list.append(1.0 / (rank + 1))
        total_nll -= float(np.log(probs[chosen] + 1e-12))

        cat_bucket = per_cat.setdefault(cat, {"t1": 0, "n": 0})
        cat_bucket["t1"] += correct
        cat_bucket["n"] += 1

        key = "repeat" if meta.get("is_repeat", False) else "novel"
        per_bucket[key]["t1"] += correct
        per_bucket[key]["n"] += 1

    test_nll_per_event = total_nll / n
    n_for_ic = train_n_events if train_n_events is not None else n
    k = max(fitted.n_params, 1)
    aic = 2 * k + 2 * n_for_ic * test_nll_per_event
    bic = np.log(max(n_for_ic, 2)) * k + 2 * n_for_ic * test_nll_per_event

    metrics = {
        "top1": top1 / n,
        "top5": top5 / n,
        "mrr": float(np.mean(rr_list)),
        "test_nll": test_nll_per_event,
        "aic": float(aic),
        "bic": float(bic),
        "n_events": n,
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
