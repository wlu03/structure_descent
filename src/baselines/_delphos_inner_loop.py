"""Flat (non-hierarchical) MNL inner loop for the Delphos baseline.

This is the ablation branch of ``old_pipeline/src/inner_loop.py``'s
``fit_weights_no_hierarchy``: one global weight vector, no per-category
or per-customer deviations. See the Option B rationale in
``docs/llm_baselines/delphos_baseline.md`` for why the hierarchical fit
is intentionally dropped.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from ._delphos_dsl import DSLStructure


def fit_weights_flat(
    structure: DSLStructure,
    features_list: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
    sigma: float = 10.0,
    maxiter: int = 500,
    ftol: float = 1e-9,
    event_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit a flat conditional-logit model.

    Objective::

        NLL(w) = sum_e  -log_softmax(X_e @ w)[chosen_e]
                 + (1 / (2 * sigma**2)) * ||w||_2^2

    Returns
    -------
    np.ndarray
        Shape ``(n_terms,)`` where ``n_terms == len(structure.terms)``.
    """
    n_terms = len(structure.terms)
    if n_terms == 0:
        return np.zeros(0, dtype=float)

    ew = (
        np.asarray(event_weights, dtype=float)
        if event_weights is not None
        else np.ones(len(features_list), dtype=float)
    )

    def objective(x: np.ndarray) -> float:
        nll = 0.0
        for idx, (feats, chosen) in enumerate(zip(features_list, chosen_indices)):
            lp = log_softmax(feats @ x)
            nll -= ew[idx] * float(lp[int(chosen)])
        reg = float(np.dot(x, x)) / (2.0 * sigma ** 2)
        return nll + reg

    result = minimize(
        objective,
        x0=np.zeros(n_terms, dtype=float),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return np.asarray(result.x, dtype=float)


def flat_loglik(
    features_list: Sequence[np.ndarray],
    chosen_indices: Sequence[int],
    weights: np.ndarray,
) -> float:
    """Signed log-likelihood (not negative NLL) of a flat-MNL fit."""
    total = 0.0
    for feats, chosen in zip(features_list, chosen_indices):
        lp = log_softmax(feats @ weights)
        total += float(lp[int(chosen)])
    return total


def null_loglik(features_list: Sequence[np.ndarray]) -> float:
    """Equal-probability null log-likelihood, summed per-event.

    ``LL0 = sum_i log(1 / |A(s_i)|)`` — respects variable |A(s_i)|.
    Returns ``nan`` if any event has an empty choice set.
    """
    total = 0.0
    for feats in features_list:
        n_alts_i = int(np.asarray(feats).shape[0])
        if n_alts_i <= 0:
            return float("nan")
        total += float(np.log(1.0 / n_alts_i))
    return total


__all__ = [
    "fit_weights_flat",
    "flat_loglik",
    "null_loglik",
]
