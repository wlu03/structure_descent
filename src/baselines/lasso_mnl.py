"""
LASSO-MNL baseline: L1-regularized conditional-logit (multinomial logit)
over the shared expanded feature pool.

Objective (exactly matching src/inner_loop.py's conditional-logit NLL):

    L(w) = sum_e  -log_softmax(X_e @ w)[chosen_e]  +  alpha * ||w||_1

where X_e is the expanded feature pool for event e (shape n_alts x n_expanded),
and one global weight vector w is shared across all events, customers, and
categories. No hierarchy, no group effects — this is the pure "statistical
floor" baseline.

Solver: **FISTA with backtracking line search** (Beck & Teboulle 2009).
We compute the smooth NLL gradient analytically (softmax minus one-hot, the
standard conditional-logit gradient form), then apply the soft-thresholding
proximal operator after each gradient step. No smoothed-L1 surrogate is used.

Regularization path: alpha is tuned on a held-out validation split over a
logarithmic grid. The alpha with the lowest validation NLL (unregularized
NLL under the fitted weights) is selected. The chosen alpha is exposed on
the fitted object and included in its description.

Sparsity: after the optimizer converges we hard-threshold weights with
|w| < 1e-6 to exactly zero; n_params counts the number of surviving
non-zero coefficients so AIC/BIC correctly reward sparsity.

Interface: implements the Baseline / FittedBaseline protocols from
src/baselines/base.py. Consumes and produces features via the shared
expand_batch() helper from feature_pool.py so the expansion is identical
to every other regression baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from .base import BaselineEventBatch, FittedBaseline
from .feature_pool import expand_batch


# -----------------------------------------------------------------------------
# Core numerics: conditional-logit NLL + gradient, FISTA with backtracking
# -----------------------------------------------------------------------------


def _nll_and_grad(
    w: np.ndarray,
    expanded_list: List[np.ndarray],
    chosen_indices: Sequence[int],
) -> Tuple[float, np.ndarray]:
    """Conditional-logit negative log-likelihood and its gradient.

    NLL(w) = sum_e -log softmax(X_e @ w)[chosen_e]
    d/dw   = sum_e  X_e^T (softmax(X_e @ w) - e_chosen)

    This is the standard conditional-logit gradient (the analytic form
    referenced in the task spec) — probability minus one-hot, contracted
    with the feature matrix.
    """
    nll = 0.0
    grad = np.zeros_like(w)
    for feats, chosen in zip(expanded_list, chosen_indices):
        utilities = feats @ w                       # (n_alts,)
        lp = log_softmax(utilities)                 # numerically stable
        nll -= float(lp[chosen])
        p = np.exp(lp)                              # softmax probs
        p[chosen] -= 1.0                            # softmax - one-hot
        grad += feats.T @ p
    return nll, grad


def _nll_only(
    w: np.ndarray,
    expanded_list: List[np.ndarray],
    chosen_indices: Sequence[int],
) -> float:
    """Unregularized conditional-logit NLL only — used for validation scoring."""
    nll = 0.0
    for feats, chosen in zip(expanded_list, chosen_indices):
        lp = log_softmax(feats @ w)
        nll -= float(lp[chosen])
    return nll


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Element-wise soft-thresholding prox operator for L1 with step `thresh`."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def _fit_temperature(
    val_logits_list: List[np.ndarray],
    chosen_indices: Sequence[int],
    t_lo: float = 0.01,
    t_hi: float = 1000.0,
) -> float:
    """Post-hoc temperature scaling (Guo et al. 2017, "On Calibration of
    Modern Neural Networks").

    Fits one scalar T > 0 by minimizing val NLL of ``logits / T``. Since
    scaling logits by a positive constant does not change argmax ordering,
    top-k metrics are invariant; only the softmax distribution sharpens
    (T<1) or flattens (T>1).

    A degenerate val set (no events, or all zero-logit slates) short-circuits
    to T=1.0.
    """
    if len(val_logits_list) == 0:
        return 1.0

    # Short-circuit if all logits are numerically zero (nothing to calibrate).
    if all(float(np.max(np.abs(lg))) < 1e-12 for lg in val_logits_list):
        return 1.0

    def _neg_log_lik(log_T: float) -> float:
        T = float(np.exp(log_T))
        total = 0.0
        for lg, ch in zip(val_logits_list, chosen_indices):
            total -= float(log_softmax(lg / T)[ch])
        return total

    # Optimize in log-T space for numerical stability, then clamp to bounds.
    res = minimize_scalar(
        _neg_log_lik,
        bounds=(float(np.log(t_lo)), float(np.log(t_hi))),
        method="bounded",
        options={"xatol": 1e-6},
    )
    T = float(np.exp(res.x))
    return float(np.clip(T, t_lo, t_hi))


def _fista(
    expanded_list: List[np.ndarray],
    chosen_indices: Sequence[int],
    alpha: float,
    n_features: int,
    max_iter: int = 500,
    tol: float = 1e-7,
    w_init: np.ndarray | None = None,
    L_init: float = 1.0,
    eta: float = 2.0,
) -> np.ndarray:
    """FISTA with backtracking line search for the L1-regularized MNL.

    Minimizes F(w) = f(w) + g(w) where f is the smooth conditional-logit
    NLL and g(w) = alpha * ||w||_1. Uses the accelerated momentum step
    with backtracking on the Lipschitz estimate L.
    """
    w = np.zeros(n_features) if w_init is None else w_init.copy()
    y = w.copy()
    t = 1.0
    L = float(L_init)

    prev_obj = _nll_only(w, expanded_list, chosen_indices) + alpha * np.sum(np.abs(w))

    for _ in range(max_iter):
        f_y, g_y = _nll_and_grad(y, expanded_list, chosen_indices)

        # Backtracking: find smallest L such that the majorization holds.
        while True:
            step = 1.0 / L
            w_next = _soft_threshold(y - step * g_y, alpha * step)
            diff = w_next - y
            f_next = _nll_only(w_next, expanded_list, chosen_indices)
            # Quadratic upper bound check
            Q = f_y + float(g_y @ diff) + 0.5 * L * float(diff @ diff)
            if f_next <= Q + 1e-12:
                break
            L *= eta
            if L > 1e12:
                break

        # Nesterov momentum update
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w = w_next
        t = t_next

        obj = f_next + alpha * float(np.sum(np.abs(w)))
        if abs(prev_obj - obj) <= tol * max(1.0, abs(prev_obj)):
            break
        prev_obj = obj

    return w


# -----------------------------------------------------------------------------
# Public API: LassoMnl (fit) and LassoMnlFitted (score)
# -----------------------------------------------------------------------------


@dataclass
class LassoMnlFitted:
    """Fitted LASSO-MNL, conforming to the FittedBaseline protocol."""

    name: str
    weights: np.ndarray
    feature_names: List[str]
    alpha: float
    include_interactions: bool
    train_nll: float
    temperature: float = 1.0

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        expanded_list, _ = expand_batch(batch, self.include_interactions)
        return [(feats @ self.weights) / self.temperature for feats in expanded_list]

    @property
    def n_params(self) -> int:
        # Hard-threshold: |w| < 1e-6 counts as zero for sparsity accounting.
        return int(np.sum(np.abs(self.weights) >= 1e-6))

    @property
    def description(self) -> str:
        nz = self.n_params
        total = int(self.weights.shape[0])
        return f"LASSO-MNL alpha={self.alpha:.3g} {nz}/{total} nonzero"


class LassoMnl:
    """LASSO-MNL baseline (conditional logit with L1 via FISTA).

    Parameters
    ----------
    alpha_grid : tuple of float
        Regularization strengths to search. The best alpha is selected by
        validation NLL (unregularized) on the held-out split.
    include_interactions : bool
        Forwarded to expand_batch(); toggles pairwise x_i * x_j terms.
    max_iter : int
        Maximum FISTA iterations per alpha.
    tol : float
        Relative objective tolerance for FISTA convergence.
    """

    name = "LASSO-MNL"

    def __init__(
        self,
        alpha_grid: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        include_interactions: bool = True,
        max_iter: int = 500,
        tol: float = 1e-7,
    ):
        self.alpha_grid = tuple(alpha_grid)
        self.include_interactions = include_interactions
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> LassoMnlFitted:
        train_exp, feature_names = expand_batch(train, self.include_interactions)
        val_exp, _ = expand_batch(val, self.include_interactions)
        n_features = len(feature_names)

        best_alpha = None
        best_val_nll = np.inf
        best_w = None
        warm_start: np.ndarray | None = None

        # Walk from most-regularized to least-regularized so warm-starts
        # progressively de-sparsify the solution (standard LASSO path trick).
        for alpha in sorted(self.alpha_grid, reverse=True):
            w = _fista(
                expanded_list=train_exp,
                chosen_indices=train.chosen_indices,
                alpha=float(alpha),
                n_features=n_features,
                max_iter=self.max_iter,
                tol=self.tol,
                w_init=warm_start,
            )
            val_nll = _nll_only(w, val_exp, val.chosen_indices)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_alpha = float(alpha)
                best_w = w.copy()
            warm_start = w

        assert best_w is not None and best_alpha is not None

        # Final hard-threshold for clean sparsity accounting.
        best_w = np.where(np.abs(best_w) < 1e-6, 0.0, best_w)
        train_nll = _nll_only(best_w, train_exp, train.chosen_indices)

        # Post-hoc temperature scaling on the validation set (Guo et al. 2017).
        # FISTA applies no magnitude regularization to surviving weights, so
        # extreme logit spreads can blow up NLL while argmax stays correct.
        # Dividing logits by a learned T > 0 recalibrates without changing top-k.
        val_logits_list = [X_e @ best_w for X_e in val_exp]
        temperature = _fit_temperature(val_logits_list, val.chosen_indices)

        return LassoMnlFitted(
            name=self.name,
            weights=best_w,
            feature_names=feature_names,
            alpha=best_alpha,
            include_interactions=self.include_interactions,
            train_nll=float(train_nll),
            temperature=float(temperature),
        )
