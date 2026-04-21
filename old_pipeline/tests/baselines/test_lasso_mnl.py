"""
Unit tests for LASSO-MNL baseline.

Verifies:
  1. Smoke test — it actually learns (top-1 well above chance on synthetic data).
  2. Sparsity  — very large alpha produces a mostly-zero weight vector.
  3. Protocol  — the fitted object is recognized as a FittedBaseline.
  4. Alpha     — the selected alpha is one of the candidates in the search grid.
"""

from __future__ import annotations

import numpy as np

from old_pipeline.src.baselines import FittedBaseline, evaluate_baseline
from old_pipeline.src.baselines._synthetic import make_synthetic_batch
from old_pipeline.src.baselines.feature_pool import expanded_pool_size
from old_pipeline.src.baselines.lasso_mnl import LassoMnl, LassoMnlFitted


def _fit_small(alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1), include_interactions=True):
    train = make_synthetic_batch(
        n_events=300, n_alts=10, seed=11, signal_strength=1.5
    )
    val = make_synthetic_batch(
        n_events=120, n_alts=10, seed=12, signal_strength=1.5,
        true_weights=train.true_weights,
    )
    model = LassoMnl(
        alpha_grid=alpha_grid,
        include_interactions=include_interactions,
        max_iter=400,
        tol=1e-6,
    )
    fitted = model.fit(train, val)
    return train, val, fitted


def test_lasso_mnl_smoke_learns_signal():
    """Fit on synthetic MNL data and verify top-1 accuracy well above chance.

    Train and test come from the SAME true-weight vector but DIFFERENT seeds
    (so the alternatives and realized choices differ), mirroring real train/
    test splits from one underlying generative model.
    """
    train = make_synthetic_batch(
        n_events=300, n_alts=10, seed=101, signal_strength=1.5
    )
    val = make_synthetic_batch(
        n_events=120, n_alts=10, seed=102, signal_strength=1.5,
        true_weights=train.true_weights,
    )
    test = make_synthetic_batch(
        n_events=300, n_alts=10, seed=103, signal_strength=1.5,
        true_weights=train.true_weights,
    )

    fitted = LassoMnl(
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        include_interactions=True,
        max_iter=400,
        tol=1e-6,
    ).fit(train, val)

    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    # Chance = 1/n_alts = 0.10. We require well above that.
    assert report.metrics["top1"] > 0.35, (
        f"LASSO-MNL top1={report.metrics['top1']:.3f} not above 0.35; "
        f"description={fitted.description}"
    )
    # NLL must be below the uniform baseline log(10) ~= 2.3026.
    assert report.metrics["test_nll"] < np.log(10)


def test_lasso_mnl_high_alpha_induces_sparsity():
    """A large L1 penalty should zero out most coefficients."""
    train = make_synthetic_batch(
        n_events=200, n_alts=10, seed=201, signal_strength=1.5
    )
    val = make_synthetic_batch(
        n_events=80, n_alts=10, seed=202, signal_strength=1.5,
        true_weights=train.true_weights,
    )

    # NLL is a *sum* over events (not a mean), so the L1 penalty must be
    # large in absolute terms to dominate ~200 events worth of gradient.
    # alpha=50 is well above the noise floor and should zero out most
    # coefficients via soft-thresholding.
    fitted = LassoMnl(
        alpha_grid=(50.0,),
        include_interactions=True,
        max_iter=400,
        tol=1e-6,
    ).fit(train, val)

    pool = expanded_pool_size(train.n_base_terms, include_interactions=True)
    assert fitted.weights.shape == (pool,)
    assert fitted.n_params <= pool // 2, (
        f"alpha=1.0 only zeroed {pool - fitted.n_params}/{pool} coeffs; "
        f"expected at least half to be zero"
    )


def test_lasso_mnl_fitted_protocol_conformance():
    """LassoMnlFitted must satisfy the FittedBaseline runtime protocol."""
    _, _, fitted = _fit_small()
    assert isinstance(fitted, LassoMnlFitted)
    assert isinstance(fitted, FittedBaseline)
    # Required members
    assert fitted.name == "LASSO-MNL"
    assert isinstance(fitted.description, str)
    assert fitted.n_params >= 0
    # score_events shape contract
    train = make_synthetic_batch(n_events=10, n_alts=10, seed=301)
    scores = fitted.score_events(train)
    assert len(scores) == train.n_events
    for s in scores:
        assert s.shape == (train.n_alternatives,)


def test_lasso_mnl_alpha_selected_from_grid():
    """The fitted baseline's alpha must be one of the values in the grid."""
    grid = (1e-4, 5e-4, 2e-3, 1e-2, 5e-2)
    train = make_synthetic_batch(
        n_events=250, n_alts=10, seed=401, signal_strength=1.5
    )
    val = make_synthetic_batch(
        n_events=100, n_alts=10, seed=402, signal_strength=1.5,
        true_weights=train.true_weights,
    )
    fitted = LassoMnl(
        alpha_grid=grid,
        include_interactions=True,
        max_iter=300,
        tol=1e-6,
    ).fit(train, val)
    assert fitted.alpha in grid, (
        f"alpha={fitted.alpha} not in tuning grid {grid}"
    )
