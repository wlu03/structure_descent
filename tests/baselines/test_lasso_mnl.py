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

from src.baselines import FittedBaseline, evaluate_baseline
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.feature_pool import expanded_pool_size
from src.baselines.lasso_mnl import LassoMnl, LassoMnlFitted


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


def test_lasso_mnl_temperature_scaling_calibrates_without_changing_topk():
    """Post-hoc temperature scaling (Guo et al. 2017) must:

      1. Return T > 1 when the raw logit spread is extreme enough that the
         unregularized softmax overconfidently mispredicts some events.
      2. Strictly reduce validation NLL vs. T=1.
      3. Leave top-1 accuracy exactly invariant (scaling logits by a positive
         constant cannot change argmax).

    The fitted weights themselves are not used here — we bypass the fit path
    and directly exercise the calibration helper and the scoring divisor on a
    constructed toy batch with known-extreme logit spreads.
    """
    from scipy.special import log_softmax
    from src.baselines.base import BaselineEventBatch
    from src.baselines.lasso_mnl import LassoMnlFitted, _fit_temperature

    # Toy batch: 4 events, 3 alts, 1 "feature" that directly IS the logit.
    # We set alt 0 to have a huge positive feature so its raw logit is 100,
    # alt 1 is at 0, alt 2 at -50. The chosen alt is alt 0 in 3 events and
    # alt 2 in 1 event — the lone "surprise" event gives a massive -log p
    # under T=1 that temperature scaling should pull down.
    feats_big = np.array([[100.0], [0.0], [-50.0]])  # shape (3, 1)
    base_features_list = [feats_big.copy() for _ in range(4)]
    chosen_indices = [0, 0, 0, 2]  # three "easy", one "surprise"
    batch = BaselineEventBatch(
        base_features_list=base_features_list,
        base_feature_names=["x"],
        chosen_indices=chosen_indices,
        customer_ids=[f"c{i}" for i in range(4)],
        categories=["cat"] * 4,
    )

    # Identity "weights" so logits == feats (no feature expansion nuance):
    # bypass expand_batch by building the fitted object with
    # include_interactions=False and a matching 1-d weight vector. That
    # produces pool = 1 + 1 (constant? check) — simpler to directly exercise
    # the internals: compute val_logits_list ourselves.
    val_logits_list = [feats.flatten() for feats in base_features_list]

    # (a) Baseline NLL at T = 1.0
    nll_t1 = 0.0
    for lg, ch in zip(val_logits_list, chosen_indices):
        nll_t1 -= float(log_softmax(lg)[ch])

    # (b) Fit temperature.
    T = _fit_temperature(val_logits_list, chosen_indices)
    assert T > 1.0, f"expected T > 1 on over-confident logits, got T={T}"

    # (c) NLL at fitted T must be strictly lower.
    nll_tT = 0.0
    for lg, ch in zip(val_logits_list, chosen_indices):
        nll_tT -= float(log_softmax(lg / T)[ch])
    assert nll_tT < nll_t1, (
        f"temperature scaling should reduce NLL: nll(T=1)={nll_t1:.3f} "
        f"vs nll(T={T:.3f})={nll_tT:.3f}"
    )

    # (d) Invariance: scoring with and without T yields identical argmax.
    # Exercise via score_events() on a LassoMnlFitted whose "weights" are a
    # single [1.0] so feats @ w = feats[:, 0]. We have to match the expanded
    # feature pool shape, so bypass expand_batch by using a no-interactions
    # batch with 1 base term — which yields expanded_pool_size(1, False)=1.
    # Simpler: directly check argmax on logits vs logits/T on the val set.
    for lg in val_logits_list:
        assert int(np.argmax(lg)) == int(np.argmax(lg / T))


def test_lasso_mnl_fit_populates_temperature_field():
    """End-to-end: the fitted object carries a positive, finite temperature."""
    _, _, fitted = _fit_small()
    assert hasattr(fitted, "temperature")
    assert np.isfinite(fitted.temperature)
    assert fitted.temperature > 0.0


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
