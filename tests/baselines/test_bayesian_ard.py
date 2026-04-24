"""
Unit tests for Bayesian ARD baseline.

Verifies:
  1. Smoke test — NUTS inference on a small synthetic batch learns the signal
     (top-1 well above chance on a held-out batch).
  2. Protocol conformance — BayesianARDFitted satisfies FittedBaseline.
  3. Pruning actually prunes — given the vague Gamma(1e-3, 1e-3) hyperprior
     and a small batch, at least one coefficient should end up below the
     pruning threshold.
  4. Posterior precisions learned — posterior_alpha is finite and the max
     alpha has moved appreciably above the tiny prior mean (a0/b0 = 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import FittedBaseline, evaluate_baseline
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.bayesian_ard import BayesianARD, BayesianARDFitted


@pytest.fixture(scope="module")
def fitted_smoke():
    """Fit a single Bayesian ARD model once and reuse across fast tests."""
    train = make_synthetic_batch(
        n_events=150, n_alts=6, seed=401, signal_strength=1.5,
    )
    val = make_synthetic_batch(
        n_events=60, n_alts=6, seed=402, signal_strength=1.5,
        true_weights=train.true_weights,
    )
    # include_interactions=True gives 102 features (36 + 66 pairwise). The
    # synthetic generator only uses the 12 base features for its true utility,
    # so the 66 pairwise terms carry no signal and the ARD prior should shrink
    # at least some of them below the pruning threshold.
    # inference="nuts" keeps the smoke test fast. alpha_threshold is set to
    # 100 (not the paper default of 1e3) because the small warmup (100
    # samples) on 150 events doesn't push as many alphas all the way past
    # 1e3, but it's still large enough that irrelevant features get pruned
    # while signal-carrying ones survive. A stricter threshold (e.g. 10)
    # over-prunes on a tiny batch.
    model = BayesianARD(
        include_interactions=True,
        n_warmup=100,
        n_samples=100,
        inference="nuts",
        seed=0,
        alpha_threshold=100.0,
    )
    fitted = model.fit(train, val)
    return train, val, fitted


def test_bayesian_ard_smoke_learns_signal(fitted_smoke):
    """Fit on synthetic MNL data and verify top-1 well above chance.

    With n_alts=6, chance = 1/6 ~= 0.167. We require top-1 > 0.25.
    """
    train, _, fitted = fitted_smoke
    test = make_synthetic_batch(
        n_events=200, n_alts=6, seed=403, signal_strength=1.5,
        true_weights=train.true_weights,
    )
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)
    assert report.metrics["top1"] > 0.25, (
        f"Bayesian-ARD top1={report.metrics['top1']:.3f} not above 0.25; "
        f"description={fitted.description}"
    )
    # Sanity: NLL below the uniform baseline log(6).
    assert report.metrics["test_nll"] < np.log(6)


def test_bayesian_ard_fitted_protocol_conformance(fitted_smoke):
    """BayesianARDFitted must satisfy the FittedBaseline runtime protocol."""
    _, _, fitted = fitted_smoke
    assert isinstance(fitted, BayesianARDFitted)
    assert isinstance(fitted, FittedBaseline)
    assert fitted.name == "Bayesian-ARD"
    assert isinstance(fitted.description, str)
    assert fitted.n_params >= 0

    # score_events shape contract
    tiny = make_synthetic_batch(n_events=5, n_alts=6, seed=999)
    scores = fitted.score_events(tiny)
    assert len(scores) == tiny.n_events
    for s in scores:
        assert s.shape == (tiny.n_alternatives,)


def test_bayesian_ard_pruning_actually_prunes(fitted_smoke):
    """Vague ARD prior + small batch should prune at least one coefficient."""
    _, _, fitted = fitted_smoke
    assert fitted.n_pruned > 0, (
        f"n_pruned={fitted.n_pruned}; expected pruning with vague ARD prior. "
        f"description={fitted.description}"
    )
    # And the non-pruned count should be strictly less than the total pool.
    total = fitted.posterior_mean_weights.shape[0]
    assert fitted.n_params < total


def test_bayesian_ard_temperature_scaling_calibrates_without_changing_topk():
    """Post-hoc temperature scaling (Guo et al. 2017) must:

      1. Return T > 1 when raw logits are over-confident.
      2. Strictly reduce val NLL vs. T = 1.
      3. Preserve argmax (top-1 invariance).

    Directly exercises the module-level _fit_temperature helper on a toy
    logit batch — independent of the NumPyro fit path, which is covered by
    the smoke test.
    """
    from scipy.special import log_softmax
    from src.baselines.bayesian_ard import _fit_temperature

    # Same toy setup as the LASSO-MNL calibration test: 4 events, 3 alts,
    # extreme logit spread (100, 0, -50). One event has a "surprise" chosen
    # alt so the raw NLL under T=1 is catastrophic; calibrated T must fix it.
    raw_logits = np.array([100.0, 0.0, -50.0])
    val_logits_list = [raw_logits.copy() for _ in range(4)]
    chosen_indices = [0, 0, 0, 2]

    nll_t1 = 0.0
    for lg, ch in zip(val_logits_list, chosen_indices):
        nll_t1 -= float(log_softmax(lg)[ch])

    T = _fit_temperature(val_logits_list, chosen_indices)
    assert T > 1.0, f"expected T > 1 on over-confident logits, got T={T}"

    nll_tT = 0.0
    for lg, ch in zip(val_logits_list, chosen_indices):
        nll_tT -= float(log_softmax(lg / T)[ch])
    assert nll_tT < nll_t1, (
        f"temperature scaling should reduce NLL: nll(T=1)={nll_t1:.3f} "
        f"vs nll(T={T:.3f})={nll_tT:.3f}"
    )

    for lg in val_logits_list:
        assert int(np.argmax(lg)) == int(np.argmax(lg / T))


def test_bayesian_ard_fit_populates_temperature_field(fitted_smoke):
    """End-to-end: the fitted object carries a positive, finite temperature."""
    _, _, fitted = fitted_smoke
    assert hasattr(fitted, "temperature")
    assert np.isfinite(fitted.temperature)
    assert fitted.temperature > 0.0


def test_bayesian_ard_posterior_precisions_learned(fitted_smoke):
    """posterior_alpha must be finite and updated from the tiny prior mean."""
    _, _, fitted = fitted_smoke
    alpha = fitted.posterior_alpha
    assert alpha.shape == fitted.posterior_mean_weights.shape
    assert np.all(np.isfinite(alpha)), "posterior_alpha has non-finite entries"
    assert np.all(alpha > 0.0), "precisions must be strictly positive"
    # Prior mean a0/b0 = 1.0. After observing data at least one alpha_j should
    # have moved well past 1.0 (either way), confirming the hyperprior updated.
    assert float(np.max(alpha)) > 1.0, (
        f"max(posterior_alpha)={float(np.max(alpha)):.3g} suggests alphas "
        f"never updated past the prior mean of 1.0"
    )
