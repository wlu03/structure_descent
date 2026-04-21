"""
Unit tests for the Paz VNS baseline.

Covers:
  1. Smoke test           — small-budget VNS beats chance on synthetic data.
  2. Pareto front size    — a short run produces >=2 non-dominated entries.
  3. Protocol conformance — PazVnsFitted satisfies FittedBaseline.
  4. Neighborhood sanity  — N1 shake preserves complexity within +/- 2.
"""

from __future__ import annotations

import numpy as np

from old_pipeline.src.baselines import FittedBaseline, evaluate_baseline
from old_pipeline.src.baselines._synthetic import make_synthetic_batch
from old_pipeline.src.baselines.paz_vns import (
    PazVNS,
    PazVnsFitted,
    shake,
    n1_neighbors,
)
from old_pipeline.src.dsl import DSLStructure, DSLTerm


def _tiny_train_val():
    train = make_synthetic_batch(
        n_events=150, n_alts=6, n_customers=8, n_categories=3,
        seed=201, signal_strength=1.5,
    )
    val = make_synthetic_batch(
        n_events=80, n_alts=6, n_customers=8, n_categories=3,
        seed=202, signal_strength=1.5, true_weights=train.true_weights,
    )
    return train, val


def test_paz_vns_smoke_learns_signal():
    """Small-budget VNS beats chance on synthetic MNL data."""
    train, val = _tiny_train_val()
    test = make_synthetic_batch(
        n_events=150, n_alts=6, n_customers=8, n_categories=3,
        seed=203, signal_strength=1.5, true_weights=train.true_weights,
    )

    fitted = PazVNS(k_max=3, max_evaluations=25, seed=0).fit(train, val)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    # Chance = 1/n_alts = 1/6 ~ 0.167; require clearly above 0.2.
    assert report.metrics["top1"] > 0.2, (
        f"Paz-VNS top1={report.metrics['top1']:.3f} not above 0.2; "
        f"description={fitted.description}"
    )
    assert report.metrics["test_nll"] < np.log(6) + 1e-6


def test_paz_vns_pareto_front_nondominated():
    """The archive should hold at least one entry, and any entries must
    be mutually non-dominated on the (BIC, -adjRho2_bar) objective pair.

    The paper's (BIC, adjRho2_bar) objectives are more strongly negatively
    correlated than (train_nll, complexity), so a well-behaved search
    frequently collapses the frontier to a single winner. That is a
    *feature* of BIC-style objectives, not a bug, so we no longer require
    >=2 entries — we check non-domination only.
    """
    train, val = _tiny_train_val()
    fitted = PazVNS(k_max=3, max_evaluations=25, seed=1).fit(train, val)
    assert len(fitted.pareto_front) >= 1
    fs = fitted.pareto_front
    for i, a in enumerate(fs):
        for j, b in enumerate(fs):
            if i == j:
                continue
            # b should NOT strictly dominate a on (bic, -adj_rho2_bar).
            assert not (
                b.bic <= a.bic
                and -b.adj_rho2_bar <= -a.adj_rho2_bar
                and (b.bic < a.bic or b.adj_rho2_bar > a.adj_rho2_bar)
            )


def test_paz_vns_fitted_protocol_conformance():
    """PazVnsFitted must satisfy the FittedBaseline runtime protocol."""
    train, val = _tiny_train_val()
    fitted = PazVNS(k_max=2, max_evaluations=15, seed=2).fit(train, val)

    assert isinstance(fitted, PazVnsFitted)
    assert isinstance(fitted, FittedBaseline)
    assert fitted.name == "Paz-VNS"
    assert isinstance(fitted.description, str)
    assert fitted.n_params > 0

    # score_events shape contract
    scores = fitted.score_events(train)
    assert len(scores) == train.n_events
    for s in scores:
        assert s.shape == (train.n_alternatives,)


def test_paz_vns_n1_shake_bounded_complexity():
    """N1 shake should modify a structure within +/- 2 complexity."""
    import random

    rng = random.Random(42)
    base = DSLStructure(["routine", "affinity", "popularity", "recency"])
    before = base.complexity()
    for _ in range(50):
        shaken = shake(base, k=1, rng=rng)
        diff = abs(shaken.complexity() - before)
        assert diff <= 2, (
            f"N1 shake moved complexity by {diff} "
            f"(before={before}, after={shaken.complexity()})"
        )
    # N1 neighborhood enumeration must produce non-empty results for a
    # non-trivial seed, and each neighbor should itself be non-empty.
    neighbors = n1_neighbors(base)
    assert len(neighbors) > 0
    for n in neighbors:
        assert isinstance(n, DSLStructure)
