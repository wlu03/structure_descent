"""
Unit tests for the Delphos baseline.

Covers:
  1. LL0 formula                    — empty-state estimate has LL0 matching
                                      n_events * log(1/n_alts).
  2. State -> DSLStructure decoder  — (1,'linear'), (2,'log'), (3,'box-cox')
                                      round-trips to the expected 3 terms.
  3. Smoke test (end-to-end DQN)    — small-budget Delphos beats chance on
                                      synthetic MNL data.
  4. Protocol conformance           — DelphosFitted satisfies FittedBaseline.
  5. Cache effectiveness            — training yields >= 1 estimator cache hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from old_pipeline.src.baselines import FittedBaseline, evaluate_baseline
from old_pipeline.src.baselines._synthetic import make_synthetic_batch
from old_pipeline.src.baselines.delphos import (
    Delphos,
    DelphosFitted,
    _estimate_from_state,
    _state_to_structure,
)
from old_pipeline.src.dsl import ALL_TERMS, DSLStructure, DSLTerm


# -----------------------------------------------------------------------------
# 1. LL0 unit test
# -----------------------------------------------------------------------------


def test_ll0_matches_equal_probability_null():
    """LL0 must equal sum_i log(1/|A(s_i)|) within tight tolerance."""
    train = make_synthetic_batch(
        n_events=80, n_alts=6, n_customers=6, n_categories=3, seed=101,
    )
    val = make_synthetic_batch(
        n_events=40, n_alts=6, n_customers=6, n_categories=3, seed=202,
    )
    df = _estimate_from_state([], train, val=val)  # empty state -> fallback
    expected = sum(
        np.log(1.0 / feats.shape[0]) for feats in train.base_features_list
    )
    actual = float(df["LL0"].iloc[0])
    assert abs(actual - expected) < 1e-9, (
        f"LL0 mismatch: expected {expected:.6f}, got {actual:.6f}"
    )
    # LLC equals LL0 for this DSL (no alternative-specific constants).
    assert abs(float(df["LLC"].iloc[0]) - expected) < 1e-9
    # And successful estimation of the fallback ('routine') structure.
    assert bool(df["successfulEstimation"].iloc[0])
    # LLout should be finite when val is provided.
    assert np.isfinite(df["LLout"].iloc[0])


# -----------------------------------------------------------------------------
# 2. State -> DSLStructure decoder
# -----------------------------------------------------------------------------


def test_state_to_structure_decodes_three_transformations():
    """[(1, linear), (2, log), (3, box-cox)] decodes to the expected 3 terms."""
    state = [(1, "linear"), (2, "log"), (3, "box-cox")]
    s = _state_to_structure(state)
    assert isinstance(s, DSLStructure)
    assert len(s.terms) == 3

    # var=1 -> ALL_TERMS[0] (linear -> simple term)
    base0 = ALL_TERMS[0]
    assert s.terms[0].name == base0 and not s.terms[0].is_compound

    # var=2 -> ALL_TERMS[1] wrapped in log_transform
    base1 = ALL_TERMS[1]
    assert s.terms[1].name == "log_transform"
    assert s.terms[1].args == [base1]

    # var=3 -> ALL_TERMS[2] wrapped in power(exponent=0.5)
    base2 = ALL_TERMS[2]
    assert s.terms[2].name == "power"
    assert s.terms[2].args == [base2]
    assert s.terms[2].kwargs.get("exponent") == pytest.approx(0.5)


def test_state_to_structure_empty_falls_back_to_routine():
    s = _state_to_structure([])
    assert len(s.terms) == 1
    assert s.terms[0].name == "routine"


def test_state_to_structure_drops_asc_var_zero():
    """var=0 (ASC) should be dropped per the documented deviation."""
    state = [(0, "linear"), (1, "linear")]
    s = _state_to_structure(state)
    assert len(s.terms) == 1
    assert s.terms[0].name == ALL_TERMS[0]


# -----------------------------------------------------------------------------
# 3. Smoke test: DQN beats chance on synthetic data
# -----------------------------------------------------------------------------


def _tiny_synthetic_batches():
    train = make_synthetic_batch(
        n_events=120, n_alts=10, n_customers=8, n_categories=3,
        seed=501, signal_strength=1.5,
    )
    val = make_synthetic_batch(
        n_events=60, n_alts=10, n_customers=8, n_categories=3,
        seed=502, signal_strength=1.5, true_weights=train.true_weights,
    )
    test = make_synthetic_batch(
        n_events=120, n_alts=10, n_customers=8, n_categories=3,
        seed=503, signal_strength=1.5, true_weights=train.true_weights,
    )
    return train, val, test


def test_delphos_smoke_learns_signal():
    """Delphos DQN should beat chance (1/n_alts = 0.10) on synthetic MNL."""
    train, val, test = _tiny_synthetic_batches()
    baseline = Delphos(
        num_episodes=30,
        early_stop_window=10,
        patience=5,
        seed=0,
    )
    fitted = baseline.fit(train, val)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    # Chance = 1/n_alts = 0.10. Require clearly above 0.20.
    assert report.metrics["top1"] > 0.20, (
        f"Delphos top1={report.metrics['top1']:.3f} not above 0.20; "
        f"description={fitted.description}"
    )
    # Structure should not be empty.
    assert len(fitted.best_structure.terms) >= 1


# -----------------------------------------------------------------------------
# 4. Protocol conformance
# -----------------------------------------------------------------------------


def test_delphos_fitted_protocol_conformance():
    """DelphosFitted must satisfy the FittedBaseline runtime protocol."""
    train, val, _ = _tiny_synthetic_batches()
    baseline = Delphos(
        num_episodes=20,
        early_stop_window=10,
        patience=5,
        seed=1,
    )
    fitted = baseline.fit(train, val)

    assert isinstance(fitted, DelphosFitted)
    assert isinstance(fitted, FittedBaseline)
    assert fitted.name == "Delphos"
    assert isinstance(fitted.description, str)
    assert fitted.n_params > 0

    scores = fitted.score_events(train)
    assert len(scores) == train.n_events
    for s in scores:
        assert s.shape == (train.n_alternatives,)


# -----------------------------------------------------------------------------
# 5. Cache effectiveness
# -----------------------------------------------------------------------------


def test_delphos_cache_records_hits_during_training():
    """With only a few episodes the DQN will revisit at least one identical
    state, so the in-memory estimate cache should register >= 1 hit."""
    train, val, _ = _tiny_synthetic_batches()
    baseline = Delphos(
        num_episodes=25,
        early_stop_window=10,
        patience=5,
        seed=2,
    )
    fitted = baseline.fit(train, val)
    learner = baseline.learner_
    assert learner is not None
    # Total lookups == hits + misses
    total = learner.cache_hits + learner.cache_misses
    assert total >= fitted.n_episodes_run, (
        f"Cache lookup count {total} is less than episodes run "
        f"{fitted.n_episodes_run}"
    )
    assert learner.cache_hits >= 1, (
        f"Expected at least one cache hit over {fitted.n_episodes_run} "
        f"episodes; got hits={learner.cache_hits}, misses={learner.cache_misses}"
    )
    # And the number of unique states equals the number of cache misses.
    assert len(learner._estimate_cache) == learner.cache_misses
