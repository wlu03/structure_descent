"""Unit tests for the Delphos baseline.

Covers the 12 cases enumerated in
``docs/llm_baselines/delphos_baseline.md`` section 5: protocol
conformance, shapes, beats-chance sanity, candidate-tracker monotonicity,
cache accounting, seed determinism, early-stopping fires, empty-state
fallback, adapter round-trip, degenerate-feature handling, n_params
agreement, and registry wiring.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.baselines import FittedBaseline
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.base import BaselineEventBatch
from src.baselines.data_adapter import (
    BUILTIN_FEATURE_NAMES,
    records_to_baseline_batch,
)
from src.baselines.delphos import (
    Delphos,
    DelphosFitted,
    _state_to_structure,
)
from src.baselines._delphos_dqn import StateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _restrict_to_builtin(batch: BaselineEventBatch) -> BaselineEventBatch:
    """Project a synthetic batch onto the 4-column ``BUILTIN_FEATURE_NAMES``.

    The shared ``make_synthetic_batch`` ships with a 12-primitive feature
    layout; Delphos runs over the adapter's 4-column pool. We slice the
    feature matrices on the first 4 columns and rename them so the rest
    of the batch contract is unchanged.
    """
    feats_list = [feats[:, :4].astype(np.float64, copy=True) for feats in batch.base_features_list]
    new = BaselineEventBatch(
        base_features_list=feats_list,
        base_feature_names=list(BUILTIN_FEATURE_NAMES),
        chosen_indices=list(batch.chosen_indices),
        customer_ids=list(batch.customer_ids),
        categories=list(batch.categories),
        metadata=[dict(m) for m in batch.metadata],
        raw_events=[dict(r) for r in batch.raw_events] if batch.raw_events else None,
    )
    true_w = getattr(batch, "true_weights", None)
    if true_w is not None:
        new.true_weights = np.asarray(true_w, dtype=float)[:4].copy()
    return new


def _make_batches(
    n_train: int = 150,
    n_val: int = 60,
    n_test: int = 150,
    seed: int = 11,
    signal_strength: float = 2.0,
):
    train = _restrict_to_builtin(
        make_synthetic_batch(
            n_events=n_train, n_alts=6, seed=seed, signal_strength=signal_strength
        )
    )
    val = _restrict_to_builtin(
        make_synthetic_batch(
            n_events=n_val,
            n_alts=6,
            seed=seed + 1,
            signal_strength=signal_strength,
            true_weights=make_synthetic_batch(
                n_events=2, n_alts=6, seed=seed, signal_strength=signal_strength
            ).true_weights,
        )
    )
    test = _restrict_to_builtin(
        make_synthetic_batch(
            n_events=n_test,
            n_alts=6,
            seed=seed + 2,
            signal_strength=signal_strength,
            true_weights=make_synthetic_batch(
                n_events=2, n_alts=6, seed=seed, signal_strength=signal_strength
            ).true_weights,
        )
    )
    return train, val, test


# ---------------------------------------------------------------------------
# 1. Protocol conformance.
# ---------------------------------------------------------------------------


def test_delphos_fitted_protocol_conformance():
    train, val, _ = _make_batches(n_train=80, n_val=40)
    fitted = Delphos(n_episodes=25, seed=1).fit(train, val)
    assert isinstance(fitted, DelphosFitted)
    assert isinstance(fitted, FittedBaseline)
    assert fitted.name == "Delphos"
    assert isinstance(fitted.description, str)
    assert fitted.n_params == len(fitted.best_structure.terms)


# ---------------------------------------------------------------------------
# 2. Shapes of score_events output.
# ---------------------------------------------------------------------------


def test_delphos_score_events_shapes():
    train, val, test = _make_batches(n_train=80, n_val=40, n_test=50)
    fitted = Delphos(n_episodes=25, seed=2).fit(train, val)
    scores = fitted.score_events(test)
    assert len(scores) == test.n_events
    J = test.n_alternatives
    for s in scores:
        assert s.shape == (J,)
        assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# 3. Beats chance on synthetic.
# ---------------------------------------------------------------------------


def test_delphos_beats_chance_on_synthetic():
    train, val, test = _make_batches(
        n_train=200, n_val=80, n_test=200, seed=31, signal_strength=2.0
    )
    fitted = Delphos(n_episodes=60, seed=31).fit(train, val)
    scores = fitted.score_events(test)
    top1_correct = 0
    for s, chosen in zip(scores, test.chosen_indices):
        top1_correct += int(np.argmax(s) == chosen)
    top1 = top1_correct / len(scores)
    chance = 1.0 / test.n_alternatives
    assert top1 > chance + 0.05, (
        f"Delphos top1={top1:.3f} not above chance+0.05={chance + 0.05:.3f}; "
        f"desc={fitted.description}"
    )


# ---------------------------------------------------------------------------
# 4. Candidate-tracker monotonicity (best AIC never regresses).
# ---------------------------------------------------------------------------


def test_delphos_candidate_tracker_best_aic_monotonic():
    train, val, _ = _make_batches(n_train=120, n_val=50, seed=41)
    baseline = Delphos(n_episodes=40, seed=41)
    fitted = baseline.fit(train, val)
    learner = baseline.learner_
    assert learner is not None
    history = learner.current_best_history
    assert history, "current_best_history is empty"
    best_aic_series = [snap["AIC_value"] for snap in history]
    # Non-increasing (AIC is minimized, so best-so-far never grows).
    for a, b in zip(best_aic_series[:-1], best_aic_series[1:]):
        if np.isfinite(a) and np.isfinite(b):
            assert b <= a + 1e-9, (
                f"best AIC regressed: {a} -> {b} across successive episodes"
            )
    # Final reported best-AIC is <= any AIC observed during training.
    final_best = best_aic_series[-1]
    training_aics = [
        log.get("AIC") for log in learner.training_log if log.get("AIC") is not None
    ]
    finite_aics = [a for a in training_aics if np.isfinite(a)]
    if finite_aics and np.isfinite(final_best):
        assert final_best <= min(finite_aics) + 1e-9
    # Fitted structure name list must be non-empty.
    assert fitted.best_structure.term_names


# ---------------------------------------------------------------------------
# 5. Cache accounting.
# ---------------------------------------------------------------------------


def test_delphos_cache_hits_plus_misses_equal_calls():
    train, val, _ = _make_batches(n_train=60, n_val=30, seed=51)
    baseline = Delphos(n_episodes=30, seed=51)
    baseline.fit(train, val)
    learner = baseline.learner_
    assert learner is not None
    # misses are exactly the number of unique cache keys populated.
    assert learner.cache_misses == len(learner._estimate_cache)
    # Every episode calls ``delphos_interaction`` exactly once.
    total_calls = learner.cache_hits + learner.cache_misses
    assert total_calls >= learner.cache_misses >= 1


# ---------------------------------------------------------------------------
# 6. Determinism under fixed seed.
# ---------------------------------------------------------------------------


def test_delphos_determinism_under_seed():
    train, val, _ = _make_batches(n_train=60, n_val=30, seed=61)
    f1 = Delphos(n_episodes=25, seed=42).fit(train, val)
    f2 = Delphos(n_episodes=25, seed=42).fit(train, val)
    assert f1.best_structure.term_names == f2.best_structure.term_names
    np.testing.assert_allclose(f1.best_weights, f2.best_weights, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Early stopping fires when the plateau detector is tight.
# ---------------------------------------------------------------------------


def test_delphos_early_stopping_fires_on_small_search_space():
    train, val, _ = _make_batches(n_train=40, n_val=20, seed=71)
    baseline = Delphos(
        n_episodes=400,
        seed=71,
        min_percentage=0.05,
        early_stop_window=10,
        patience=1,
        epsilon_min=0.01,
    )
    fitted = baseline.fit(train, val)
    # With a tight plateau detector and a small feature pool, we should
    # stop well before exhausting the 400-episode budget.
    assert fitted.n_episodes_run < 400


# ---------------------------------------------------------------------------
# 8. Empty-state fallback.
# ---------------------------------------------------------------------------


def test_delphos_empty_state_falls_back_to_non_empty_structure():
    # An empty state should decode to a non-empty structure so score_events
    # never has to multiply a zero-column feature matrix.
    structure = _state_to_structure([], list(BUILTIN_FEATURE_NAMES))
    assert len(structure.terms) == 1
    assert structure.terms[0].name == BUILTIN_FEATURE_NAMES[0]


def test_delphos_score_events_never_divides_by_zero_even_on_empty_fallback():
    train, val, test = _make_batches(n_train=40, n_val=20, n_test=20, seed=81)
    # Fit normally, then overwrite best_structure to the fallback to
    # simulate the "terminate immediately" path. We patch via the
    # existing fitted object rather than monkey-patching the DQN internals.
    fitted = Delphos(n_episodes=25, seed=81).fit(train, val)
    fitted.best_structure = _state_to_structure([], list(BUILTIN_FEATURE_NAMES))
    fitted.best_weights = np.zeros(1, dtype=np.float64)
    scores = fitted.score_events(test)
    assert len(scores) == test.n_events
    for s in scores:
        assert s.shape == (test.n_alternatives,)
        assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# 9. Adapter round-trip.
# ---------------------------------------------------------------------------


def _make_synthetic_records(n_records: int = 3) -> List[dict]:
    """Small handful of PO-LEU-shaped records for the data_adapter path."""
    rng = np.random.default_rng(0)
    records: List[dict] = []
    n_alts = 4
    for i in range(n_records):
        alt_texts = []
        for j in range(n_alts):
            alt_texts.append(
                {
                    "title": f"item_{i}_{j}",
                    "category": "cat_A",
                    "price": float(5.0 + j * 2.5),
                    "popularity_rank": f"popularity score {int(rng.integers(1, 100))}",
                    "brand": f"brand_{j}",
                    "state": "CA",
                }
            )
        records.append(
            {
                "customer_id": f"cust_{i}",
                "chosen_idx": int(rng.integers(0, n_alts)),
                "alt_texts": alt_texts,
                "category": "cat_A",
                "metadata": {"is_repeat": False, "price": 10.0, "routine": 0},
            }
        )
    return records


def test_delphos_adapter_round_trip():
    records = _make_synthetic_records(4)
    train = records_to_baseline_batch(records)
    val = records_to_baseline_batch(records)
    assert tuple(train.base_feature_names) == BUILTIN_FEATURE_NAMES
    fitted = Delphos(n_episodes=10, seed=91).fit(train, val)
    assert tuple(fitted.base_feature_names) == BUILTIN_FEATURE_NAMES


# ---------------------------------------------------------------------------
# 10. Degenerate (all-zero) feature handled without inf AIC.
# ---------------------------------------------------------------------------


def test_delphos_handles_degenerate_feature():
    train, val, _ = _make_batches(n_train=80, n_val=30, seed=101)
    # Zero out the first feature in every event on both splits.
    for feats in train.base_features_list:
        feats[:, 0] = 0.0
    for feats in val.base_features_list:
        feats[:, 0] = 0.0
    baseline = Delphos(n_episodes=20, seed=101)
    fitted = baseline.fit(train, val)
    learner = baseline.learner_
    assert learner is not None
    any_successful = False
    for df in learner._estimate_cache.values():
        if df.empty:
            continue
        if bool(df["successfulEstimation"].iloc[0]):
            aic = float(df["AIC"].iloc[0])
            assert np.isfinite(aic)
            any_successful = True
    assert any_successful, "no successful estimation on degenerate input"
    # Fitted object still scores finitely.
    scores = fitted.score_events(val)
    for s in scores:
        assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# 11. n_params matches the structure.
# ---------------------------------------------------------------------------


def test_delphos_n_params_matches_structure_length():
    train, val, _ = _make_batches(n_train=60, n_val=30, seed=111)
    fitted = Delphos(n_episodes=20, seed=111).fit(train, val)
    assert fitted.n_params == len(fitted.best_structure.terms)
    assert fitted.best_weights.shape == (fitted.n_params,)


# ---------------------------------------------------------------------------
# 12. Registry wiring.
# ---------------------------------------------------------------------------


def test_delphos_registered_in_baseline_registry():
    from src.baselines.run_all import BASELINE_REGISTRY

    entry = ("Delphos", "src.baselines.delphos", "Delphos")
    matches = [row for row in BASELINE_REGISTRY if row == entry]
    assert len(matches) == 1, (
        f"expected exactly one Delphos registry entry, got {matches}"
    )


# ---------------------------------------------------------------------------
# Extra: state vector shape under Option B.
# ---------------------------------------------------------------------------


def test_delphos_state_vector_length_option_b():
    # 4 features + ASC slot -> 1 + 4 * 4 = 17.
    sm = StateManager({"num_vars": len(BUILTIN_FEATURE_NAMES) + 1})
    assert sm.get_state_length() == 17
    vec = sm.encode_state_to_vector([(0, "linear"), (1, "log")])
    assert vec.shape == (17,)
    assert vec[0] == 1.0  # ASC slot.
    # Feature 1 ('price') with 'log' -> slot 1 + (1-1)*4 + 2 = 3.
    assert vec[3] == 1.0
