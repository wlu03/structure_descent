"""
Smoke tests for the classical ML baselines.

Each of the three baselines (RandomForest, HistGradientBoosting, MLP) is
fit on a synthetic batch with a clear signal and must beat chance on a
separate held-out synthetic batch. Protocol conformance is also checked.
"""

from __future__ import annotations

import numpy as np

from src.baselines import BaselineEventBatch, FittedBaseline, evaluate_baseline
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.classical_ml import (
    ClassicalMLFitted,
    GradientBoostingChoice,
    MLPChoice,
    RandomForestChoice,
    _batch_to_long_format,
)


def _shared_triplet(seed_base: int, signal_strength: float = 2.0,
                    n_train: int = 600, n_val: int = 150, n_test: int = 250):
    """
    Build (train, val, test) synthetic batches that all share the SAME
    underlying true_weights vector. This is critical for classical ML
    baselines — `make_synthetic_batch` normally draws a fresh weight
    vector per seed, so train/val/test otherwise come from different
    models and nothing generalizes.
    """
    rng = np.random.default_rng(seed_base)
    # 12 base features matches _DEFAULT_BASE_NAMES in _synthetic.py
    true_w = rng.normal(0.0, 1.0, size=12) * signal_strength
    train = make_synthetic_batch(n_events=n_train, seed=seed_base + 1, true_weights=true_w)
    val = make_synthetic_batch(n_events=n_val, seed=seed_base + 2, true_weights=true_w)
    test = make_synthetic_batch(n_events=n_test, seed=seed_base + 3, true_weights=true_w)
    return train, val, test


# ── long-format flattening ──────────────────────────────────────────────────


def test_batch_to_long_format_shapes_and_labels():
    batch = make_synthetic_batch(n_events=20, n_alts=10, seed=11)
    X, y, sw = _batch_to_long_format(batch)
    assert X.shape == (20 * 10, batch.n_base_terms)
    assert y.shape == (20 * 10,)
    assert sw.shape == (20 * 10,)
    # sample_weight: every row weighted 1/n_alts so each event contributes 1.0.
    import numpy as _np
    assert _np.allclose(sw, 1.0 / 10)
    # Exactly one chosen alternative per event → exactly 20 positives.
    assert int(y.sum()) == 20
    # The positive at event e should be at row e*n_alts + chosen_indices[e].
    for e, chosen in enumerate(batch.chosen_indices):
        assert y[e * 10 + chosen] == 1
        # Every other row in the same event should be 0.
        for a in range(10):
            if a != chosen:
                assert y[e * 10 + a] == 0


def test_batch_to_long_format_rejects_empty():
    import pytest
    empty = BaselineEventBatch(
        base_features_list=[],
        base_feature_names=["a", "b"],
        chosen_indices=[],
        customer_ids=[],
        categories=[],
    )
    with pytest.raises(ValueError, match="empty"):
        _batch_to_long_format(empty)


# ── RandomForest ─────────────────────────────────────────────────────────────


def test_random_forest_smoke_learns_signal():
    train, val, test = _shared_triplet(seed_base=110)

    fitted = RandomForestChoice(n_estimators=80, max_depth=8, random_state=0).fit(train, val)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    assert isinstance(fitted, FittedBaseline)
    assert isinstance(fitted, ClassicalMLFitted)
    assert report.metrics["top1"] > 0.20, f"RF top-1 = {report.metrics['top1']:.3f}"
    assert fitted.n_params > 0


def test_random_forest_description_mentions_trees():
    batch = make_synthetic_batch(n_events=80, seed=114)
    fitted = RandomForestChoice(n_estimators=10, max_depth=3, random_state=0).fit(batch, batch)
    desc = fitted.description
    assert "RandomForest" in desc
    assert "n_trees=10" in desc
    assert "leaves=" in desc


# ── GradientBoosting ─────────────────────────────────────────────────────────


def test_gradient_boosting_smoke_learns_signal():
    train, val, test = _shared_triplet(seed_base=120)

    fitted = GradientBoostingChoice(
        max_iter=120, max_depth=5, learning_rate=0.1, random_state=0
    ).fit(train, val)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    assert isinstance(fitted, FittedBaseline)
    assert report.metrics["top1"] > 0.20, f"HGB top-1 = {report.metrics['top1']:.3f}"
    assert fitted.n_params > 0


def test_gradient_boosting_description_mentions_iterations():
    batch = make_synthetic_batch(n_events=80, seed=124)
    fitted = GradientBoostingChoice(max_iter=25, max_depth=4, random_state=0).fit(batch, batch)
    desc = fitted.description
    assert "GradientBoosting" in desc
    assert "max_iter=25" in desc
    assert "leaves=" in desc


# ── MLP ──────────────────────────────────────────────────────────────────────


def test_mlp_smoke_learns_signal():
    train, val, test = _shared_triplet(seed_base=130)

    fitted = MLPChoice(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        random_state=0,
    ).fit(train, val)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    assert isinstance(fitted, FittedBaseline)
    assert report.metrics["top1"] > 0.20, f"MLP top-1 = {report.metrics['top1']:.3f}"
    assert fitted.n_params > 0


def test_mlp_description_mentions_architecture():
    batch = make_synthetic_batch(n_events=160, seed=134)
    fitted = MLPChoice(hidden_layer_sizes=(16, 8), max_iter=100, random_state=0).fit(batch, batch)
    desc = fitted.description
    assert "MLP" in desc
    assert "16 x 8" in desc
    assert "params=" in desc


# ── Cross-baseline sanity ────────────────────────────────────────────────────


def test_all_three_produce_comparable_score_shapes():
    """All three baselines should return List[ndarray[n_alts]] from score_events."""
    batch = make_synthetic_batch(n_events=40, n_alts=10, seed=141)
    baselines = [
        RandomForestChoice(n_estimators=10, max_depth=3, random_state=0),
        GradientBoostingChoice(max_iter=20, max_depth=3, random_state=0),
        MLPChoice(hidden_layer_sizes=(8,), max_iter=60, random_state=0),
    ]
    for bl in baselines:
        fitted = bl.fit(batch, batch)
        scores = fitted.score_events(batch)
        assert len(scores) == 40
        for s in scores:
            assert s.shape == (10,)
            assert np.all(np.isfinite(s))
