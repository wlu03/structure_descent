"""Tests for the popularity baseline and the NLL uplift post-processing pass.

Covers both granularities supported by :class:`PopularityBaseline`:
ASIN-level (when ``raw_events`` exposes ``choice_asins``) and the
(category, within-event position) fallback used for synthetic batches
or anywhere global item IDs aren't plumbed through.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from src.baselines import (
    BaselineEventBatch,
    BaselineReport,
    FittedBaseline,
    evaluate_baseline,
)
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.popularity import (
    PopularityBaseline,
    PopularityFitted,
    _event_has_asins,
)
from src.baselines.run_all import (
    UPLIFT_EXTRA_KEY,
    annotate_uplift_vs_popularity,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _batch_with_asins(
    events_asin_chosen: List[tuple[List[str], int, str]],
) -> BaselineEventBatch:
    """Construct a :class:`BaselineEventBatch` carrying ``choice_asins``.

    Each entry is ``(asin_list, chosen_idx, category)``; all events must
    share the same ``J = len(asin_list)``.
    """
    if not events_asin_chosen:
        raise ValueError("empty fixture")
    J = len(events_asin_chosen[0][0])
    n_feat = 3
    base_list: List[np.ndarray] = []
    chosen: List[int] = []
    cats: List[str] = []
    raw_events: List[dict] = []
    for asins, ci, cat in events_asin_chosen:
        assert len(asins) == J, "uniform J required"
        base_list.append(np.zeros((J, n_feat), dtype=np.float32))
        chosen.append(int(ci))
        cats.append(cat)
        raw_events.append(
            {
                "customer_id": "c0",
                "choice_asins": list(asins),
                "chosen_idx": int(ci),
                "category": cat,
            }
        )
    return BaselineEventBatch(
        base_features_list=base_list,
        base_feature_names=[f"f{i}" for i in range(n_feat)],
        chosen_indices=chosen,
        customer_ids=[f"c{i}" for i in range(len(events_asin_chosen))],
        categories=cats,
        raw_events=raw_events,
    )


def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D utility vector."""
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


# ── ASIN-granularity behaviour ──────────────────────────────────────────────


def test_event_has_asins_detection():
    with_asins = _batch_with_asins(
        [
            (["A", "B", "C"], 0, "cat0"),
            (["A", "B", "D"], 1, "cat0"),
        ]
    )
    assert _event_has_asins(with_asins)

    # ``make_synthetic_batch`` now populates ``raw_events`` with text
    # fields (alt_texts, c_d, order_date) so ST-MLP and the LLM rankers
    # can run against it, but it does NOT set ``choice_asins``. The
    # popularity baseline's ASIN-mode detection must therefore still
    # report False on synthetic and fall back to (category, slot) keys.
    no_asins = make_synthetic_batch(n_events=5, n_alts=4, seed=1)
    assert no_asins.raw_events is not None
    assert "choice_asins" not in no_asins.raw_events[0]
    assert _event_has_asins(no_asins) is False


def test_popularity_asin_counts_match_training_frequency():
    # A chosen 3 times, B chosen 1 time. Unseen alternative D is 0.
    train = _batch_with_asins(
        [
            (["A", "B", "C"], 0, "cat0"),
            (["A", "B", "C"], 0, "cat0"),
            (["A", "B", "C"], 0, "cat0"),
            (["A", "B", "C"], 1, "cat0"),
        ]
    )
    fitted = PopularityBaseline(alpha=1.0).fit(train, train)
    assert isinstance(fitted, FittedBaseline)
    assert isinstance(fitted, PopularityFitted)
    assert fitted.granularity == "asin"
    assert fitted.asin_counts == {"A": 3, "B": 1}  # C never chosen -> absent

    # Score a test event with alternatives [A, B, D]. D has count 0 →
    # Laplace-smoothed score log(1). A -> log(4), B -> log(2).
    test = _batch_with_asins([(["A", "B", "D"], 0, "cat0")])
    scores = fitted.score_events(test)
    assert len(scores) == 1
    np.testing.assert_allclose(
        scores[0],
        np.asarray([math.log(4.0), math.log(2.0), math.log(1.0)]),
        rtol=1e-6,
    )
    # After softmax the probabilities are proportional to (count+1).
    probs = _scores_to_probs(scores[0])
    np.testing.assert_allclose(probs, np.asarray([4, 2, 1]) / 7.0, rtol=1e-6)


def test_popularity_laplace_prevents_minus_inf_for_unseen():
    train = _batch_with_asins([(["A", "B"], 0, "cat0")])
    fitted = PopularityBaseline(alpha=1.0).fit(train, train)
    # Score a test event whose alternatives are ALL unseen.
    test = _batch_with_asins([(["X", "Y"], 0, "cat0")])
    scores = fitted.score_events(test)
    assert np.all(np.isfinite(scores[0]))
    # All zero counts + alpha=1 → uniform over alternatives after softmax.
    np.testing.assert_allclose(
        _scores_to_probs(scores[0]), np.full(2, 0.5), rtol=1e-6
    )


def test_popularity_asin_mode_requires_choice_asins_in_test_batch():
    train = _batch_with_asins([(["A", "B"], 0, "cat0")])
    fitted = PopularityBaseline(alpha=1.0).fit(train, train)
    # Strip raw_events from the test batch.
    test_no_raw = BaselineEventBatch(
        base_features_list=[np.zeros((2, 1), dtype=np.float32)],
        base_feature_names=["f"],
        chosen_indices=[0],
        customer_ids=["c0"],
        categories=["cat0"],
        raw_events=None,
    )
    with pytest.raises(ValueError, match="choice_asins"):
        fitted.score_events(test_no_raw)


def test_popularity_forced_asin_without_choice_asins_raises_at_fit():
    # Synthetic ``raw_events`` does not carry ``choice_asins``; the
    # ASIN-mode baseline must refuse to fit on it with a clear error.
    batch = make_synthetic_batch(n_events=10, n_alts=4, seed=2)
    assert batch.raw_events is not None
    assert "choice_asins" not in batch.raw_events[0]
    with pytest.raises(ValueError, match="choice_asins"):
        PopularityBaseline(granularity="asin").fit(batch, batch)


def test_popularity_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        PopularityBaseline(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        PopularityBaseline(alpha=-1.0)


def test_popularity_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="granularity"):
        PopularityBaseline(granularity="per_customer")


def test_popularity_fit_rejects_empty_batch():
    empty = BaselineEventBatch(
        base_features_list=[],
        base_feature_names=["f"],
        chosen_indices=[],
        customer_ids=[],
        categories=[],
    )
    with pytest.raises(ValueError, match="empty"):
        PopularityBaseline().fit(empty, empty)


# ── category-position fallback ──────────────────────────────────────────────


def test_popularity_falls_back_to_category_position_on_synthetic():
    # Synthetic batches have raw_events=None, so auto mode must fall back.
    train = make_synthetic_batch(n_events=400, n_alts=6, seed=31, signal_strength=2.0)
    fitted = PopularityBaseline(alpha=1.0).fit(train, train)
    assert fitted.granularity == "category_position"
    assert fitted.n_params > 0
    assert fitted.asin_counts == {}
    assert len(fitted.cat_pos_counts) == fitted.n_params


def test_popularity_category_position_scores_are_log_count_plus_alpha():
    # Hand-crafted batch: 3 events in "cat0", 2 events in "cat1".
    rng = np.random.default_rng(0)
    n_feat = 2
    J = 3
    events = [
        ("cat0", 0),
        ("cat0", 0),
        ("cat0", 2),
        ("cat1", 1),
        ("cat1", 1),
    ]
    batch = BaselineEventBatch(
        base_features_list=[rng.normal(size=(J, n_feat)) for _ in events],
        base_feature_names=["f0", "f1"],
        chosen_indices=[c for _, c in events],
        customer_ids=[f"c{i}" for i in range(len(events))],
        categories=[cat for cat, _ in events],
    )
    fitted = PopularityBaseline(alpha=1.0, granularity="category_position").fit(
        batch, batch
    )
    # Score one test event in cat0. Expected log-counts:
    #   pos 0: log(2+1)=log 3, pos 1: log(0+1)=log 1, pos 2: log(1+1)=log 2.
    test = BaselineEventBatch(
        base_features_list=[np.zeros((J, n_feat), dtype=np.float32)],
        base_feature_names=["f0", "f1"],
        chosen_indices=[0],
        customer_ids=["cx"],
        categories=["cat0"],
    )
    scores = fitted.score_events(test)[0]
    np.testing.assert_allclose(
        scores, np.asarray([math.log(3.0), math.log(1.0), math.log(2.0)]), rtol=1e-6
    )


def test_popularity_unseen_category_scores_uniform():
    # Train on cat0 only; score on a test event whose category is unseen.
    J = 4
    batch = BaselineEventBatch(
        base_features_list=[np.zeros((J, 1), dtype=np.float32) for _ in range(3)],
        base_feature_names=["f"],
        chosen_indices=[0, 1, 2],
        customer_ids=["a", "b", "c"],
        categories=["cat0"] * 3,
    )
    fitted = PopularityBaseline(alpha=1.0, granularity="category_position").fit(
        batch, batch
    )
    test = BaselineEventBatch(
        base_features_list=[np.zeros((J, 1), dtype=np.float32)],
        base_feature_names=["f"],
        chosen_indices=[0],
        customer_ids=["z"],
        categories=["cat_unseen"],
    )
    scores = fitted.score_events(test)[0]
    np.testing.assert_allclose(scores, np.full(J, math.log(1.0)), rtol=1e-6)
    np.testing.assert_allclose(
        _scores_to_probs(scores), np.full(J, 1.0 / J), rtol=1e-6
    )


# ── evaluate_baseline integration ───────────────────────────────────────────


def test_evaluate_popularity_produces_well_formed_report():
    train = make_synthetic_batch(n_events=300, n_alts=5, seed=41, signal_strength=1.5)
    test = make_synthetic_batch(n_events=120, n_alts=5, seed=42, signal_strength=1.5)
    fitted = PopularityBaseline(alpha=1.0).fit(train, train)
    report = evaluate_baseline(fitted, test, train_n_events=train.n_events)

    assert isinstance(report, BaselineReport)
    assert report.name == "Popularity"
    assert report.n_params > 0
    assert set(report.metrics.keys()) >= {
        "top1", "top5", "mrr", "test_nll", "aic", "bic", "n_events"
    }
    assert report.metrics["n_events"] == 120
    assert np.isfinite(report.metrics["test_nll"])


def test_popularity_beats_uniform_on_nondegenerate_training():
    """Popularity should beat the pure-uniform reference.

    Evaluated on the SAME non-degenerate training batch as the test set
    (so the empirical distribution matches). A correctly smoothed
    popularity model must hit top-1 > 1/J and NLL < log J.
    """
    J = 6
    # Biased synthetic: all events share category "cat0" so category_pos
    # falls back to a pure per-position prior, and we rig chosen_idx to
    # concentrate mass on position 0.
    rng = np.random.default_rng(7)
    n_events = 400
    chosen = rng.integers(0, 2, size=n_events)  # positions {0, 1} only
    batch = BaselineEventBatch(
        base_features_list=[
            rng.normal(size=(J, 2)).astype(np.float32) for _ in range(n_events)
        ],
        base_feature_names=["f0", "f1"],
        chosen_indices=[int(c) for c in chosen],
        customer_ids=[f"c{i % 50}" for i in range(n_events)],
        categories=["cat0"] * n_events,
    )
    fitted = PopularityBaseline(alpha=1.0).fit(batch, batch)
    report = evaluate_baseline(fitted, batch, train_n_events=batch.n_events)

    # Uniform references:
    uniform_top1 = 1.0 / J
    uniform_nll = math.log(J)
    assert report.metrics["top1"] > uniform_top1, (
        f"popularity top1 {report.metrics['top1']:.3f} should beat "
        f"uniform {uniform_top1:.3f}"
    )
    assert report.metrics["test_nll"] < uniform_nll, (
        f"popularity NLL {report.metrics['test_nll']:.4f} should beat "
        f"uniform {uniform_nll:.4f}"
    )


# ── uplift post-processing ──────────────────────────────────────────────────


def _mk_report(name: str, test_nll: float) -> BaselineReport:
    return BaselineReport(
        name=name,
        n_params=1,
        metrics={
            "top1": 0.1,
            "top5": 0.5,
            "mrr": 0.2,
            "test_nll": test_nll,
            "aic": 0.0,
            "bic": 0.0,
            "n_events": 100,
        },
    )


def test_annotate_uplift_writes_to_extra_not_metrics():
    pop = _mk_report("Popularity", test_nll=2.0)
    better = _mk_report("LASSO-MNL", test_nll=1.6)
    worse = _mk_report("Worse", test_nll=2.3)
    reports = {"Popularity": pop, "LASSO-MNL": better, "Worse": worse}
    rows = [
        {"name": "Popularity", "test_nll": 2.0},
        {"name": "LASSO-MNL", "test_nll": 1.6},
        {"name": "Worse", "test_nll": 2.3},
    ]

    annotate_uplift_vs_popularity(reports, rows)

    # Popularity itself gets uplift = 0 exactly.
    assert pop.extra[UPLIFT_EXTRA_KEY] == pytest.approx(0.0)
    # Models that beat popularity get positive uplift (pop_nll - model_nll).
    assert better.extra[UPLIFT_EXTRA_KEY] == pytest.approx(0.4)
    # Models worse than popularity get negative uplift.
    assert worse.extra[UPLIFT_EXTRA_KEY] == pytest.approx(-0.3)
    # Uplift must live in `extra`, not in `metrics` (another agent
    # owns `metrics`).
    for r in reports.values():
        assert UPLIFT_EXTRA_KEY not in r.metrics
    # Rows are mirrored so the summary table can render the column.
    row_by_name = {row["name"]: row for row in rows}
    assert row_by_name["LASSO-MNL"][UPLIFT_EXTRA_KEY] == pytest.approx(0.4)
    assert row_by_name["Worse"][UPLIFT_EXTRA_KEY] == pytest.approx(-0.3)


def test_annotate_uplift_is_noop_when_popularity_missing():
    other = _mk_report("LASSO-MNL", test_nll=1.6)
    reports = {"LASSO-MNL": other}
    rows = [{"name": "LASSO-MNL", "test_nll": 1.6}]
    annotate_uplift_vs_popularity(reports, rows)
    assert UPLIFT_EXTRA_KEY not in other.extra
    assert UPLIFT_EXTRA_KEY not in rows[0]
