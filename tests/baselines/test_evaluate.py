"""Unit tests for :func:`src.baselines.evaluate.evaluate_baseline`.

Coverage focuses on the paper-grade instrumentation added per
``docs/paper_evaluation_additions.md`` (additions 1 + 2):

* ``per_event_nll``  — length = n_events, mean equals ``test_nll``.
* ``per_event_topk_correct`` — length = n_events, parallel to NLL.
* ``per_customer_nll`` — grouped aggregation matches per-event values.

These are *invariants* downstream significance/segmentation scripts
consume; breaking them silently corrupts the paper table.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.baselines._synthetic import make_synthetic_batch
from src.baselines.base import BaselineEventBatch, BaselineReport
from src.baselines.evaluate import evaluate_baseline


class _DeterministicFitted:
    """A fitted baseline that returns caller-supplied per-event score arrays.

    Lets us pin an exact expected ``per_event_nll`` from a known
    logits tensor and verify the harness threads it through unchanged.
    """

    name = "Deterministic"

    def __init__(self, score_matrix: np.ndarray):
        self._scores = np.asarray(score_matrix, dtype=np.float32)

    def score_events(self, batch: BaselineEventBatch):
        return [self._scores[i] for i in range(batch.n_events)]

    @property
    def n_params(self) -> int:
        return 0

    @property
    def description(self) -> str:
        return "deterministic-fitted for unit tests"


class _ZeroFitted:
    """Returns all-zero utility vectors (uniform softmax)."""

    name = "Zero"

    def __init__(self, n_alts: int):
        self._n_alts = n_alts

    def score_events(self, batch: BaselineEventBatch):
        return [np.zeros(self._n_alts) for _ in range(batch.n_events)]

    @property
    def n_params(self) -> int:
        return 0

    @property
    def description(self) -> str:
        return "zero-utility"


# ---------------------------------------------------------------------------
# Invariant 1: lengths match n_events
# ---------------------------------------------------------------------------


def test_per_event_arrays_length_matches_n_events():
    batch = make_synthetic_batch(n_events=37, n_alts=6, seed=101)
    fitted = _ZeroFitted(n_alts=6)
    report = evaluate_baseline(fitted, batch, train_n_events=37)

    assert len(report.per_event_nll) == batch.n_events == 37
    assert len(report.per_event_topk_correct) == batch.n_events == 37
    assert report.metrics["n_events"] == 37


# ---------------------------------------------------------------------------
# Invariant 2: mean(per_event_nll) == test_nll (within 1e-6)
# ---------------------------------------------------------------------------


def test_mean_per_event_nll_matches_aggregate_test_nll():
    """The critical invariant. Downstream bootstrap resampling relies
    on ``per_event_nll`` averaging back to the reported aggregate.
    """
    batch = make_synthetic_batch(n_events=80, n_alts=5, seed=7)
    rng = np.random.default_rng(0)
    # A non-trivial scoring signal so NLL varies across events.
    scores = rng.normal(0.0, 1.0, size=(batch.n_events, 5)).astype(np.float32)
    fitted = _DeterministicFitted(score_matrix=scores)
    report = evaluate_baseline(fitted, batch, train_n_events=batch.n_events)

    mean_nll = float(np.mean(report.per_event_nll))
    aggregate = float(report.metrics["test_nll"])
    assert abs(mean_nll - aggregate) < 1e-6, (
        f"per-event mean {mean_nll} deviates from aggregate {aggregate}"
    )


def test_mean_per_event_nll_matches_for_zero_baseline():
    """Uniform softmax → every per-event NLL equals ``log(n_alts)``."""
    batch = make_synthetic_batch(n_events=50, n_alts=7, seed=99)
    fitted = _ZeroFitted(n_alts=7)
    report = evaluate_baseline(fitted, batch, train_n_events=50)

    expected = float(np.log(7))
    for v in report.per_event_nll:
        assert abs(v - expected) < 1e-6
    assert abs(float(np.mean(report.per_event_nll)) - report.metrics["test_nll"]) < 1e-6


# ---------------------------------------------------------------------------
# Invariant 3: per_event_topk_correct is parallel to per_event_nll
# ---------------------------------------------------------------------------


def test_per_event_topk_correct_matches_argmax():
    """``per_event_topk_correct[i] == (argmax(scores[i]) == chosen[i])``."""
    batch = make_synthetic_batch(n_events=30, n_alts=4, seed=55)
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(batch.n_events, 4)).astype(np.float32)
    fitted = _DeterministicFitted(score_matrix=scores)
    report = evaluate_baseline(fitted, batch, train_n_events=30)

    expected = [
        (int(np.argmax(scores[i])) == int(batch.chosen_indices[i]))
        for i in range(batch.n_events)
    ]
    assert report.per_event_topk_correct == expected
    # Aggregate top1 equals the mean of the per-event flags.
    mean_t1 = float(np.mean(report.per_event_topk_correct))
    assert abs(mean_t1 - report.metrics["top1"]) < 1e-6


# ---------------------------------------------------------------------------
# Invariant 4: per_customer_nll keys are a subset of batch.customer_ids
# ---------------------------------------------------------------------------


def test_per_customer_keys_subset_of_batch_customer_ids():
    batch = make_synthetic_batch(
        n_events=60, n_alts=4, n_customers=8, seed=3
    )
    fitted = _ZeroFitted(n_alts=4)
    report = evaluate_baseline(fitted, batch, train_n_events=60)

    batch_cids = set(str(c) for c in batch.customer_ids)
    assert set(report.per_customer_nll.keys()).issubset(batch_cids)
    # Every customer that appears in the test batch should be in the
    # per_customer_nll dict (complete accounting).
    assert set(report.per_customer_nll.keys()) == batch_cids


# ---------------------------------------------------------------------------
# Invariant 5: per-customer aggregation equals local mean of per-event NLL
# ---------------------------------------------------------------------------


def test_per_customer_nll_matches_local_mean():
    """For every customer k: per_customer_nll[k].nll == mean of per-event
    NLL restricted to events where customer_id == k. Same invariant on
    n_events and top1.
    """
    batch = make_synthetic_batch(
        n_events=50, n_alts=5, n_customers=6, seed=21
    )
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(batch.n_events, 5)).astype(np.float32)
    fitted = _DeterministicFitted(score_matrix=scores)
    report = evaluate_baseline(fitted, batch, train_n_events=50)

    nll_arr = np.asarray(report.per_event_nll, dtype=np.float64)
    t1_arr = np.asarray(report.per_event_topk_correct, dtype=np.float64)
    cid_arr = np.asarray([str(c) for c in batch.customer_ids])

    for cid, entry in report.per_customer_nll.items():
        mask = cid_arr == cid
        n_c = int(mask.sum())
        assert entry["n_events"] == n_c
        expected_nll = float(nll_arr[mask].mean())
        assert abs(entry["nll"] - expected_nll) < 1e-9
        expected_t1 = float(t1_arr[mask].sum() / n_c)
        assert abs(entry["top1"] - expected_t1) < 1e-9


# ---------------------------------------------------------------------------
# Serialization sanity: every per-event element is a plain Python scalar
# ---------------------------------------------------------------------------


def test_per_event_fields_are_json_scalars():
    """CSV / JSON writer expects ``float`` and ``bool`` — not numpy scalars."""
    batch = make_synthetic_batch(n_events=10, n_alts=4, seed=4)
    fitted = _ZeroFitted(n_alts=4)
    report = evaluate_baseline(fitted, batch, train_n_events=10)

    for v in report.per_event_nll:
        assert type(v) is float
    for v in report.per_event_topk_correct:
        assert type(v) is bool
    for cid, entry in report.per_customer_nll.items():
        assert type(cid) is str
        assert type(entry["nll"]) is float
        assert type(entry["n_events"]) is int
        assert type(entry["top1"]) is float
