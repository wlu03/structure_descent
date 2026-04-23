"""
Scaffold-level smoke test.

Exercises the baseline interface end-to-end using a trivial ZeroBaseline
(returns all-zero utility scores). This must pass before any real baseline
is considered working — it verifies the harness contract itself.
"""

from __future__ import annotations

import numpy as np

from src.baselines import (
    BaselineEventBatch,
    BaselineReport,
    FittedBaseline,
    evaluate_baseline,
    expand_batch,
)
from src.baselines._synthetic import make_synthetic_batch
from src.baselines.feature_pool import build_expanded_pool, expanded_pool_size


class _ZeroFitted:
    """A fitted baseline that always returns a zero score vector.

    Under the harness's tie-breaking convention (rank counts only strictly
    greater alternatives), a zero score vector yields top-1 = 100% because
    nothing is strictly greater than the chosen alternative. This matches
    the convention used in src/evaluation.py and we verify it here.
    """

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
        return "zero-utility reference baseline"


class _RandomFitted:
    """A fitted baseline that returns iid uniform random scores per event.

    Tests that the harness actually computes chance-level metrics when the
    baseline has no signal. Expected top-1 ~ 1 / n_alts.
    """

    name = "Random"

    def __init__(self, n_alts: int, seed: int = 0):
        self._n_alts = n_alts
        self._rng = np.random.default_rng(seed)

    def score_events(self, batch: BaselineEventBatch):
        return [self._rng.normal(size=self._n_alts) for _ in range(batch.n_events)]

    @property
    def n_params(self) -> int:
        return 0

    @property
    def description(self) -> str:
        return "random-utility reference baseline"


def test_synthetic_batch_shapes():
    batch = make_synthetic_batch(n_events=50, n_alts=10, seed=1)
    assert batch.n_events == 50
    assert batch.n_alternatives == 10
    assert batch.n_base_terms == 12
    assert len(batch.chosen_indices) == 50
    assert len(batch.customer_ids) == 50
    assert all(0 <= c < 10 for c in batch.chosen_indices)
    assert all(f.shape == (10, 12) for f in batch.base_features_list)


def test_feature_pool_expansion():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 12))
    names = [f"f{i}" for i in range(12)]

    exp, exp_names = build_expanded_pool(base, names, include_interactions=True)
    assert exp.shape == (10, expanded_pool_size(12, include_interactions=True))
    assert len(exp_names) == exp.shape[1]
    # First 12 columns should be the identity
    np.testing.assert_allclose(exp[:, :12], base)

    exp_no_x, _ = build_expanded_pool(base, names, include_interactions=False)
    assert exp_no_x.shape == (10, expanded_pool_size(12, include_interactions=False))


def test_expand_batch_roundtrip():
    batch = make_synthetic_batch(n_events=20, n_alts=10, seed=2)
    expanded_list, names = expand_batch(batch, include_interactions=True)
    assert len(expanded_list) == 20
    assert len(names) == expanded_pool_size(12, include_interactions=True)
    for exp in expanded_list:
        assert exp.shape == (10, len(names))


def test_evaluate_zero_baseline_documents_tie_convention():
    """
    Unified tie convention (post Phase-1 migration): top-k uses
    ``torch.topk`` with lowest-index-first tie-break, same as
    :func:`src.eval.metrics.topk_accuracy`. A zero-utility baseline on
    ``n_alts=10`` therefore reports top-1 ≈ 1/10 (the chosen alt wins
    only when its index is 0) and top-5 ≈ 5/10 — *not* 100% as under
    the old biased "ties favor chosen" convention.

    MRR uses the unbiased ``1 + strictly_greater + 0.5 * ties``
    formula, so a zero-baseline reports MRR ≈ ``2 / (n_alts + 1) =
    2/11 ≈ 0.1818``. NLL is ``log(n_alts)`` since softmax of zeros is
    uniform.
    """
    batch = make_synthetic_batch(n_events=100, n_alts=10, seed=3)
    fitted = _ZeroFitted(n_alts=10)
    report = evaluate_baseline(fitted, batch, train_n_events=100)

    assert isinstance(report, BaselineReport)
    assert report.name == "Zero"
    assert report.n_params == 0
    assert set(report.metrics.keys()) >= {
        "top1", "top5", "mrr", "test_nll", "aic", "bic", "n_events"
    }
    # Under argmax-ties-lowest-index, top-1 = P(chosen_idx == 0).
    # The synthetic sampler draws chosen indices ~ uniform over n_alts,
    # so the expected fraction is 1/10 ≈ 0.07–0.15 at n=100.
    assert 0.02 <= report.metrics["top1"] <= 0.20, (
        f"top1 {report.metrics['top1']!r} not near 0.1 under the new "
        "lowest-index tie-break"
    )
    # top-5: chosen_idx in {0..4}, expected 0.5 for uniform labels.
    assert 0.30 <= report.metrics["top5"] <= 0.70
    # Unbiased MRR on an all-equal row = 2 / (n_alts + 1).
    expected_mrr = 2.0 / 11.0
    assert abs(report.metrics["mrr"] - expected_mrr) < 0.02
    assert abs(report.metrics["test_nll"] - np.log(10)) < 1e-6
    assert report.metrics["n_events"] == 100


def test_evaluate_random_baseline_near_chance():
    """A random baseline should achieve ~1/n_alts top-1 accuracy."""
    batch = make_synthetic_batch(n_events=2000, n_alts=10, seed=5)
    fitted = _RandomFitted(n_alts=10, seed=17)
    report = evaluate_baseline(fitted, batch, train_n_events=2000)
    assert 0.05 < report.metrics["top1"] < 0.20
    assert 0.40 < report.metrics["top5"] < 0.60


def test_fitted_baseline_protocol_check():
    fitted = _ZeroFitted(n_alts=10)
    assert isinstance(fitted, FittedBaseline)
