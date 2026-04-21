"""Offline tests for outer-loop prompt improvements (items 1, 2, 4, 5, 6, 8).

Runs without any LLM API calls or real training. Uses hand-built mocks for
HierarchicalWeights, DSLStructure, metrics, residuals, history, and proposal_log.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from old_pipeline.src.dsl import DSLStructure, DSLTerm  # noqa: E402
from old_pipeline.src.inner_loop import HierarchicalWeights  # noqa: E402
from old_pipeline.src.outer_loop import (  # noqa: E402
    generate_diagnostics,
    apply_proposal,
    _detect_rejection_pattern,
    _format_proposal_history,
)


# ──────────────────────────────────────────────────────────────────────────
# Mocks
# ──────────────────────────────────────────────────────────────────────────

def make_mock_structure() -> DSLStructure:
    return DSLStructure(terms=["routine", "affinity", "popularity"])


def make_mock_weights(structure: DSLStructure) -> HierarchicalWeights:
    # routine: HIGH (|w|>1.0), affinity: LOW (|w|<0.1 → removal candidate),
    # popularity: MEDIUM (0.3 ≤ |w| ≤ 1.0)
    theta_g = np.array([1.82, 0.05, 0.64])
    theta_c = {
        "HEADPHONES":   np.array([0.10, 0.80, -0.05]),
        "FLASH_MEMORY": np.array([0.40, 0.05, -0.20]),
        "KITCHEN":      np.array([-0.30, 0.02, 0.01]),
    }
    delta_i = {}
    return HierarchicalWeights(theta_g=theta_g, theta_c=theta_c, delta_i=delta_i)


def make_mock_metrics() -> dict:
    return {"top1": 0.082, "top5": 0.31, "mrr": 0.154, "val_nll": 4.12}


def make_mock_residuals() -> dict:
    return {
        "slices": [
            {"name": "HEADPHONES",   "top1": 0.084, "n_events": 12300,
             "pct_first_buy_errors": 0.64, "pct_popular_errors": 0.31},
            {"name": "DISHWARE_BOWL", "top1": 0.101, "n_events": 3400,
             "pct_first_buy_errors": 0.18, "pct_popular_errors": 0.52},
            {"name": "FLASH_MEMORY",  "top1": 0.182, "n_events": 15700,
             "pct_first_buy_errors": 0.12, "pct_popular_errors": 0.47},
        ],
        "overall_patterns": [
            "Model accuracy drops 40% on items bought for the first time",
            "Errors concentrate in low-popularity (rank > 500) items",
        ],
    }


def make_mock_history() -> list:
    return [
        {"iteration": 0, "structure": "S = routine", "score": -4250.0, "accepted": True},
        {"iteration": 1, "structure": "S = routine + popularity",
         "score": -4220.0, "accepted": True,
         "changes": [{"op": "ADD", "term": "popularity"}]},
        {"iteration": 2, "structure": "S = routine + popularity + affinity",
         "score": -4170.5, "accepted": True,
         "changes": [{"op": "ADD", "term": "interaction(routine, recency)"}]},
    ]


def make_mock_proposal_log_with_pattern() -> list:
    """3 rejected proposals that all use `power(...)` → rejection pattern should trigger."""
    return [
        {
            "iteration": 1,
            "changes": [{"op": "ADD", "term": "power(affinity, exponent=2)"}],
            "reasoning": "capture diminishing returns on affinity",
            "hypothesis": "diminishing returns on category loyalty",
            "accepted": False,
            "score_delta": -0.45,
        },
        {
            "iteration": 2,
            "changes": [{"op": "ADD", "term": "power(popularity, exponent=0.5)"}],
            "reasoning": "smoothing popularity",
            "hypothesis": "popular items are saturating",
            "accepted": False,
            "score_delta": -1.20,
        },
        {
            "iteration": 3,
            "changes": [{"op": "ADD", "term": "power(recency, exponent=2)"}],
            "reasoning": "bend recency curve",
            "hypothesis": "recency decays too linearly",
            "accepted": False,
            "score_delta": -0.60,
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# Diagnostics rendering
# ──────────────────────────────────────────────────────────────────────────

def test_generate_diagnostics_contains_all_sections():
    structure = make_mock_structure()
    weights = make_mock_weights(structure)
    metrics = make_mock_metrics()
    residuals = make_mock_residuals()
    history = make_mock_history()
    proposal_log = make_mock_proposal_log_with_pattern()

    report = generate_diagnostics(
        structure=structure,
        posterior_score=-4170.5,
        metrics=metrics,
        residuals=residuals,
        proposal_log=proposal_log,
        weights=weights,
        history=history,
        iteration=3,
        n_iterations=10,
    )

    # Item 1: fitted global weights with importance
    assert "weight=" in report
    assert "importance=HIGH" in report  # routine has |w|=1.82 > 1.0
    assert "top contributor" in report
    # affinity |w|=0.05 is below removal-candidate threshold
    assert "candidate for removal" in report
    # Category deviations block present
    assert "Category-level deviations" in report

    # Item 5: trajectory
    assert "Trajectory" in report
    assert "Best score so far" in report
    assert "iter 0" in report
    assert "(iteration 3 of 10)" in report

    # Item 2: structured residual table
    assert "top-1" in report
    assert "HEADPHONES" in report
    assert "Key patterns:" in report

    # Item 6: rejection pattern
    assert "Pattern detected" in report
    assert "power" in report

    # Item 4: hypothesis-first reasoning template
    assert "HYPOTHESIS" in report
    assert "MECHANISM" in report
    assert "CANDIDATES" in report

    # Item 8: candidates + selected schema
    assert "candidates" in report
    assert "selected" in report


def test_generate_diagnostics_backwards_compat_no_weights_no_history():
    """Old-style callers (no weights, no history, old residual dict) still work."""
    structure = make_mock_structure()
    metrics = make_mock_metrics()
    old_residuals = {
        "Category 'HEADPHONES' top-1": "8.4% (n=12300)",
        "Repeat purchase top-1":       "30.0% (n=5000)",
    }
    report = generate_diagnostics(
        structure=structure,
        posterior_score=-4200.0,
        metrics=metrics,
        residuals=old_residuals,
        proposal_log=None,
        weights=None,
        history=None,
    )
    assert "Current structure:" in report  # single-line fallback
    assert "Posterior score: -4200.00" in report
    assert "Error analysis:" in report
    assert "HEADPHONES" in report  # rendered via old fallback


# ──────────────────────────────────────────────────────────────────────────
# Rejection pattern detector (item 6)
# ──────────────────────────────────────────────────────────────────────────

def test_rejection_pattern_detects_combinator():
    rejected = make_mock_proposal_log_with_pattern()
    msg = _detect_rejection_pattern(rejected)
    assert msg is not None
    assert "power" in msg
    assert "combinator" in msg.lower() or "combinators" in msg.lower()


def test_rejection_pattern_returns_none_when_below_threshold():
    rejected = [
        {"iteration": 1, "changes": [{"op": "ADD", "term": "power(affinity)"}]},
        {"iteration": 2, "changes": [{"op": "ADD", "term": "ratio(a, b)"}]},
    ]
    # Only 2 rejections → below the min-3 threshold
    assert _detect_rejection_pattern(rejected) is None


def test_format_proposal_history_includes_hypothesis():
    log = make_mock_proposal_log_with_pattern()
    text = _format_proposal_history(log)
    assert "hypothesis was:" in text
    assert "Pattern detected" in text


# ──────────────────────────────────────────────────────────────────────────
# apply_proposal — item 8 schema
# ──────────────────────────────────────────────────────────────────────────

def test_apply_proposal_candidates_selected():
    structure = make_mock_structure()
    proposal = {
        "hypothesis": "popular products win too often",
        "mechanism":  "threshold on popularity",
        "candidates": [
            {"changes": [{"op": "ADD", "term": "recency", "reason": "..."}],
             "expected_impact": "HIGH",   "rationale": "a"},
            {"changes": [{"op": "ADD", "term": "time_match", "reason": "..."}],
             "expected_impact": "MEDIUM", "rationale": "b"},
            {"changes": [{"op": "ADD", "term": "novelty", "reason": "..."}],
             "expected_impact": "LOW",    "rationale": "c"},
        ],
        "selected":  0,
        "reasoning": "top-ranked best addresses the failure mode",
    }
    new_structure = apply_proposal(structure, proposal)
    assert new_structure is not None
    assert "recency" in [t.name for t in new_structure.terms]
    assert "time_match" not in [t.name for t in new_structure.terms]
    assert "novelty" not in [t.name for t in new_structure.terms]


def test_apply_proposal_candidates_invalid_selected_returns_none():
    """Caveat: out-of-range `selected` → None (grammar-fail, not fallback)."""
    structure = make_mock_structure()
    proposal = {
        "candidates": [
            {"changes": [{"op": "ADD", "term": "recency"}]},
        ],
        "selected": 7,
    }
    assert apply_proposal(structure, proposal) is None


def test_apply_proposal_old_schema_fallback():
    """Old-style `changes` at top level still works."""
    structure = make_mock_structure()
    proposal = {
        "changes":   [{"op": "ADD", "term": "recency", "reason": "..."}],
        "reasoning": "recency missing",
    }
    new_structure = apply_proposal(structure, proposal)
    assert new_structure is not None
    assert "recency" in [t.name for t in new_structure.terms]
