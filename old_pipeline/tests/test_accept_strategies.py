"""
Unit tests for pluggable accept strategies.

These tests deliberately avoid importing the LLM / data pipeline — each
strategy is exercised directly against hand-crafted score sequences.
"""

import math

import numpy as np
import pytest

from old_pipeline.src.accept_strategies import (
    AcceptStrategy,
    GreedyAccept,
    SimulatedAnnealingAccept,
    ThresholdAccept,
    LateAcceptanceHillClimbing,
)


# ── GreedyAccept ──────────────────────────────────────────────────────────────

class TestGreedyAccept:
    def test_accepts_uphill(self):
        g = GreedyAccept()
        assert g.decide(current_score=0.0, new_score=1.0,
                        iteration=0, n_iterations=10) is True

    def test_rejects_downhill(self):
        g = GreedyAccept()
        assert g.decide(current_score=1.0, new_score=0.5,
                        iteration=0, n_iterations=10) is False

    def test_rejects_equal(self):
        g = GreedyAccept()
        assert g.decide(current_score=1.0, new_score=1.0,
                        iteration=0, n_iterations=10) is False

    def test_config(self):
        assert GreedyAccept().config() == {"name": "greedy"}


# ── SimulatedAnnealingAccept ──────────────────────────────────────────────────

class TestSimulatedAnnealingAccept:
    def test_always_accepts_uphill(self):
        sa = SimulatedAnnealingAccept(T_start=0.5, T_end=0.1, seed=0)
        for it in range(10):
            assert sa.decide(current_score=0.0, new_score=0.1,
                             iteration=it, n_iterations=10) is True

    def test_uphill_does_not_consume_rng(self):
        """Uphill moves must be deterministic regardless of RNG state."""
        sa1 = SimulatedAnnealingAccept(T_start=1.0, T_end=0.0, seed=123)
        sa2 = SimulatedAnnealingAccept(T_start=1.0, T_end=0.0, seed=123)
        # Only sa2 sees a series of uphill moves first.
        for _ in range(50):
            sa2.decide(current_score=0.0, new_score=1.0,
                       iteration=0, n_iterations=10)
        # RNG state should be unchanged → same downhill decision as sa1.
        d1 = sa1.decide(current_score=1.0, new_score=0.5,
                        iteration=0, n_iterations=10)
        d2 = sa2.decide(current_score=1.0, new_score=0.5,
                        iteration=0, n_iterations=10)
        assert d1 == d2

    def test_downhill_acceptance_rate(self):
        """
        delta = -0.5, T = 1.0 → p_accept = exp(-0.5) ≈ 0.6065.
        Empirically accept rate should match within a few percentage points
        over 10,000 draws.
        """
        sa = SimulatedAnnealingAccept(T_start=1.0, T_end=1.0, seed=7)
        accepts = 0
        trials = 10_000
        for _ in range(trials):
            if sa.decide(current_score=1.0, new_score=0.5,
                         iteration=0, n_iterations=10):
                accepts += 1
        expected = math.exp(-0.5)
        observed = accepts / trials
        assert abs(observed - expected) < 0.02, (observed, expected)

    def test_final_temperature(self):
        sa = SimulatedAnnealingAccept(T_start=2.0, T_end=0.1, seed=0)
        T_final = sa._temperature(iteration=10, n_iterations=10)
        assert T_final == pytest.approx(0.1, abs=1e-9)

    def test_config(self):
        sa = SimulatedAnnealingAccept(T_start=3.0, T_end=0.5)
        assert sa.config() == {"name": "simulated_annealing",
                               "T_start": 3.0, "T_end": 0.5}


# ── ThresholdAccept ───────────────────────────────────────────────────────────

class TestThresholdAccept:
    def test_early_iteration_tolerates_downhill(self):
        ta = ThresholdAccept(tau_start=2.0, tau_end=0.0)
        # iter 0, tau = 2.0 → accept delta = -1.0 (within threshold)
        assert ta.decide(current_score=5.0, new_score=4.0,
                         iteration=0, n_iterations=10) is True

    def test_early_iteration_rejects_large_downhill(self):
        ta = ThresholdAccept(tau_start=2.0, tau_end=0.0)
        # iter 0, tau = 2.0 → reject delta = -3.0 (beyond threshold)
        assert ta.decide(current_score=5.0, new_score=2.0,
                         iteration=0, n_iterations=10) is False

    def test_final_iteration_is_greedy(self):
        ta = ThresholdAccept(tau_start=2.0, tau_end=0.0)
        # iter = n_iterations → tau = 0
        assert ta.decide(current_score=1.0, new_score=1.1,
                         iteration=10, n_iterations=10) is True
        assert ta.decide(current_score=1.0, new_score=0.9,
                         iteration=10, n_iterations=10) is False

    def test_config(self):
        ta = ThresholdAccept(tau_start=5.0, tau_end=0.0)
        assert ta.config() == {"name": "threshold_accept",
                               "tau_start": 5.0, "tau_end": 0.0}


# ── LateAcceptanceHillClimbing ────────────────────────────────────────────────

class TestLateAcceptanceHillClimbing:
    def test_empty_history_is_greedy(self):
        lahc = LateAcceptanceHillClimbing(history_length=3)
        assert lahc.decide(current_score=0.0, new_score=1.0,
                           iteration=0, n_iterations=10) is True
        assert lahc.decide(current_score=1.0, new_score=0.5,
                           iteration=0, n_iterations=10) is False

    def test_compares_against_oldest(self):
        lahc = LateAcceptanceHillClimbing(history_length=3)
        # Fill history with [1.0, 2.0, 3.0]
        lahc.on_accept(1.0, iteration=0)
        lahc.on_accept(2.0, iteration=1)
        lahc.on_accept(3.0, iteration=2)
        assert lahc._history == [1.0, 2.0, 3.0]

        # Oldest = 1.0. 1.5 beats oldest → accept. 0.5 does not → reject.
        # current_score is irrelevant for LAHC's decision.
        assert lahc.decide(current_score=3.0, new_score=1.5,
                           iteration=3, n_iterations=10) is True
        assert lahc.decide(current_score=3.0, new_score=0.5,
                           iteration=3, n_iterations=10) is False

    def test_circular_buffer_drops_oldest(self):
        lahc = LateAcceptanceHillClimbing(history_length=3)
        lahc.on_accept(1.0, iteration=0)
        lahc.on_accept(2.0, iteration=1)
        lahc.on_accept(3.0, iteration=2)
        lahc.on_accept(4.0, iteration=3)  # overflow → drops 1.0
        assert lahc._history == [2.0, 3.0, 4.0]
        lahc.on_accept(5.0, iteration=4)  # drops 2.0
        assert lahc._history == [3.0, 4.0, 5.0]

    def test_reject_does_not_modify_history(self):
        lahc = LateAcceptanceHillClimbing(history_length=3)
        lahc.on_accept(1.0, iteration=0)
        lahc.on_reject(999.0, iteration=1)
        assert lahc._history == [1.0]

    def test_reset_clears_history(self):
        lahc = LateAcceptanceHillClimbing(history_length=3)
        lahc.on_accept(1.0, iteration=0)
        lahc.on_accept(2.0, iteration=1)
        lahc.reset()
        assert lahc._history == []
        # After reset it's greedy again.
        assert lahc.decide(current_score=0.0, new_score=1.0,
                           iteration=0, n_iterations=10) is True

    def test_config(self):
        lahc = LateAcceptanceHillClimbing(history_length=7)
        assert lahc.config() == {"name": "late_acceptance", "history_length": 7}
