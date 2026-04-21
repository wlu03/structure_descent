"""
Pluggable accept strategies for the Structure Descent outer loop.

Each strategy decides whether a proposed structure S' replaces the current
structure S given their scores. Swap strategies to run ablation comparisons
between greedy, simulated annealing, threshold accepting, and late acceptance
hill climbing.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AcceptStrategy(ABC):
    """Decides whether a proposed structure replaces the current one."""

    name: str  # human-readable identifier, shown in logs / saved in proposal_detail

    @abstractmethod
    def decide(
        self,
        current_score: float,
        new_score: float,
        iteration: int,
        n_iterations: int,
    ) -> bool:
        """Return True to accept, False to reject."""
        ...

    def on_accept(self, new_score: float, iteration: int) -> None:
        """Hook for strategies with internal state (e.g. late acceptance queue)."""
        pass

    def on_reject(self, new_score: float, iteration: int) -> None:
        """Hook for strategies that update state on rejection too."""
        pass

    def reset(self) -> None:
        """Clear internal state when resuming from a checkpoint without history."""
        pass

    def config(self) -> dict:
        """Return a JSON-serializable dict of the strategy's hyperparameters."""
        return {"name": self.name}


class GreedyAccept(AcceptStrategy):
    """Pure hill climbing: accept only strict improvements."""

    name = "greedy"

    def decide(
        self,
        current_score: float,
        new_score: float,
        iteration: int,
        n_iterations: int,
    ) -> bool:
        return new_score > current_score


class SimulatedAnnealingAccept(AcceptStrategy):
    """
    Metropolis-Hastings accept with a linear cooling schedule.

    Uphill moves (delta > 0) are always accepted deterministically (no RNG
    draw), so identical LLM proposals yield identical accept decisions on
    uphill moves across runs. Downhill moves are accepted with probability
    exp(delta / T_i).
    """

    name = "simulated_annealing"

    def __init__(self, T_start: float = 2.0, T_end: float = 0.1, seed: int = 42):
        self.T_start = T_start
        self.T_end = T_end
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _temperature(self, iteration: int, n_iterations: int) -> float:
        frac = iteration / max(1, n_iterations)
        return self.T_start + (self.T_end - self.T_start) * frac

    def decide(
        self,
        current_score: float,
        new_score: float,
        iteration: int,
        n_iterations: int,
    ) -> bool:
        delta = new_score - current_score
        if delta > 0:
            return True
        T_i = self._temperature(iteration, n_iterations)
        if T_i <= 0:
            return False
        p_accept = float(np.exp(delta / T_i))
        return bool(self._rng.random() < p_accept)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def config(self) -> dict:
        return {
            "name": self.name,
            "T_start": self.T_start,
            "T_end": self.T_end,
        }


class ThresholdAccept(AcceptStrategy):
    """
    Deterministic relative of simulated annealing. Accept iff
    new_score - current_score > -tau_i, where tau linearly decreases from
    tau_start to tau_end over n_iterations.
    """

    name = "threshold_accept"

    def __init__(self, tau_start: float = 5.0, tau_end: float = 0.0):
        self.tau_start = tau_start
        self.tau_end = tau_end

    def _threshold(self, iteration: int, n_iterations: int) -> float:
        frac = iteration / max(1, n_iterations)
        return self.tau_start + (self.tau_end - self.tau_start) * frac

    def decide(
        self,
        current_score: float,
        new_score: float,
        iteration: int,
        n_iterations: int,
    ) -> bool:
        tau_i = self._threshold(iteration, n_iterations)
        return (new_score - current_score) > -tau_i

    def config(self) -> dict:
        return {
            "name": self.name,
            "tau_start": self.tau_start,
            "tau_end": self.tau_end,
        }


class LateAcceptanceHillClimbing(AcceptStrategy):
    """
    Late acceptance hill climbing (Burke & Bykov, 2008).

    Maintains a circular buffer of the last L accepted scores and compares
    new proposals against the *oldest* stored score. This stale comparison
    creates a grace period: proposals that improve the recent trajectory are
    accepted even if they don't beat the global best.
    """

    name = "late_acceptance"

    def __init__(self, history_length: int = 5):
        self.L = history_length
        self._history: list[float] = []

    def decide(
        self,
        current_score: float,
        new_score: float,
        iteration: int,
        n_iterations: int,
    ) -> bool:
        if not self._history:
            # Empty history → fall back to greedy.
            return new_score > current_score
        oldest = self._history[0]
        return new_score > oldest

    def on_accept(self, new_score: float, iteration: int) -> None:
        self._history.append(new_score)
        if len(self._history) > self.L:
            self._history.pop(0)

    def reset(self) -> None:
        self._history = []

    def config(self) -> dict:
        return {
            "name": self.name,
            "history_length": self.L,
        }
