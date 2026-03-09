"""
DSL feature functions for the Amazon e-commerce domain.

Layer 1 — Universal Behavioral Primitives (appear in all domains):
  routine, recency, novelty, popularity, affinity, time_match

Layer 2 — Amazon-Specific Feature Functions:
  price_sensitivity, rating_signal, brand_affinity,
  price_rank, delivery_speed, co_purchase

Layer 3 — Combinators (domain-agnostic):
  interaction, split_by, threshold, log_transform, decay

A DSLStructure is a list of active term names. The inner loop reads the
feature function specified by S and fits weights θ to each active term.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


def routine(purchase_count: np.ndarray) -> np.ndarray:
    """How many times the customer previously bought this item."""
    return purchase_count.astype(float)


def recency(days_since_last: np.ndarray) -> np.ndarray:
    """Inverse time since last purchase — higher means more recent."""
    return 1.0 / (1.0 + np.asarray(days_since_last, dtype=float))


def novelty(is_new: np.ndarray) -> np.ndarray:
    """1 if item never purchased by this customer before, else 0."""
    return np.asarray(is_new, dtype=float)


def popularity(purchase_counts: np.ndarray) -> np.ndarray:
    """Log-scaled aggregate purchase frequency across all customers."""
    return np.log1p(np.asarray(purchase_counts, dtype=float))


def affinity(category_purchase_counts: np.ndarray) -> np.ndarray:
    """Customer's historical preference for this item's category."""
    return np.log1p(np.asarray(category_purchase_counts, dtype=float))


def time_match(cat_hour_counts: np.ndarray) -> np.ndarray:
    """Whether this category is typically purchased at this time of day."""
    return np.asarray(cat_hour_counts, dtype=float)


def price_sensitivity(item_price: np.ndarray, category_avg_price: np.ndarray) -> np.ndarray:
    """Negative price ratio relative to category average (cheaper = higher utility)."""
    ratio = np.asarray(item_price, dtype=float) / (np.asarray(category_avg_price, dtype=float) + 1e-8)
    return -(ratio - 1.0)


def rating_signal(rating: np.ndarray, review_count: np.ndarray) -> np.ndarray:
    """Star rating weighted by log(review_count) for credibility."""
    return np.asarray(rating, dtype=float) * np.log1p(np.asarray(review_count, dtype=float))


def brand_affinity(brand_purchase_counts: np.ndarray) -> np.ndarray:
    """Customer's log-scaled preference for this brand based on history."""
    return np.log1p(np.asarray(brand_purchase_counts, dtype=float))


def price_rank(item_price: np.ndarray, session_prices: np.ndarray) -> np.ndarray:
    """
    Fraction of session items this item is cheaper than.
    item_price: [n_alts], session_prices: [n_alts] (prices in current session)
    Returns 1 - rank_fraction so that cheaper items score higher.
    """
    item_price = np.asarray(item_price, dtype=float)
    session_prices = np.asarray(session_prices, dtype=float)
    rank_frac = np.mean(session_prices < item_price)
    return 1.0 - rank_frac * np.ones_like(item_price)


def delivery_speed(is_prime: np.ndarray) -> np.ndarray:
    """1 if Prime/fast shipping available, 0 otherwise."""
    return np.asarray(is_prime, dtype=float)


def co_purchase(co_purchase_freq: np.ndarray) -> np.ndarray:
    """Log-scaled frequency of co-purchase with recent items."""
    return np.log1p(np.asarray(co_purchase_freq, dtype=float))


def interaction(term_a: np.ndarray, term_b: np.ndarray) -> np.ndarray:
    """Element-wise product of two terms (interaction effect)."""
    return np.asarray(term_a, dtype=float) * np.asarray(term_b, dtype=float)


def split_by(term: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Term value when condition is True, 0 otherwise (separate weight fitted)."""
    return np.asarray(term, dtype=float) * np.asarray(condition, dtype=float)


def threshold(term: np.ndarray, cutoff: float) -> np.ndarray:
    """Binary indicator: 1 if term > cutoff."""
    return (np.asarray(term, dtype=float) > cutoff).astype(float)


def log_transform(term: np.ndarray) -> np.ndarray:
    """Signed log scaling: log1p(|x|) * sign(x)."""
    t = np.asarray(term, dtype=float)
    return np.log1p(np.abs(t)) * np.sign(t)


def decay(term: np.ndarray, time_delta: np.ndarray, halflife: float = 30.0) -> np.ndarray:
    """Exponential decay of term over time: term * exp(-ln2 * Δt / halflife)."""
    return np.asarray(term, dtype=float) * np.exp(-np.log(2) * np.asarray(time_delta, dtype=float) / halflife)


LAYER1_PRIMITIVES = ["routine", "recency", "novelty", "popularity", "affinity", "time_match"]
LAYER2_AMAZON = [
    "price_sensitivity", "rating_signal", "brand_affinity",
    "price_rank", "delivery_speed", "co_purchase",
]
LAYER3_COMBINATORS = ["interaction", "split_by", "threshold", "log_transform", "decay"]

ALL_TERMS = LAYER1_PRIMITIVES + LAYER2_AMAZON
ALL_DSL = ALL_TERMS + LAYER3_COMBINATORS


@dataclass
class DSLStructure:
    """A model structure S — an ordered list of active DSL term names."""

    terms: List[str]

    def __post_init__(self):
        self.terms = list(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return "S = " + " + ".join(self.terms) if self.terms else "S = (empty)"

    def complexity(self) -> int:
        """L(S): number of terms."""
        return len(self.terms)

    def log_prior(self) -> float:
        """log p(S) = -L(S) * ln(2)  (complexity prior: p(S) ∝ 2^{-L(S)})."""
        return -self.complexity() * np.log(2)

    def add_term(self, term: str) -> "DSLStructure":
        if term in self.terms:
            return self
        return DSLStructure(self.terms + [term])

    def remove_term(self, term: str) -> "DSLStructure":
        return DSLStructure([t for t in self.terms if t != term])

    def to_dict(self) -> dict:
        return {"terms": self.terms}

    @classmethod
    def from_dict(cls, d: dict) -> "DSLStructure":
        return cls(d["terms"])

    @classmethod
    def initial(cls) -> "DSLStructure":
        """Starting structure: simplest behaviorally-motivated baseline."""
        return cls(["routine", "affinity"])
