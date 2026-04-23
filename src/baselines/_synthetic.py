"""
Synthetic BaselineEventBatch for unit tests and smoke tests.

Generates choice events where alternatives are drawn from a known
multinomial logit model with a random "true" weight vector. A correctly
implemented baseline should beat chance on this data.

Used by:
  - tests/baselines/test_scaffold.py
  - Individual baseline unit tests (each baseline should include its own
    test that fits on a synthetic batch and checks top-1 > chance)
"""

from __future__ import annotations

import numpy as np

from .base import BaselineEventBatch


_DEFAULT_BASE_NAMES = [
    "routine",
    "recency",
    "novelty",
    "popularity",
    "affinity",
    "time_match",
    "price_sensitivity",
    "rating_signal",
    "brand_affinity",
    "price_rank",
    "delivery_speed",
    "co_purchase",
]


def make_synthetic_batch(
    n_events: int = 200,
    n_alts: int = 10,
    n_customers: int = 20,
    n_categories: int = 5,
    seed: int = 0,
    signal_strength: float = 1.5,
    true_weights: np.ndarray | None = None,
) -> BaselineEventBatch:
    """
    Generate a synthetic BaselineEventBatch from a known MNL.

    A random "true" weight vector is drawn; per event, per alternative
    features are drawn iid from N(0, 1); utilities are computed as
    features @ true_weights + gumbel_noise; the argmax is the chosen
    alternative.

    Parameters
    ----------
    n_events : int
    n_alts : int
    n_customers : int
        Number of distinct customer ids to assign round-robin.
    n_categories : int
    seed : int
    signal_strength : float
        Multiplier applied to the drawn weight vector. Higher values yield
        a cleaner signal; use 0 for pure noise.
    true_weights : optional np.ndarray
        Override the randomly drawn weights. Shape must be (n_base,) where
        n_base matches len(_DEFAULT_BASE_NAMES).

    Returns
    -------
    BaselineEventBatch
    """
    rng = np.random.default_rng(seed)
    base_names = list(_DEFAULT_BASE_NAMES)
    n_base = len(base_names)

    if true_weights is None:
        true_weights = rng.normal(0.0, 1.0, size=n_base) * signal_strength
    else:
        true_weights = np.asarray(true_weights, dtype=float)
        if true_weights.shape != (n_base,):
            raise ValueError(
                f"true_weights shape {true_weights.shape} != ({n_base},)"
            )

    base_features_list: list = []
    chosen_indices: list = []
    customer_ids: list = []
    categories: list = []
    metadata: list = []

    for i in range(n_events):
        feats = rng.normal(0.0, 1.0, size=(n_alts, n_base))
        gumbel_noise = -np.log(-np.log(rng.uniform(size=n_alts) + 1e-12) + 1e-12)
        utilities = feats @ true_weights + gumbel_noise
        chosen = int(np.argmax(utilities))

        base_features_list.append(feats)
        chosen_indices.append(chosen)
        customer_ids.append(f"cust_{i % n_customers}")
        categories.append(f"cat_{i % n_categories}")
        metadata.append({
            "is_repeat": bool(rng.random() < 0.3),
            "price": float(rng.uniform(1.0, 100.0)),
            "routine": int(rng.integers(0, 5)),
        })

    batch = BaselineEventBatch(
        base_features_list=base_features_list,
        base_feature_names=base_names,
        chosen_indices=chosen_indices,
        customer_ids=customer_ids,
        categories=categories,
        metadata=metadata,
    )
    batch.true_weights = true_weights  # attach for baselines that want to peek
    return batch
