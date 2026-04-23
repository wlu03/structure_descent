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
    # raw_events carries the text fields baselines like ST-MLP and the
    # frozen-LLM rankers need (``alt_texts``, ``c_d``, ``order_date``,
    # etc.). Populating it lets those baselines run end-to-end on
    # synthetic data — they still enforce their own shape constraints
    # (e.g. the LLM rankers require J=4 and will error cleanly at J=10).
    raw_events: list = []

    # Synthetic calendar: stamp events on consecutive days so baselines
    # that read order_date (e.g. FewShotICLRanker's timeline) have a
    # coherent per-customer temporal ordering.
    import pandas as _pd

    base_date = _pd.Timestamp("2024-01-01")

    for i in range(n_events):
        feats = rng.normal(0.0, 1.0, size=(n_alts, n_base))
        gumbel_noise = -np.log(-np.log(rng.uniform(size=n_alts) + 1e-12) + 1e-12)
        utilities = feats @ true_weights + gumbel_noise
        chosen = int(np.argmax(utilities))

        base_features_list.append(feats)
        chosen_indices.append(chosen)
        cid = f"cust_{i % n_customers}"
        cat = f"cat_{i % n_categories}"
        customer_ids.append(cid)
        categories.append(cat)
        is_repeat = bool(rng.random() < 0.3)
        price = float(rng.uniform(1.0, 100.0))
        metadata.append({
            "is_repeat": is_repeat,
            "price": price,
            "routine": int(rng.integers(0, 5)),
        })

        # Per-alt text dicts matching the 7-key schema ST-MLP renders.
        # The per-alt fields use *deterministic* functions of (i, j) — not
        # fresh rng draws — so adding these dicts to ``raw_events`` does
        # not perturb the event-loop rng stream. Pre-existing tests
        # (e.g. ``test_bayesian_ard_smoke_learns_signal``) pin the rng
        # sequence that produces ``base_features_list``; drawing
        # additional alt_text samples here would shift all subsequent
        # feats and break the pin.
        alt_texts = [
            {
                "title": f"synth_item_{i:05d}_alt_{j:02d}",
                "category": cat,
                "price": float(10.0 + 5.0 * j),
                "popularity_rank": float((j + 1) / (n_alts + 1)),
                "brand": f"synth_brand_{j % 10}",
                # The chosen alt inherits the event's is_repeat flag; negatives
                # are False (mirrors the adapter's per-alt convention).
                "is_repeat": is_repeat if j == chosen else False,
                "state": "CA",
            }
            for j in range(n_alts)
        ]
        raw_events.append({
            "customer_id": cid,
            "order_date": base_date + _pd.Timedelta(days=i),
            "category": cat,
            "chosen_idx": chosen,
            "alt_texts": alt_texts,
            # A short stationary c_d so LLM rankers have something to
            # prompt on; real runs populate this from the PO-LEU
            # context builder.
            "c_d": f"Customer {cid}; shopping in {cat}.",
        })

    batch = BaselineEventBatch(
        base_features_list=base_features_list,
        base_feature_names=base_names,
        chosen_indices=chosen_indices,
        customer_ids=customer_ids,
        categories=categories,
        metadata=metadata,
        raw_events=raw_events,
    )
    batch.true_weights = true_weights  # attach for baselines that want to peek
    return batch
