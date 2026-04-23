"""
Expanded feature pool for regression-style baselines (LASSO-MNL, Bayesian ARD).

Every regression baseline consumes the SAME expanded pool so the comparison
is apples-to-apples. Any baseline that needs more or fewer terms should
document the deviation explicitly.

Expansion (applied per choice event, per alternative row):
  1. Identity                  — the 12 DSL base features
  2. Signed log1p              — log1p(|x|) * sign(x)            [per feature]
  3. Signed square             — sign(x) * x**2                  [per feature]
  4. Pairwise interactions     — x_i * x_j for i < j             [optional]

The expansion is deterministic and depends only on the base feature matrix,
so two baselines that set include_interactions=True will see identical
feature matrices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaselineEventBatch


def build_expanded_pool(
    base_feats: np.ndarray,
    base_names: List[str],
    include_interactions: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Expand a single [n_alts x n_base] base-feature matrix into the canonical
    expanded pool.

    Parameters
    ----------
    base_feats : np.ndarray
        Shape (n_alts, n_base). One row per alternative.
    base_names : list of str
        Column names for base_feats.
    include_interactions : bool
        If True, append all pairwise products x_i * x_j for i < j.

    Returns
    -------
    expanded : np.ndarray
        Shape (n_alts, n_expanded).
    names : list of str
        Expanded feature names, parallel to columns of `expanded`.
    """
    if base_feats.ndim != 2:
        raise ValueError(f"base_feats must be 2D, got shape {base_feats.shape}")

    n_alts, n_base = base_feats.shape
    if len(base_names) != n_base:
        raise ValueError(
            f"base_names length {len(base_names)} != n_base {n_base}"
        )

    columns: List[np.ndarray] = []
    names: List[str] = []

    # 1. Identity
    for i, nm in enumerate(base_names):
        columns.append(base_feats[:, i])
        names.append(nm)

    # 2. Signed log1p
    for i, nm in enumerate(base_names):
        x = base_feats[:, i]
        columns.append(np.log1p(np.abs(x)) * np.sign(x))
        names.append(f"log1p_{nm}")

    # 3. Signed square
    for i, nm in enumerate(base_names):
        x = base_feats[:, i]
        columns.append(np.sign(x) * x * x)
        names.append(f"sq_{nm}")

    # 4. Pairwise interactions
    if include_interactions:
        for i in range(n_base):
            for j in range(i + 1, n_base):
                columns.append(base_feats[:, i] * base_feats[:, j])
                names.append(f"{base_names[i]}_x_{base_names[j]}")

    return np.column_stack(columns), names


def expand_batch(
    batch: BaselineEventBatch,
    include_interactions: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Apply build_expanded_pool to every event in a batch.

    Returns
    -------
    expanded_list : list of np.ndarray
        One [n_alts x n_expanded] matrix per event.
    names : list of str
        Expanded feature names (shared across all events).
    """
    expanded_list: List[np.ndarray] = []
    names: List[str] = []
    for base in batch.base_features_list:
        exp, nm = build_expanded_pool(base, batch.base_feature_names, include_interactions)
        if not names:
            names = nm
        expanded_list.append(exp)
    return expanded_list, names


def expanded_pool_size(
    n_base: int,
    include_interactions: bool = True,
) -> int:
    """Analytical size of the expanded pool given n_base features."""
    size = 3 * n_base  # identity + log1p + square
    if include_interactions:
        size += n_base * (n_base - 1) // 2
    return size
