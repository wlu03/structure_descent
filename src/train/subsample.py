"""
customer subsampling via leverage scores.
picks a diverse subset of customers so we dont have to train on everyone.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Tuple


def build_customer_profiles(df, n_pca_components=20):
    """builds a behavior vector per customer from their purchase history.
    returns a dataframe + standardized numpy matrix."""

    grouped = df.groupby("customer_id")

    # category distribution -- pivot to get counts per category then normalize
    cat_counts = df.pivot_table(
        index="customer_id", columns="category",
        values="asin", aggfunc="count", fill_value=0
    )
    cat_fracs = cat_counts.div(cat_counts.sum(axis=1), axis=0)

    # pca on category fracs -- clip components so we dont blow up
    n_components = min(n_pca_components, cat_fracs.shape[1] - 1, cat_fracs.shape[0] - 1)
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components)
    cat_pca = pca.fit_transform(cat_fracs.values)
    pca_cols = [f"cat_pca_{i}" for i in range(n_components)]

    # scalar features per customer
    repeat_rate = grouped.apply(lambda g: (g["routine"] > 0).mean()).rename("repeat_rate")
    mean_recency = grouped.apply(
        lambda g: (1.0 / (1.0 + g["recency_days"])).mean()
    ).rename("mean_recency")
    novelty_rate = grouped.apply(lambda g: (g["novelty"] == 1).mean()).rename("novelty_rate")
    purchase_freq = grouped.size().apply(lambda x: np.log1p(x)).rename("purchase_freq")

    # shannon entropy of category distribution
    cat_fracs_arr = cat_fracs.values
    cat_fracs_safe = np.where(cat_fracs_arr > 0, cat_fracs_arr, 1.0)
    entropy = -(cat_fracs_arr * np.log(cat_fracs_safe)).sum(axis=1)
    cat_entropy = pd.Series(entropy, index=cat_fracs.index, name="cat_entropy")

    scalar_features = pd.concat(
        [repeat_rate, mean_recency, novelty_rate, purchase_freq, cat_entropy], axis=1
    )
    scalar_features = scalar_features.loc[cat_fracs.index]

    pca_df = pd.DataFrame(cat_pca, index=cat_fracs.index, columns=pca_cols)
    profiles_df = pd.concat([pca_df, scalar_features], axis=1)

    # standardize so leverage scores make sense
    matrix = profiles_df.values.astype(np.float64)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds < 1e-12] = 1.0
    matrix = (matrix - means) / stds

    return profiles_df, matrix


def compute_leverage_scores(X):
    """h_i = ||U[i,:]||^2 from thin svd. filters out tiny singular values."""

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    mask = S > 1e-6
    U_eff = U[:, mask]
    leverage = (U_eff ** 2).sum(axis=1)
    return leverage


def subsample_customers(df, n_customers=500, n_pca_components=20,
                        min_per_category_cluster=2, seed=42):
    """picks a diverse subset using leverage scores + kmeans stratification.
    returns (selected_ids, importance_weights)."""

    rng = np.random.default_rng(seed)
    profiles_df, profile_matrix = build_customer_profiles(df, n_pca_components)
    customer_ids = profiles_df.index.values
    n_total = len(customer_ids)
    n_customers = min(n_customers, n_total)

    leverage = compute_leverage_scores(profile_matrix)
    leverage = np.clip(leverage, 1e-10, None)

    # cluster customers so we can guarantee coverage of different behavior types
    n_clusters = max(n_customers // 5, 2)
    n_clusters = min(n_clusters, n_total)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(profile_matrix)

    # grab top leverage customers from each cluster first
    selected_set = set()
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        member_leverage = leverage[members]
        top_indices = members[np.argsort(-member_leverage)]
        n_pick = min(min_per_category_cluster, len(members))
        for idx in top_indices[:n_pick]:
            selected_set.add(idx)

    # fill the rest by sampling proportional to leverage
    remaining_budget = n_customers - len(selected_set)
    if remaining_budget > 0:
        candidates = np.array([i for i in range(n_total) if i not in selected_set])
        if len(candidates) > 0:
            cand_leverage = leverage[candidates]
            probs = cand_leverage / cand_leverage.sum()
            n_fill = min(remaining_budget, len(candidates))
            chosen = rng.choice(candidates, size=n_fill, replace=False, p=probs)
            selected_set.update(chosen)

    selected_indices = np.array(sorted(selected_set))
    selected_ids = customer_ids[selected_indices]

    # importance weights -- inverse probability, normalized so sum = n_total
    sel_leverage = leverage[selected_indices]
    probs_effective = sel_leverage / leverage.sum()
    raw_weights = 1.0 / (n_customers * probs_effective)
    weights = raw_weights * (n_total / raw_weights.sum())

    return selected_ids, weights


def random_subsample_customers(df, n_customers=500, seed=42):
    """picks a uniformly random subset of customers with ω=1 weights.
    for dev iteration and representative held-out evaluation — unlike the
    leverage-score path, the resulting subset is population-representative,
    so per-event metrics like top-1 / NLL / ECE don't need re-weighting.
    returns (selected_ids, importance_weights)."""

    rng = np.random.default_rng(seed)
    customer_ids = np.array(sorted(df["customer_id"].unique()))
    n_total = len(customer_ids)
    n_pick = min(int(n_customers), n_total)
    if n_pick <= 0:
        raise ValueError(f"n_customers must be positive; got {n_customers}.")

    selected_ids = rng.choice(customer_ids, size=n_pick, replace=False)
    selected_ids = np.array(sorted(selected_ids))
    # ω=1: a random sample is already representative, no inverse-probability
    # correction needed. Matches §9.1 default when subsampling is off.
    weights = np.ones(n_pick, dtype=np.float64)
    return selected_ids, weights


def apply_subsample(df, selected_ids, weights):
    """Filter df to selected customers and return per-event importance weights.

    Returns:
        filtered_df: rows of df belonging to the selected customers, with index
            reset so that filtered_df.iloc[i] aligns with event_weights[i].
        event_weights: np.ndarray of shape (len(filtered_df),) with dtype float;
            event_weights[i] is the importance weight of filtered_df.iloc[i]'s
            customer.
    """
    customer_weight_map = dict(zip(selected_ids, weights))
    filtered_df = df[df["customer_id"].isin(set(selected_ids))].copy().reset_index(drop=True)
    event_weights = filtered_df["customer_id"].map(customer_weight_map).to_numpy(dtype=float)
    return filtered_df, event_weights
