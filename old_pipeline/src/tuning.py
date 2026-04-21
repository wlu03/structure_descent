"""
Hyperparameter tuning for the inner loop's Gaussian-prior std devs.

Grid-searches (sigma, tau, nu) on a fixed seed structure by fitting on a
training split and scoring unweighted held-out log-likelihood on a val split.
The triple with the highest held-out log-likelihood is returned.

Scoring is pure data fit: no parameter prior, no structure prior. These are
not what we are tuning for — we want the triple that best predicts held-out
data, period.
"""

import time
import numpy as np
from scipy.special import log_softmax

from .inner_loop import fit_weights


def val_log_likelihood(weights, val_data) -> float:
    """Unweighted held-out log-likelihood under fitted hierarchical weights.

    Unseen customers/categories fall back to zeros inside
    `HierarchicalWeights.get_weights`, which is the correct behavior.
    """
    features_list = val_data["features_list"]
    chosen_indices = val_data["chosen_indices"]
    customer_ids = val_data["customer_ids"]
    categories = val_data["categories"]

    total = 0.0
    for feats, chosen, cid, cat in zip(
        features_list, chosen_indices, customer_ids, categories
    ):
        w = weights.get_weights(cid, cat)
        total += log_softmax(feats @ w)[chosen]
    return float(total)


def tune_hyperparameters(
    seed_structure,
    train_data,
    val_data,
    grid_sigma=(5.0, 10.0, 20.0),
    grid_tau=(0.5, 1.0, 2.0),
    grid_nu=(0.25, 0.5, 1.0, 2.0),
    verbose=True,
):
    """Grid-search (sigma, tau, nu) by fitting on train, scoring on val.

    Args:
        seed_structure:  DSLStructure to hold fixed across the grid.
        train_data:      dict with features_list, chosen_indices, customer_ids,
                         categories, event_weights (event_weights may be None).
        val_data:        same shape as train_data, minus event_weights
                         (val is always scored uniformly).
        grid_sigma/tau/nu: iterables of candidate values.
        verbose:         print per-grid-point progress.

    Returns:
        best:    {"sigma": float, "tau": float, "nu": float, "val_log_lik": float}
        results: list of per-grid-point dicts with
                 sigma/tau/nu/val_log_lik/fit_time.
    """
    train_features = train_data["features_list"]
    train_chosen = train_data["chosen_indices"]
    train_cids = train_data["customer_ids"]
    train_cats = train_data["categories"]
    train_ew = train_data.get("event_weights")

    results = []
    best = None

    for sigma in grid_sigma:
        for tau in grid_tau:
            for nu in grid_nu:
                t0 = time.time()
                weights = fit_weights(
                    seed_structure,
                    train_features,
                    train_chosen,
                    train_cids,
                    train_cats,
                    sigma=sigma,
                    tau=tau,
                    nu=nu,
                    event_weights=train_ew,
                )
                val_ll = val_log_likelihood(weights, val_data)
                fit_time = time.time() - t0

                row = {
                    "sigma": float(sigma),
                    "tau": float(tau),
                    "nu": float(nu),
                    "val_log_lik": float(val_ll),
                    "fit_time": float(fit_time),
                }
                results.append(row)

                if verbose:
                    print(
                        f"  sigma={sigma:>5.2f} tau={tau:>5.2f} nu={nu:>5.2f}"
                        f"  val_ll={val_ll:+.2f}  ({fit_time:.1f}s)"
                    )

                if best is None or val_ll > best["val_log_lik"]:
                    best = {
                        "sigma": float(sigma),
                        "tau": float(tau),
                        "nu": float(nu),
                        "val_log_lik": float(val_ll),
                    }

    if verbose:
        print(
            f"\nBest: sigma={best['sigma']} tau={best['tau']} nu={best['nu']}"
            f"  val_ll={best['val_log_lik']:+.2f}"
        )
    return best, results
