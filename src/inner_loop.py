"""
Inner loop: continuous weight fitting for a fixed DSL structure S.

Hierarchical weight decomposition:
  θ_{i,c} = θ_g + θ_c + Δ_i

  θ_g : global weights shared across all users
  θ_c : category-level deviations
  Δ_i : individual-level deviations (regularized toward 0)

MAP objective:
  L(θ) = -Σ log P(a* | s_t)
        + (1/2σ²)||θ_g||²
        + (1/2τ²) Σ_c ||θ_c||²
        + (1/2ν²) Σ_i ||Δ_i||²

Solved with scipy L-BFGS-B.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .dsl import DSLStructure


@dataclass
class HierarchicalWeights:
    theta_g: np.ndarray                   # [n_terms]  global
    theta_c: Dict[str, np.ndarray]        # {category: [n_terms]}  context
    delta_i: Dict[str, np.ndarray]        # {customer_id: [n_terms]}  individual

    def get_weights(self, customer_id: str, category: str) -> np.ndarray:
        """θ_{i,c} = θ_g + θ_c + Δ_i"""
        tc = self.theta_c.get(category, np.zeros_like(self.theta_g))
        di = self.delta_i.get(customer_id, np.zeros_like(self.theta_g))
        return self.theta_g + tc + di


@dataclass
class FitResult:
    structure: DSLStructure
    weights: HierarchicalWeights
    train_nll: float
    posterior_score: float


def fit_weights(
    structure: DSLStructure,
    features_list: List[np.ndarray],   # [n_events] of [n_alts, n_terms]
    chosen_indices: List[int],          # [n_events] index of chosen item
    customer_ids: List[str],            # [n_events]
    categories: List[str],              # [n_events]
    sigma: float = 10.0,
    tau: float = 1.0,
    nu: float = 0.5,
    verbose: bool = False,
) -> HierarchicalWeights:
    """
    Fit hierarchical logit weights for a fixed structure via L-BFGS-B.
    """
    n_terms = len(structure.terms)
    unique_cats = sorted(set(categories))
    unique_custs = sorted(set(customer_ids))
    n_cats = len(unique_cats)
    n_custs = len(unique_custs)
    cat_idx = {c: i for i, c in enumerate(unique_cats)}
    cust_idx = {c: i for i, c in enumerate(unique_custs)}

    # Flat parameter layout:
    # [theta_g | theta_c (n_cats × n_terms) | delta_i (n_custs × n_terms)]
    total_params = n_terms * (1 + n_cats + n_custs)

    def unpack(x: np.ndarray):
        tg = x[:n_terms]
        tc = x[n_terms: n_terms + n_cats * n_terms].reshape(n_cats, n_terms)
        di = x[n_terms + n_cats * n_terms:].reshape(n_custs, n_terms)
        return tg, tc, di

    def objective(x: np.ndarray) -> float:
        tg, tc, di = unpack(x)
        nll = 0.0
        for feats, chosen, cid, cat in zip(features_list, chosen_indices, customer_ids, categories):
            w = tg + tc[cat_idx[cat]] + di[cust_idx[cid]]
            log_probs = log_softmax(feats @ w)
            nll -= log_probs[chosen]

        reg = (
            np.dot(tg, tg) / (2 * sigma ** 2)
            + np.sum(tc ** 2) / (2 * tau ** 2)
            + np.sum(di ** 2) / (2 * nu ** 2)
        )
        return nll + reg

    result = minimize(
        objective,
        x0=np.zeros(total_params),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9, "disp": verbose},
    )

    tg, tc_arr, di_arr = unpack(result.x)

    return HierarchicalWeights(
        theta_g=tg,
        theta_c={c: tc_arr[cat_idx[c]] for c in unique_cats},
        delta_i={c: di_arr[cust_idx[c]] for c in unique_custs},
    )


def compute_posterior_score(
    structure: DSLStructure,
    weights: HierarchicalWeights,
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    customer_ids: List[str],
    categories: List[str],
    sigma: float = 10.0,
    tau: float = 1.0,
    nu: float = 0.5,
) -> float:
    """
    Score(S, θ) = log p(D|θ,S) + log p(θ|S) + log p(S)
    """
    log_likelihood = 0.0
    for feats, chosen, cid, cat in zip(features_list, chosen_indices, customer_ids, categories):
        w = weights.get_weights(cid, cat)
        log_probs = log_softmax(feats @ w)
        log_likelihood += log_probs[chosen]

    log_p_theta = (
        -np.dot(weights.theta_g, weights.theta_g) / (2 * sigma ** 2)
        - sum(np.dot(v, v) for v in weights.theta_c.values()) / (2 * tau ** 2)
        - sum(np.dot(v, v) for v in weights.delta_i.values()) / (2 * nu ** 2)
    )

    return log_likelihood + log_p_theta + structure.log_prior()


def run_inner_loop(
    structure: DSLStructure,
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    customer_ids: List[str],
    categories: List[str],
    **kwargs,
) -> Tuple[HierarchicalWeights, float]:
    """Convenience wrapper: fit weights and return (weights, posterior_score)."""
    weights = fit_weights(structure, features_list, chosen_indices, customer_ids, categories, **kwargs)
    score = compute_posterior_score(structure, weights, features_list, chosen_indices, customer_ids, categories)
    return weights, score


def fit_weights_no_hierarchy(
    structure: DSLStructure,
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    sigma: float = 10.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Ablation: fit a single global weight vector (no θ_c or Δ_i).
    Paper: "No hierarchy — single-level weights ablation showing hierarchy
    captures individual/context variation."

    Returns theta_g: np.ndarray [n_terms]
    """
    from scipy.optimize import minimize
    from scipy.special import log_softmax

    n_terms = len(structure.terms)

    def objective(x: np.ndarray) -> float:
        nll = 0.0
        for feats, chosen in zip(features_list, chosen_indices):
            log_probs = log_softmax(feats @ x)
            nll -= log_probs[chosen]
        reg = np.dot(x, x) / (2 * sigma ** 2)
        return nll + reg

    result = minimize(
        objective,
        x0=np.zeros(n_terms),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9, "disp": verbose},
    )
    return result.x
