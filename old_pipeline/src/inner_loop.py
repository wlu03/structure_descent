"""
inner loop -- fits hierarchical weights for a fixed dsl structure.

weight decomposition: theta_{i,c} = theta_g + theta_c + delta_i
  theta_g: global weights shared across all users
  theta_c: category-level deviations
  delta_i: individual deviations (regularized toward 0)

map objective solved with scipy l-bfgs-b.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .dsl import DSLStructure


@dataclass
class HierarchicalWeights:
    theta_g: np.ndarray                   # [n_terms] global
    theta_c: Dict[str, np.ndarray]        # {category: [n_terms]} context
    delta_i: Dict[str, np.ndarray]        # {customer_id: [n_terms]} individual

    def get_weights(self, customer_id, category):
        tc = self.theta_c.get(category, np.zeros_like(self.theta_g))
        di = self.delta_i.get(customer_id, np.zeros_like(self.theta_g))
        return self.theta_g + tc + di


@dataclass
class FitResult:
    structure: DSLStructure
    weights: HierarchicalWeights
    train_nll: float
    posterior_score: float


def fit_weights(structure, features_list, chosen_indices, customer_ids, categories,
                sigma=10.0, tau=1.0, nu=0.5, verbose=False, event_weights=None):
    """fit hierarchical logit weights via l-bfgs-b."""

    n_terms = len(structure.terms)
    unique_cats = sorted(set(categories))
    unique_custs = sorted(set(customer_ids))
    n_cats = len(unique_cats)
    n_custs = len(unique_custs)
    cat_idx = {c: i for i, c in enumerate(unique_cats)}
    cust_idx = {c: i for i, c in enumerate(unique_custs)}

    # flat param layout: [theta_g | theta_c (n_cats x n_terms) | delta_i (n_custs x n_terms)]
    total_params = n_terms * (1 + n_cats + n_custs)

    def unpack(x):
        tg = x[:n_terms]
        tc = x[n_terms: n_terms + n_cats * n_terms].reshape(n_cats, n_terms)
        di = x[n_terms + n_cats * n_terms:].reshape(n_custs, n_terms)
        return tg, tc, di

    ew = event_weights if event_weights is not None else np.ones(len(features_list))

    def objective(x):
        tg, tc, di = unpack(x)
        nll = 0.0
        for idx, (feats, chosen, cid, cat) in enumerate(zip(
                features_list, chosen_indices, customer_ids, categories)):
            w = tg + tc[cat_idx[cat]] + di[cust_idx[cid]]
            log_probs = log_softmax(feats @ w)
            nll -= ew[idx] * log_probs[chosen]

        reg = (np.dot(tg, tg) / (2 * sigma ** 2)
               + np.sum(tc ** 2) / (2 * tau ** 2)
               + np.sum(di ** 2) / (2 * nu ** 2))
        return nll + reg

    result = minimize(objective, x0=np.zeros(total_params), method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-9, "disp": verbose})

    tg, tc_arr, di_arr = unpack(result.x)

    return HierarchicalWeights(
        theta_g=tg,
        theta_c={c: tc_arr[cat_idx[c]] for c in unique_cats},
        delta_i={c: di_arr[cust_idx[c]] for c in unique_custs},
    )


def compute_posterior_score(structure, weights, features_list, chosen_indices,
                            customer_ids, categories, sigma=10.0, tau=1.0, nu=0.5,
                            event_weights=None):
    """score(S, theta) = log p(D|theta,S) + log p(theta|S) + log p(S)"""

    ew = event_weights if event_weights is not None else np.ones(len(features_list))
    log_likelihood = 0.0
    for idx, (feats, chosen, cid, cat) in enumerate(zip(
            features_list, chosen_indices, customer_ids, categories)):
        w = weights.get_weights(cid, cat)
        log_probs = log_softmax(feats @ w)
        log_likelihood += ew[idx] * log_probs[chosen]

    log_p_theta = (
        -np.dot(weights.theta_g, weights.theta_g) / (2 * sigma ** 2)
        - sum(np.dot(v, v) for v in weights.theta_c.values()) / (2 * tau ** 2)
        - sum(np.dot(v, v) for v in weights.delta_i.values()) / (2 * nu ** 2)
    )
    return log_likelihood + log_p_theta + structure.log_prior()


def run_inner_loop(structure, features_list, chosen_indices, customer_ids, categories,
                   event_weights=None, **kwargs):
    """convenience wrapper -- fit weights and return (weights, posterior_score)."""

    weights = fit_weights(structure, features_list, chosen_indices,
                          customer_ids, categories, event_weights=event_weights, **kwargs)
    score = compute_posterior_score(structure, weights, features_list, chosen_indices,
                                   customer_ids, categories, event_weights=event_weights, **kwargs)
    return weights, score


def fit_weights_no_hierarchy(structure, features_list, chosen_indices,
                             sigma=10.0, verbose=False, event_weights=None):
    """ablation -- single global weight vector, no theta_c or delta_i."""

    n_terms = len(structure.terms)
    ew = event_weights if event_weights is not None else np.ones(len(features_list))

    def objective(x):
        nll = 0.0
        for idx, (feats, chosen) in enumerate(zip(features_list, chosen_indices)):
            log_probs = log_softmax(feats @ x)
            nll -= ew[idx] * log_probs[chosen]
        reg = np.dot(x, x) / (2 * sigma ** 2)
        return nll + reg

    result = minimize(objective, x0=np.zeros(n_terms), method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-9, "disp": verbose})
    return result.x
