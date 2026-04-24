"""Delphos baseline: DQN-based utility-structure search.

Reference
---------
Nova, G., Hess, S., van Cranenburgh, S. (2025). "Delphos: A reinforcement
learning framework for assisting discrete choice model specification."
arXiv:2506.06410.

Scope
-----
This baseline runs Delphos against the shared 4-column feature pool the
rest of the suite consumes (``src.baselines.data_adapter``'s
:data:`BUILTIN_FEATURE_NAMES`). The search space is 4 variables (``price,
popularity_rank, log1p_price, price_rank``) times 3 transformations
(``linear / log / box-cox``) = 12 atomic terms. ``box-cox`` is mapped to
a fixed ``power(exponent=0.5)`` transform; no lambda is learned.

What is replaced (the estimator)
--------------------------------
Delphos's ``delphos_interaction`` in the paper delegates to R/Apollo for
the inner MNL estimation. Here the estimator fits a **flat** MNL
(no per-category / per-customer deviations) via
:func:`src.baselines._delphos_inner_loop.fit_weights_flat`. The design
doc justifies the flat-vs-hierarchical choice -- it keeps AIC / BIC
comparable to LASSO-MNL and collapses per-episode cost by ~30x.

Deviations from the paper (documented)
--------------------------------------
* ASC slot (``var == 0``) is encoded in the DQN state but dropped when
  decoding to a :class:`DSLStructure`. The flat MNL does not carry
  alternative-specific constants.
* ``box-cox`` -> fixed ``power(exponent=0.5)`` (no learned lambda).
* ``specific`` taste and categorical covariates are unreachable in this
  action space (Option B only).
* ``rewards.csv`` persistence stripped; the estimator cache lives in
  memory.
* No behavioral sign check (Delphos itself omits Eq. 17 in its own
  ``reward_function``).

Interface: implements :class:`src.baselines.base.Baseline` /
:class:`src.baselines.base.FittedBaseline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import BaselineEventBatch, FittedBaseline
from .data_adapter import BUILTIN_FEATURE_NAMES
from ._delphos_dsl import DSLStructure, DSLTerm, build_structure_features
from ._delphos_dqn import DQNLearner, _REWARD_COLUMNS
from ._delphos_inner_loop import fit_weights_flat, flat_loglik, null_loglik


# ---------------------------------------------------------------------------
# State <-> DSLStructure bridge.
# ---------------------------------------------------------------------------


def _state_to_structure(
    state: List[Tuple[int, str]],
    feature_names: Sequence[str],
) -> DSLStructure:
    """Decode a Delphos state list into a :class:`DSLStructure`.

    ``var == 0`` is the ASC slot and is skipped (no ASCs in the flat MNL).
    For ``var in [1, len(feature_names)]``, ``base_name =
    feature_names[var - 1]``. The transformation maps to:

    * ``linear``  -> ``DSLTerm(base_name)``
    * ``log``     -> ``DSLTerm('log_transform', args=[base_name])``
    * ``box-cox`` -> ``DSLTerm('power', args=[base_name], kwargs={'exponent': 0.5})``

    If the decoded structure is empty, fall back to
    ``DSLStructure([feature_names[0]])`` so the flat-MNL fit is
    well-defined (guards ``score_events`` against zero-column features).
    """
    terms: List[DSLTerm] = []
    for entry in state:
        if len(entry) < 2:
            continue
        var, trans = int(entry[0]), str(entry[1])
        if var == 0:
            continue
        if not (1 <= var <= len(feature_names)):
            continue
        base_name = feature_names[var - 1]
        if trans == "linear":
            terms.append(DSLTerm(name=base_name))
        elif trans == "log":
            terms.append(DSLTerm(name="log_transform", args=[base_name]))
        elif trans == "box-cox":
            terms.append(
                DSLTerm(
                    name="power",
                    args=[base_name],
                    kwargs={"exponent": 0.5},
                )
            )
    if not terms:
        return DSLStructure([feature_names[0]])
    return DSLStructure(terms)


def _features_for_structure(
    structure: DSLStructure,
    batch: BaselineEventBatch,
) -> List[np.ndarray]:
    names = batch.base_feature_names
    return [
        build_structure_features(structure, feats, names)
        for feats in batch.base_features_list
    ]


# ---------------------------------------------------------------------------
# Estimator: fits flat MNL, packages metrics for Delphos's reward function.
# ---------------------------------------------------------------------------


def _estimate_from_state(
    state: List[Tuple[int, str]],
    train: BaselineEventBatch,
    val: Optional[BaselineEventBatch],
    feature_names: Sequence[str],
    sigma: float,
) -> pd.DataFrame:
    """Decode ``state`` to a structure, fit flat MNL, return a single-row DF.

    The DataFrame matches the Delphos reward-function column contract
    (:data:`src.baselines._delphos_dqn._REWARD_COLUMNS`). A failed or
    non-finite fit is reported with ``successfulEstimation=False`` and
    the downstream reward is zero.
    """
    structure = _state_to_structure(state, feature_names)

    n_events = int(train.n_events)
    LL0 = null_loglik(train.base_features_list)
    # No ASCs: constants-only MNL collapses to the uniform baseline.
    LLC = LL0

    row: Dict[str, Any] = {c: np.nan for c in _REWARD_COLUMNS}
    row["successfulEstimation"] = False
    row["LL0"] = LL0
    row["LLC"] = LLC
    row["numParams"] = int(len(structure.terms))

    try:
        feats_list = _features_for_structure(structure, train)
        weights = fit_weights_flat(
            structure,
            feats_list,
            train.chosen_indices,
            sigma=float(sigma),
        )
        ll = flat_loglik(feats_list, train.chosen_indices, weights)
    except Exception:
        return pd.DataFrame([row])

    if not np.isfinite(ll):
        return pd.DataFrame([row])

    num_params = int(len(structure.terms))

    if val is not None and val.n_events > 0:
        try:
            val_feats = _features_for_structure(structure, val)
            ll_out = flat_loglik(val_feats, val.chosen_indices, weights)
        except Exception:
            ll_out = float("nan")
    else:
        ll_out = float("nan")

    aic = 2.0 * num_params - 2.0 * ll
    bic = float(np.log(max(n_events, 1))) * num_params - 2.0 * ll
    if LL0 != 0.0 and np.isfinite(LL0):
        rho2_0 = 1.0 - ll / LL0
        adj_rho2_0 = 1.0 - (ll - num_params) / LL0
    else:
        rho2_0 = float("nan")
        adj_rho2_0 = float("nan")
    if LLC != 0.0 and np.isfinite(LLC):
        rho2_C = 1.0 - ll / LLC
        adj_rho2_C = 1.0 - (ll - num_params) / LLC
    else:
        rho2_C = float("nan")
        adj_rho2_C = float("nan")

    row.update(
        {
            "numParams": num_params,
            "successfulEstimation": True,
            "LL0": LL0,
            "LLC": LLC,
            "LLout": ll_out,
            "rho2_0": rho2_0,
            "adjRho2_0": adj_rho2_0,
            "rho2_C": rho2_C,
            "adjRho2_C": adj_rho2_C,
            "AIC": aic,
            "BIC": bic,
        }
    )
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# FittedBaseline.
# ---------------------------------------------------------------------------


@dataclass
class DelphosFitted:
    """Fitted Delphos model conforming to :class:`FittedBaseline`.

    Attributes
    ----------
    best_structure
        The single DSL structure selected by the candidate tracker.
    best_weights
        Flat-MNL coefficient vector, shape ``(len(best_structure.terms),)``.
    base_feature_names
        Copy of ``train.base_feature_names`` from fit time, used by
        :meth:`score_events` to recompute structure features.
    n_episodes_run
        Number of episodes actually executed (``<= num_episodes``).
    candidates_evaluated
        Size of the final estimation cache (unique specifications seen).
    train_nll, val_nll
        Negative log-likelihood of ``best_structure`` + ``best_weights``
        on the train / val batches. ``val_nll`` is ``nan`` if the
        validation batch was empty at fit time.
    """

    name: str
    best_structure: DSLStructure
    best_weights: np.ndarray
    base_feature_names: List[str]
    n_episodes_run: int
    candidates_evaluated: int
    train_nll: float
    val_nll: float

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Score each event's alternatives via ``struct_feats @ best_weights``."""
        names = batch.base_feature_names
        out: List[np.ndarray] = []
        for feats in batch.base_features_list:
            struct_feats = build_structure_features(self.best_structure, feats, names)
            out.append(np.asarray(struct_feats @ self.best_weights, dtype=np.float64))
        return out

    @property
    def n_params(self) -> int:
        return int(len(self.best_structure.terms))

    @property
    def description(self) -> str:
        return (
            f"Delphos DQN {self.best_structure} "
            f"episodes={self.n_episodes_run} "
            f"evals={self.candidates_evaluated}"
        )


# ---------------------------------------------------------------------------
# Baseline.
# ---------------------------------------------------------------------------


class Delphos:
    """DQN-based utility-specification search over the shared 4-feature pool.

    Parameters
    ----------
    n_episodes
        Number of training episodes. Paper uses 50k; 300 / 1500 are the
        pilot / paper budgets recommended in the design doc.
    feature_names
        Optional override; defaults to
        :data:`src.baselines.data_adapter.BUILTIN_FEATURE_NAMES`.
    hidden_layers
        Hidden layer sizes for the DQN's policy / target MLPs.
    gamma
        Discount factor used both for Q-learning and for the
        ``'exponential'`` reward-distribution mode.
    batch_size
        Experience-replay mini-batch size.
    target_update_freq
        Episodes between target-network hard copies.
    reward_weights
        Delphos reward-metric weights. Default ``{'AIC': 1.0}``.
    reward_distribution_mode
        One of ``'exponential'``, ``'linear'``, ``'uniform'``.
    epsilon_min
        Floor on the epsilon-greedy exploration rate.
    min_percentage
        Fraction of ``n_episodes`` before early stopping is considered.
    early_stop_window
        Rolling-window length for the plateau detector.
    patience
        Number of windows without improvement before stopping.
    sigma
        L2 prior std. on the flat-MNL coefficients.
    seed
        RNG seed.
    """

    name = "Delphos"

    def __init__(
        self,
        n_episodes: int = 300,
        feature_names: Optional[Sequence[str]] = None,
        hidden_layers: Tuple[int, ...] = (128, 64),
        gamma: float = 0.9,
        batch_size: int = 64,
        target_update_freq: int = 10,
        reward_weights: Optional[Dict[str, float]] = None,
        reward_distribution_mode: str = "exponential",
        epsilon_min: float = 0.01,
        min_percentage: float = 0.1,
        early_stop_window: int = 50,
        patience: int = 10,
        sigma: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.n_episodes = int(n_episodes)
        self.feature_names = (
            tuple(feature_names) if feature_names is not None else BUILTIN_FEATURE_NAMES
        )
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.reward_weights = dict(reward_weights) if reward_weights else {"AIC": 1.0}
        self.reward_distribution_mode = str(reward_distribution_mode)
        self.epsilon_min = float(epsilon_min)
        self.min_percentage = float(min_percentage)
        self.early_stop_window = int(early_stop_window)
        self.patience = int(patience)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.learner_: Optional[DQNLearner] = None

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> DelphosFitted:
        """Train the DQN against the estimator and return a fitted baseline."""
        feature_names = list(self.feature_names)
        # Validate against the training batch's feature pool; Delphos
        # references columns by name via ``build_structure_features``.
        missing = [n for n in feature_names if n not in train.base_feature_names]
        if missing:
            raise ValueError(
                "Delphos feature_names not present in train.base_feature_names: "
                f"{missing}. Got base_feature_names={train.base_feature_names}."
            )

        # State space: ASC at slot 0 + one slot per feature name.
        state_space_params = {
            "num_vars": len(feature_names) + 1,
            "transformations": ["linear", "log", "box-cox"],
            "taste": ["generic"],
            "covariates": [],
        }

        def estimator(state: List[Tuple[int, str]]) -> pd.DataFrame:
            return _estimate_from_state(
                state, train, val, feature_names, sigma=self.sigma
            )

        learner = DQNLearner(
            state_space_params=state_space_params,
            num_episodes=self.n_episodes,
            estimate_fn=estimator,
            hidden_layers=self.hidden_layers,
            discount_factor=self.gamma,
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq,
            patience=self.patience,
            early_stop_window=self.early_stop_window,
            min_percentage=self.min_percentage,
            reward_weights=self.reward_weights,
            reward_distribution=self.reward_distribution_mode,
            epsilon_min=self.epsilon_min,
            seed=self.seed,
        )
        self.learner_ = learner

        total_episodes = learner.train()

        # Pick the best candidate. Delphos's tracker stores best per metric.
        metric = learner.metric
        best_info = learner.best_candidates.get(metric, {})
        best_repr = best_info.get("representation")
        if best_repr is None:
            # Fallback: scan the estimation cache for any successful row.
            best_val = np.inf
            for rep, df in learner._estimate_cache.items():
                if df.empty:
                    continue
                if not bool(df["successfulEstimation"].iloc[0]):
                    continue
                aic_val = float(df["AIC"].iloc[0])
                if np.isfinite(aic_val) and aic_val < best_val:
                    best_val = aic_val
                    best_repr = rep
        if best_repr is None:
            best_state: List[Tuple[int, str]] = []
        else:
            best_state = learner.state_manager.decode_string_to_state(best_repr)

        best_structure = _state_to_structure(best_state, feature_names)

        # Re-fit flat weights on train for the final artifact.
        train_feats = _features_for_structure(best_structure, train)
        best_weights = fit_weights_flat(
            best_structure,
            train_feats,
            train.chosen_indices,
            sigma=self.sigma,
        )
        train_ll = flat_loglik(train_feats, train.chosen_indices, best_weights)
        train_nll = float(-train_ll)

        if val is not None and val.n_events > 0:
            val_feats = _features_for_structure(best_structure, val)
            val_ll = flat_loglik(val_feats, val.chosen_indices, best_weights)
            val_nll = float(-val_ll)
        else:
            val_nll = float("nan")

        return DelphosFitted(
            name=self.name,
            best_structure=best_structure,
            best_weights=np.asarray(best_weights, dtype=np.float64),
            base_feature_names=list(train.base_feature_names),
            n_episodes_run=int(total_episodes),
            candidates_evaluated=int(len(learner._estimate_cache)),
            train_nll=train_nll,
            val_nll=val_nll,
        )


__all__ = [
    "Delphos",
    "DelphosFitted",
]
