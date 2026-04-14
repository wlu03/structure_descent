"""
Classical ML baselines for discrete choice prediction.

These are flexible-fit *predictive ceilings*, not faithful discrete choice
models. They replace the blocked RUMBoost slot in the baseline suite with
three off-the-shelf sklearn classifiers from distinct ML families:

  - RandomForestChoice        (bagging, sklearn.ensemble.RandomForestClassifier)
  - GradientBoostingChoice    (boosting, sklearn.ensemble.HistGradientBoostingClassifier)
  - MLPChoice                 (neural, sklearn.neural_network.MLPClassifier)

## Positioning

These are NOT faithful to any specific discrete-choice paper. They are
flexible ML models used to *bound* how much predictive signal is in the
data. The comparison story is:

  "Our LLM-guided interpretable utility search achieves X% of a black-box
   ML ceiling while using Y× fewer parameters and producing
   economically-interpretable coefficients."

In that story, RandomForest / GradientBoosting / MLP are the ceiling.
They should predict at least as well as any structured baseline on any
task with enough data — if they do not, the dataset has no signal and
the entire comparison is suspect.

## Method: binary classification over alternatives

A choice event with n_alts alternatives is flattened into n_alts rows,
one per alternative. The alternative at `chosen_indices[e]` gets `y=1`;
all others get `y=0`. The classifier is fit on the concatenation of all
such rows across the training batch.

At prediction time, each alternative's raw per-alternative features are
fed through `predict_proba`. The score returned to the shared harness is
`log P(y=1 | x_i)`, so the harness's softmax normalizer produces
choice-set probabilities:

    P(alt = i | event e) = p_i / sum_j p_j

where `p_i = P(y=1 | x_i)`. This is the standard "rank by propensity"
approach to turning a binary classifier into a choice model. It has no
utility-maximization story, no invariance to alternative rescaling, and
no interpretable coefficients — that's the point.

## Features

All three baselines consume the **raw base features** from the batch
(shape `[n_alts, n_base_terms]` per event). No feature-pool expansion is
applied here, unlike LASSO-MNL / Bayesian ARD which operate on the
expanded pool. The rationale is that tree-based and neural models learn
their own nonlinear transformations and interactions, so pre-expanding
features is redundant noise for them. This deviation is intentional and
matches how ML ceilings are reported in most choice-modeling papers.

## n_params accounting

The `n_params` property returns a rough effective-parameter count used
for the AIC/BIC columns in the shared report. These counts are model-
dependent and should not be compared directly across families:

  - RandomForest: sum of leaves across trees
  - HistGradientBoosting: sum of leaves across all boosted predictors
  - MLP: total number of weights + biases

The AIC/BIC values are useful for a same-family model-selection lens
(e.g. "did adding trees help?"), not for cross-family model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import BaselineEventBatch, FittedBaseline


def _batch_to_long_format(batch: BaselineEventBatch):
    """
    Flatten a BaselineEventBatch into a long-format (X, y, sample_weight) triple.

    Returns
    -------
    X : np.ndarray, shape (n_events * n_alts, n_base_terms)
        Row e*n_alts + a is the feature vector for alternative a in event e.
    y : np.ndarray, shape (n_events * n_alts,)
        Binary chosen indicator.
    sample_weight : np.ndarray, shape (n_events * n_alts,)
        Per-row weight. Every row gets 1/n_alts so each EVENT contributes
        mass 1.0 regardless of choice-set size. This corrects the raw
        binary-classification view without warping base-rate calibration
        the way class_weight="balanced" does, so NLL/AIC/BIC reported
        through the harness stay honest.
    """
    n_events = batch.n_events
    n_alts = batch.n_alternatives
    if n_events == 0:
        raise ValueError("Cannot flatten an empty BaselineEventBatch.")

    X_parts = []
    y_parts = []
    for feats, chosen in zip(batch.base_features_list, batch.chosen_indices):
        feats_arr = np.asarray(feats, dtype=float)
        if feats_arr.shape[0] != n_alts:
            raise ValueError(
                f"Inconsistent n_alts: event has shape {feats_arr.shape}, "
                f"expected ({n_alts}, n_base_terms)."
            )
        X_parts.append(feats_arr)
        y_row = np.zeros(n_alts, dtype=int)
        y_row[int(chosen)] = 1
        y_parts.append(y_row)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    sample_weight = np.full(X.shape[0], 1.0 / max(n_alts, 1), dtype=float)
    return X, y, sample_weight


def _val_nll(fitted: "ClassicalMLFitted", val: BaselineEventBatch) -> float:
    """Compute validation NLL by re-softmaxing per-event scores (matches the harness)."""
    if val.n_events == 0:
        return float("inf")
    total = 0.0
    from scipy.special import log_softmax as _log_softmax
    scores_list = fitted.score_events(val)
    for scores, chosen in zip(scores_list, val.chosen_indices):
        lp = _log_softmax(np.asarray(scores, dtype=float))
        total -= float(lp[int(chosen)])
    return total / max(val.n_events, 1)


@dataclass
class ClassicalMLFitted:
    """
    Shared FittedBaseline wrapper for all classical ML baselines.

    Stores a fitted sklearn-like classifier exposing `predict_proba`. The
    score_events method feeds each event's per-alternative feature matrix
    directly through `predict_proba` and returns log-probabilities for
    the chosen class.
    """

    name: str
    classifier: object
    n_base_terms: int
    n_alternatives: int
    family: str
    n_params_estimate: int
    description_str: str

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        scores_list: List[np.ndarray] = []
        for feats in batch.base_features_list:
            feats_arr = np.asarray(feats, dtype=float)
            proba = self.classifier.predict_proba(feats_arr)
            # Binary classifier: proba has shape (n_alts, 2), columns [P(y=0), P(y=1)]
            if proba.shape[1] == 2:
                p_chosen = proba[:, 1]
            else:
                # One-class edge case: classifier saw only y=0 at fit time
                p_chosen = np.full(feats_arr.shape[0], 1e-6)
            log_p = np.log(p_chosen + 1e-12)
            scores_list.append(log_p)
        return scores_list

    @property
    def n_params(self) -> int:
        return int(self.n_params_estimate)

    @property
    def description(self) -> str:
        return self.description_str


# ── Shared fit logic ─────────────────────────────────────────────────────────


class _ClassicalMLBase:
    """
    Base class implementing the `Baseline` protocol via a subclass hook.

    Subclasses override `_build_classifier()` (returns an unfit sklearn
    estimator exposing fit / predict_proba) and `_describe(classifier)`
    (returns a one-line summary). The fit method handles flattening,
    fitting, and wrapping in a ClassicalMLFitted.
    """

    name: str = ""
    family: str = ""

    def _build_classifier(self):
        raise NotImplementedError

    def _describe(self, classifier) -> str:
        return f"{self.name} (no description)"

    def _n_params_estimate(self, classifier) -> int:
        return 0

    def _hyperparam_grid(self):
        """Override in subclasses. Return a list of dicts (empty = no tuning)."""
        return []

    def _with_params(self, **overrides):
        """Return a fresh instance with the given hyperparameter overrides."""
        cls = type(self)
        params = {
            k: getattr(self, k)
            for k in cls.__init__.__code__.co_varnames[1:]
            if hasattr(self, k)
        }
        params.update(overrides)
        trial = cls(**params)
        # Tuning is opt-in; a trial instance should not recursively tune.
        trial.tune = False
        return trial

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> ClassicalMLFitted:
        X, y, sw = _batch_to_long_format(train)

        tune = getattr(self, "tune", False)
        grid = self._hyperparam_grid() if tune else []
        # val is used only when tuning is explicitly requested.
        if not grid or val is train or val.n_events == 0:
            classifier = self._build_classifier()
            self._safe_fit(classifier, X, y, sw)
            return ClassicalMLFitted(
                name=self.name,
                classifier=classifier,
                n_base_terms=train.n_base_terms,
                n_alternatives=train.n_alternatives,
                family=self.family,
                n_params_estimate=self._n_params_estimate(classifier),
                description_str=self._describe(classifier),
            )

        best_fitted: Optional[ClassicalMLFitted] = None
        best_val_nll = float("inf")
        for overrides in grid:
            trial = self._with_params(**overrides)
            trial_clf = trial._build_classifier()
            trial._safe_fit(trial_clf, X, y, sw)
            trial_fitted = ClassicalMLFitted(
                name=trial.name,
                classifier=trial_clf,
                n_base_terms=train.n_base_terms,
                n_alternatives=train.n_alternatives,
                family=trial.family,
                n_params_estimate=trial._n_params_estimate(trial_clf),
                description_str=trial._describe(trial_clf),
            )
            nll = _val_nll(trial_fitted, val)
            if nll < best_val_nll:
                best_val_nll = nll
                best_fitted = trial_fitted

        assert best_fitted is not None
        return best_fitted

    def _safe_fit(self, classifier, X, y, sw) -> None:
        """Fit with sample_weight if the estimator supports it, else fall back.

        Pipelines raise ValueError on bare ``sample_weight``; bare classifiers
        that don't support sample_weight raise TypeError. Both fall back to
        the unweighted fit (logged via the family description).
        """
        try:
            classifier.fit(X, y, sample_weight=sw)
        except (TypeError, ValueError):
            classifier.fit(X, y)


# ── 1. RandomForest ──────────────────────────────────────────────────────────


class RandomForestChoice(_ClassicalMLBase):
    """
    Random forest classifier over (event, alternative) pairs.

    Bagging of decision trees. We correct the (n_alts - 1):1 class
    imbalance via ``sample_weight = 1/n_alts`` per row rather than
    ``class_weight="balanced"`` — the latter warps predicted
    probabilities away from the base rate and poisons NLL/AIC/BIC.
    """

    name = "RandomForest"
    family = "bagging+trees"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        random_state: int = 0,
        tune: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tune = tune

    def _build_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )

    def _hyperparam_grid(self):
        return [
            {"n_estimators": 100, "max_depth": 6},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 300, "max_depth": None},
        ]

    def _describe(self, clf) -> str:
        depths = [est.get_depth() for est in clf.estimators_]
        mean_depth = float(np.mean(depths)) if depths else 0.0
        return (
            f"RandomForest n_trees={clf.n_estimators} "
            f"mean_depth={mean_depth:.1f} "
            f"leaves={self._n_params_estimate(clf)}"
        )

    def _n_params_estimate(self, clf) -> int:
        return int(sum(est.get_n_leaves() for est in clf.estimators_))


# ── 2. HistGradientBoosting ──────────────────────────────────────────────────


class GradientBoostingChoice(_ClassicalMLBase):
    """
    Histogram-based gradient boosting classifier.

    Sklearn's HistGradientBoostingClassifier. Class imbalance is corrected
    via ``sample_weight = 1/n_alts`` per row (see RandomForestChoice).
    """

    name = "GradientBoosting"
    family = "boosting+trees"

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: Optional[int] = 6,
        learning_rate: float = 0.1,
        l2_regularization: float = 0.0,
        random_state: int = 0,
        tune: bool = False,
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.tune = tune

    def _build_classifier(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    def _hyperparam_grid(self):
        return [
            {"max_iter": 200, "max_depth": 4, "learning_rate": 0.1},
            {"max_iter": 200, "max_depth": 8, "learning_rate": 0.05},
            {"max_iter": 500, "max_depth": 6, "learning_rate": 0.05},
        ]

    def _describe(self, clf) -> str:
        return (
            f"GradientBoosting max_iter={clf.max_iter} "
            f"max_depth={clf.max_depth} "
            f"lr={clf.learning_rate} "
            f"leaves={self._n_params_estimate(clf)}"
        )

    def _n_params_estimate(self, clf) -> int:
        # Each iteration produces one tree for binary classification. Each
        # tree exposes a `nodes` structured array; leaf nodes are where
        # `is_leaf` is True. We approximate by counting leaves across all
        # fitted predictors.
        total = 0
        if hasattr(clf, "_predictors"):
            for iter_predictors in clf._predictors:
                for tree_predictor in iter_predictors:
                    try:
                        nodes = tree_predictor.nodes
                        total += int(np.sum(nodes["is_leaf"]))
                    except Exception:
                        # Older sklearn versions use a different structure
                        total += 1
        return int(total)


# ── 3. MLP ───────────────────────────────────────────────────────────────────


class MLPChoice(_ClassicalMLBase):
    """
    Multi-layer perceptron classifier (neural network).

    Standard sklearn MLPClassifier wrapped in a Pipeline with
    StandardScaler — neural nets need scaled inputs, and the base DSL
    features have wildly different scales (log-transformed counts vs.
    binary flags vs. unnormalized prices). No explicit class balancing
    is applied: sklearn's MLPClassifier does not accept sample_weight or
    class_weight. The ranking metrics (top-1/top-5/MRR) are invariant to
    monotone rescaling of the decision boundary, so imbalance matters
    less for ranking than for calibrated NLL.
    """

    name = "MLP"
    family = "neural"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (32, 16),
        activation: str = "relu",
        alpha: float = 1e-4,
        max_iter: int = 200,
        random_state: int = 0,
        tune: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.tune = tune

    def _build_classifier(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state,
            )),
        ])

    def _hyperparam_grid(self):
        return [
            {"hidden_layer_sizes": (32,), "alpha": 1e-4},
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
        ]

    def _describe(self, clf) -> str:
        mlp = clf.named_steps["mlp"]
        arch = " x ".join(str(h) for h in self.hidden_layer_sizes)
        return (
            f"MLP arch={arch} "
            f"activation={self.activation} "
            f"n_iter={getattr(mlp, 'n_iter_', '?')} "
            f"params={self._n_params_estimate(clf)}"
        )

    def _n_params_estimate(self, clf) -> int:
        mlp = clf.named_steps["mlp"]
        if not hasattr(mlp, "coefs_"):
            return 0
        total = 0
        for w in mlp.coefs_:
            total += int(np.size(w))
        for b in mlp.intercepts_:
            total += int(np.size(b))
        return total
