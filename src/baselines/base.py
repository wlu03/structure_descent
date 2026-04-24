"""
Shared protocols and dataclasses for the baseline suite.

Every baseline consumes BaselineEventBatch instances and produces an object
conforming to the FittedBaseline protocol. The harness in evaluate.py then
converts that into a BaselineReport with standardized metrics.

Why a score_events interface (instead of a weight_fn):
  Different baselines produce different kinds of fitted artifacts:
    - LASSO-MNL / Bayesian ARD → flat coefficient vector over expanded pool
    - RUMBoost                 → gradient-boosted trees over raw attributes
    - Paz VNS / DUET / SD      → DSLStructure + HierarchicalWeights
    - Delphos                  → RL policy over action sequences → structure
  A score_events(batch) -> List[np.ndarray[n_alts]] callable is the only
  representation all of them can share without leaking internal structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class BaselineEventBatch:
    """
    Shared input format for every baseline.

    All list fields are parallel and have length n_events. base_features_list
    matches the 12-column (or whatever cardinality) DSL base-feature layout:
    one [n_alts x n_base_terms] matrix per choice event.

    Fields
    ------
    base_features_list : list of np.ndarray
        Per-event base feature matrices, shape (n_alts, n_base_terms).
    base_feature_names : list of str
        Column names for base_features_list (matches DSL base term names).
    chosen_indices : list of int
        Index of the chosen alternative per event, in [0, n_alts).
    customer_ids : list of str
        Customer id per event (needed for hierarchical baselines and
        breakdowns).
    categories : list of str
        Product category per event (needed for hierarchical baselines and
        breakdowns).
    metadata : list of dict
        Per-event metadata. Conventional keys: 'is_repeat' (bool),
        'price' (float), 'routine' (int). Used by breakdowns; baselines
        may read additional keys they need.
    raw_events : optional list of dict
        The original choice-event dicts from data_prep.build_choice_sets,
        carried through for baselines that need raw attributes (RUMBoost,
        frequency baselines). None if not available.
    """

    base_features_list: List[np.ndarray]
    base_feature_names: List[str]
    chosen_indices: List[int]
    customer_ids: List[str]
    categories: List[str]
    metadata: List[dict] = field(default_factory=list)
    raw_events: Optional[List[dict]] = None

    def __post_init__(self):
        n = len(self.base_features_list)
        if not (len(self.chosen_indices) == len(self.customer_ids) == len(self.categories) == n):
            raise ValueError(
                "BaselineEventBatch parallel lists must all have length n_events. "
                f"Got base={n} chosen={len(self.chosen_indices)} "
                f"cids={len(self.customer_ids)} cats={len(self.categories)}"
            )
        if self.metadata and len(self.metadata) != n:
            raise ValueError(
                f"metadata length {len(self.metadata)} != n_events {n}"
            )
        if not self.metadata:
            self.metadata = [{} for _ in range(n)]

    @property
    def n_events(self) -> int:
        return len(self.base_features_list)

    @property
    def n_base_terms(self) -> int:
        return len(self.base_feature_names)

    @property
    def n_alternatives(self) -> int:
        if not self.base_features_list:
            return 0
        return int(self.base_features_list[0].shape[0])

    def subset(self, indices) -> "BaselineEventBatch":
        """Return a new batch containing only the events at the given indices."""
        idx = list(indices)
        return BaselineEventBatch(
            base_features_list=[self.base_features_list[i] for i in idx],
            base_feature_names=self.base_feature_names,
            chosen_indices=[self.chosen_indices[i] for i in idx],
            customer_ids=[self.customer_ids[i] for i in idx],
            categories=[self.categories[i] for i in idx],
            metadata=[self.metadata[i] for i in idx],
            raw_events=[self.raw_events[i] for i in idx] if self.raw_events else None,
        )


@dataclass
class BaselineReport:
    """
    Standardized output of evaluate_baseline.

    metrics keys (always present):
        top1, top5, mrr, test_nll, aic, bic, n_events

    Additional per-event / per-customer fields (populated iff the
    baseline completes evaluation successfully; see
    ``docs/paper_evaluation_additions.md`` for the contract):

    per_event_nll : list[float]
        Parallel to the test batch: ``-log_softmax(logits)[chosen_idx]``
        for each event. ``mean(per_event_nll) == metrics['test_nll']``
        up to float error.
    per_event_topk_correct : list[bool]
        Per-event top-1 correctness flag (``argmax(logits) == chosen``,
        lowest-index tie-break matching :func:`topk_accuracy`).
    per_customer_nll : dict[str, dict]
        Groups the above by ``batch.customer_ids``. Each value is
        ``{"nll": float, "n_events": int, "top1": float}``.
    """

    name: str
    n_params: int
    metrics: Dict[str, float]
    per_category: Optional["object"] = None            # pandas DataFrame
    per_repeat_novel: Optional["object"] = None        # pandas DataFrame
    sign_check: Optional[Dict[str, float]] = None
    fit_time_seconds: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)
    per_event_nll: List[float] = field(default_factory=list)
    per_event_topk_correct: List[bool] = field(default_factory=list)
    per_customer_nll: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"[{self.name}] "
            f"top1={m['top1']:.1%}  "
            f"top5={m['top5']:.1%}  "
            f"MRR={m['mrr']:.4f}  "
            f"NLL={m['test_nll']:.4f}  "
            f"AIC={m['aic']:.1f}  "
            f"BIC={m['bic']:.1f}  "
            f"params={self.n_params}  "
            f"n={m['n_events']}"
        )


@runtime_checkable
class FittedBaseline(Protocol):
    """
    Interface every baseline must expose after fitting.

    Attributes
    ----------
    name : str
        Short identifier (e.g. "LASSO-MNL").

    Methods
    -------
    score_events(batch) -> list of np.ndarray
        For each event in batch, return a 1-D array of length n_alts giving
        the utility score for each alternative. The harness converts these
        into top-1 / top-5 / MRR / NLL.

    n_params : int
        Number of *effective* parameters (for AIC/BIC). For sparse methods,
        this should be the count of non-zero coefficients.

    description : str
        Human-readable one-line summary of the fitted model.
    """

    name: str

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]: ...

    @property
    def n_params(self) -> int: ...

    @property
    def description(self) -> str: ...


class Baseline(Protocol):
    """
    Fit-time protocol. Each baseline module exports a class implementing this.

    Usage:
        baseline = LassoMnl()
        fitted = baseline.fit(train_batch, val_batch)
        report = evaluate_baseline(fitted, test_batch)
    """

    name: str

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> FittedBaseline: ...
