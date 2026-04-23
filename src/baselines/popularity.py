"""Popularity baseline: score alternatives by training-set choice frequency.

Why this baseline exists
------------------------
NLL is interpretable only relative to a baseline. Comparing against
uniform (``log J``) is a trivially easy reference. The more honest
question is: does the model beat "always predict the most popular
alternative"? That is what this baseline provides, and what the
``nll_uplift_vs_popularity = popularity_nll - model_nll`` metric
(computed in :mod:`src.baselines.run_all`) measures — nats of
information gained over a frequency-based predictor that ignores
features entirely.

Data-model choice (what does "popularity" mean here?)
-----------------------------------------------------
The :class:`~src.baselines.base.BaselineEventBatch` does not carry a
global item identifier on its own; alternatives are indexed by their
**position within an event** (``0..J-1``). However, when the batch
originated from the PO-LEU adapter, ``raw_events[e]["choice_asins"]``
provides a global ASIN per alternative. We prefer the richest stable
axis available and fall back cleanly:

1. If ``batch.raw_events`` contains per-event ``choice_asins`` and
   ``chosen_idx``, count **global item (ASIN) purchase frequency**. At
   test time each alternative is scored by how often its ASIN was
   chosen in training (Laplace smoothed).
2. Otherwise, fall back to **(category, within-event position)**
   counts — a per-category position prior — which still captures
   structural regularity (e.g. "category X shoppers disproportionately
   buy the alternative presented in slot 3"). If categories are
   degenerate, this collapses to a pure per-position prior.

Both paths use Laplace smoothing (``count + 1``) so unseen items /
positions never produce ``-inf``. The fitted object records which
granularity it used so the description is honest.

Scoring
-------
For each alternative ``a`` in event ``e`` we return

    score[a] = log(count[a] + alpha)

where ``alpha`` is the Laplace smoothing constant (default ``1.0``).
The shared harness applies ``log_softmax`` on top of these scores, so
the implied choice probability is

    P(a | e) ∝ count[a] + alpha.

Scores are real logits (not log-probabilities). ``log(count + alpha)``
is the correct pre-softmax quantity: after exponentiation and
normalization it yields the smoothed empirical distribution.

Parameter count (n_params for AIC/BIC)
--------------------------------------
We set ``n_params = number of distinct keys with observed counts``.
Argument: AIC/BIC are meant to penalize effective model capacity, and
a popularity table with K distinct entries fits K free parameters
(one log-rate per key). Reporting ``n_params = 1`` would hide the fact
that the per-ASIN variant can have thousands of effective parameters
relative to the per-position variant's ``J``. Honest counts let the
reader see AIC/BIC trade-offs across popularity variants.

This means popularity will typically *lose* on AIC/BIC to feature-based
baselines at similar NLL — that's the correct story: if a parametric
model matches popularity's NLL with 30 coefficients, it is favored.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .base import BaselineEventBatch, FittedBaseline


# ── key extraction ────────────────────────────────────────────────────────────


def _event_has_asins(batch: BaselineEventBatch) -> bool:
    """Return True iff raw_events carries per-event ``choice_asins``.

    We require the key to be present and the value to be a sequence of
    length ``n_alternatives`` on every event. A partial presence is
    treated as absent — mixing ASIN-level and fallback-level keying
    across events would silently bias counts.
    """
    if batch.raw_events is None:
        return False
    expected_J = batch.n_alternatives
    for rec in batch.raw_events:
        asins = rec.get("choice_asins") if isinstance(rec, Mapping) else None
        if asins is None:
            return False
        try:
            if len(asins) != expected_J:
                return False
        except TypeError:
            return False
    return True


# ── fitted object ─────────────────────────────────────────────────────────────


@dataclass
class PopularityFitted:
    """Fitted popularity table wrapped as a :class:`FittedBaseline`.

    Attributes
    ----------
    granularity
        ``"asin"`` when per-ASIN counts were learned;
        ``"category_position"`` when falling back.
    asin_counts
        Populated iff ``granularity == "asin"``. Maps ASIN to chosen
        count in the training batch.
    cat_pos_counts
        Populated iff ``granularity == "category_position"``. Maps
        ``(category, position)`` to chosen count.
    alpha
        Laplace smoothing constant added to every count before taking
        log.
    n_keys
        Number of distinct keys with observed counts (rows of the
        learned table). Drives :attr:`n_params`.
    """

    name: str
    granularity: str
    alpha: float
    n_keys: int
    asin_counts: Dict[str, int] = field(default_factory=dict)
    cat_pos_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        if self.granularity == "asin":
            return self._score_by_asin(batch)
        return self._score_by_cat_pos(batch)

    def _score_by_asin(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        if batch.raw_events is None:
            raise ValueError(
                "PopularityFitted fitted on ASIN-level counts cannot score a "
                "batch without raw_events / choice_asins. Re-fit on a "
                "batch whose raw_events carry choice_asins, or ensure the "
                "evaluation batch was produced by the same adapter."
            )
        out: List[np.ndarray] = []
        for rec in batch.raw_events:
            asins = rec.get("choice_asins") if isinstance(rec, Mapping) else None
            if asins is None or len(asins) != batch.n_alternatives:
                raise ValueError(
                    "PopularityFitted(asin).score_events: event missing "
                    "choice_asins or length mismatch."
                )
            counts = np.asarray(
                [self.asin_counts.get(str(a), 0) for a in asins],
                dtype=np.float64,
            )
            out.append(np.log(counts + self.alpha))
        return out

    def _score_by_cat_pos(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        J = batch.n_alternatives
        for cat in batch.categories:
            counts = np.asarray(
                [self.cat_pos_counts.get((str(cat), j), 0) for j in range(J)],
                dtype=np.float64,
            )
            out.append(np.log(counts + self.alpha))
        return out

    @property
    def n_params(self) -> int:
        return int(self.n_keys)

    @property
    def description(self) -> str:
        return (
            f"Popularity granularity={self.granularity} "
            f"alpha={self.alpha:g} n_keys={self.n_keys}"
        )


# ── fit-time class ────────────────────────────────────────────────────────────


class PopularityBaseline:
    """Count-based popularity predictor (conditional-logit compatible).

    Parameters
    ----------
    alpha
        Laplace smoothing constant. ``1.0`` (default) is the standard
        add-one prior; values in ``(0, 1]`` trade calibration at low
        counts against aggressiveness. A strictly positive alpha is
        required so unseen alternatives never score ``-inf``.
    granularity
        ``"auto"`` (default) uses ASIN-level counts when
        ``raw_events`` has ``choice_asins``, else falls back to
        ``(category, position)``. Override with ``"asin"`` or
        ``"category_position"`` to force a specific mode — fitting
        with ``"asin"`` on a batch lacking ``choice_asins`` raises.
    """

    name = "Popularity"

    def __init__(self, alpha: float = 1.0, granularity: str = "auto"):
        if alpha <= 0.0:
            raise ValueError(
                f"PopularityBaseline requires alpha > 0 for Laplace "
                f"smoothing; got alpha={alpha!r}."
            )
        if granularity not in ("auto", "asin", "category_position"):
            raise ValueError(
                f"Unknown granularity {granularity!r}; expected one of "
                "'auto', 'asin', 'category_position'."
            )
        self.alpha = float(alpha)
        self.granularity = granularity

    def fit(
        self,
        train: BaselineEventBatch,
        val: Optional[BaselineEventBatch] = None,
    ) -> PopularityFitted:
        """Tally training-set chosen-alternative frequencies."""
        if train.n_events == 0:
            raise ValueError("PopularityBaseline.fit received an empty train batch")

        want_asin = self.granularity == "asin" or (
            self.granularity == "auto" and _event_has_asins(train)
        )
        if self.granularity == "asin" and not _event_has_asins(train):
            raise ValueError(
                "PopularityBaseline(granularity='asin') requires every "
                "training record to expose raw_events['choice_asins'] "
                "with length n_alternatives. The batch does not."
            )

        if want_asin:
            asin_counts: Counter[str] = Counter()
            assert train.raw_events is not None  # narrowed by _event_has_asins
            for rec, chosen in zip(train.raw_events, train.chosen_indices):
                asins = rec["choice_asins"]
                asin_counts[str(asins[int(chosen)])] += 1
            return PopularityFitted(
                name=self.name,
                granularity="asin",
                alpha=self.alpha,
                n_keys=len(asin_counts),
                asin_counts=dict(asin_counts),
            )

        cat_pos_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        for cat, chosen in zip(train.categories, train.chosen_indices):
            cat_pos_counts[(str(cat), int(chosen))] += 1
        return PopularityFitted(
            name=self.name,
            granularity="category_position",
            alpha=self.alpha,
            n_keys=len(cat_pos_counts),
            cat_pos_counts=dict(cat_pos_counts),
        )


__all__ = ["PopularityBaseline", "PopularityFitted"]
