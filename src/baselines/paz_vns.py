"""
Paz VNS baseline: Variable Neighborhood Search over DSL utility specifications.

Reference
---------
Paz, A., Arteaga, C., Cobos, C. (2019). "Assisted specification of discrete
choice models." Journal of Choice Modelling 30, 100137. See also the 2021
Transportation Research Part B follow-up by the same group.

Method
------
Paz et al. frame utility specification as a multi-objective combinatorial
optimization problem. The paper optimizes goodness-of-fit indicators rather
than the raw training NLL. We follow that convention with the pair:

    minimize   f(S) = ( BIC(S),  -adjRho2_bar(S) )

where
    BIC(S)          = 2 * train_nll(S) + K(S) * log(n_events)
    adjRho2_bar(S)  = 1 - (train_nll(S) - K(S)) / LL0
    LL0             = sum_i log(1 / |A(s_i)|)
    K(S)            = effective hierarchical parameter count, namely
                      n_terms * (1 + n_contexts + n_individuals_with_delta)

Minimizing BIC captures the paper's preference for parsimonious models;
minimizing -adjRho2_bar captures fit quality relative to the null.

The search follows the classic Mladenovic-Hansen VNS skeleton:

    1. Shake(S, k)         — select a random perturbation from the k-th
                             neighborhood N_k. k indexes WHICH neighborhood
                             to draw from, not how many edits to apply.
    2. LocalImprove(S')    — hill-climb across N1 and N2 using strict Pareto
                             domination on f(S). Capped by local_budget.
    3. Accept              — if Pareto-accepted into the front, reset k=1;
                             otherwise increment k. After k > k_max with no
                             progress, restart from a random front member.

Neighborhoods (DSL-adapted analogues of the paper's variable-inclusion /
random-parameter / distribution neighborhoods):

    N1  — one add / drop / swap of a simple term
    N2  — one add / drop of a compound term
    N3  — a composed perturbation of two atomic edits in sequence (shake only)

Behavioral sign check
---------------------
Paz et al. reject specifications whose coefficients violate economic priors
(e.g. positive price coefficient). We replicate this as a configurable
expected-sign filter. A solution whose global theta_g for any checked
feature has the wrong sign is excluded from the Pareto archive but
logged via ``PazVnsFitted.sign_violations``. The default set covers
``price_sensitivity`` (expected sign: -1) on the Amazon e-commerce domain.

Interface: implements Baseline / FittedBaseline from src/baselines/base.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import log_softmax

from ..dsl import (
    ALL_TERMS,
    BINARY_COMBINATORS,
    DSLStructure,
    DSLTerm,
    LAYER3_COMBINATORS,
    UNARY_COMBINATORS,
    build_structure_features,
)
from ..inner_loop import HierarchicalWeights, fit_weights
from .base import BaselineEventBatch, FittedBaseline


# Default expected signs for the Amazon e-commerce domain.
# Keys are DSL simple-term names; values are ±1.
_DEFAULT_EXPECTED_SIGNS: Dict[str, int] = {
    "price_sensitivity": -1,
}

# Local-search budget cap (maximum neighbors explored per _local_improve call).
_DEFAULT_LOCAL_BUDGET = 40

# Number of composed edits used by the N3 shake.
_N3_SHAKE_EDITS = 2


# -----------------------------------------------------------------------------
# Helpers: structure features, NLL, and effective parameter count
# -----------------------------------------------------------------------------


def _features_for_structure(
    structure: DSLStructure,
    batch: BaselineEventBatch,
) -> List[np.ndarray]:
    """Build the per-event structure feature matrices for one candidate S."""
    names = batch.base_feature_names
    return [
        build_structure_features(structure, base_feats, names)
        for base_feats in batch.base_features_list
    ]


def _nll_from_weights(
    features_list: List[np.ndarray],
    chosen_indices: List[int],
    customer_ids: List[str],
    categories: List[str],
    weights: HierarchicalWeights,
) -> float:
    """Unregularized conditional-logit NLL (sum over events)."""
    total = 0.0
    for feats, chosen, cid, cat in zip(
        features_list, chosen_indices, customer_ids, categories
    ):
        w = weights.get_weights(cid, cat)
        lp = log_softmax(feats @ w)
        total -= float(lp[chosen])
    return total


def _null_loglik(batch: BaselineEventBatch) -> float:
    """LL0 = sum_i log(1/|A(s_i)|), respecting per-event choice-set size."""
    total = 0.0
    for feats in batch.base_features_list:
        n = int(feats.shape[0])
        if n <= 0:
            return float("nan")
        total += math.log(1.0 / n)
    return total


def _effective_n_params(structure: DSLStructure, weights: HierarchicalWeights) -> int:
    """Hierarchical param count matching fit_weights' flat layout."""
    n_terms = len(structure.terms)
    n_cats = len(weights.theta_c)
    n_delta_i = len(weights.delta_i)
    return int(n_terms * (1 + n_cats + n_delta_i))


def _canonical_key(structure: DSLStructure) -> str:
    """Order-independent canonical key for Pareto-archive dedup."""
    parts = sorted(repr(t) for t in structure.terms)
    return "|".join(parts)


def _check_signs(
    structure: DSLStructure,
    weights: HierarchicalWeights,
    expected_signs: Dict[str, int],
) -> List[str]:
    """Return a list of names whose global theta_g has the wrong sign."""
    if not expected_signs:
        return []
    violations: List[str] = []
    for idx, term in enumerate(structure.terms):
        if term.is_compound:
            continue
        expected = expected_signs.get(term.name)
        if expected is None:
            continue
        w = float(weights.theta_g[idx])
        if expected > 0 and w < 0:
            violations.append(term.name)
        elif expected < 0 and w > 0:
            violations.append(term.name)
    return violations


# -----------------------------------------------------------------------------
# Atomic edits: N1 (simple) and N2 (compound)
# -----------------------------------------------------------------------------


def _simple_term_names_in(structure: DSLStructure) -> List[str]:
    return [t.name for t in structure.simple_terms]


def _available_simple_terms(structure: DSLStructure) -> List[str]:
    present = set(_simple_term_names_in(structure))
    return [t for t in ALL_TERMS if t not in present]


def _random_compound_term(rng: random.Random) -> DSLTerm:
    """Draw a compound term by picking a combinator then its arg(s)."""
    combinator = rng.choice(LAYER3_COMBINATORS)
    if combinator in BINARY_COMBINATORS:
        a, b = rng.sample(ALL_TERMS, 2)
        return DSLTerm(name=combinator, args=[a, b])
    a = rng.choice(ALL_TERMS)
    term = DSLTerm(name=combinator, args=[a])
    if combinator == "threshold":
        term.kwargs = {"cutoff": float(rng.choice([0.5, 1.0, 2.0, 3.0]))}
    elif combinator == "power":
        term.kwargs = {"exponent": float(rng.choice([0.5, 2.0, 3.0]))}
    elif combinator == "decay":
        term.kwargs = {"halflife": float(rng.choice([7.0, 30.0, 90.0]))}
    return term


def _n1_edit(structure: DSLStructure, rng: random.Random) -> DSLStructure:
    """Single atomic edit on simple terms: add / drop / swap."""
    simple_present = structure.simple_terms
    available = _available_simple_terms(structure)

    moves: List[str] = []
    if available:
        moves.append("add")
    if simple_present:
        moves.append("drop")
    if simple_present and available:
        moves.append("swap")
    if not moves:
        return structure

    move = rng.choice(moves)
    if move == "add":
        new_name = rng.choice(available)
        return structure.add_term(DSLTerm(name=new_name))
    if move == "drop":
        victim = rng.choice(simple_present)
        return structure.remove_term(victim)
    victim = rng.choice(simple_present)
    new_name = rng.choice(available)
    dropped = structure.remove_term(victim)
    return dropped.add_term(DSLTerm(name=new_name))


def _n2_edit(structure: DSLStructure, rng: random.Random) -> DSLStructure:
    """Single atomic edit on compound terms: add or drop one."""
    compound_present = structure.compound_terms
    moves = ["add"]
    if compound_present:
        moves.append("drop")
    move = rng.choice(moves)
    if move == "add":
        for _ in range(8):
            term = _random_compound_term(rng)
            if term not in structure.terms:
                return structure.add_term(term)
        return structure
    victim = rng.choice(compound_present)
    return structure.remove_term(victim)


def _atomic_edit(structure: DSLStructure, rng: random.Random) -> DSLStructure:
    """Pick between N1 and N2 edits uniformly-ish."""
    if rng.random() < 0.6:
        return _n1_edit(structure, rng)
    return _n2_edit(structure, rng)


def shake(structure: DSLStructure, k: int, rng: random.Random) -> DSLStructure:
    """Paz VNS shake: draw one perturbation from the k-th neighborhood.

    - k=1  -> N1 (single simple-term edit)
    - k=2  -> N2 (single compound-term edit)
    - k>=3 -> N3 (composed perturbation of _N3_SHAKE_EDITS atomic edits)

    Unlike a naive k-repeated edit, the neighborhood index is separate
    from the perturbation magnitude. N3 always applies exactly
    ``_N3_SHAKE_EDITS`` edits regardless of how far k has climbed.
    """
    if len(structure) == 0:
        seed = DSLTerm(name=rng.choice(ALL_TERMS))
        structure = DSLStructure([seed])

    if k <= 1:
        return _n1_edit(structure, rng)
    if k == 2:
        return _n2_edit(structure, rng)
    out = structure
    for _ in range(_N3_SHAKE_EDITS):
        out = _atomic_edit(out, rng)
        if len(out) == 0:
            out = out.add_term(DSLTerm(name=rng.choice(ALL_TERMS)))
    return out


def n1_neighbors(structure: DSLStructure) -> List[DSLStructure]:
    """Deterministic enumeration of the N1 neighborhood."""
    out: List[DSLStructure] = []
    simple_present = list(structure.simple_terms)
    available = _available_simple_terms(structure)

    for new_name in available:
        out.append(structure.add_term(DSLTerm(name=new_name)))
    for victim in simple_present:
        out.append(structure.remove_term(victim))
    for victim in simple_present:
        for new_name in available:
            dropped = structure.remove_term(victim)
            out.append(dropped.add_term(DSLTerm(name=new_name)))
    return out


def n2_neighbors(
    structure: DSLStructure,
    rng: random.Random,
    n_adds: int = 6,
) -> List[DSLStructure]:
    """Sampled N2 neighborhood.

    Drops are enumerated (one per compound term present); adds are drawn
    randomly because the combinator argument space is too large to
    enumerate.
    """
    out: List[DSLStructure] = []
    for victim in list(structure.compound_terms):
        out.append(structure.remove_term(victim))
    tried = set(repr(t) for t in structure.terms)
    attempts = 0
    while len([s for s in out if len(s) > len(structure) - len(structure.compound_terms)]) < n_adds and attempts < 4 * n_adds:
        attempts += 1
        term = _random_compound_term(rng)
        if repr(term) in tried:
            continue
        tried.add(repr(term))
        out.append(structure.add_term(term))
    return out


# -----------------------------------------------------------------------------
# Pareto front over (BIC, -adjRho2_bar)
# -----------------------------------------------------------------------------


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Does objective vector ``a`` Pareto-dominate ``b``? (minimize both)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


@dataclass
class ParetoEntry:
    structure: DSLStructure
    weights: HierarchicalWeights
    train_nll: float
    complexity: int
    n_params: int
    bic: float
    adj_rho2_bar: float
    val_nll: float = float("inf")
    sign_ok: bool = True

    @property
    def objective(self) -> Tuple[float, float]:
        return (self.bic, -self.adj_rho2_bar)


def _update_pareto(front: List[ParetoEntry], entry: ParetoEntry) -> bool:
    """Try to insert ``entry`` into the Pareto front; return True if accepted."""
    obj = entry.objective
    key = _canonical_key(entry.structure)
    for existing in front:
        if _canonical_key(existing.structure) == key:
            return False
        if _dominates(existing.objective, obj):
            return False
    front[:] = [e for e in front if not _dominates(obj, e.objective)]
    front.append(entry)
    return True


# -----------------------------------------------------------------------------
# FittedBaseline
# -----------------------------------------------------------------------------


@dataclass
class PazVnsFitted:
    """Fitted Paz-VNS baseline, conforming to the FittedBaseline protocol."""

    name: str
    structure: DSLStructure
    weights: HierarchicalWeights
    train_nll: float
    val_nll: float
    pareto_front: List[ParetoEntry]
    base_feature_names: List[str]
    n_categories_fit: int = 0
    n_customers_fit: int = 0
    n_evaluations: int = 0
    sign_violations: List[Tuple[str, List[str]]] = field(default_factory=list)

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        names = batch.base_feature_names
        scores: List[np.ndarray] = []
        for feats, cid, cat in zip(
            batch.base_features_list, batch.customer_ids, batch.categories
        ):
            struct_feats = build_structure_features(self.structure, feats, names)
            w = self.weights.get_weights(cid, cat)
            scores.append(struct_feats @ w)
        return scores

    @property
    def n_params(self) -> int:
        n_terms = len(self.structure.terms)
        return int(n_terms * (1 + self.n_categories_fit + self.n_customers_fit))

    @property
    def description(self) -> str:
        return (
            f"Paz-VNS S={self.structure} "
            f"|L|={self.structure.complexity()} "
            f"train_nll={self.train_nll:.3f} "
            f"val_nll={self.val_nll:.3f} "
            f"pareto={len(self.pareto_front)} "
            f"evals={self.n_evaluations} "
            f"signViols={len(self.sign_violations)}"
        )


# -----------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------


class PazVNS:
    """Variable Neighborhood Search over DSL structures (Paz et al. 2019).

    Parameters
    ----------
    k_max : int
        Maximum neighborhood index used by ``shake``.
    max_evaluations : int
        Cap on inner-loop fits. Each candidate structure evaluation counts
        as one.
    seed : int
        RNG seed for shake/local-improvement.
    max_restarts : int
        Restart rounds from the current Pareto front before early
        termination.
    improve_iters : int
        Maximum outer passes per local improvement call.
    local_budget : int
        Hard cap on neighbors explored inside a single local improvement.
    expected_signs : dict, optional
        Mapping {simple_term_name: +1 or -1}. Solutions whose global
        theta_g for any listed term has the wrong sign are excluded from
        the Pareto archive. Defaults to
        ``{"price_sensitivity": -1}``. Pass ``{}`` to disable.
    fit_kwargs : optional dict
        Forwarded to ``inner_loop.fit_weights``.
    """

    name = "Paz-VNS"

    def __init__(
        self,
        k_max: int = 3,
        max_evaluations: int = 150,
        seed: int = 0,
        max_restarts: int = 3,
        improve_iters: int = 6,
        local_budget: int = _DEFAULT_LOCAL_BUDGET,
        expected_signs: Optional[Dict[str, int]] = None,
        fit_kwargs: Optional[dict] = None,
    ):
        self.k_max = int(k_max)
        self.max_evaluations = int(max_evaluations)
        self.seed = int(seed)
        self.max_restarts = int(max_restarts)
        self.improve_iters = int(improve_iters)
        self.local_budget = int(local_budget)
        self.expected_signs = (
            dict(_DEFAULT_EXPECTED_SIGNS) if expected_signs is None
            else dict(expected_signs)
        )
        self.fit_kwargs = dict(fit_kwargs or {})

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _build_evaluator(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
        sign_violations: List[Tuple[str, List[str]]],
    ) -> Callable[[DSLStructure], Optional[ParetoEntry]]:
        cache: dict = {}
        state = {"evals": 0}

        ll0_train = _null_loglik(train)
        n_events_train = train.n_events
        log_n_train = float(np.log(max(n_events_train, 1)))

        def evaluate(structure: DSLStructure) -> Optional[ParetoEntry]:
            if len(structure) == 0:
                return None
            key = _canonical_key(structure)
            if key in cache:
                return cache[key]
            if state["evals"] >= self.max_evaluations:
                return None
            state["evals"] += 1

            train_feats = _features_for_structure(structure, train)
            try:
                weights = fit_weights(
                    structure,
                    train_feats,
                    train.chosen_indices,
                    train.customer_ids,
                    train.categories,
                    **self.fit_kwargs,
                )
            except Exception:
                cache[key] = None
                return None

            train_nll = _nll_from_weights(
                train_feats,
                train.chosen_indices,
                train.customer_ids,
                train.categories,
                weights,
            )
            val_feats = _features_for_structure(structure, val)
            val_nll = _nll_from_weights(
                val_feats,
                val.chosen_indices,
                val.customer_ids,
                val.categories,
                weights,
            )

            n_params = _effective_n_params(structure, weights)
            bic = 2.0 * train_nll + n_params * log_n_train
            # Note: train_nll in this module is the *positive* NLL, so the
            # log-likelihood is -train_nll. adjRho2_bar therefore uses
            # (-train_nll - n_params) / LL0 which is 1 + (train_nll + K)/LL0
            # because LL0 is negative.
            if ll0_train != 0.0 and math.isfinite(ll0_train):
                adj_rho2_bar = 1.0 - ((-train_nll) - n_params) / ll0_train
            else:
                adj_rho2_bar = float("nan")

            violations = _check_signs(structure, weights, self.expected_signs)
            sign_ok = len(violations) == 0
            if not sign_ok:
                sign_violations.append((_canonical_key(structure), violations))

            entry = ParetoEntry(
                structure=structure,
                weights=weights,
                train_nll=float(train_nll),
                complexity=int(structure.complexity()),
                n_params=int(n_params),
                bic=float(bic),
                adj_rho2_bar=float(adj_rho2_bar),
                val_nll=float(val_nll),
                sign_ok=sign_ok,
            )
            cache[key] = entry
            return entry

        evaluate.state = state  # type: ignore[attr-defined]
        return evaluate

    # ------------------------------------------------------------------
    # Local improvement: sweep N1 ∪ N2 with budget cap
    # ------------------------------------------------------------------

    def _local_improve(
        self,
        entry: ParetoEntry,
        evaluate: Callable[[DSLStructure], Optional[ParetoEntry]],
        rng: random.Random,
    ) -> ParetoEntry:
        """Hill climb across N1 and N2 using strict Pareto dominance."""
        current = entry
        budget_used = 0
        for _ in range(self.improve_iters):
            neighbors = n1_neighbors(current.structure) + n2_neighbors(
                current.structure, rng
            )
            rng.shuffle(neighbors)
            improved = False
            for cand in neighbors:
                if budget_used >= self.local_budget:
                    return current
                if evaluate.state["evals"] >= self.max_evaluations:  # type: ignore[attr-defined]
                    return current
                budget_used += 1
                out = evaluate(cand)
                if out is None or not out.sign_ok:
                    continue
                if _dominates(out.objective, current.objective):
                    current = out
                    improved = True
                    break
            if not improved:
                break
        return current

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> PazVnsFitted:
        rng = random.Random(self.seed)
        sign_violations: List[Tuple[str, List[str]]] = []
        evaluate = self._build_evaluator(train, val, sign_violations)

        pareto: List[ParetoEntry] = []

        def try_archive(entry: Optional[ParetoEntry]) -> bool:
            if entry is None or not entry.sign_ok:
                return False
            return _update_pareto(pareto, entry)

        incumbent = evaluate(DSLStructure.initial())
        if incumbent is None:
            incumbent = evaluate(DSLStructure([DSLTerm(rng.choice(ALL_TERMS))]))
        try_archive(incumbent)

        unproductive_restarts = 0

        while evaluate.state["evals"] < self.max_evaluations:  # type: ignore[attr-defined]
            if incumbent is None:
                break
            k = 1
            made_progress = False
            while k <= self.k_max:
                if evaluate.state["evals"] >= self.max_evaluations:  # type: ignore[attr-defined]
                    break
                shaken_struct = shake(incumbent.structure, k, rng)
                shaken_entry = evaluate(shaken_struct)
                if shaken_entry is None or not shaken_entry.sign_ok:
                    k += 1
                    continue
                improved = self._local_improve(shaken_entry, evaluate, rng)
                accepted = try_archive(improved)
                dominates_incumbent = _dominates(
                    improved.objective, incumbent.objective
                )
                if accepted or dominates_incumbent:
                    if improved.sign_ok:
                        incumbent = improved
                    k = 1
                    made_progress = True
                else:
                    k += 1

            if made_progress:
                unproductive_restarts = 0
            else:
                unproductive_restarts += 1
                if unproductive_restarts >= self.max_restarts:
                    break
                if pareto:
                    incumbent = rng.choice(pareto)

        if not pareto:
            raise RuntimeError(
                "Paz-VNS failed to evaluate any sign-valid structure — check "
                "that the training batch contains at least one valid event "
                "and that expected_signs are not over-restrictive."
            )

        best = min(pareto, key=lambda e: (e.val_nll, e.complexity))

        return PazVnsFitted(
            name=self.name,
            structure=best.structure,
            weights=best.weights,
            train_nll=best.train_nll,
            val_nll=best.val_nll,
            pareto_front=list(pareto),
            base_feature_names=list(train.base_feature_names),
            n_categories_fit=len(set(train.categories)),
            n_customers_fit=len(set(train.customer_ids)),
            n_evaluations=int(evaluate.state["evals"]),  # type: ignore[attr-defined]
            sign_violations=list(sign_violations),
        )
