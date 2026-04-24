"""Minimal DSL vendored for the Delphos baseline (Option B).

This module is a private, trimmed-down port of ``old_pipeline/src/dsl.py``
restricted to the 4-feature pool produced by
:mod:`src.baselines.data_adapter` (``price``, ``popularity_rank``,
``log1p_price``, ``price_rank``). The three Delphos transformations map
onto a pair of unary combinators:

* ``linear``   -> ``DSLTerm(base_name)``
* ``log``      -> ``DSLTerm('log_transform', args=[base_name])``
* ``box-cox``  -> ``DSLTerm('power', args=[base_name], kwargs={'exponent': 0.5})``

Binary combinators, Layer-2 Amazon-specific primitives, and the 12-atom
registry have all been dropped -- Delphos's action space cannot reach
them under Option B. See ``docs/llm_baselines/delphos_baseline.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


# ---------------------------------------------------------------------------
# Unary transformations reachable from a Delphos ``(var, trans)`` action.
# ---------------------------------------------------------------------------


def _log_transform(x: np.ndarray) -> np.ndarray:
    """Sign-preserving ``log1p`` transform: ``log1p(|x|) * sign(x)``."""
    t = np.asarray(x, dtype=float)
    return np.log1p(np.abs(t)) * np.sign(t)


def _power(x: np.ndarray, exponent: float = 2.0) -> np.ndarray:
    """Sign-preserving power transform: ``|x|**e * sign(x)``."""
    t = np.asarray(x, dtype=float)
    return np.abs(t) ** float(exponent) * np.sign(t)


# ---------------------------------------------------------------------------
# Term + structure dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class DSLTerm:
    """A single term in a structure.

    Simple:   ``DSLTerm("price")``
    Unary:    ``DSLTerm("log_transform", args=["price"])``
    Power:    ``DSLTerm("power", args=["price"], kwargs={"exponent": 0.5})``
    """

    name: str
    args: List[str] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

    @property
    def is_compound(self) -> bool:
        return len(self.args) > 0

    @property
    def display_name(self) -> str:
        if not self.is_compound:
            return self.name
        args_str = ", ".join(self.args)
        if self.kwargs:
            kw_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
            return f"{self.name}({args_str}, {kw_str})"
        return f"{self.name}({args_str})"

    @property
    def key(self) -> str:
        return self.display_name

    def __repr__(self) -> str:
        return self.display_name

    def __eq__(self, other) -> bool:
        if isinstance(other, DSLTerm):
            return self.key == other.key
        return False

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass
class DSLStructure:
    """An ordered list of DSL terms (simple or unary-compound)."""

    terms: List[Union[str, DSLTerm]]

    def __post_init__(self):
        normalized: List[DSLTerm] = []
        for t in self.terms:
            if isinstance(t, str):
                normalized.append(DSLTerm(name=t))
            elif isinstance(t, DSLTerm):
                normalized.append(t)
            else:
                normalized.append(DSLTerm(name=str(t)))
        self.terms = normalized

    @property
    def term_names(self) -> List[str]:
        return [t.display_name for t in self.terms]

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        if not self.terms:
            return "S = (empty)"
        return "S = " + " + ".join(t.display_name for t in self.terms)


# ---------------------------------------------------------------------------
# Compound-feature dispatch.
# ---------------------------------------------------------------------------


def _compute_compound_feature(
    term: DSLTerm, base_features: dict[str, np.ndarray]
) -> np.ndarray:
    """Compute a compound term's column from base feature columns."""
    if term.name == "log_transform":
        if len(term.args) != 1:
            raise ValueError(
                f"log_transform requires 1 arg, got {len(term.args)}: {term.args}"
            )
        a = base_features.get(term.args[0])
        if a is None:
            raise KeyError(f"unknown base term: {term.args[0]!r}")
        return _log_transform(a)
    if term.name == "power":
        if len(term.args) != 1:
            raise ValueError(
                f"power requires 1 arg, got {len(term.args)}: {term.args}"
            )
        a = base_features.get(term.args[0])
        if a is None:
            raise KeyError(f"unknown base term: {term.args[0]!r}")
        exponent = float(term.kwargs.get("exponent", 2.0))
        return _power(a, exponent)
    raise ValueError(f"Unknown combinator for Option B Delphos: {term.name}")


def build_structure_features(
    structure: DSLStructure,
    full_base_features: np.ndarray,
    all_term_names: list[str],
) -> np.ndarray:
    """Assemble the ``(n_alts, n_structure_terms)`` feature matrix.

    ``full_base_features`` is the ``(n_alts, n_base_terms)`` matrix
    carried by :class:`BaselineEventBatch`; ``all_term_names`` names its
    columns in order.
    """
    term_to_idx = {t: i for i, t in enumerate(all_term_names)}
    n_alts = int(full_base_features.shape[0])
    columns: list[np.ndarray] = []

    for term in structure.terms:
        if not term.is_compound:
            idx = term_to_idx.get(term.name)
            if idx is not None:
                columns.append(full_base_features[:, idx].astype(float, copy=False))
            else:
                columns.append(np.zeros(n_alts, dtype=float))
        else:
            base_dict = {
                name: full_base_features[:, i].astype(float, copy=False)
                for name, i in term_to_idx.items()
            }
            columns.append(_compute_compound_feature(term, base_dict))

    if not columns:
        return np.zeros((n_alts, 0), dtype=float)
    return np.column_stack(columns)


__all__ = [
    "DSLTerm",
    "DSLStructure",
    "build_structure_features",
]
