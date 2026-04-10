"""
DSL feature functions for the Amazon e-commerce domain.

Layer 1 — Universal Behavioral Primitives (appear in all domains):
  routine, recency, novelty, popularity, affinity, time_match

Layer 2 — Amazon-Specific Feature Functions:
  price_sensitivity, rating_signal, brand_affinity,
  price_rank, delivery_speed, co_purchase

Layer 3 — Combinators (domain-agnostic):
  interaction, split_by, threshold, log_transform, decay,
  ratio, power, difference

A DSLStructure contains both simple terms and compound terms.
Simple terms reference a base feature column.
Compound terms apply a combinator to one or two base terms.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ── Layer 1/2 feature functions (unchanged) ──────────────────────────────────

def routine(purchase_count: np.ndarray) -> np.ndarray:
    return purchase_count.astype(float)

def recency(days_since_last: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(days_since_last, dtype=float))

def novelty(is_new: np.ndarray) -> np.ndarray:
    return np.asarray(is_new, dtype=float)

def popularity(purchase_counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(purchase_counts, dtype=float))

def affinity(category_purchase_counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(category_purchase_counts, dtype=float))

def time_match(cat_hour_counts: np.ndarray) -> np.ndarray:
    return np.asarray(cat_hour_counts, dtype=float)

def price_sensitivity(item_price: np.ndarray, category_avg_price: np.ndarray) -> np.ndarray:
    ratio = np.asarray(item_price, dtype=float) / (np.asarray(category_avg_price, dtype=float) + 1e-8)
    return -(ratio - 1.0)

def rating_signal(rating: np.ndarray, review_count: np.ndarray) -> np.ndarray:
    return np.asarray(rating, dtype=float) * np.log1p(np.asarray(review_count, dtype=float))

def brand_affinity(brand_purchase_counts: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(brand_purchase_counts, dtype=float))

def price_rank(item_price: np.ndarray, session_prices: np.ndarray) -> np.ndarray:
    item_price = np.asarray(item_price, dtype=float)
    session_prices = np.asarray(session_prices, dtype=float)
    rank_frac = np.mean(session_prices < item_price)
    return 1.0 - rank_frac * np.ones_like(item_price)

def delivery_speed(is_prime: np.ndarray) -> np.ndarray:
    return np.asarray(is_prime, dtype=float)

def co_purchase(co_purchase_freq: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(co_purchase_freq, dtype=float))


# ── Layer 3 combinator functions ─────────────────────────────────────────────

def interaction(term_a: np.ndarray, term_b: np.ndarray) -> np.ndarray:
    return np.asarray(term_a, dtype=float) * np.asarray(term_b, dtype=float)

def split_by(term: np.ndarray, condition: np.ndarray) -> np.ndarray:
    return np.asarray(term, dtype=float) * np.asarray(condition, dtype=float)

def threshold(term: np.ndarray, cutoff: float) -> np.ndarray:
    return (np.asarray(term, dtype=float) > cutoff).astype(float)

def log_transform(term: np.ndarray) -> np.ndarray:
    t = np.asarray(term, dtype=float)
    return np.log1p(np.abs(t)) * np.sign(t)

def decay(term: np.ndarray, time_delta: np.ndarray, halflife: float = 30.0) -> np.ndarray:
    return np.asarray(term, dtype=float) * np.exp(-np.log(2) * np.asarray(time_delta, dtype=float) / halflife)

def ratio(term_a: np.ndarray, term_b: np.ndarray) -> np.ndarray:
    """Ratio of two terms: a / (b + epsilon) for numerical stability."""
    return np.asarray(term_a, dtype=float) / (np.asarray(term_b, dtype=float) + 1e-8)

def power(term: np.ndarray, exponent: float = 2.0) -> np.ndarray:
    """Power transform: |x|^n * sign(x) for signed scaling."""
    t = np.asarray(term, dtype=float)
    return np.abs(t) ** exponent * np.sign(t)

def difference(term_a: np.ndarray, term_b: np.ndarray) -> np.ndarray:
    """Explicit difference: a - b, fitted with a single weight."""
    return np.asarray(term_a, dtype=float) - np.asarray(term_b, dtype=float)


# ── Term registries ──────────────────────────────────────────────────────────

LAYER1_PRIMITIVES = ["routine", "recency", "novelty", "popularity", "affinity", "time_match"]
LAYER2_AMAZON = [
    "price_sensitivity", "rating_signal", "brand_affinity",
    "price_rank", "delivery_speed", "co_purchase",
]
LAYER3_COMBINATORS = ["interaction", "split_by", "threshold", "log_transform", "decay", "ratio", "power", "difference"]

ALL_TERMS = LAYER1_PRIMITIVES + LAYER2_AMAZON
ALL_DSL = ALL_TERMS + LAYER3_COMBINATORS

# Combinators that take 2 base terms as arguments
BINARY_COMBINATORS = {"interaction", "split_by", "ratio", "difference"}
# Combinators that take 1 base term
UNARY_COMBINATORS = {"log_transform", "threshold", "decay", "power"}


# ── Term representation ──────────────────────────────────────────────────────

@dataclass
class DSLTerm:
    """A single term in a structure — either simple or compound.

    Simple:   DSLTerm("routine")
    Compound: DSLTerm("interaction", args=["routine", "recency"])
              DSLTerm("log_transform", args=["popularity"])
              DSLTerm("threshold", args=["routine"], kwargs={"cutoff": 3.0})
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
        """Unique key for dedup and lookup."""
        return self.display_name

    def to_dict(self) -> dict:
        d = {"name": self.name}
        if self.args:
            d["args"] = self.args
        if self.kwargs:
            d["kwargs"] = self.kwargs
        return d

    @classmethod
    def from_dict(cls, d: Union[dict, str]) -> "DSLTerm":
        if isinstance(d, str):
            return cls(name=d)
        return cls(
            name=d["name"],
            args=d.get("args", []),
            kwargs=d.get("kwargs", {}),
        )

    @classmethod
    def parse(cls, s: str) -> "DSLTerm":
        """Parse a string like 'interaction(routine, recency)' or 'popularity'."""
        import re
        s = s.strip()
        match = re.match(r'^(\w+)\((.+)\)$', s)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = [a.strip() for a in args_str.split(",")]
            # Separate kwargs (e.g., cutoff=3.0)
            positional = []
            kw = {}
            for a in args:
                if "=" in a:
                    k, v = a.split("=", 1)
                    try:
                        kw[k.strip()] = float(v.strip())
                    except ValueError:
                        kw[k.strip()] = v.strip()
                else:
                    positional.append(a)
            return cls(name=name, args=positional, kwargs=kw)
        return cls(name=s)

    def __repr__(self) -> str:
        return self.display_name

    def __eq__(self, other) -> bool:
        if isinstance(other, DSLTerm):
            return self.key == other.key
        return False

    def __hash__(self) -> int:
        return hash(self.key)


# ── Structure representation ─────────────────────────────────────────────────

@dataclass
class DSLStructure:
    """A model structure S — an ordered list of DSL terms (simple or compound)."""

    terms: List[Union[str, DSLTerm]]

    def __post_init__(self):
        # Normalize: convert any plain strings to DSLTerm objects
        normalized = []
        for t in self.terms:
            if isinstance(t, str):
                normalized.append(DSLTerm(name=t))
            elif isinstance(t, DSLTerm):
                normalized.append(t)
            elif isinstance(t, dict):
                normalized.append(DSLTerm.from_dict(t))
            else:
                normalized.append(DSLTerm(name=str(t)))
        self.terms = normalized

    @property
    def term_names(self) -> List[str]:
        """Simple term names only (for backward compat with inner loop)."""
        return [t.name if not t.is_compound else t.display_name for t in self.terms]

    @property
    def simple_terms(self) -> List[DSLTerm]:
        return [t for t in self.terms if not t.is_compound]

    @property
    def compound_terms(self) -> List[DSLTerm]:
        return [t for t in self.terms if t.is_compound]

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        if not self.terms:
            return "S = (empty)"
        return "S = " + " + ".join(t.display_name for t in self.terms)

    def complexity(self) -> int:
        """L(S): simple terms count 1, compound terms count 1 + n_args."""
        total = 0
        for t in self.terms:
            if t.is_compound:
                total += 1 + len(t.args)
            else:
                total += 1
        return total

    def log_prior(self) -> float:
        """log p(S) = -L(S) * ln(2)"""
        return -self.complexity() * np.log(2)

    def add_term(self, term: Union[str, DSLTerm]) -> "DSLStructure":
        if isinstance(term, str):
            term = DSLTerm(name=term)
        if term in self.terms:
            return self
        return DSLStructure(self.terms + [term])

    def remove_term(self, term: Union[str, DSLTerm]) -> "DSLStructure":
        if isinstance(term, str):
            # Try matching by display_name or name
            return DSLStructure([t for t in self.terms if t.name != term and t.display_name != term])
        return DSLStructure([t for t in self.terms if t != term])

    def to_dict(self) -> dict:
        return {"terms": [t.to_dict() for t in self.terms]}

    @classmethod
    def from_dict(cls, d: dict) -> "DSLStructure":
        terms = d.get("terms", [])
        parsed = []
        for t in terms:
            if isinstance(t, str):
                parsed.append(DSLTerm(name=t))
            elif isinstance(t, dict):
                parsed.append(DSLTerm.from_dict(t))
            else:
                parsed.append(DSLTerm(name=str(t)))
        return cls(parsed)

    @classmethod
    def initial(cls) -> "DSLStructure":
        """Starting structure: simplest behaviorally-motivated baseline."""
        return cls(["routine", "affinity"])

    def get_base_terms_needed(self) -> set[str]:
        """All base (Layer 1/2) term names needed to compute this structure's features."""
        needed = set()
        for t in self.terms:
            if t.is_compound:
                for arg in t.args:
                    if arg in ALL_TERMS:
                        needed.add(arg)
            else:
                if t.name in ALL_TERMS:
                    needed.add(t.name)
        return needed


# ── Compound feature computation ─────────────────────────────────────────────

def compute_compound_feature(term: DSLTerm, base_features: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute a compound term's feature values from base feature columns.

    Args:
        term: a compound DSLTerm (e.g., DSLTerm("interaction", args=["routine", "recency"]))
        base_features: dict mapping base term names to 1D arrays of shape [n_alts]

    Returns:
        1D array of shape [n_alts]
    """
    if term.name == "interaction":
        if len(term.args) != 2:
            raise ValueError(f"interaction requires 2 args, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        b = base_features.get(term.args[1], np.zeros(1))
        return interaction(a, b)

    elif term.name == "split_by":
        if len(term.args) != 2:
            raise ValueError(f"split_by requires 2 args, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        b = base_features.get(term.args[1], np.zeros(1))
        return split_by(a, b)

    elif term.name == "log_transform":
        if len(term.args) != 1:
            raise ValueError(f"log_transform requires 1 arg, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        return log_transform(a)

    elif term.name == "threshold":
        if len(term.args) < 1:
            raise ValueError(f"threshold requires at least 1 arg, got {len(term.args)}")
        a = base_features.get(term.args[0], np.zeros(1))
        cutoff = term.kwargs.get("cutoff", 1.0)
        return threshold(a, float(cutoff))

    elif term.name == "decay":
        if len(term.args) < 1:
            raise ValueError(f"decay requires at least 1 arg, got {len(term.args)}")
        a = base_features.get(term.args[0], np.zeros(1))
        # Use recency_days as the time delta if available
        time_delta = base_features.get("recency", np.zeros_like(a))
        halflife = term.kwargs.get("halflife", 30.0)
        return decay(a, time_delta, float(halflife))

    elif term.name == "ratio":
        if len(term.args) != 2:
            raise ValueError(f"ratio requires 2 args, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        b = base_features.get(term.args[1], np.zeros(1))
        return ratio(a, b)

    elif term.name == "power":
        if len(term.args) != 1:
            raise ValueError(f"power requires 1 arg, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        exponent = term.kwargs.get("exponent", 2.0)
        return power(a, float(exponent))

    elif term.name == "difference":
        if len(term.args) != 2:
            raise ValueError(f"difference requires 2 args, got {len(term.args)}: {term.args}")
        a = base_features.get(term.args[0], np.zeros(1))
        b = base_features.get(term.args[1], np.zeros(1))
        return difference(a, b)

    else:
        raise ValueError(f"Unknown combinator: {term.name}")


def build_structure_features(
    structure: DSLStructure,
    full_base_features: np.ndarray,
    all_term_names: list[str],
) -> np.ndarray:
    """
    Build the feature matrix for a structure from pre-computed base features.

    Args:
        structure: DSLStructure with simple and/or compound terms
        full_base_features: [n_alts x n_base_terms] matrix (all 12 base terms)
        all_term_names: list of base term names matching columns of full_base_features

    Returns:
        [n_alts x n_structure_terms] matrix
    """
    term_to_idx = {t: i for i, t in enumerate(all_term_names)}
    n_alts = full_base_features.shape[0]
    columns = []

    for term in structure.terms:
        if not term.is_compound:
            # Simple term — slice from base features
            idx = term_to_idx.get(term.name)
            if idx is not None:
                columns.append(full_base_features[:, idx])
            else:
                columns.append(np.zeros(n_alts))
        else:
            # Compound term — compute from base features
            base_dict = {name: full_base_features[:, i] for name, i in term_to_idx.items()}
            try:
                col = compute_compound_feature(term, base_dict)
                columns.append(col)
            except Exception as e:
                print(f"  [DSL] Warning: failed to compute {term.display_name}: {e}")
                columns.append(np.zeros(n_alts))

    if not columns:
        return np.zeros((n_alts, 0))
    return np.column_stack(columns)
