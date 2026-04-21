"""Person feature vector ``z_d`` builder (redesign.md §2.1, orchestrator override).

This module turns tidy customer-level survey/state columns into the
26-dimensional ``z_d`` feature matrix consumed by the weight and salience
networks. The encoding breakdown is fixed by the orchestrator resolution
recorded in ``NOTES.md`` ("## Wave 1 — data prep ... p = 26 reconciliation"):

    age_bucket     (6, one-hot)
    income_bucket  (5, one-hot)
    household_size (5, one-hot over {1, 2, 3, 4, 5+})     <- orchestrator override
    has_kids       (1, 0/1)
    city_size      (4, one-hot)
    education      (1, standardized scalar)
    health_rating  (1, standardized scalar)
    risk_tolerance (1, standardized scalar)
    purchase_frequency (1, log1p then standardized scalar)
    novelty_rate   (1, pass-through in [0, 1])
    ───────────────────────────────────────────
    total                                                                26

Standardization statistics and category vocabularies are fit on the training
split only (§2.1 "Standardization") and frozen inside :class:`PersonFeatureStats`
so that val/test splits use identical encodings. ``novelty_rate`` is left
un-standardized per §2.1 row 10 (spec lists it as a rate in [0, 1]).

Pure functions; no file I/O, no globals, no RNG. Only dependencies are
``numpy``, ``pandas``, and :mod:`dataclasses`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #

#: Raw input columns expected in the DataFrame, in the spec's row order (§2.1).
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "age_bucket",
    "income_bucket",
    "household_size",
    "has_kids",
    "city_size",
    "education",
    "health_rating",
    "risk_tolerance",
    "purchase_frequency",
    "novelty_rate",
)

#: Scalar columns that get standardized (mean/std fit on train).
#: ``purchase_frequency`` is log1p'd *before* standardization (§2.1 row 9).
#: ``novelty_rate`` is NOT standardized — spec lists it as a rate in [0, 1]
#: and the orchestrator resolution keeps it pass-through.
#: ``household_size`` is NOT standardized under the orchestrator override —
#: it is a 5-bin one-hot.
#: ``has_kids`` stays binary (§2.1 row 4).
_STANDARDIZED_COLUMNS: tuple[str, ...] = (
    "education",
    "health_rating",
    "risk_tolerance",
    "purchase_frequency",
)

#: Canonical household-size one-hot vocabulary (orchestrator override).
_HOUSEHOLD_SIZE_CATEGORIES: tuple[str, ...] = ("1", "2", "3", "4", "5+")


# --------------------------------------------------------------------------- #
# Stats container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PersonFeatureStats:
    """Frozen statistics produced by :func:`fit_person_features`.

    Attributes
    ----------
    means, stds:
        Per-column mean and std for the standardized scalar features, in the
        order given by ``_STANDARDIZED_COLUMNS`` (education, health_rating,
        risk_tolerance, purchase_frequency). ``purchase_frequency`` stores
        the mean/std of ``log1p(raw_count)``. ``household_size`` is NOT
        included here because it is a one-hot under the orchestrator
        override; ``novelty_rate`` is NOT included because it is
        pass-through.
    age_categories, income_categories, household_size_categories,
    city_size_categories:
        Canonical vocabulary for each categorical column. ``age_categories``,
        ``income_categories``, ``city_size_categories`` are learned from
        the training split; ``household_size_categories`` is the fixed
        ``["1","2","3","4","5+"]`` vocabulary from the orchestrator
        resolution. Order here fixes the one-hot slot order in the output.
    feature_columns:
        The 26 output column names in order; ``transform_person_features``
        emits columns in exactly this order.
    """

    means: np.ndarray
    stds: np.ndarray
    age_categories: list[str]
    income_categories: list[str]
    household_size_categories: list[str]
    city_size_categories: list[str]
    feature_columns: list[str] = field(default_factory=list)

    # -- JSON round-trip ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (lists and plain floats)."""
        return {
            "means": [float(x) for x in np.asarray(self.means).tolist()],
            "stds": [float(x) for x in np.asarray(self.stds).tolist()],
            "age_categories": list(self.age_categories),
            "income_categories": list(self.income_categories),
            "household_size_categories": list(self.household_size_categories),
            "city_size_categories": list(self.city_size_categories),
            "feature_columns": list(self.feature_columns),
            "standardized_columns": list(_STANDARDIZED_COLUMNS),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonFeatureStats":
        """Inverse of :meth:`to_dict`."""
        return cls(
            means=np.asarray(d["means"], dtype=np.float64),
            stds=np.asarray(d["stds"], dtype=np.float64),
            age_categories=list(d["age_categories"]),
            income_categories=list(d["income_categories"]),
            household_size_categories=list(d["household_size_categories"]),
            city_size_categories=list(d["city_size_categories"]),
            feature_columns=list(d["feature_columns"]),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Expected all of {list(_REQUIRED_COLUMNS)}."
        )


def _canonical_categories(series: pd.Series) -> list[str]:
    """Return sorted, stringified unique categories (deterministic order)."""
    values = series.dropna().unique().tolist()
    return sorted([str(v) for v in values])


def _bucket_household_size(hs: int | float) -> str:
    """Map a raw household-size value to one of the 5 one-hot bins.

    Bins are ``{"1", "2", "3", "4", "5+"}``. Any integer value ``>= 5``
    folds into ``"5+"``. ``hs`` must be a non-negative integer — negative
    values or non-integer floats raise ``ValueError``. Floats that are
    exactly integer-valued (e.g. ``3.0``) are accepted.

    Parameters
    ----------
    hs:
        Raw household-size value from survey data.

    Returns
    -------
    str
        The matching bucket label.

    Raises
    ------
    ValueError
        If ``hs`` is negative, non-finite, or not integer-valued.
    """
    if hs is None:
        raise ValueError("household_size must be a non-negative integer; got None.")
    # Reject booleans explicitly — bool is an int subclass in Python but is
    # not a meaningful household size.
    if isinstance(hs, bool):
        raise ValueError(
            f"household_size must be a non-negative integer; got bool {hs!r}."
        )
    if isinstance(hs, (int, np.integer)):
        value = int(hs)
    elif isinstance(hs, (float, np.floating)):
        if not math.isfinite(float(hs)):
            raise ValueError(
                f"household_size must be a finite integer; got {hs!r}."
            )
        if float(hs) != int(hs):
            raise ValueError(
                f"household_size must be integer-valued; got {hs!r}."
            )
        value = int(hs)
    else:
        raise ValueError(
            f"household_size must be an int or integer-valued float; "
            f"got {type(hs).__name__} {hs!r}."
        )

    if value < 1:
        raise ValueError(
            f"household_size must be >= 1; got {value!r}."
        )
    if value >= 5:
        return "5+"
    return str(value)


def _build_feature_columns(
    age_categories: list[str],
    income_categories: list[str],
    household_size_categories: list[str],
    city_size_categories: list[str],
) -> list[str]:
    """Assemble the 26 output column names in canonical order (orchestrator table)."""
    cols: list[str] = []
    cols.extend(f"age_bucket={c}" for c in age_categories)
    cols.extend(f"income_bucket={c}" for c in income_categories)
    cols.extend(f"household_size={c}" for c in household_size_categories)  # one-hot
    cols.append("has_kids")                # binary 0/1
    cols.extend(f"city_size={c}" for c in city_size_categories)
    cols.append("education")               # standardized
    cols.append("health_rating")           # standardized
    cols.append("risk_tolerance")          # standardized
    cols.append("purchase_frequency")      # log1p then standardized
    cols.append("novelty_rate")            # raw rate in [0,1]
    return cols


def _one_hot(series: pd.Series, categories: list[str], prefix: str) -> np.ndarray:
    """One-hot encode ``series`` against ``categories``; raise on unknowns."""
    str_vals = series.astype(str).to_numpy()
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n = len(str_vals)
    k = len(categories)
    out = np.zeros((n, k), dtype=np.float32)
    for row, v in enumerate(str_vals):
        if v not in cat_to_idx:
            raise ValueError(
                f"Unknown category {v!r} encountered in column {prefix!r}. "
                f"Known categories learned on train split: {categories}."
            )
        out[row, cat_to_idx[v]] = 1.0
    return out


def _household_size_one_hot(series: pd.Series) -> np.ndarray:
    """One-hot encode household_size via ``_bucket_household_size``."""
    n = len(series)
    k = len(_HOUSEHOLD_SIZE_CATEGORIES)
    cat_to_idx = {c: i for i, c in enumerate(_HOUSEHOLD_SIZE_CATEGORIES)}
    out = np.zeros((n, k), dtype=np.float32)
    for row_i, raw in enumerate(series.to_list()):
        bucket = _bucket_household_size(raw)
        out[row_i, cat_to_idx[bucket]] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def fit_person_features(df: pd.DataFrame) -> PersonFeatureStats:
    """Fit standardization stats and category vocabularies on a training split.

    Parameters
    ----------
    df:
        Tidy pandas DataFrame. Must contain every column in
        :data:`_REQUIRED_COLUMNS`.

    Returns
    -------
    PersonFeatureStats
        Frozen container with means/stds for the standardized scalars and
        canonical category vocabularies. The 26-element ``feature_columns``
        list fixes the column order used by :func:`transform_person_features`.

    Notes
    -----
    Deterministic, no RNG. Standardization is fit on this split ONLY;
    val/test should call :func:`transform_person_features` with the returned
    stats (redesign.md §2.1 "Standardization"). ``household_size`` is
    validated through :func:`_bucket_household_size` during the fit so bad
    values surface here rather than silently at transform time.
    """
    _validate_columns(df)

    # Categorical vocabularies (learned from training split).
    age_categories = _canonical_categories(df["age_bucket"])
    income_categories = _canonical_categories(df["income_bucket"])
    city_size_categories = _canonical_categories(df["city_size"])

    # Household size has a fixed vocabulary from the orchestrator override;
    # validate that every raw value is in-range during the fit so errors
    # surface early.
    for raw in df["household_size"].to_list():
        _bucket_household_size(raw)
    household_size_categories = list(_HOUSEHOLD_SIZE_CATEGORIES)

    # log1p purchase_frequency before computing mean/std.
    pf_log = np.log1p(df["purchase_frequency"].to_numpy(dtype=np.float64))

    means = np.empty(len(_STANDARDIZED_COLUMNS), dtype=np.float64)
    stds = np.empty(len(_STANDARDIZED_COLUMNS), dtype=np.float64)
    for i, col in enumerate(_STANDARDIZED_COLUMNS):
        if col == "purchase_frequency":
            vals = pf_log
        else:
            vals = df[col].to_numpy(dtype=np.float64)
        means[i] = float(np.mean(vals))
        # population std (ddof=0); guard against zero to avoid div-by-zero.
        std = float(np.std(vals, ddof=0))
        stds[i] = std if std > 0.0 else 1.0

    feature_columns = _build_feature_columns(
        age_categories,
        income_categories,
        household_size_categories,
        city_size_categories,
    )

    return PersonFeatureStats(
        means=means,
        stds=stds,
        age_categories=age_categories,
        income_categories=income_categories,
        household_size_categories=household_size_categories,
        city_size_categories=city_size_categories,
        feature_columns=feature_columns,
    )


def transform_person_features(
    df: pd.DataFrame, stats: PersonFeatureStats
) -> np.ndarray:
    """Encode ``df`` into a ``(n, 26)`` float32 matrix using frozen ``stats``.

    Parameters
    ----------
    df:
        Tidy DataFrame with all columns in :data:`_REQUIRED_COLUMNS`.
    stats:
        Output of :func:`fit_person_features`, fit on the training split.

    Returns
    -------
    np.ndarray
        Shape ``(len(df), 26)``, dtype ``float32``. Column order matches
        ``stats.feature_columns``.

    Raises
    ------
    ValueError
        If any row has a category not seen during fit, any row has an
        invalid ``household_size`` (see :func:`_bucket_household_size`),
        or required columns are missing.
    """
    _validate_columns(df)

    n = len(df)

    age_oh = _one_hot(df["age_bucket"], stats.age_categories, "age_bucket")
    income_oh = _one_hot(df["income_bucket"], stats.income_categories, "income_bucket")
    household_oh = _household_size_one_hot(df["household_size"])
    city_oh = _one_hot(df["city_size"], stats.city_size_categories, "city_size")

    # Binary has_kids: cast to {0,1} float32; reject anything else.
    hk_raw = df["has_kids"].to_numpy()
    has_kids = np.asarray(hk_raw, dtype=np.int64)
    if not np.all(np.isin(has_kids, (0, 1))):
        bad = np.unique(has_kids[~np.isin(has_kids, (0, 1))])
        raise ValueError(
            f"has_kids must be binary 0/1; found values {bad.tolist()!r}."
        )
    has_kids = has_kids.astype(np.float32).reshape(n, 1)

    # Standardized scalars (log1p applied to purchase_frequency first).
    std_mat = np.empty((n, len(_STANDARDIZED_COLUMNS)), dtype=np.float64)
    for i, col in enumerate(_STANDARDIZED_COLUMNS):
        if col == "purchase_frequency":
            vals = np.log1p(df[col].to_numpy(dtype=np.float64))
        else:
            vals = df[col].to_numpy(dtype=np.float64)
        std_mat[:, i] = (vals - stats.means[i]) / stats.stds[i]

    idx = {col: i for i, col in enumerate(_STANDARDIZED_COLUMNS)}
    education = std_mat[:, idx["education"]].reshape(n, 1)
    health_rating = std_mat[:, idx["health_rating"]].reshape(n, 1)
    risk_tolerance = std_mat[:, idx["risk_tolerance"]].reshape(n, 1)
    purchase_frequency = std_mat[:, idx["purchase_frequency"]].reshape(n, 1)

    # novelty_rate: left as-is in [0, 1] (orchestrator resolution).
    novelty_rate = df["novelty_rate"].to_numpy(dtype=np.float64).reshape(n, 1)

    out = np.concatenate(
        [
            age_oh,                              # (n, 6)
            income_oh,                           # (n, 5)
            household_oh,                        # (n, 5)
            has_kids,                            # (n, 1)
            city_oh,                             # (n, 4)
            education.astype(np.float32),        # (n, 1)
            health_rating.astype(np.float32),    # (n, 1)
            risk_tolerance.astype(np.float32),   # (n, 1)
            purchase_frequency.astype(np.float32),  # (n, 1)
            novelty_rate.astype(np.float32),     # (n, 1)
        ],
        axis=1,
    )

    if out.shape[1] != len(stats.feature_columns):
        # Defensive: internal invariant — should never trip for a correctly fit
        # stats object.
        raise ValueError(
            f"Encoded width {out.shape[1]} does not match "
            f"stats.feature_columns length {len(stats.feature_columns)}."
        )

    return out.astype(np.float32, copy=False)


def fit_transform_person_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, PersonFeatureStats]:
    """Convenience: fit on ``df`` and return the encoded matrix plus stats.

    Returns
    -------
    (np.ndarray, PersonFeatureStats)
        The ``(n, 26)`` float32 matrix and the fitted stats object.
    """
    stats = fit_person_features(df)
    return transform_person_features(df, stats), stats
