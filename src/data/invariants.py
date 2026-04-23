"""Strict between-stage data invariants for the PO-LEU data pipeline.

Each stage of the data pipeline (load -> clean -> survey_join ->
state_features -> split -> attach_train_popularity) has a dedicated
validator that runs *after* the stage's transform and *before* the next
stage starts. Failures raise :class:`InvariantError`, an
``AssertionError`` subclass carrying structured context for debugging:

- ``invariant_name``: a short identifier for the rule that was broken
  (e.g. ``"customer_id_non_null"``).
- ``stage``: the pipeline stage where the check fires (e.g. ``"clean"``).
- ``column``: the offending column, when applicable.
- ``offending``: a ``head(5)`` sample of the offending rows (if any).

The module is pure: no file I/O, no RNG, no globals, no side effects on
import. Only dependencies are :mod:`pandas`, :mod:`numpy`, and stdlib
:mod:`logging`.

See Wave 8 design doc section 3 for the binding policy.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Error type
# --------------------------------------------------------------------------- #


class InvariantError(AssertionError):
    """Raised when a between-stage data invariant fails.

    Carries structured context for debugging:

    - ``invariant_name``: the rule identifier (e.g. ``"customer_id_non_null"``).
    - ``stage``: the pipeline stage that was being validated.
    - ``column``: optional column name implicated by the failure.
    - ``offending``: optional ``pd.DataFrame`` of up to 5 offending rows.

    The ``__str__`` form embeds the structured fields plus a ``to_string``
    dump of ``offending`` (when present) so that ``pytest`` failure output
    is self-contained.
    """

    def __init__(
        self,
        invariant_name: str,
        stage: str,
        message: str,
        *,
        column: str | None = None,
        offending: pd.DataFrame | None = None,
    ) -> None:
        self.invariant_name = invariant_name
        self.stage = stage
        self.message = message
        self.column = column
        self.offending = offending

        parts: list[str] = [
            f"[invariant={invariant_name!s}] [stage={stage!s}]",
        ]
        if column is not None:
            parts.append(f"[column={column!s}]")
        parts.append(message)
        if offending is not None and not offending.empty:
            parts.append("offending rows (head 5):\n" + offending.head(5).to_string())
        super().__init__(" ".join(parts))


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _raise(
    invariant_name: str,
    stage: str,
    message: str,
    *,
    column: str | None = None,
    offending: pd.DataFrame | None = None,
) -> None:
    """Internal helper to construct and raise an :class:`InvariantError`."""
    raise InvariantError(
        invariant_name,
        stage,
        message,
        column=column,
        offending=offending,
    )


def assert_columns_present(
    df: pd.DataFrame,
    required: list[str],
    *,
    invariant_name: str,
    stage: str,
) -> None:
    """Assert every name in ``required`` appears as a column of ``df``.

    Parameters
    ----------
    df:
        DataFrame under test.
    required:
        List of expected column names.
    invariant_name, stage:
        Structured context copied into any raised :class:`InvariantError`.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        _raise(
            invariant_name,
            stage,
            f"missing required columns: {missing!r}; "
            f"actual columns: {list(df.columns)!r}.",
        )


def assert_no_nan(
    df: pd.DataFrame,
    column: str,
    *,
    invariant_name: str,
    stage: str,
) -> None:
    """Assert that ``df[column]`` contains no null/NaN/NaT values.

    ``pd.isna`` is used so datetime NaT, float NaN, and object None all
    count as offenders. The first 5 offending rows are attached to the
    raised :class:`InvariantError`.
    """
    if column not in df.columns:
        _raise(
            invariant_name,
            stage,
            f"column {column!r} not in DataFrame; cannot check NaN.",
            column=column,
        )
    mask = df[column].isna()
    if mask.any():
        _raise(
            invariant_name,
            stage,
            f"column {column!r} has {int(mask.sum())} null/NaN/NaT rows.",
            column=column,
            offending=df.loc[mask].head(5),
        )


def assert_non_negative(
    df: pd.DataFrame,
    column: str,
    *,
    invariant_name: str,
    stage: str,
    allow_sentinel: float | None = None,
) -> None:
    """Assert ``df[column] >= 0`` (optionally allowing one sentinel value).

    NaN is *not* tolerated — callers that want "NaN OR sentinel" semantics
    should assert non-null separately. ``allow_sentinel`` lets recency-like
    columns keep the v1 ``999`` sentinel (for customers with no prior
    purchases).

    Raises :class:`InvariantError` if any non-sentinel value is ``< 0`` or
    if any value is NaN.
    """
    if column not in df.columns:
        _raise(
            invariant_name,
            stage,
            f"column {column!r} not in DataFrame; cannot check non-negativity.",
            column=column,
        )
    series = df[column]
    nan_mask = series.isna()
    if nan_mask.any():
        _raise(
            invariant_name,
            stage,
            f"column {column!r} has {int(nan_mask.sum())} NaN values; "
            f"NaN is not a valid non-negative value.",
            column=column,
            offending=df.loc[nan_mask].head(5),
        )

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = numeric.isna() & ~nan_mask
        _raise(
            invariant_name,
            stage,
            f"column {column!r} has non-numeric entries; cannot check "
            f"non-negativity.",
            column=column,
            offending=df.loc[bad].head(5),
        )

    neg_mask = numeric < 0
    if allow_sentinel is not None:
        neg_mask = neg_mask & (numeric != allow_sentinel)

    if neg_mask.any():
        _raise(
            invariant_name,
            stage,
            f"column {column!r} has {int(neg_mask.sum())} negative values"
            + (f" (sentinel {allow_sentinel!r} allowed)" if allow_sentinel is not None else "")
            + ".",
            column=column,
            offending=df.loc[neg_mask].head(5),
        )


def assert_dtype(
    df: pd.DataFrame,
    column: str,
    expected: str,
    *,
    invariant_name: str,
    stage: str,
) -> None:
    """Assert that ``df[column].dtype`` matches ``expected`` (string form).

    ``expected`` uses the pandas dtype string representation, e.g.
    ``"datetime64[ns]"``, ``"int64"``, ``"float64"``. The comparison is a
    simple ``str(dtype) == expected`` so callers can rely on the exact
    match they specify. For object/string columns, pass ``"object"``.
    """
    if column not in df.columns:
        _raise(
            invariant_name,
            stage,
            f"column {column!r} not in DataFrame; cannot check dtype.",
            column=column,
        )
    actual = str(df[column].dtype)
    if actual != expected:
        _raise(
            invariant_name,
            stage,
            f"column {column!r} dtype {actual!r} does not match "
            f"expected {expected!r}.",
            column=column,
            offending=df[[column]].head(5),
        )


def assert_values_in_set(
    df: pd.DataFrame,
    column: str,
    allowed: set,
    *,
    invariant_name: str,
    stage: str,
) -> None:
    """Assert every value of ``df[column]`` is in ``allowed`` (no NaN)."""
    if column not in df.columns:
        _raise(
            invariant_name,
            stage,
            f"column {column!r} not in DataFrame; cannot check value set.",
            column=column,
        )
    allowed_set = set(allowed)
    series = df[column]
    # NaN is never in any set; treat NaN as a violation for set-membership.
    bad_mask = ~series.isin(allowed_set) | series.isna()
    if bad_mask.any():
        offending_values = sorted(
            {repr(v) for v in series.loc[bad_mask].unique().tolist()}
        )
        _raise(
            invariant_name,
            stage,
            f"column {column!r} has values outside allowed set {sorted(allowed_set)!r}; "
            f"offending unique values: {offending_values}.",
            column=column,
            offending=df.loc[bad_mask].head(5),
        )


# --------------------------------------------------------------------------- #
# Per-stage validators
# --------------------------------------------------------------------------- #


def validate_loaded(
    events_df: pd.DataFrame,
    persons_df: pd.DataFrame,
    schema: "DatasetSchema",  # noqa: F821 — forward reference avoids circular import
) -> None:
    """Post-load validator. Runs after ``load.py`` and before ``clean.py``.

    Checks:

    - For every canonical destination in ``schema.events_column_map``, at
      least one of its raw source keys is present in ``events_df``. The
      schema may list multiple raw keys mapping to the same canonical
      (e.g., Amazon's ``"ASIN/ISBN (Product Code)"`` and ``"ASIN/ISBN"``
      both mapping to ``asin`` for older export compatibility); any one
      of them satisfies the invariant
      (``invariant_name="events_column_map_canonical_satisfied"``).
    - ``persons_df`` contains ``schema.persons_id_column``
      (``invariant_name="persons_id_column_present"``).

    Parameters
    ----------
    events_df, persons_df:
        Raw DataFrames as loaded from disk.
    schema:
        Duck-typed object exposing ``events_column_map`` (``dict``) and
        ``persons_id_column`` (``str``). Typed as forward-ref string to
        avoid a circular import against ``schema_map``.
    """
    stage = "load"

    # Build the inverse: {canonical -> set of raw keys that map to it}.
    # For each canonical destination, at least one raw key must be present.
    canonical_to_raws: dict[str, list[str]] = {}
    for raw, canonical in schema.events_column_map.items():
        canonical_to_raws.setdefault(canonical, []).append(raw)

    unsatisfied: list[tuple[str, list[str]]] = []
    for canonical, raws in canonical_to_raws.items():
        if not any(r in events_df.columns for r in raws):
            unsatisfied.append((canonical, raws))

    if unsatisfied:
        missing_summary = "; ".join(
            f"{can!r} needs one of {raws}" for can, raws in unsatisfied
        )
        raise InvariantError(
            invariant_name="events_column_map_canonical_satisfied",
            stage=stage,
            message=(
                f"no raw source key present for canonical columns: "
                f"{missing_summary}. Actual raw columns: "
                f"{list(events_df.columns)}."
            ),
        )

    assert_columns_present(
        persons_df,
        [schema.persons_id_column],
        invariant_name="persons_id_column_present",
        stage=stage,
    )


def validate_cleaned(
    events_df: pd.DataFrame,
    schema: "DatasetSchema",  # noqa: F821
) -> None:
    """Post-clean validator. Runs after ``clean.py``.

    Required canonical columns on ``events_df`` after clean:
    ``customer_id``, ``order_date``, ``price``, ``asin``, ``category``.

    Checks:

    - Canonical columns present (``canonical_columns_present``).
    - ``customer_id`` dtype is ``object`` (``customer_id_dtype``).
    - ``customer_id`` non-null (``customer_id_non_null``).
    - ``order_date`` dtype is ``datetime64[ns]`` (``order_date_dtype``).
    - ``order_date`` has no ``NaT`` (``order_date_no_nat``).
    - ``price`` dtype is ``float64`` (``price_dtype``).
    - ``price`` non-negative (``price_non_negative``).
    - ``asin`` non-null (``asin_non_null``).
    - ``category`` non-null (``category_non_null``).

    The ``schema`` parameter is accepted for API symmetry with the other
    validators; it is not currently used but may gate future stricter
    checks (e.g. dropna_subset coverage).
    """
    stage = "clean"
    _ = schema  # reserved for future stricter checks.

    assert_columns_present(
        events_df,
        ["customer_id", "order_date", "price", "asin", "category"],
        invariant_name="canonical_columns_present",
        stage=stage,
    )

    assert_dtype(
        events_df,
        "customer_id",
        "object",
        invariant_name="customer_id_dtype",
        stage=stage,
    )
    assert_no_nan(
        events_df,
        "customer_id",
        invariant_name="customer_id_non_null",
        stage=stage,
    )

    assert_dtype(
        events_df,
        "order_date",
        "datetime64[ns]",
        invariant_name="order_date_dtype",
        stage=stage,
    )
    assert_no_nan(
        events_df,
        "order_date",
        invariant_name="order_date_no_nat",
        stage=stage,
    )

    assert_dtype(
        events_df,
        "price",
        "float64",
        invariant_name="price_dtype",
        stage=stage,
    )
    assert_non_negative(
        events_df,
        "price",
        invariant_name="price_non_negative",
        stage=stage,
    )

    assert_no_nan(
        events_df,
        "asin",
        invariant_name="asin_non_null",
        stage=stage,
    )
    assert_no_nan(
        events_df,
        "category",
        invariant_name="category_non_null",
        stage=stage,
    )


#: Threshold above which the survey-join miss-rate escalates from WARNING
#: to a hard :class:`InvariantError`. Expressed as a fraction in [0, 1].
_JOIN_MISS_RATE_THRESHOLD: float = 0.05


def validate_joined(
    events_df: pd.DataFrame,
    schema: "DatasetSchema",  # noqa: F821
) -> None:
    """Post-survey-join validator.

    Checks:

    - ``customer_id`` non-null on events (``customer_id_non_null``).
    - Every ``customer_id`` row is "joined" — i.e. has *some* non-null
      survey column beyond the canonical events columns
      (``no_orphan_customers``). Orphan customers are those whose survey
      columns are entirely null. If the orphan fraction is above
      :data:`_JOIN_MISS_RATE_THRESHOLD` (5%) the validator raises; at or
      below the threshold it logs a ``WARNING``.

    The survey columns are inferred as "every column on ``events_df``
    that is NOT a known canonical events column". That keeps the check
    agnostic to the exact z_d mapping.
    """
    stage = "survey_join"
    _ = schema  # reserved for future stricter checks.

    assert_columns_present(
        events_df,
        ["customer_id"],
        invariant_name="customer_id_present",
        stage=stage,
    )
    assert_no_nan(
        events_df,
        "customer_id",
        invariant_name="customer_id_non_null",
        stage=stage,
    )

    canonical_events_cols = {
        "customer_id",
        "order_date",
        "price",
        "quantity",
        "state",
        "title",
        "asin",
        "category",
    }
    survey_cols: list[str] = [
        c for c in events_df.columns if c not in canonical_events_cols
    ]

    if not survey_cols:
        # No joined survey columns at all — treat as a total join miss.
        _raise(
            "no_orphan_customers",
            stage,
            "no survey columns present on events_df after join; "
            "expected at least one joined survey column.",
        )

    # A row is "orphaned" if every survey column is null on that row.
    orphan_row_mask = events_df[survey_cols].isna().all(axis=1)
    n_rows = len(events_df)
    if n_rows == 0:
        return
    orphan_fraction = float(orphan_row_mask.sum()) / float(n_rows)

    if orphan_fraction > _JOIN_MISS_RATE_THRESHOLD:
        _raise(
            "no_orphan_customers",
            stage,
            f"survey-join miss rate {orphan_fraction:.4f} exceeds threshold "
            f"{_JOIN_MISS_RATE_THRESHOLD:.4f}; "
            f"{int(orphan_row_mask.sum())} of {n_rows} event rows have "
            f"entirely null survey columns.",
            offending=events_df.loc[orphan_row_mask].head(5),
        )
    elif orphan_row_mask.any():
        logger.warning(
            "survey-join miss rate %.4f (%d of %d rows) is within the %.4f "
            "threshold; %d orphan rows will carry null survey columns.",
            orphan_fraction,
            int(orphan_row_mask.sum()),
            n_rows,
            _JOIN_MISS_RATE_THRESHOLD,
            int(orphan_row_mask.sum()),
        )


#: The v1 "never purchased before" sentinel for ``recency_days`` (see
#: ``compute_state_features``). Validators accept this value alongside
#: real non-negative recencies.
_RECENCY_SENTINEL: float = 999.0


def validate_state_features(events_df: pd.DataFrame) -> None:
    """Post ``compute_state_features`` validator.

    Checks:

    - Columns ``routine``, ``novelty``, ``cat_affinity``, ``recency_days``
      are present (``state_feature_columns_present``).
    - ``routine``, ``novelty``, ``cat_affinity`` are non-negative and
      non-NaN (``routine_non_negative``, ``novelty_non_negative``,
      ``cat_affinity_non_negative``).
    - ``recency_days`` is non-negative and non-NaN, with the ``999``
      sentinel allowed (``recency_days_non_negative``).
    """
    stage = "state_features"

    assert_columns_present(
        events_df,
        ["routine", "novelty", "cat_affinity", "recency_days"],
        invariant_name="state_feature_columns_present",
        stage=stage,
    )

    assert_non_negative(
        events_df,
        "routine",
        invariant_name="routine_non_negative",
        stage=stage,
    )
    assert_non_negative(
        events_df,
        "novelty",
        invariant_name="novelty_non_negative",
        stage=stage,
    )
    assert_non_negative(
        events_df,
        "cat_affinity",
        invariant_name="cat_affinity_non_negative",
        stage=stage,
    )
    assert_non_negative(
        events_df,
        "recency_days",
        invariant_name="recency_days_non_negative",
        stage=stage,
        allow_sentinel=_RECENCY_SENTINEL,
    )


_SPLIT_VALUES: set[str] = {"train", "val", "test"}


def validate_split(events_df: pd.DataFrame) -> None:
    """Post-temporal-split validator.

    Checks:

    - ``split`` column present (``split_column_present``).
    - Every ``split`` value is in ``{"train", "val", "test"}``
      (``split_values_subset``).
    - Every ``customer_id`` has at least one ``"train"`` row
      (``every_customer_has_train_row``).
    """
    stage = "split"

    assert_columns_present(
        events_df,
        ["split", "customer_id"],
        invariant_name="split_column_present",
        stage=stage,
    )

    assert_values_in_set(
        events_df,
        "split",
        _SPLIT_VALUES,
        invariant_name="split_values_subset",
        stage=stage,
    )

    customers_with_train: set = set(
        events_df.loc[events_df["split"] == "train", "customer_id"].unique().tolist()
    )
    all_customers: set = set(events_df["customer_id"].unique().tolist())
    missing_train: set = all_customers - customers_with_train
    if missing_train:
        offending = events_df.loc[events_df["customer_id"].isin(missing_train)].head(5)
        _raise(
            "every_customer_has_train_row",
            stage,
            f"{len(missing_train)} customer_id(s) have no train rows: "
            f"{sorted(list(missing_train))[:10]!r}"
            + (" ..." if len(missing_train) > 10 else ""),
            column="customer_id",
            offending=offending,
        )


def validate_cold_start_split(events_df: pd.DataFrame) -> None:
    """Post-cold-start-split validator.

    The cold-start split partitions *customers* into disjoint train / val /
    test groups (every event for a given customer lands in the same
    split). This differs from :func:`validate_split`, which assumes
    within-customer splits and therefore requires every customer to
    have a train row — an invariant cold-start deliberately violates.

    Checks:

    - ``split`` column present.
    - Every ``split`` value is in ``{"train", "val", "test"}``.
    - **Every customer_id is fully contained in exactly one split**
      (``customer_split_disjoint``). This is the core cold-start
      contract: no event leakage for any held-out customer.
    - At least one customer exists in the ``"train"`` split
      (``cold_start_train_nonempty``). Val/test may legitimately be
      empty for tiny datasets; a zero-train run would crash downstream
      so we fail loud.
    """
    stage = "cold_start_split"

    assert_columns_present(
        events_df,
        ["split", "customer_id"],
        invariant_name="split_column_present",
        stage=stage,
    )

    assert_values_in_set(
        events_df,
        "split",
        _SPLIT_VALUES,
        invariant_name="split_values_subset",
        stage=stage,
    )

    # Disjointness: a customer must appear under exactly one split label.
    per_customer = events_df.groupby("customer_id")["split"].nunique()
    offending_ids = per_customer[per_customer > 1].index.tolist()
    if offending_ids:
        offending = events_df.loc[
            events_df["customer_id"].isin(offending_ids[:5])
        ].head(20)
        _raise(
            "customer_split_disjoint",
            stage,
            f"{len(offending_ids)} customer_id(s) appear under more than "
            f"one split label: {sorted(offending_ids)[:10]!r}"
            + (" ..." if len(offending_ids) > 10 else ""),
            column="customer_id",
            offending=offending,
        )

    train_customers = events_df.loc[
        events_df["split"] == "train", "customer_id"
    ].unique()
    if len(train_customers) == 0:
        _raise(
            "cold_start_train_nonempty",
            stage,
            "cold-start split produced 0 train customers; cannot fit a "
            "downstream baseline. Check val_customer_frac + "
            "test_customer_frac do not sum to ~1.0.",
            column="customer_id",
            offending=None,
        )


def validate_popularity(events_df: pd.DataFrame) -> None:
    """Post ``attach_train_popularity`` validator.

    Checks:

    - ``popularity`` column present (``popularity_column_present``).
    - ``popularity`` dtype ``int64`` (``popularity_dtype``).
    - ``popularity`` non-negative (``popularity_non_negative``).
    - Logs ``WARNING`` with the count of ``popularity == 0`` rows
      (unseen ASINs). Does not raise.
    """
    stage = "popularity"

    assert_columns_present(
        events_df,
        ["popularity"],
        invariant_name="popularity_column_present",
        stage=stage,
    )

    assert_dtype(
        events_df,
        "popularity",
        "int64",
        invariant_name="popularity_dtype",
        stage=stage,
    )

    assert_non_negative(
        events_df,
        "popularity",
        invariant_name="popularity_non_negative",
        stage=stage,
    )

    n_zero = int((events_df["popularity"] == 0).sum())
    if n_zero > 0:
        logger.warning(
            "popularity == 0 on %d of %d event rows (unseen ASINs outside "
            "the train window).",
            n_zero,
            len(events_df),
        )


__all__ = [
    "InvariantError",
    "assert_columns_present",
    "assert_no_nan",
    "assert_non_negative",
    "assert_dtype",
    "assert_values_in_set",
    "validate_loaded",
    "validate_cleaned",
    "validate_joined",
    "validate_state_features",
    "validate_split",
    "validate_popularity",
]
