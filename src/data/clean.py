"""Schema-driven cleaning of a raw events DataFrame (Wave 8, design doc §1).

Ports the cleaning logic of ``old_pipeline/src/data_prep.py::clean_purchases``
(lines 34-63) onto the canonical schema produced by
:mod:`src.data.schema_map`. The heavy lifting (column rename, ``dropna`` on
the required subset, dtype coercion, category-null fill) already lives in
:func:`src.data.schema_map.translate_events`; this module adds the
post-translate deterministic sort, a ``reset_index``, logging, and the
post-stage invariant check.

Pure, deterministic, no module-level side effects.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.invariants import validate_cleaned
from src.data.schema_map import DatasetSchema, translate_events


__all__ = ["clean_events"]


logger = logging.getLogger(__name__)


# Fraction of rows that may be dropped by ``translate_events`` (via the
# schema's ``dropna_subset``) before we upgrade the log-line from INFO to
# WARNING. Same threshold as the survey-join miss-rate in
# :mod:`src.data.invariants` (5%).
_DROP_WARNING_THRESHOLD: float = 0.05


def clean_events(
    events_raw: pd.DataFrame,
    schema: DatasetSchema,
) -> pd.DataFrame:
    """Return a canonical-column events DataFrame, sorted by (customer, date).

    Wraps :func:`src.data.schema_map.translate_events` with a post-sort and
    post-validation step. No-op-idempotent: cleaning an already-clean
    DataFrame is a shape-preserving pass (modulo the sort).

    Shape contract
    --------------
    In
        DataFrame with raw dataset column names (per
        ``schema.events_column_map`` keys).
    Out
        DataFrame with canonical columns
        ``{"customer_id", "order_date", "price", "asin", "category", ...}``
        (per ``schema.events_column_map`` values), sorted by
        ``(customer_id, order_date)`` and with a fresh ``RangeIndex``.

    Logging
    -------
    Emits a single INFO line on completion giving row / customer / category
    counts. If ``translate_events``'s ``dropna`` step removed more than 5%
    of rows, a WARNING is emitted instead of the INFO line.

    Raises
    ------
    InvariantError
        From :func:`src.data.invariants.validate_cleaned`, if the cleaned
        DataFrame fails any post-stage check (canonical columns present,
        ``order_date`` dtype ``datetime64[ns]``, non-negative ``price``, etc).
    """
    n_before = len(events_raw)

    # All the real work (rename, dropna_subset, dtype_coerce, category
    # null-fill) lives in translate_events. Do NOT re-implement any of it.
    cleaned = translate_events(events_raw, schema)

    # Deterministic sort by (customer_id, order_date), then fresh RangeIndex
    # so downstream positional indexing is stable.
    cleaned = cleaned.sort_values(
        ["customer_id", "order_date"], kind="mergesort"
    ).reset_index(drop=True)

    # Post-stage validation: fails loudly if translate_events + sort didn't
    # produce a schema-compliant frame (e.g. price still has NaNs, or the
    # dtype coerce block was misconfigured).
    validate_cleaned(cleaned, schema)

    n_after = len(cleaned)
    n_dropped = n_before - n_after
    drop_fraction = (n_dropped / n_before) if n_before > 0 else 0.0

    n_customers = int(cleaned["customer_id"].nunique()) if "customer_id" in cleaned.columns else 0
    n_categories = int(cleaned["category"].nunique()) if "category" in cleaned.columns else 0

    summary = (
        "cleaned %d events, %d customers, %d categories "
        "(dropped %d of %d rows, %.4f)"
    )
    if drop_fraction > _DROP_WARNING_THRESHOLD:
        logger.warning(
            summary,
            n_after,
            n_customers,
            n_categories,
            n_dropped,
            n_before,
            drop_fraction,
        )
    else:
        logger.info(
            summary,
            n_after,
            n_customers,
            n_categories,
            n_dropped,
            n_before,
            drop_fraction,
        )

    return cleaned
