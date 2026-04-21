"""Per-customer temporal split for the PO-LEU data layer (Wave 8, design doc §1).

Ports ``old_pipeline/src/data_prep.py::temporal_split`` (lines 496-520) onto
the canonical schema produced by :mod:`src.data.schema_map`. Each customer's
events are sorted earliest to latest and assigned three contiguous blocks:
``train`` (earliest), ``val`` (next), ``test`` (most recent). The v1
edge-case logic is preserved verbatim — ``max(1, int(n * frac))`` and the
``n_test + n_val >= n`` collapse to ``(1, 0)`` for small customers.

Pure, deterministic, no module-level side effects.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.invariants import validate_split
from src.data.schema_map import DatasetSchema


__all__ = ["temporal_split"]


logger = logging.getLogger(__name__)


def temporal_split(
    events_df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    *,
    val_frac: float | None = None,
    test_frac: float | None = None,
) -> pd.DataFrame:
    """Per-customer temporal split.

    Either ``schema`` or both of ``val_frac``/``test_frac`` must be given.
    When both are present, explicit kwargs win. When only ``schema`` is
    given, uses ``schema.val_frac`` and ``schema.test_frac``.

    Per-customer allocation (v1 logic, preserved exactly)::

        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val

    Labels are assigned in order earliest -> latest: ``n_train`` ``"train"``
    rows, then ``n_val`` ``"val"`` rows, then ``n_test`` ``"test"`` rows.

    Runs :func:`src.data.invariants.validate_split` before returning.

    Parameters
    ----------
    events_df:
        Cleaned events DataFrame carrying ``customer_id`` and
        ``order_date`` canonical columns.
    schema:
        Optional :class:`DatasetSchema`; supplies default fractions via
        ``schema.val_frac`` / ``schema.test_frac``.
    val_frac, test_frac:
        Explicit fractions; when supplied they override the schema
        defaults.

    Returns
    -------
    pd.DataFrame
        A copy of ``events_df`` sorted by ``(customer_id, order_date)``
        with a fresh ``RangeIndex`` and a new ``split`` column whose
        values are in ``{"train", "val", "test"}``.

    Raises
    ------
    ValueError
        If neither ``schema`` nor both of ``val_frac``/``test_frac`` are
        provided.
    InvariantError
        From :func:`src.data.invariants.validate_split` — e.g. if any
        customer ends up with no ``"train"`` row.
    """
    # Resolve (val_frac, test_frac) from kwargs with fallback to schema.
    if val_frac is None:
        if schema is None:
            raise ValueError(
                "temporal_split: either `schema` or explicit "
                "`val_frac`/`test_frac` kwargs must be provided."
            )
        val_frac = float(schema.val_frac)
    else:
        val_frac = float(val_frac)

    if test_frac is None:
        if schema is None:
            raise ValueError(
                "temporal_split: either `schema` or explicit "
                "`val_frac`/`test_frac` kwargs must be provided."
            )
        test_frac = float(schema.test_frac)
    else:
        test_frac = float(test_frac)

    # Deterministic per-customer temporal ordering.
    df = (
        events_df.sort_values(["customer_id", "order_date"], kind="mergesort")
        .copy()
        .reset_index(drop=True)
    )

    def assign_splits(n: int) -> list[str]:
        # v1 edge-case logic — preserved EXACTLY.
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val
        return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    split_labels: list[str] = []
    for _, grp in df.groupby("customer_id", sort=False):
        split_labels.extend(assign_splits(len(grp.index)))

    df["split"] = split_labels

    counts = df["split"].value_counts()
    n_train = int(counts.get("train", 0))
    n_val = int(counts.get("val", 0))
    n_test = int(counts.get("test", 0))

    # Post-stage validation: asserts split values subset of
    # {train, val, test} and every customer has >= 1 train row.
    validate_split(df)

    logger.info(
        "temporal_split: val_frac=%.4f, test_frac=%.4f -> "
        "train=%d, val=%d, test=%d (total=%d).",
        val_frac,
        test_frac,
        n_train,
        n_val,
        n_test,
        len(df),
    )

    return df
