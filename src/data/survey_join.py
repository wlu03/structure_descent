"""Left-join raw survey (persons) columns onto cleaned events (Wave 8, §2).

Ported from ``old_pipeline/src/data_prep.py::join_survey`` (lines 66-99).
The v1 function was hard-coded to the Amazon ``"Survey ResponseID"`` id
column and always printed progress to stdout; this module parameterizes
the id column via :class:`~src.data.schema_map.DatasetSchema.persons_id_column`
and switches all reporting to :mod:`logging`.

Scope
-----

The v1 pipeline performed four sub-steps in ``join_survey``:

1. Rename ``schema.persons_id_column`` to canonical ``customer_id``.
2. Drop persons columns that are entirely NaN.
3. Drop duplicate ``customer_id`` rows in persons (keep first).
4. Snake_case every persons column name except the join key.

Those behaviours are preserved **verbatim** here. This module deliberately
does **not** apply the ``translate_persons`` z_d translation: z_d-column
production is a separate concern that consumes the *raw* persons
DataFrame at a different stage.

Q4 decision (NOTES.md "Wave 8" — orphan events policy)
------------------------------------------------------

Customers present in ``events_df`` but missing from ``persons_raw`` remain
in the output. They keep their event columns and carry ``NaN`` in every
joined survey column, so they still contribute to popularity counts and
state features. They *will* be excluded from the training set later, at
the point where z_d features are built (null survey columns cannot produce
a z_d row). A WARNING is emitted at join time with the orphan count and
percentage so the exclusion is visible in pipeline logs.

Invariant hook
--------------

:func:`src.data.invariants.validate_joined` runs before the joined
DataFrame is returned. If the orphan fraction exceeds the 5% hard
threshold encoded in ``invariants.py``, ``InvariantError`` fires — at or
below the threshold, only a WARNING is emitted (both from ``join_survey``
itself and from ``validate_joined``).
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.data.invariants import validate_joined
from src.data.schema_map import DatasetSchema


__all__ = ["join_survey"]


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# v1-compatible snake_case helper (ported verbatim from data_prep.py:87-90).
# --------------------------------------------------------------------------- #


def _snake_case(name: str) -> str:
    """Normalize a column name to lower-snake_case.

    Ported verbatim from v1 ``old_pipeline/src/data_prep.py::_snake`` (lines
    87-90). Strips surrounding whitespace, lowercases, collapses any run of
    non-alphanumeric characters to a single underscore, and trims leading
    and trailing underscores.

    Examples
    --------
    ``"Q-demos-age"``    -> ``"q_demos_age"``
    ``"  Foo Bar!!  "``  -> ``"foo_bar"``
    """
    s = name.strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return s.strip("_")


# --------------------------------------------------------------------------- #
# join_survey
# --------------------------------------------------------------------------- #


def join_survey(
    events_df: pd.DataFrame,
    persons_raw: pd.DataFrame,
    schema: DatasetSchema,
) -> pd.DataFrame:
    """Left-join raw survey columns onto cleaned events by ``customer_id``.

    Parameters
    ----------
    events_df:
        Cleaned events DataFrame. Must carry the canonical ``customer_id``
        column (produced by the clean stage via
        :func:`src.data.schema_map.translate_events` plus load-time
        ``parse_dates``).
    persons_raw:
        Raw persons DataFrame as read from disk. Must carry
        ``schema.persons_id_column``. z_d translation has **not** been
        applied — that is the job of
        :func:`src.data.schema_map.translate_persons`.
    schema:
        The :class:`DatasetSchema` whose ``persons_id_column`` is renamed
        to canonical ``customer_id`` before the merge.

    Returns
    -------
    pd.DataFrame
        Events with all snake_cased, deduped, non-all-null persons columns
        merged on by ``customer_id``. Row count equals
        ``len(events_df)`` — left-join preserves every event row. Events
        whose ``customer_id`` has no matching persons row carry ``NaN`` in
        every joined survey column.

    Raises
    ------
    KeyError
        If ``events_df`` lacks a ``customer_id`` column or ``persons_raw``
        lacks ``schema.persons_id_column``.
    InvariantError
        (via :func:`validate_joined`) if the post-join orphan-row fraction
        exceeds the 5% hard threshold.
    """
    if "customer_id" not in events_df.columns:
        raise KeyError(
            "join_survey: events_df must carry a 'customer_id' column; "
            f"got columns {list(events_df.columns)!r}."
        )

    id_col = schema.persons_id_column
    if id_col not in persons_raw.columns:
        raise KeyError(
            f"join_survey: persons_raw missing id column {id_col!r}; "
            f"got columns {list(persons_raw.columns)!r}."
        )

    n_events = len(events_df)
    n_persons_in = len(persons_raw)

    # Work on a copy — never mutate caller's DataFrames.
    persons = persons_raw.copy()

    # Step 1 — rename id column to canonical `customer_id`.
    if id_col != "customer_id":
        # If a stray `customer_id` column already exists, drop it so the
        # rename does not collide and produce two columns of the same name.
        if "customer_id" in persons.columns:
            persons = persons.drop(columns=["customer_id"])
        persons = persons.rename(columns={id_col: "customer_id"})
    persons["customer_id"] = persons["customer_id"].astype(str)

    # Step 2 — drop all-null columns (v1 behaviour).
    all_null_cols = [c for c in persons.columns if persons[c].isna().all()]
    if all_null_cols:
        persons = persons.drop(columns=all_null_cols)

    # Step 3 — dedupe on customer_id, keep first (v1 behaviour).
    n_persons_predupe = len(persons)
    persons = persons.drop_duplicates(subset=["customer_id"], keep="first")
    n_persons_postdupe = len(persons)
    n_persons_dropped_dupe = n_persons_predupe - n_persons_postdupe

    # Step 4 — snake_case every column except the join key.
    persons.columns = [
        c if c == "customer_id" else _snake_case(c) for c in persons.columns
    ]

    # Step 5 — left-merge onto events. `customer_id` must be comparable; we
    # don't mutate events_df, but we do align dtype on the merge key locally.
    events_for_merge = events_df
    if events_df["customer_id"].dtype != persons["customer_id"].dtype:
        events_for_merge = events_df.copy()
        events_for_merge["customer_id"] = events_for_merge["customer_id"].astype(
            str
        )

    before_cols = set(events_for_merge.columns)
    merged = events_for_merge.merge(persons, on="customer_id", how="left")
    new_cols = [c for c in merged.columns if c not in before_cols]

    # Orphan report: count events whose customer_id is not in persons.
    persons_ids = set(persons["customer_id"].unique().tolist())
    event_ids = merged["customer_id"]
    orphan_mask = ~event_ids.isin(persons_ids)
    n_events_orphan = int(orphan_mask.sum())
    orphan_pct = (100.0 * n_events_orphan / n_events) if n_events else 0.0

    if n_events_orphan > 0:
        logger.warning(
            "join_survey: %d of %d event rows (%.2f%%) have no matching "
            "persons row and will carry NaN in every joined survey column.",
            n_events_orphan,
            n_events,
            orphan_pct,
        )

    logger.info(
        "join_survey: merged %d persons (was %d raw; %d dedupe-dropped; "
        "%d all-null columns dropped) into %d events; added %d new columns.",
        n_persons_postdupe,
        n_persons_in,
        n_persons_dropped_dupe,
        len(all_null_cols),
        n_events,
        len(new_cols),
    )

    # Final invariant gate.
    validate_joined(merged, schema)

    return merged
