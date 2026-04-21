"""Pure path-driven loader for the PO-LEU data layer (Wave 8, design doc §1).

Reads the two raw CSVs (events + persons) declared on a
:class:`~src.data.schema_map.DatasetSchema` and returns them untouched —
no column renames, no dtype coercion, no dropna; those transforms belong
to ``src/data/clean.py``. The only post-read work is calling
:func:`src.data.invariants.validate_loaded`, which guarantees any
schema/CSV mismatch surfaces immediately as an :class:`InvariantError`.

The v1 module (``old_pipeline/src/data_prep.py``) used a
module-level ``DATA_DIR = Path(__file__).parent.parent / "amazon_ecom"``
side effect and ``print``-based stage reporting; both are dropped here.
Paths are schema-driven (overridable per-call for fixture tests) and
progress is reported via :mod:`logging`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.invariants import validate_loaded
from src.data.schema_map import DatasetSchema


__all__ = ["load"]


logger = logging.getLogger(__name__)


def load(
    schema: DatasetSchema,
    *,
    events_path: Path | None = None,
    persons_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw (events, persons) DataFrames per schema.

    Parameters
    ----------
    schema:
        :class:`DatasetSchema` produced by
        :func:`src.data.schema_map.load_schema`. Provides default paths
        (``schema.events_path``, ``schema.persons_path``) and the list of
        datetime columns to parse (``schema.events_parse_dates``).
    events_path, persons_path:
        Override the schema paths. Useful for tests against fixtures.

    Returns
    -------
    (events_raw, persons_raw):
        Raw DataFrames exactly as read from disk, before any column
        renames or cleaning. :func:`validate_loaded` is run before the
        return so any schema/CSV mismatch raises ``InvariantError``
        immediately.

    Raises
    ------
    InvariantError:
        (from :func:`src.data.invariants.validate_loaded`) if either CSV
        is missing columns the schema requires.
    FileNotFoundError:
        If a CSV path does not exist.
    """
    ev_path = Path(events_path) if events_path is not None else Path(schema.events_path)
    ps_path = Path(persons_path) if persons_path is not None else Path(schema.persons_path)

    if not ev_path.exists():
        raise FileNotFoundError(f"events CSV not found: {ev_path}")
    if not ps_path.exists():
        raise FileNotFoundError(f"persons CSV not found: {ps_path}")

    # Peek at the events header to prune parse_dates down to columns that are
    # actually present. This avoids pandas raising a bare ValueError on a
    # missing parse_dates column before we ever get to ``validate_loaded`` —
    # we want the structured :class:`InvariantError` to surface instead.
    header_cols = list(pd.read_csv(ev_path, nrows=0).columns)
    parse_dates_all = list(schema.events_parse_dates) if schema.events_parse_dates else []
    parse_dates = [c for c in parse_dates_all if c in header_cols] or None

    events_raw = pd.read_csv(ev_path, parse_dates=parse_dates)
    persons_raw = pd.read_csv(ps_path)

    logger.debug(
        "load: events_raw.shape=%s, persons_raw.shape=%s (paths: events=%s, persons=%s).",
        events_raw.shape,
        persons_raw.shape,
        ev_path,
        ps_path,
    )

    if events_raw.empty:
        logger.warning("load: events DataFrame is empty (path=%s).", ev_path)
    if persons_raw.empty:
        logger.warning("load: persons DataFrame is empty (path=%s).", ps_path)

    # Strict schema ↔ CSV shape check before handing back to the caller.
    validate_loaded(events_raw, persons_raw, schema)

    logger.info(
        "load: loaded %d events, %d persons.",
        len(events_raw),
        len(persons_raw),
    )
    return events_raw, persons_raw
