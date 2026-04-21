"""Thin adapter layer over the Wave-8 schema_map/load (Wave 9, design doc §2-§3).

A :class:`DatasetAdapter` bundles everything the downstream PO-LEU pipeline
needs to know about a specific dataset into a small, duck-typed Protocol:
how to load its events and persons frames, which raw columns become canonical
``customer_id`` / ``order_date`` / ``asin`` / ``category`` / ``price`` /
``title`` / ``popularity``, how to translate the survey columns into canonical
``z_d``, which z_d columns are worth rendering in the ``c_d`` context string
and which are sentinels that would be identical across customers, and how to
render alternative metadata for the LLM generator.

The default implementation :class:`YamlAdapter` is a one-line factory over the
Wave-8 YAML schema format — dataset #2 costs zero Python. The bundled
:func:`AmazonAdapter` factory is the canonical example.

Design constraints (locked by the Wave 9 brief):

* No re-implementation of z_d translation — :meth:`translate_z_d` delegates to
  :func:`src.data.schema_map.translate_persons`.
* No re-implementation of loading — the first call to ``load_events`` /
  ``load_persons`` invokes :func:`src.data.load.load` once and caches *both*
  frames; subsequent calls hit the cache. Callers commonly want both, and the
  1.5M-row Amazon CSV would be wasteful to re-read.
* The :meth:`suppress_fields_for_c_d` rule is computed once at ``__init__``
  and cached on the instance, so hot-path callers pay O(1).
* Wave 10 wires a real popularity-percentile renderer; this module ships the
  ``_popularity_percentile_fn`` slot and a DEBUG-logged fallback so the
  pre-Wave-10 smoke test is exercisable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import pandas as pd

from src.data.load import load
from src.data.schema_map import DatasetSchema, load_schema, translate_persons


__all__ = [
    "DatasetAdapter",
    "YamlAdapter",
    "AmazonAdapter",
]


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol surface
# --------------------------------------------------------------------------- #


@runtime_checkable
class DatasetAdapter(Protocol):
    """Duck-typed contract for a dataset's data-layer adapter.

    Seven core methods describe how to materialize the canonical per-event
    and per-customer frames, plus a helper (:meth:`suppress_fields_for_c_d`)
    that returns the z_d columns the context-string renderer should skip.

    Implementations are expected to be pure apart from per-instance caching
    of ``load()`` results. Callers may rely on
    ``isinstance(obj, DatasetAdapter)`` as a runtime smoke test (the
    Protocol is ``@runtime_checkable``).
    """

    name: str
    schema: "DatasetSchema"

    def load_events(self) -> pd.DataFrame:
        """Return the raw events DataFrame (pre-clean). Cached after first call."""
        ...

    def load_persons(self) -> pd.DataFrame:
        """Return the raw persons DataFrame (pre-translate). Cached after first call."""
        ...

    def event_column_map(self) -> dict[str, str]:
        """Return the raw -> canonical event column rename map."""
        ...

    def person_id_column(self) -> str:
        """Return the raw persons id-column name (e.g. ``"Survey ResponseID"``)."""
        ...

    def translate_z_d(
        self,
        persons_raw: pd.DataFrame,
        *,
        training_events: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Translate raw persons -> canonical z_d columns + ``customer_id``."""
        ...

    def derived_z_d_columns(self) -> tuple[str, ...]:
        """Canonical z_d column names whose values come from events aggregation."""
        ...

    def alt_text(self, event_row: Mapping[str, Any]) -> dict[str, Any]:
        """Render the LLM-generator-visible alternative metadata dict."""
        ...

    def suppress_fields_for_c_d(self) -> tuple[str, ...]:
        """Canonical z_d column names that should be skipped in ``c_d``."""
        ...


# --------------------------------------------------------------------------- #
# Default YAML-backed implementation
# --------------------------------------------------------------------------- #


# Canonical alt-text keys the LLM prompt expects (§3.2).
_ALT_KEYS: tuple[str, ...] = ("title", "category", "price", "popularity_rank")


class YamlAdapter:
    """Default :class:`DatasetAdapter` backed by a Wave-8 dataset YAML.

    Parameters
    ----------
    yaml_path:
        Path to a dataset YAML (see ``configs/datasets/amazon.yaml``).

    Attributes
    ----------
    name:
        ``self.schema.name``.
    schema:
        Frozen :class:`DatasetSchema` parsed from the YAML.
    yaml_path:
        The path the schema was parsed from (kept for diagnostics).

    Notes
    -----
    * ``load_events`` / ``load_persons`` share a single cache; first call to
      *either* invokes :func:`src.data.load.load` which returns both frames,
      and both are stored on the instance. The second call (for either
      frame) hits the cache.
    * :meth:`suppress_fields_for_c_d` is computed once in ``__init__`` and
      memoised, so downstream callers pay O(1).
    * ``_popularity_percentile_fn`` is populated by Wave-10
      ``alt_rendering.build_popularity_percentiles(adapter)``. Until then,
      :meth:`alt_text` returns the ``"popularity score N"`` stub.
    """

    def __init__(self, yaml_path: Path | str) -> None:
        self.yaml_path: Path = Path(yaml_path)
        self.schema: DatasetSchema = load_schema(self.yaml_path)
        self.name: str = self.schema.name
        # Cached raw frames. Populated lazily by load_events / load_persons.
        self._events_cache: pd.DataFrame | None = None
        self._persons_cache: pd.DataFrame | None = None
        # Populated by Wave 10 alt_rendering.build_popularity_percentiles(adapter).
        # Until then, alt_text returns the "popularity score N" stub.
        self._popularity_percentile_fn: Callable[[int], str] | None = None
        # Precompute once; suppress_fields_for_c_d returns the cached tuple.
        self._suppress_fields_cache: tuple[str, ...] = self._compute_suppress_fields()

    # ------------------------------------------------------------------ #
    # Loader contract (with shared caching)
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Lazy-load both raw frames on first access; idempotent afterwards."""
        if self._events_cache is None or self._persons_cache is None:
            events_raw, persons_raw = load(self.schema)
            self._events_cache = events_raw
            self._persons_cache = persons_raw

    def load_events(self) -> pd.DataFrame:
        """Return the raw events DataFrame; cached after first call.

        Implementation: first call runs :func:`src.data.load.load` once,
        storing both frames on the adapter. Subsequent calls (on either
        ``load_events`` or ``load_persons``) hit the cache.
        """
        self._ensure_loaded()
        assert self._events_cache is not None  # for the type-checker
        return self._events_cache

    def load_persons(self) -> pd.DataFrame:
        """Return the raw persons DataFrame; cached after first call.

        Shares the cache with :meth:`load_events`.
        """
        self._ensure_loaded()
        assert self._persons_cache is not None  # for the type-checker
        return self._persons_cache

    # ------------------------------------------------------------------ #
    # Static schema-derived projections
    # ------------------------------------------------------------------ #

    def event_column_map(self) -> dict[str, str]:
        """Raw -> canonical event column rename map (copy)."""
        return dict(self.schema.events_column_map)

    def person_id_column(self) -> str:
        """Raw persons id-column name (e.g. ``"Survey ResponseID"``)."""
        return self.schema.persons_id_column

    def derived_z_d_columns(self) -> tuple[str, ...]:
        """Canonical z_d column names with ``kind == "derived_from_events"``.

        Preserves YAML order. For Amazon this is
        ``("purchase_frequency", "novelty_rate")``.
        """
        return tuple(
            spec.canonical_column
            for spec in self.schema.z_d_mapping
            if spec.kind == "derived_from_events"
        )

    # ------------------------------------------------------------------ #
    # z_d translation (delegates to schema_map)
    # ------------------------------------------------------------------ #

    def translate_z_d(
        self,
        persons_raw: pd.DataFrame,
        *,
        training_events: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Translate raw persons to canonical z_d columns.

        Delegates to :func:`src.data.schema_map.translate_persons`; no
        logic duplication. Any z_d field with ``kind ==
        "derived_from_events"`` requires ``training_events`` — passing
        ``None`` in that case raises ``ValueError`` inside
        ``translate_persons``.
        """
        return translate_persons(
            persons_raw,
            self.schema,
            training_events=training_events,
        )

    # ------------------------------------------------------------------ #
    # Alt-text rendering
    # ------------------------------------------------------------------ #

    def alt_text(self, event_row: Mapping[str, Any]) -> dict[str, Any]:
        """Render the LLM-generator-visible alternative metadata.

        Returns a dict with keys ``title``, ``category``, ``price``, and
        ``popularity_rank``. ``popularity_rank`` uses
        ``self._popularity_percentile_fn(popularity)`` when that callable
        has been wired by Wave 10; until then it is a stub
        ``"popularity score N"`` string (a DEBUG log records the
        fallback).
        """
        title = event_row.get("title", "")
        category = event_row.get("category", "")
        price = event_row.get("price", 0.0)
        popularity = int(event_row.get("popularity", 0) or 0)

        fn = self._popularity_percentile_fn
        if fn is not None:
            popularity_rank = fn(popularity)
        else:
            logger.debug(
                "alt_text: no popularity_percentile_fn wired yet (pre-Wave-10); "
                "emitting 'popularity score %d' stub.",
                popularity,
            )
            popularity_rank = f"popularity score {popularity}"

        return {
            "title": title,
            "category": category,
            "price": price,
            "popularity_rank": popularity_rank,
        }

    # ------------------------------------------------------------------ #
    # suppress_fields_for_c_d
    # ------------------------------------------------------------------ #

    def suppress_fields_for_c_d(self) -> tuple[str, ...]:
        """Canonical z_d column names that ``build_context_string`` should skip.

        Cached — the rule is evaluated once in ``__init__``. Rule:

        * ``kind == "constant"`` -> suppress (one value for every customer,
          so the rendered clause carries no per-customer signal).
        * ``kind == "external_lookup"`` AND lookup CSV is empty/header-only
          or missing -> suppress (every row falls through to the same
          fallback, so the rendered clause is identical across customers).
        * Any other kind -> NOT suppressed.

        Unknown names are silently ignored by ``build_context_string``,
        so adapter-suppress drift cannot break the renderer.
        """
        return self._suppress_fields_cache

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compute_suppress_fields(self) -> tuple[str, ...]:
        """Walk ``schema.z_d_mapping`` once and apply the suppress rule."""
        names: list[str] = []
        for spec in self.schema.z_d_mapping:
            if spec.kind == "constant":
                names.append(spec.canonical_column)
                continue
            if spec.kind == "external_lookup":
                if self._external_lookup_is_empty(spec.lookup_path):
                    names.append(spec.canonical_column)
                continue
            # All other kinds: not suppressed.
        return tuple(names)

    @staticmethod
    def _external_lookup_is_empty(lookup_path: Path | None) -> bool:
        """True iff the external-lookup CSV is missing, unreadable, or
        header-only (zero data rows)."""
        if lookup_path is None:
            return True
        p = Path(lookup_path)
        if not p.exists():
            return True
        try:
            table = pd.read_csv(p)
        except Exception:  # noqa: BLE001 — any read failure => treat as empty
            return True
        return len(table) == 0


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def AmazonAdapter() -> YamlAdapter:  # noqa: N802 — factory styled as a class ctor
    """Return a :class:`YamlAdapter` bound to ``configs/datasets/amazon.yaml``.

    The factory's name is capitalized deliberately so callers read it as
    ``adapter = AmazonAdapter()``, mirroring the ergonomics of a class
    constructor. This is the canonical way to obtain the default Amazon
    adapter.
    """
    return YamlAdapter("configs/datasets/amazon.yaml")
