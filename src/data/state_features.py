"""Per-event state features + train-only popularity broadcast (Wave 8, design doc §5).

Pure port of ``old_pipeline/src/data_prep.py::compute_state_features`` and
``attach_train_popularity`` (and their ``_BRAND_STOPWORDS`` / brand-map
helpers), preserving v1 numeric behavior row-for-row. Stage ``print``
statements are replaced by :mod:`logging`, and each public function calls
the sibling :mod:`src.data.invariants` validator before returning.

Determinism: no RNG, no file I/O, no globals touched at runtime. The
``_BRAND_STOPWORDS`` set is a module-level constant lifted verbatim from
v1; it is Amazon-flavored and will be generalized if/when a second
dataset requires title-based brand inference (noted in ``NOTES.md``).

The v1 behavior around missing prior purchases is preserved: the first
purchase of a ``(customer_id, asin)`` pair has ``recency_days = 999``
(the sentinel that :func:`invariants.validate_state_features` accepts
via ``assert_non_negative(..., allow_sentinel=999.0)``).
"""

from __future__ import annotations

import logging
import re
from typing import Dict

import numpy as np
import pandas as pd

from src.data.invariants import (
    assert_columns_present,
    validate_popularity,
    validate_state_features,
)


__all__ = [
    "compute_state_features",
    "attach_train_popularity",
    "attach_train_brand_map",
]


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Brand helpers (verbatim port from v1)
# --------------------------------------------------------------------------- #


#: First-token stopword list for the v1 brand heuristic. Lifted verbatim
#: from ``old_pipeline/src/data_prep.py``; the list is Amazon-flavored
#: (adjective buzzwords + pack/set/kit marketing tokens) and will be
#: generalized in a later wave if a second dataset's titles need brand
#: inference.
_BRAND_STOPWORDS: set[str] = {
    "premium", "new", "upgraded", "upgrade", "original", "genuine", "authentic",
    "professional", "pro", "deluxe", "luxury", "ultra", "super", "best",
    "the", "a", "an", "for", "with", "by", "of",
    "2-pack", "3-pack", "4-pack", "6-pack", "10-pack", "pack",
    "set", "kit", "bundle", "case", "box",
}


def _first_brand_token(title: str) -> str:
    """Return the first title token that looks like a brand name.

    Skips: empty tokens, members of :data:`_BRAND_STOPWORDS`, purely
    numeric / punctuation tokens (``re.fullmatch(r"[\\d\\.\\-x/]+")``),
    and compact count-codes like ``"2pk"`` / ``"10ct"``
    (``re.fullmatch(r"\\d+[a-z]{1,3}")``). Returns an empty string if no
    candidate token is found.

    Tokens are lower-cased and stripped of surrounding punctuation before
    the stopword / regex filters are applied.
    """
    if not isinstance(title, str) or not title:
        return ""
    for tok in title.split():
        t = tok.lower().strip(",.:;!?()[]{}\"'")
        if not t:
            continue
        if t in _BRAND_STOPWORDS:
            continue
        if re.fullmatch(r"[\d\.\-x/]+", t):
            continue
        if re.fullmatch(r"\d+[a-z]{1,3}", t):
            continue
        return t
    return ""


def _build_asin_brand_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build a ``{asin: mode_brand_token}`` map from ``df["title"]``.

    Applies :func:`_first_brand_token` to each non-null title, groups by
    ``asin``, and takes the most-frequent token per ASIN (``idxmax`` on
    ``value_counts``). ASINs whose titles never produce a non-empty
    token are absent from the map — callers typically fill these to
    ``"unknown_brand"`` after an ambiguity pass.
    """
    tokens = df["title"].fillna("").map(_first_brand_token)
    tmp = pd.DataFrame({"asin": df["asin"].to_numpy(), "tok": tokens.to_numpy()})
    tmp = tmp[tmp["tok"] != ""]
    if len(tmp) == 0:
        return {}
    mode = (
        tmp.groupby("asin")["tok"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    return mode


# --------------------------------------------------------------------------- #
# Public: compute_state_features
# --------------------------------------------------------------------------- #


def compute_state_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-event history features to ``events_df``.

    Adds columns (all per-row, computed from prior events of the same
    customer only, so there is no leakage from future rows):

    - ``routine``: cumulative count of prior purchases of the same
      ``(customer_id, asin)`` pair (``0`` for the first occurrence).
    - ``novelty``: ``1`` iff this is the first purchase of that pair,
      else ``0``.
    - ``recency_days``: days since the last purchase of the same
      ``(customer_id, asin)``. First-time pairs receive the v1 sentinel
      ``999`` (which :func:`invariants.validate_state_features`
      whitelists via ``allow_sentinel=999.0``).
    - ``cat_affinity``: cumulative count of prior purchases by the same
      customer in the same ``category``.
    - ``cat_count_7d`` / ``cat_count_30d``: time-based rolling event
      counts within ``(customer_id, category)`` over the preceding
      7-day / 30-day windows (inclusive of the current event; matches
      pandas ``DataFrame.rolling("7D", on=order_date)`` semantics).

    Preserves the input row count. Returns a new DataFrame (input is
    ``.copy()``-ed), re-sorted by ``(customer_id, order_date)`` on exit.
    Runs :func:`invariants.validate_state_features` before returning.

    Popularity is *not* computed here — see
    :func:`attach_train_popularity`, which must be called after the
    temporal split to prevent val/test counts from leaking into train.

    Brand is *not* computed here either — see
    :func:`attach_train_brand_map`, which must be called after the split
    so that the per-ASIN brand mode is derived from ``split == "train"``
    rows only. Computing it pre-split would leak a test-customer's title
    spelling into the brand signal visible at training (finding F4 of
    the leakage audit). This function returns a frame *without* a
    ``brand`` column; downstream code must call
    :func:`attach_train_brand_map` before reading ``event_row["brand"]``.
    """
    logger.info("compute_state_features: starting on %d events.", len(events_df))
    df = events_df.copy().sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    logger.info("compute_state_features: routine (repeat purchase count).")
    df["routine"] = df.groupby(["customer_id", "asin"]).cumcount()
    df["novelty"] = (df["routine"] == 0).astype(int)

    logger.info("compute_state_features: recency_days (days since last purchase).")
    df["last_purchase_date"] = df.groupby(["customer_id", "asin"])["order_date"].shift(1)
    df["recency_days"] = (df["order_date"] - df["last_purchase_date"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(999)

    logger.info("compute_state_features: cat_affinity (category purchase count).")
    df["cat_affinity"] = df.groupby(["customer_id", "category"]).cumcount()

    logger.info(
        "compute_state_features: cat_count_7d / cat_count_30d "
        "(time-based rolling counts)."
    )
    df["_cat_event"] = 1
    df = df.sort_values(["customer_id", "category", "order_date"]).reset_index(drop=True)
    rolled7 = (
        df.groupby(["customer_id", "category"], sort=False)
        .rolling("7D", on="order_date")["_cat_event"].sum()
        .reset_index(level=[0, 1], drop=True)
    )
    rolled30 = (
        df.groupby(["customer_id", "category"], sort=False)
        .rolling("30D", on="order_date")["_cat_event"].sum()
        .reset_index(level=[0, 1], drop=True)
    )
    df["cat_count_7d"] = rolled7.values
    df["cat_count_30d"] = rolled30.values
    df = df.drop(columns=["_cat_event"])
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    # Brand is no longer set here — see attach_train_brand_map (post-split).
    # Any pre-existing ``brand`` column from a caller-supplied frame is
    # cleared so that the pre-split output is unambiguously brand-less and
    # the downstream helper is the single source of truth.
    if "brand" in df.columns:
        df = df.drop(columns=["brand"])

    repeat_rate = float((df["routine"] > 0).mean()) if len(df) else 0.0
    logger.debug(
        "compute_state_features: %d events, repeat_rate=%.4f (brand attached post-split).",
        len(df),
        repeat_rate,
    )
    logger.info(
        "compute_state_features: done (%d events, repeat_rate=%.1f%%; "
        "brand attached post-split via attach_train_brand_map).",
        len(df),
        100.0 * repeat_rate,
    )

    validate_state_features(df)
    return df


# --------------------------------------------------------------------------- #
# Public: attach_train_popularity
# --------------------------------------------------------------------------- #


def attach_train_popularity(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute train-only ASIN popularity and broadcast to every row.

    Popularity is the count of ``split == "train"`` rows per ``asin``.
    Every row receives that count as ``popularity`` (val/test rows get
    the train-derived value — a legitimate "what the model knew at
    training time"). ASINs that never appear in train are mapped to
    ``popularity = 0``.

    Requires the ``split`` column to already exist; the pre-check raises
    :class:`~src.data.invariants.InvariantError` with
    ``stage="state_features"`` if it is missing, and the error message
    points the caller at :mod:`src.data.temporal_split`.

    Runs :func:`invariants.validate_popularity` before returning.
    """
    # Pre-condition: split must exist. Point the caller at the upstream
    # temporal_split stage (v1 raised KeyError; we upgrade to a
    # structured InvariantError via the shared helper).
    assert_columns_present(
        events_df,
        ["split"],
        invariant_name="split_required_for_popularity",
        stage="state_features",
    )

    logger.info(
        "attach_train_popularity: computing train-only ASIN popularity "
        "over %d events.",
        len(events_df),
    )
    df = events_df.copy()

    train_mask = df["split"] == "train"
    pop = (
        df.loc[train_mask]
        .groupby("asin")
        .size()
        .rename("popularity")
    )
    df = df.drop(columns=["popularity"], errors="ignore")
    df = df.join(pop, on="asin")
    df["popularity"] = df["popularity"].fillna(0).astype(np.int64)

    n_unseen = int((df["popularity"] == 0).sum())
    logger.debug(
        "attach_train_popularity: %d unique train ASINs, %d rows with popularity=0.",
        int(len(pop)),
        n_unseen,
    )
    logger.info(
        "attach_train_popularity: done (%d unique train ASINs, %d unseen rows).",
        int(len(pop)),
        n_unseen,
    )

    validate_popularity(df)
    return df


# --------------------------------------------------------------------------- #
# Public: attach_train_brand_map
# --------------------------------------------------------------------------- #


def attach_train_brand_map(events_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a train-only per-ASIN brand map and broadcast to every row.

    Mirrors :func:`attach_train_popularity`. The per-ASIN brand is the
    mode first-token (via :func:`_first_brand_token`) across rows with
    ``split == "train"`` only. Every row — train, val, and test — then
    receives that train-derived brand label for its ASIN, so val/test
    rows see "what the model knew about this ASIN's brand at training
    time" instead of a label that may have been inferred from their own
    (or a fellow test-customer's) title spelling.

    The ``brand`` column is attached in place. ASINs that never appear
    in train (cold-start catalogs, val/test-only ASINs) receive the
    empty string ``""`` as the brand sentinel — the downstream
    :meth:`src.data.adapter.DatasetAdapter.alt_text` path already maps
    an empty / missing brand to ``"unknown_brand"``, so no further
    plumbing is required.

    Requires the ``split`` column to already exist; the pre-check raises
    :class:`~src.data.invariants.InvariantError` with
    ``stage="state_features"`` if it is missing, and the error message
    points the caller at :mod:`src.data.split`.

    This helper deliberately does *not* apply v1's "ambiguous brand"
    collapse (singleton-token brands → ``"unknown_brand"``). The goal
    here is leakage-correctness, not v1 numeric parity: the downstream
    ``alt_text`` sentinel already handles empty / unknown cases
    gracefully, and preserving low-count tokens when they do appear in
    train is useful signal rather than noise once the map is restricted
    to the train split.
    """
    # Pre-condition: split must exist. Point the caller at the upstream
    # temporal_split / cold_start_split stage.
    assert_columns_present(
        events_df,
        ["split"],
        invariant_name="split_required_for_brand_map",
        stage="state_features",
    )

    logger.info(
        "attach_train_brand_map: computing train-only ASIN brand mode "
        "over %d events.",
        len(events_df),
    )
    df = events_df.copy()

    train_mask = df["split"] == "train"
    train_df = df.loc[train_mask, ["asin", "title"]]
    brand_map = _build_asin_brand_map(train_df) if len(train_df) else {}

    df = df.drop(columns=["brand"], errors="ignore")
    df["brand"] = df["asin"].map(brand_map).fillna("").astype(str)

    n_mapped = int((df["brand"] != "").sum())
    n_unmapped = int((df["brand"] == "").sum())
    logger.debug(
        "attach_train_brand_map: %d unique train-mapped ASINs, "
        "%d rows mapped, %d rows with brand=''.",
        int(len(brand_map)),
        n_mapped,
        n_unmapped,
    )
    logger.info(
        "attach_train_brand_map: done (%d unique train-mapped ASINs, "
        "%d rows mapped, %d rows unmapped).",
        int(len(brand_map)),
        n_mapped,
        n_unmapped,
    )

    return df
