"""Popularity-percentile alt-text renderer (Wave 10, design doc §2).

Converts raw train-only ASIN popularity counts into stable, human-readable
band labels for the LLM prompt's ``popularity_rank`` alt field: one of

    {"top 5%", "top 25%", "top 50%", "bottom 50%"}

The band thresholds are computed once over the **unique-ASIN** popularity
distribution in the training split (de-duplicated on ``asin`` so that a
single popular ASIN that appears in thousands of events contributes a
weight of 1, not its event-frequency) via the 0.95 / 0.75 / 0.50
quantiles. The returned closure is pure, picklable, and safe to
`copy.deepcopy` — callers who snapshot adapter state (e.g. for
reproducibility manifests) rely on this.

The Wave-9 :class:`~src.data.adapter.YamlAdapter` ships a
``_popularity_percentile_fn`` slot that is ``None`` at construction. This
module's :func:`register_on_adapter` is the canonical way to populate
that slot after ``attach_train_popularity`` has run.

Design constraints (locked by the Wave 10 brief):

* **Unique-ASIN basis.** Quantiles are computed after
  ``drop_duplicates(subset=[asin_column], keep="first")``; otherwise
  popular ASINs that appear in thousands of events would dominate the
  distribution and compress the thresholds toward the high end.
* **Four fixed bands.** The ladder is ``q95`` → ``q75`` → ``q50`` →
  bottom. When quantiles collapse (e.g. all ASINs have popularity 1
  → ``q95 == q75 == q50 == 1``), the first matching band wins; the
  function never crashes and always returns a valid band string.
* **Pure / picklable.** Only numpy + pandas + stdlib imports at module
  load. No side effects on import, no file I/O, no globals touched.
"""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd


__all__ = [
    "BAND_LABELS",
    "build_popularity_percentile_fn",
    "register_on_adapter",
]


logger = logging.getLogger(__name__)


#: Canonical band labels, ordered most-popular → least-popular. The ladder
#: in :func:`build_popularity_percentile_fn` returns these exact strings.
BAND_LABELS: tuple[str, str, str, str] = (
    "top 5%",
    "top 25%",
    "top 50%",
    "bottom 50%",
)


if TYPE_CHECKING:
    # Forward reference only; no runtime import to avoid a circular-import
    # risk against src.data.adapter.
    from src.data.adapter import DatasetAdapter


# --------------------------------------------------------------------------- #
# Picklable closure for the popularity-percentile function
# --------------------------------------------------------------------------- #


class _PopularityPercentileFn:
    """Picklable callable wrapping the three quantile thresholds.

    A plain ``def``-inside-function closure would not pickle across
    module-import boundaries because :mod:`pickle` serializes functions
    by fully-qualified name. This thin callable class stores the three
    thresholds as float attributes and dispatches in ``__call__``, so
    ``pickle.dumps`` / ``pickle.loads`` round-trips it cleanly.
    """

    __slots__ = ("q95", "q75", "q50")

    def __init__(self, q95: float, q75: float, q50: float) -> None:
        self.q95 = float(q95)
        self.q75 = float(q75)
        self.q50 = float(q50)

    def __call__(self, n: int) -> str:
        """Map a raw popularity count to one of the four band labels.

        The ladder is ordered from most-popular to least-popular so the
        first matching clause wins when quantiles collapse (e.g. all
        ASINs have popularity 1 → ``q95 == q75 == q50 == 1`` → any
        ``n >= 1`` lands in ``"top 5%"``).
        """
        if n >= self.q95:
            return "top 5%"
        if n >= self.q75:
            return "top 25%"
        if n >= self.q50:
            return "top 50%"
        return "bottom 50%"

    def __repr__(self) -> str:  # pragma: no cover — diagnostic only
        return (
            f"_PopularityPercentileFn(q95={self.q95!r}, "
            f"q75={self.q75!r}, q50={self.q50!r})"
        )


# --------------------------------------------------------------------------- #
# Public: build_popularity_percentile_fn
# --------------------------------------------------------------------------- #


def build_popularity_percentile_fn(
    train_events: pd.DataFrame,
    *,
    popularity_col: str = "popularity",
    asin_col: str = "asin",
) -> Callable[[int], str]:
    """Build a popularity-band renderer from the training split.

    Parameters
    ----------
    train_events:
        Events DataFrame with a populated ``popularity_col`` (from
        :func:`src.data.state_features.attach_train_popularity`) and an
        ``asin_col`` identifying each alternative.
    popularity_col:
        Name of the column carrying the raw train-only popularity count.
        Defaults to ``"popularity"``.
    asin_col:
        Name of the column carrying the ASIN identifier. Defaults to
        ``"asin"``. The frame is de-duplicated on this column before the
        quantiles are computed so each unique ASIN contributes a weight
        of 1.

    Returns
    -------
    Callable[[int], str]
        A picklable callable that maps a raw popularity count to one of
        :data:`BAND_LABELS`. Safe for ``copy.deepcopy`` and
        ``pickle.dumps``.

    Raises
    ------
    ValueError
        If ``train_events`` is empty (0 rows). The error message names
        ``popularity_col`` so the caller can locate the upstream stage
        that should have populated it.
    KeyError
        If ``popularity_col`` is absent from ``train_events.columns``.
    """
    if popularity_col not in train_events.columns:
        raise KeyError(
            f"build_popularity_percentile_fn: column {popularity_col!r} "
            f"not in train_events.columns; "
            f"has attach_train_popularity run? "
            f"available columns: {list(train_events.columns)!r}"
        )

    if len(train_events) == 0:
        raise ValueError(
            f"build_popularity_percentile_fn: train_events is empty, "
            f"cannot compute {popularity_col!r} quantiles. "
            f"Has the training split been materialized?"
        )

    # De-duplicate on asin before computing quantiles. Otherwise popular
    # ASINs that appear in thousands of events would dominate the
    # distribution and compress the thresholds. Weight should be 1 per
    # unique ASIN, not per event.
    if asin_col in train_events.columns:
        unique_events = train_events.drop_duplicates(
            subset=[asin_col], keep="first"
        )
    else:
        # The asin column is optional; if it's missing, fall back to
        # the raw distribution. This keeps the function usable against
        # pre-aggregated popularity tables.
        unique_events = train_events

    pop_series = unique_events[popularity_col].astype(float)

    # numpy.quantile with linear interpolation (pandas default).
    q95 = float(np.quantile(pop_series.to_numpy(), 0.95))
    q75 = float(np.quantile(pop_series.to_numpy(), 0.75))
    q50 = float(np.quantile(pop_series.to_numpy(), 0.50))

    logger.debug(
        "build_popularity_percentile_fn: quantiles over %d unique ASINs: "
        "q95=%.4f, q75=%.4f, q50=%.4f",
        len(unique_events),
        q95,
        q75,
        q50,
    )
    logger.info(
        "build_popularity_percentile_fn: built 4-band renderer "
        "(top 5%%, top 25%%, top 50%%, bottom 50%%) over %d unique ASINs.",
        len(unique_events),
    )

    return _PopularityPercentileFn(q95=q95, q75=q75, q50=q50)


# --------------------------------------------------------------------------- #
# Public: register_on_adapter
# --------------------------------------------------------------------------- #


def register_on_adapter(
    adapter: "DatasetAdapter",
    train_events: pd.DataFrame,
    *,
    popularity_col: str = "popularity",
    asin_col: str = "asin",
) -> None:
    """Build a popularity-percentile renderer and wire it onto ``adapter``.

    After this call, ``adapter.alt_text(row)["popularity_rank"]`` returns
    one of :data:`BAND_LABELS` instead of the Wave-9 stub
    ``f"popularity score {n}"``.

    Parameters
    ----------
    adapter:
        A :class:`~src.data.adapter.DatasetAdapter` instance (typically a
        :class:`~src.data.adapter.YamlAdapter`). The adapter must expose
        a writable ``_popularity_percentile_fn`` attribute (Wave 9's
        :class:`YamlAdapter` declares this slot in ``__init__``).
    train_events:
        See :func:`build_popularity_percentile_fn`.
    popularity_col:
        Forwarded to :func:`build_popularity_percentile_fn`.
    asin_col:
        Forwarded to :func:`build_popularity_percentile_fn`.

    Notes
    -----
    Idempotent: calling again with a fresh training split overwrites the
    existing callable. Callers doing train/val splits in a loop should
    rebuild after each split.
    """
    fn = build_popularity_percentile_fn(
        train_events,
        popularity_col=popularity_col,
        asin_col=asin_col,
    )
    adapter._popularity_percentile_fn = fn
    logger.info(
        "register_on_adapter: wired popularity-percentile fn onto adapter %r.",
        getattr(adapter, "name", type(adapter).__name__),
    )
