"""Adapter: PO-LEU :func:`build_choice_sets` records → :class:`BaselineEventBatch`.

The current pipeline's event record schema (see
:mod:`src.data.choice_sets` lines ~720–800) is:

    {
        "customer_id":      str,
        "chosen_asin":      str,
        "choice_asins":     list[str]  (length J),
        "chosen_idx":       int in [0, J),
        "z_d":              np.ndarray (p,),       # person-only
        "c_d":              str,                   # context string
        "alt_texts":        list[dict] (length J), # 7-key dicts from
                                                    # adapter.alt_text
        "chosen_features":  dict,                  # chosen-alt-only
        "order_date":       pd.Timestamp,
        "category":         str,
        "metadata":         dict,                  # is_repeat, price, routine
    }

The baseline suite's :class:`BaselineEventBatch` (see
:mod:`src.baselines.base`) expects per-event ``(J, n_base_terms)``
feature matrices. The *old* 12-DSL-primitive feature set (``routine,
recency, novelty, popularity, affinity, time_match, price_sensitivity,
rating_signal, brand_affinity, price_rank, delivery_speed,
co_purchase``) is NOT reconstructible per-alternative from the current
pipeline's outputs — six of those columns (``recency``,
``cat_affinity``, ``time_match``, ``rating_signal``,
``delivery_speed``, ``co_purchase``) are populated for the chosen
alternative only; :func:`build_choice_sets` doesn't sample them for
negatives.

Rather than fabricate zeros for negatives (which would artifactually
inflate PO-LEU's relative win), this adapter exposes a **restricted,
per-alt-verified** feature set derived purely from the 7-key
``alt_texts`` dict + the per-record ``metadata``:

    - ``price``            (float)           — from ``alt_texts[j]["price"]``
    - ``popularity_rank``  (float)           — parsed from the
                                                ``alt_texts[j]["popularity_rank"]``
                                                string ("popularity score N")
                                                or passed-through if numeric
    - ``is_repeat``        (0.0 / 1.0)       — from ``alt_texts[j]["is_repeat"]``
    - ``brand_known``      (0.0 / 1.0)       — 1 iff the alt's brand matches
                                                the chosen-alt brand for this
                                                event (proxy for brand_affinity)
    - ``log1p_price``      (float)           — ``log1p(max(price, 0))``
    - ``price_rank``       (float, 0..1)     — within-event dense rank of
                                                ``price``, scaled to [0, 1]

Baselines that previously required the full 12-primitive pool see this
restricted 6-column pool. For LASSO-MNL / Bayesian-ARD the
:func:`expand_batch` nonlinear expansion still runs on top, yielding
``3*6 + C(6,2) = 18 + 15 = 33`` columns with interactions.

Callers can pass ``extra_feature_fn(record, alt_idx) -> dict[str,
float]`` to inject additional per-alt columns (for example, a
``cat_affinity`` lookup computed from an external aggregation). The
returned keys become named columns appended after the 6 built-in ones.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from .base import BaselineEventBatch

# Canonical order of built-in columns — downstream baselines rely on
# this order being stable (e.g. DUET's monotonicity features reference
# column names; LASSO-MNL's expanded pool is deterministic per order).
BUILTIN_FEATURE_NAMES: tuple[str, ...] = (
    "price",
    "popularity_rank",
    "is_repeat",
    "brand_known",
    "log1p_price",
    "price_rank",
)

_POPRANK_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_popularity_rank(val: Any) -> float:
    """Coerce ``alt_texts[j]["popularity_rank"]`` to a float.

    ``alt_text()`` wires either a callable-produced string (often
    ``"top X%"`` or ``"popularity score N"``) or a bare numeric. We
    grab the first numeric substring we find; missing / unparseable
    values default to 0.0 (the baseline's logit will learn to ignore
    the column if it's degenerate).
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val)
    m = _POPRANK_RE.search(s)
    if m is None:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def _price_rank(prices: np.ndarray) -> np.ndarray:
    """Dense within-event rank of ``prices``, scaled to ``[0, 1]``."""
    J = int(prices.shape[0])
    if J <= 1:
        return np.zeros(J, dtype=np.float32)
    order = np.argsort(prices, kind="stable")
    ranks = np.empty(J, dtype=np.float32)
    ranks[order] = np.arange(J, dtype=np.float32) / float(J - 1)
    return ranks


def _build_feature_matrix(
    record: Mapping[str, Any],
    *,
    extra_feature_fn: Optional[
        Callable[[Mapping[str, Any], int], Mapping[str, float]]
    ] = None,
    extra_feature_names: Sequence[str] = (),
) -> np.ndarray:
    """Build the ``(J, n_base_terms)`` matrix for one event."""
    alt_texts = record["alt_texts"]
    J = len(alt_texts)
    if J == 0:
        raise ValueError(
            f"record {record.get('customer_id', '?')} has empty alt_texts"
        )

    prices = np.asarray(
        [float(alt.get("price", 0.0) or 0.0) for alt in alt_texts],
        dtype=np.float32,
    )
    popularities = np.asarray(
        [_parse_popularity_rank(alt.get("popularity_rank", 0.0)) for alt in alt_texts],
        dtype=np.float32,
    )
    is_repeat = np.asarray(
        [1.0 if bool(alt.get("is_repeat", False)) else 0.0 for alt in alt_texts],
        dtype=np.float32,
    )
    chosen_idx = int(record["chosen_idx"])
    chosen_brand = str(alt_texts[chosen_idx].get("brand", "") or "")
    # ``brand_known``: a purchase-context proxy for brand affinity
    # available under our per-alt constraint. A learned coefficient
    # picks up whether alternatives sharing the same brand as the
    # chosen row are systematically preferred — not the same as
    # historical brand affinity but a legal per-alt signal.
    if chosen_brand and chosen_brand != "unknown_brand":
        brand_known = np.asarray(
            [
                1.0
                if str(alt.get("brand", "") or "") == chosen_brand
                else 0.0
                for alt in alt_texts
            ],
            dtype=np.float32,
        )
    else:
        brand_known = np.zeros(J, dtype=np.float32)
    log1p_price = np.log1p(np.maximum(prices, 0.0)).astype(np.float32)
    price_rank = _price_rank(prices)

    n_builtin = len(BUILTIN_FEATURE_NAMES)
    n_extra = len(extra_feature_names)
    mat = np.empty((J, n_builtin + n_extra), dtype=np.float32)
    mat[:, 0] = prices
    mat[:, 1] = popularities
    mat[:, 2] = is_repeat
    mat[:, 3] = brand_known
    mat[:, 4] = log1p_price
    mat[:, 5] = price_rank

    if n_extra:
        if extra_feature_fn is None:
            raise ValueError(
                "extra_feature_names provided but extra_feature_fn is None"
            )
        for j in range(J):
            vals = extra_feature_fn(record, j) or {}
            for k, name in enumerate(extra_feature_names):
                mat[j, n_builtin + k] = float(vals.get(name, 0.0) or 0.0)

    return mat


def records_to_baseline_batch(
    records: Iterable[Mapping[str, Any]],
    *,
    extra_feature_fn: Optional[
        Callable[[Mapping[str, Any], int], Mapping[str, float]]
    ] = None,
    extra_feature_names: Sequence[str] = (),
) -> BaselineEventBatch:
    """Convert PO-LEU :func:`build_choice_sets` records into a
    :class:`BaselineEventBatch`.

    Parameters
    ----------
    records
        Iterable of per-event dicts with the keys documented in
        :mod:`src.data.choice_sets` (``customer_id``, ``choice_asins``,
        ``chosen_idx``, ``alt_texts``, ``category``, ``metadata``, …).
    extra_feature_fn
        Optional callable ``(record, alt_idx) -> Mapping[str, float]``
        returning additional per-alt columns. Any key named in
        ``extra_feature_names`` is read from the returned mapping;
        missing keys default to 0.0.
    extra_feature_names
        Names of extra columns, appended to
        :data:`BUILTIN_FEATURE_NAMES` in the final
        ``base_feature_names``.

    Returns
    -------
    BaselineEventBatch
        With ``base_feature_names = BUILTIN_FEATURE_NAMES +
        tuple(extra_feature_names)``.

    Raises
    ------
    ValueError
        On empty records / mismatched J across events (baselines
        require uniform choice-set size).
    """
    records_list = list(records)
    if not records_list:
        raise ValueError("records_to_baseline_batch received an empty iterable")

    feature_names = list(BUILTIN_FEATURE_NAMES) + list(extra_feature_names)

    base_features_list: list[np.ndarray] = []
    chosen_indices: list[int] = []
    customer_ids: list[str] = []
    categories: list[str] = []
    metadata: list[dict] = []
    raw_events: list[dict] = []

    first_J: Optional[int] = None

    for i, rec in enumerate(records_list):
        mat = _build_feature_matrix(
            rec,
            extra_feature_fn=extra_feature_fn,
            extra_feature_names=extra_feature_names,
        )
        if first_J is None:
            first_J = mat.shape[0]
        elif mat.shape[0] != first_J:
            raise ValueError(
                f"record {i}: choice-set size {mat.shape[0]} != "
                f"{first_J} from record 0. Baselines require uniform J "
                "per batch."
            )

        base_features_list.append(mat)
        chosen_indices.append(int(rec["chosen_idx"]))
        customer_ids.append(str(rec["customer_id"]))
        categories.append(str(rec.get("category", "")))
        meta = dict(rec.get("metadata", {}) or {})
        # Ensure the breakdown keys evaluate.py reads are present.
        meta.setdefault("is_repeat", False)
        metadata.append(meta)
        raw_events.append(dict(rec))

    return BaselineEventBatch(
        base_features_list=base_features_list,
        base_feature_names=feature_names,
        chosen_indices=chosen_indices,
        customer_ids=customer_ids,
        categories=categories,
        metadata=metadata,
        raw_events=raw_events,
    )


__all__ = [
    "BUILTIN_FEATURE_NAMES",
    "records_to_baseline_batch",
]
