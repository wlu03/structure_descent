"""Per-(event, alt) geodistance for the mobility_boston dataset.

Closes the chosen-vs-negative price asymmetry identified in the leakage
audit. Previously, the chosen alternative's ``price`` field carried the
correct haversine geodistance for THIS event's (from_cbg, to_cbg) pair
while sampled negatives' ``price`` was looked up from a per-asin flat
table built from each ASIN's first-seen event — i.e. distance from some
other event's origin to the same destination. A model could exploit
that artifact to identify the chosen position.

This module provides a closure that ``build_choice_sets`` calls per
(event, alt). It overrides every alt's ``price`` field with
``haversine(event.from_cbg, alt.typical_to_cbg)`` using:

* ``centroids``: the (lon, lat) of every Boston CBG centroid, parsed
  from the basic-geographic-statistics CSV.
* ``asin_to_cbg``: the per-ASIN typical destination CBG, fit on TRAIN
  rows only so val/test events under cold-start don't leak.
* ``from_cbg_arr``: per-event from_cbg, indexed by the same integer
  position that build_choice_sets uses.

When either CBG is missing (e.g. ``home`` / ``work`` pseudo-places), the
closure returns the train-distance median as a sentinel — the same
fallback for chosen and negatives, preserving symmetry.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "haversine_km",
    "load_centroid_lookup",
    "make_per_event_alt_overrides_fn",
]


_EARTH_RADIUS_KM: float = 6371.0088
_POINT_RE: re.Pattern = re.compile(
    r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)"
)


def _normalize_cbg(value) -> str:
    """Stringify a CBG code and strip a trailing ``.0`` from float-inferred
    pandas columns. Centroid CSV keys come in as integer strings; the
    events frame's ``from_cbg`` / ``to_cbg`` arrive as float64 (CSV
    autotyping). Without this, the lookup misses every row.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        # Float CBG codes have no fractional part by construction.
        return str(int(value))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return s


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def load_centroid_lookup(path: Path) -> dict[str, tuple[float, float]]:
    """Return ``{cbg_code: (lon, lat)}`` parsed from the basic-stats CSV.

    Each CBG has one centroid (verified constant across years); we dedupe
    by keeping the first occurrence. Empty / malformed POINT strings are
    silently skipped — caller treats those as missing geodistance.
    Returns an empty dict when the file is missing.
    """
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, usecols=["CBG Code", "Centroid"])
    out: dict[str, tuple[float, float]] = {}
    for cbg_raw, point in zip(df["CBG Code"], df["Centroid"].astype(str)):
        cbg = _normalize_cbg(cbg_raw)
        if not cbg or cbg in out:
            continue
        m = _POINT_RE.search(point or "")
        if m is None:
            continue
        try:
            out[cbg] = (float(m.group(1)), float(m.group(2)))
        except ValueError:
            continue
    return out


def make_per_event_alt_overrides_fn(
    events_df: pd.DataFrame,
    centroid_path: Path,
    *,
    train_only: bool = True,
) -> Callable[[int, str], dict]:
    """Build a closure for ``build_choice_sets``'s ``per_event_alt_overrides_fn``.

    Parameters
    ----------
    events_df : pd.DataFrame
        The same frame ``build_choice_sets`` will iterate over. Must
        carry ``asin``, ``from_cbg``, ``to_cbg``, and (for the
        train-only fit) the ``split`` column.
    centroid_path : Path
        Path to ``Basic_Geographic_Statistics_CBG_Boston.csv``.
    train_only : bool, default True
        Fit the per-asin typical_to_cbg lookup on rows where
        ``split == "train"`` only. Default True; val/test rows transform
        under that fit (no leakage), matching the convention used by
        ``state_features.attach_train_popularity`` and the existing
        ``ref_df`` slicing in ``build_choice_sets``.

    Returns
    -------
    Callable[[int, str], dict]
        ``fn(event_idx, alt_asin)`` returns a dict to merge into the
        rendered ``alt_text``. Always carries a ``"price"`` key
        containing the haversine kilometers from the event's
        ``from_cbg`` to the alt's typical ``to_cbg``. Falls back to the
        train-distance median when either centroid is missing — the same
        fallback for chosen and negatives, so the path is symmetric.

    Notes
    -----
    The per-asin typical CBG is the mode of ``to_cbg`` across the
    chosen ASIN's training events. For real SafeGraph place IDs (asin
    starting with ``"sg:"``), the mode equals the actual to_cbg because
    each place has one location. For the synthetic ``"home"`` /
    ``"work"`` ASINs the mode is meaningless across customers; those
    cases mostly fall through to the median fallback (and remain
    symmetric chosen-vs-negative).
    """
    centroids = load_centroid_lookup(centroid_path)

    if train_only and "split" in events_df.columns:
        ref = events_df[events_df["split"] == "train"]
    else:
        ref = events_df

    has_ref_cbg = "to_cbg" in ref.columns and "asin" in ref.columns
    if has_ref_cbg:
        # Per-asin typical to_cbg = mode over the train rows where the
        # ASIN's to_cbg normalizes to a non-empty string.
        ref_norm = ref.assign(
            _to_cbg_norm=ref["to_cbg"].map(_normalize_cbg),
        )
        non_empty = ref_norm[ref_norm["_to_cbg_norm"].str.len() > 0]
        if not non_empty.empty:
            asin_to_cbg = (
                non_empty.groupby("asin")["_to_cbg_norm"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
                .astype(str)
                .to_dict()
            )
        else:
            asin_to_cbg = {}
    else:
        asin_to_cbg = {}

    # Per-event from_cbg, indexed by integer position in events_df.
    # Stringify with _normalize_cbg so float-typed CBG codes (CSV
    # autotyping) match the centroid keys (which are integer strings).
    if "from_cbg" in events_df.columns:
        from_cbg_arr = np.array(
            [_normalize_cbg(v) for v in events_df["from_cbg"].tolist()],
            dtype=object,
        )
    else:
        from_cbg_arr = np.array([""] * len(events_df), dtype=object)

    # Median distance over train rows whose centroids resolve — the
    # symmetric fallback for either-side missing CBG.
    train_distances: list[float] = []
    if "from_cbg" in ref.columns and "to_cbg" in ref.columns:
        for f, t in zip(ref["from_cbg"].tolist(), ref["to_cbg"].tolist()):
            fk, tk = _normalize_cbg(f), _normalize_cbg(t)
            a = centroids.get(fk)
            b = centroids.get(tk)
            if a is not None and b is not None:
                train_distances.append(
                    haversine_km(a[0], a[1], b[0], b[1])
                )
    median_km: float = (
        float(np.median(train_distances)) if train_distances else 0.0
    )

    def _override(event_idx: int, alt_asin: str) -> dict:
        if event_idx >= len(from_cbg_arr):
            return {"price": median_km}
        from_cbg = from_cbg_arr[event_idx]
        if not from_cbg:
            return {"price": median_km}
        to_cbg = asin_to_cbg.get(alt_asin, "")
        if not to_cbg:
            return {"price": median_km}
        a_pt = centroids.get(from_cbg)
        b_pt = centroids.get(to_cbg)
        if a_pt is None or b_pt is None:
            return {"price": median_km}
        return {
            "price": float(
                haversine_km(a_pt[0], a_pt[1], b_pt[0], b_pt[1])
            )
        }

    return _override
