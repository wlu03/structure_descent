"""Prepare Boston mobility-trajectory data for the PO-LEU pipeline.

Reads two JSONL files (one per year) of agent trajectories and emits two CSVs
that the pipeline can ingest:

- ``events.csv``  -- one row per *move* (the move's destination is the choice).
- ``persons.csv`` -- one row per agent (UNION of caids across the two years).

Run from the repo root:

    venv/bin/python scripts/prepare_mobility_dataset.py
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "mobility_trajectory_boston"
INPUT_FILES = [
    (2019, DATA_DIR / "veraset_boston_aligned_agents_2019.jsonl"),
    (2020, DATA_DIR / "veraset_boston_aligned_agents_2020.jsonl"),
]
EVENTS_CSV = DATA_DIR / "events.csv"
PERSONS_CSV = DATA_DIR / "persons.csv"
CBG_STATS_CSV = DATA_DIR / "Basic_Geographic_Statistics_CBG_Boston.csv"

# Earth radius in km — used for haversine geodistance between CBG centroids.
_EARTH_RADIUS_KM: float = 6371.0088

# Matches "POINT (-71.139 42.361)" — captures lon then lat.
_POINT_RE: re.Pattern = re.compile(
    r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)"
)

# NAICS-2 prefix -> human-readable industry label.
NAICS2_TO_INDUSTRY: dict[str, str] = {
    "11": "Agriculture",
    "21": "Mining",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation",
    "49": "Transportation",
    "51": "Information",
    "52": "Finance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management",
    "56": "Admin and Support",
    "61": "Education",
    "62": "Health Care",
    "71": "Arts and Entertainment",
    "72": "Food and Accommodation",
    "81": "Other Services",
    "92": "Public Administration",
}

UNKNOWN = "Unknown"

logger = logging.getLogger("prepare_mobility_dataset")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_missing(value) -> bool:
    """True for ``None``, NaN, empty strings, and the literal "None"/"nan"."""
    if value is None:
        return True
    if isinstance(value, float):
        try:
            import math

            if math.isnan(value):
                return True
        except Exception:
            pass
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"none", "nan", "null"}:
            return True
    return False


def naics_to_industry(naics_code) -> str:
    """Map a NAICS code to a human-readable industry label, else "Unknown"."""
    if _is_missing(naics_code):
        return UNKNOWN
    code = str(naics_code).strip()
    if not code:
        return UNKNOWN
    # NAICS-2 is the first two characters.
    prefix = code[:2]
    return NAICS2_TO_INDUSTRY.get(prefix, UNKNOWN)


def category_for_destination(to_place_id, to_naics_code) -> str:
    """Pick a category, special-casing the synthetic 'home' / 'work' places."""
    if isinstance(to_place_id, str):
        pid = to_place_id.strip().lower()
        if pid == "home":
            return "Home"
        if pid == "work":
            return "Workplace"
    return naics_to_industry(to_naics_code)


def make_place_label(to_place_id, category: str, to_cbg) -> str:
    """Build a human-readable place title for the destination."""
    if isinstance(to_place_id, str):
        pid = to_place_id.strip().lower()
        if pid == "home":
            return "Home"
        if pid == "work":
            return "Workplace"
    if _is_missing(to_cbg):
        return f"{category} place"
    return f"{category} place in CBG {to_cbg}"


def _to_str_or_empty(value) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def load_cbg_centroids(path: Path) -> dict[str, tuple[float, float]]:
    """Return ``{cbg_code: (lon, lat)}`` parsed from the basic-stats CSV.

    Each CBG has one centroid (verified: all years agree); we dedupe by
    keeping the first occurrence. CBGs with malformed POINT strings are
    silently skipped — caller treats those as missing geodistance.
    """
    if not path.exists():
        logger.warning(
            "CBG stats CSV not found at %s; geodistance prices unavailable.",
            path,
        )
        return {}
    df = pd.read_csv(path, usecols=["CBG Code", "Centroid"])
    centroids: dict[str, tuple[float, float]] = {}
    for cbg, point in zip(df["CBG Code"].astype(str), df["Centroid"].astype(str)):
        if cbg in centroids:
            continue
        m = _POINT_RE.search(point or "")
        if m is None:
            continue
        try:
            lon = float(m.group(1))
            lat = float(m.group(2))
        except ValueError:
            continue
        centroids[cbg] = (lon, lat)
    logger.info(
        "load_cbg_centroids: parsed %d unique CBG centroids from %s.",
        len(centroids),
        path,
    )
    return centroids


def haversine_km(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Great-circle distance in kilometers between two (lon, lat) points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def compute_distance_km(
    from_cbg, to_cbg, centroids: dict[str, tuple[float, float]]
) -> float | None:
    """Return haversine km between two CBG centroids, else ``None``.

    Returns ``None`` when either CBG is missing, malformed, or absent from
    the centroid lookup. Callers fold this into a per-event price column,
    median-imputing the Nones at write time so the downstream framework
    sees a non-NaN float.
    """
    if _is_missing(from_cbg) or _is_missing(to_cbg):
        return None
    a = centroids.get(str(from_cbg))
    b = centroids.get(str(to_cbg))
    if a is None or b is None:
        return None
    return haversine_km(a[0], a[1], b[0], b[1])


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_year(
    year: int,
    path: Path,
    centroids: dict[str, tuple[float, float]] | None = None,
):
    """Yield (event_row_dict, person_row_dict) tuples for one JSONL file.

    ``centroids`` is the optional CBG -> (lon, lat) lookup; when supplied
    each event row carries a ``distance_km`` column with the haversine
    distance from from_cbg to to_cbg (or ``NaN`` when either centroid is
    unavailable). The caller imputes the NaN before writing.

    Returns a tuple ``(event_rows, person_rows, stats)``.
    """
    event_rows: list[dict] = []
    person_rows: list[dict] = []
    skipped_empty_endpoints = 0
    naics_unknown_count = 0
    total_moves = 0
    distance_resolved = 0
    distance_missing = 0
    centroids = centroids or {}

    with path.open("r") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON on %s line %d: %s",
                    path.name,
                    line_no,
                    exc,
                )
                continue

            caid = rec.get("caid")
            if _is_missing(caid):
                continue
            caid = str(caid)

            home_demo = rec.get("home_demographic") or {}
            person_rows.append(
                {
                    "caid": caid,
                    "home_cbg": _to_str_or_empty(rec.get("home_cbg")),
                    "dominant_race": _to_str_or_empty(home_demo.get("dominant_race")),
                    "dominant_age_group": _to_str_or_empty(
                        home_demo.get("dominant_age_group")
                    ),
                    "dominant_industry": _to_str_or_empty(
                        home_demo.get("dominant_industry")
                    ),
                    "education_level_2019": _to_str_or_empty(
                        home_demo.get("education_level_2019")
                    ),
                    "income_level_3bin_2019": _to_str_or_empty(
                        home_demo.get("income_level_3bin_2019")
                    ),
                    "income_was_zero_2019": _to_str_or_empty(
                        home_demo.get("income_was_zero_2019")
                    ),
                    "year_seen": year,
                }
            )

            for move in rec.get("moves", []) or []:
                total_moves += 1
                from_pid = move.get("from_safegraph_place_id")
                to_pid = move.get("to_safegraph_place_id")
                if _is_missing(from_pid) and _is_missing(to_pid):
                    skipped_empty_endpoints += 1
                    continue

                to_naics = move.get("to_naics_code")
                category = category_for_destination(to_pid, to_naics)
                if category == UNKNOWN:
                    naics_unknown_count += 1

                to_cbg = move.get("to_cbg")
                place_label = make_place_label(to_pid, category, to_cbg)

                # Convert Unix-seconds local timestamp to ISO datetime string
                # so pandas `parse_dates` handles it trivially downstream.
                ts = move.get("to_local_timestamp")
                if _is_missing(ts):
                    order_date = ""
                else:
                    try:
                        order_date = (
                            pd.to_datetime(int(ts), unit="s")
                            .strftime("%Y-%m-%d %H:%M:%S")
                        )
                    except (TypeError, ValueError, OverflowError):
                        order_date = ""

                from_cbg = move.get("from_cbg")
                d_km = compute_distance_km(from_cbg, to_cbg, centroids)
                if d_km is None:
                    distance_missing += 1
                    distance_value = float("nan")
                else:
                    distance_resolved += 1
                    distance_value = float(d_km)

                event_rows.append(
                    {
                        "caid": caid,
                        "to_local_timestamp": order_date,
                        "to_place_id": _to_str_or_empty(to_pid),
                        "naics_industry": category,
                        # ``price`` carries the per-move cost-proxy. We keep
                        # the haversine geodistance (km) here so the
                        # framework's monotonicity prior reads "lower-cost
                        # preferred" as "shorter trip preferred" — directly
                        # relevant for mobility. NaN rows get median-
                        # imputed at write time so the schema stays float.
                        "price": distance_value,
                        "distance_km": distance_value,
                        "place_label": place_label,
                        "from_place_id": _to_str_or_empty(from_pid),
                        "from_cbg": _to_str_or_empty(from_cbg),
                        "to_cbg": _to_str_or_empty(to_cbg),
                        "year": year,
                    }
                )

    stats = {
        "year": year,
        "total_moves": total_moves,
        "skipped_empty_endpoints": skipped_empty_endpoints,
        "naics_unknown_count": naics_unknown_count,
        "n_event_rows": len(event_rows),
        "n_person_rows": len(person_rows),
        "distance_resolved": distance_resolved,
        "distance_missing": distance_missing,
    }
    return event_rows, person_rows, stats


def merge_persons(person_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union caids across years; prefer 2019 row when both exist."""
    combined = pd.concat(person_frames, ignore_index=True)
    # `year_first_seen` = earliest year per caid.
    first_seen = combined.groupby("caid", as_index=False)["year_seen"].min()
    first_seen = first_seen.rename(columns={"year_seen": "year_first_seen"})

    # Prefer 2019 row when both exist; otherwise use 2020.
    combined_sorted = combined.sort_values(
        by=["caid", "year_seen"], ascending=[True, True]
    )
    deduped = combined_sorted.drop_duplicates(subset="caid", keep="first")
    merged = deduped.merge(first_seen, on="caid", how="left")

    output_cols = [
        "caid",
        "home_cbg",
        "dominant_race",
        "dominant_age_group",
        "dominant_industry",
        "education_level_2019",
        "income_level_3bin_2019",
        "income_was_zero_2019",
        "year_first_seen",
    ]
    return merged[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    all_event_rows: list[dict] = []
    person_frames: list[pd.DataFrame] = []
    per_year_stats: list[dict] = []

    centroids = load_cbg_centroids(CBG_STATS_CSV)

    for year, path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        logger.info("Reading %s", path)
        ev_rows, p_rows, stats = extract_year(year, path, centroids=centroids)
        per_year_stats.append(stats)
        all_event_rows.extend(ev_rows)
        person_frames.append(pd.DataFrame(p_rows))
        logger.info(
            "year=%d total_moves=%d emitted_events=%d skipped_empty=%d unknown_industry=%d agents=%d",
            stats["year"],
            stats["total_moves"],
            stats["n_event_rows"],
            stats["skipped_empty_endpoints"],
            stats["naics_unknown_count"],
            stats["n_person_rows"],
        )

    events_df = pd.DataFrame(
        all_event_rows,
        columns=[
            "caid",
            "to_local_timestamp",
            "to_place_id",
            "naics_industry",
            "price",
            "distance_km",
            "place_label",
            "from_place_id",
            "from_cbg",
            "to_cbg",
            "year",
        ],
    )

    # Median-impute missing geodistance so ``price`` stays a non-NaN
    # float column (the framework's clean / dropna-subset / dtype-coerce
    # path already runs, but it operates on canonical names — keeping
    # ``price`` numeric here avoids invariant churn). When every row is
    # missing (e.g. a stripped fixture without the centroid CSV), fall
    # back to 0.0 so the column is at least well-defined.
    valid = events_df["distance_km"].dropna()
    median_km = float(valid.median()) if not valid.empty else 0.0
    n_imputed = int(events_df["distance_km"].isna().sum())
    events_df["price"] = events_df["price"].fillna(median_km)
    events_df["distance_km"] = events_df["distance_km"].fillna(median_km)
    if n_imputed:
        logger.info(
            "median-imputed %d/%d events with missing geodistance "
            "(median = %.3f km).",
            n_imputed, len(events_df), median_km,
        )
    distance_summary = {
        "n_with_distance": int(len(events_df) - n_imputed),
        "n_imputed": n_imputed,
        "median_km": median_km,
        "distance_min_km": float(events_df["distance_km"].min()),
        "distance_max_km": float(events_df["distance_km"].max()),
        "distance_mean_km": float(events_df["distance_km"].mean()),
    }

    persons_df = merge_persons(person_frames)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(EVENTS_CSV, index=False)
    persons_df.to_csv(PERSONS_CSV, index=False)

    # ---- Summary to stdout -------------------------------------------------
    industry_counts = events_df["naics_industry"].value_counts(dropna=False)
    n_unknown = int(industry_counts.get(UNKNOWN, 0))
    n_distinct_industries = int(events_df["naics_industry"].nunique(dropna=False))

    print("=" * 72)
    print("PO-LEU mobility preprocessing summary")
    print("=" * 72)
    print(f"events.csv  -> {EVENTS_CSV}")
    print(f"persons.csv -> {PERSONS_CSV}")
    print()
    for stats in per_year_stats:
        print(
            f"  year={stats['year']}  moves={stats['total_moves']:,}  "
            f"emitted={stats['n_event_rows']:,}  "
            f"skipped_empty={stats['skipped_empty_endpoints']:,}  "
            f"unknown_industry={stats['naics_unknown_count']:,}  "
            f"agents={stats['n_person_rows']:,}"
        )
    print()
    print(f"events.csv rows                  : {len(events_df):,}")
    print(f"persons.csv rows                 : {len(persons_df):,}")
    print(f"unique caids in events.csv       : {events_df['caid'].nunique():,}")
    print(f"unique caids in persons.csv      : {persons_df['caid'].nunique():,}")
    print(f"distinct industries in events.csv: {n_distinct_industries}")
    print(f"events.csv rows with industry='Unknown': {n_unknown:,}")
    print()
    print("Geodistance (price) summary:")
    print(f"  events with resolved distance   : {distance_summary['n_with_distance']:,}")
    print(f"  events median-imputed           : {distance_summary['n_imputed']:,}")
    print(f"  median km                       : {distance_summary['median_km']:.3f}")
    print(f"  min / max / mean (km)           : "
          f"{distance_summary['distance_min_km']:.3f} / "
          f"{distance_summary['distance_max_km']:.3f} / "
          f"{distance_summary['distance_mean_km']:.3f}")
    print()
    print("Industry distribution (events.csv):")
    for label, count in industry_counts.items():
        print(f"  {label:<28s} {int(count):>10,d}")
    print("=" * 72)


if __name__ == "__main__":
    main()
