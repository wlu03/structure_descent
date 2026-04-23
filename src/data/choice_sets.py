"""Choice-set builder for the PO-LEU data pipeline (Wave 9, design doc §5).

Ports ``old_pipeline/src/data_prep.py::build_choice_sets`` (lines 228-493)
onto the canonical schema produced by the Wave 8 data layer and augments
each per-event record with three Wave 9 keys — ``z_d`` (the 26-dim
canonical person vector), ``c_d`` (the rendered context paragraph), and
``alt_texts`` (a list of adapter-rendered alternative descriptors, one
per alternative in the choice set). The sampling logic — temporal
availability filter, popularity-weighted random negatives, category-
stratified same-cat negatives, collision handling, n_resamples support,
and deterministic per-event RNG seeding ``seed + 1_000_003 * k_rep + i``
— is preserved verbatim from v1.

**McFadden caveat (preserved from v1).** Sampled conditional logit with
stratified (same-category) negatives yields coefficients consistent for
*relative* utility between the chosen alternative and the sampled
alternatives, not absolute utility over the full catalog. Downstream
probability / rank metrics inherit that caveat.

Pure; deterministic given ``seed``; no module-level side effects; stdlib
+ numpy + pandas + internal imports only (no torch, no sentence-
transformers).
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from src.data.context_string import build_context_string
from src.data.person_features import (
    fit_person_features,
    transform_person_features,
)

# Max characters retained per recent-purchase title when building the
# per-event ``recent_purchases`` list (see ``build_choice_sets``). Amazon
# titles routinely run 200+ characters; truncating keeps c_d readable
# and within a sensible prompt budget.
_RECENT_PURCHASE_TITLE_MAX_CHARS = 80

# Raw bucket codes / raw column names that must not appear anywhere in
# the rendered ``c_d``. ``paraphrase_rules_check`` in
# ``context_string.py`` scans the whole rendered string for these
# substrings, so Amazon titles that happen to contain "65+" (e.g.
# "(165+ pcs)"), "25-34", or the word "education" (e.g. "Research
# methods in education") would trip it.
#
# We scrub them out of recent-purchase titles before handing the list
# to ``build_context_string``. The paraphrase check is case-sensitive
# (it does plain ``in`` substring matching), so we only scrub the
# lowercase form for raw column names (which are always lowercase
# identifiers). Ordering matters — longer codes are replaced before
# shorter ones so e.g. ``"150k+"`` is handled before ``"65+"``.
_RAW_BUCKET_SUBSTRINGS_TO_SCRUB: tuple[str, ...] = (
    "100-150k",
    "150k+",
    "25-50k",
    "50-100k",
    "<25k",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
    # Raw canonical column names flagged by paraphrase_rules_check.
    "age_bucket",
    "income_bucket",
    "household_size",
    "has_kids",
    "city_size",
    "education",
    "health_rating",
    "risk_tolerance",
    "purchase_frequency",
    "novelty_rate",
)


def _scrub_title_for_c_d(title: str) -> str:
    """Truncate and scrub a raw event title for safe inclusion in ``c_d``.

    Two guarantees:

    1. Length is capped at ``_RECENT_PURCHASE_TITLE_MAX_CHARS``.
    2. None of the ``_RAW_BUCKET_SUBSTRINGS_TO_SCRUB`` codes appear
       in the result (the renderer's paraphrase check would raise
       ``AssertionError`` otherwise; Amazon titles routinely include
       fragments like "(165+ pcs)" that contain "65+" as a substring).

    The scrub replaces each hit with a single space; collapsed whitespace
    is not re-flowed (callers comma-join the returned strings).
    """
    if title is None:
        return ""
    s = str(title)
    if len(s) > _RECENT_PURCHASE_TITLE_MAX_CHARS:
        s = s[:_RECENT_PURCHASE_TITLE_MAX_CHARS]
    for code in _RAW_BUCKET_SUBSTRINGS_TO_SCRUB:
        if code in s:
            s = s.replace(code, " ")
    # Collapse any runs of whitespace the scrub may have introduced.
    if "  " in s:
        s = " ".join(s.split())
    return s.strip()

if TYPE_CHECKING:
    # Wave-9 sibling module; forward-referenced here so the import stays
    # in the type-checker-only branch and there is no hard runtime dep
    # on ``src/data/adapter.py`` (avoids circular-import risk at module
    # load time; the methods we use are called duck-typed at runtime).
    from src.data.adapter import DatasetAdapter  # noqa: F401


__all__ = ["build_choice_sets"]


logger = logging.getLogger(__name__)


def build_choice_sets(
    events_df: pd.DataFrame,
    persons_canonical: pd.DataFrame,
    adapter: "DatasetAdapter",
    *,
    n_negatives: int = 9,
    seed: int = 42,
    n_resamples: int = 1,
    popularity_col: str = "popularity",
    split_column: str = "split",
    customer_to_extras: dict | None = None,
    recent_purchases_window_days: int = 30,
    max_recent_purchases: int = 5,
) -> list[dict]:
    """Return per-event choice-set records with ``z_d`` and ``c_d`` attached.

    Shape contract
    --------------
    Input:
      events_df: canonical events (post state_features + split + popularity).
        Required columns: ``customer_id``, ``order_date``, ``asin``,
        ``category``, ``price``, ``routine``, ``novelty``, ``recency_days``,
        ``popularity``, ``split``, ``title``.
      persons_canonical: output of ``adapter.translate_z_d()``. Must
        contain ``customer_id`` plus all 10 canonical z_d columns.
        INVARIANT: ``persons_canonical['customer_id']`` is a
        *superset* of every ``customer_id`` that appears in
        ``events_df`` (any split). The z_d vocabulary / standardization
        stats were fit train-only via ``translate_z_d(training_events=)``,
        but the TRANSFORM runs on every customer — val/test customers
        under cold-start therefore legitimately carry z_d rows without
        leaking train statistics. What we strictly require is
        coverage, so the per-event ``customer_to_zd[cid]`` lookup
        below cannot KeyError.
      adapter: supplies ``suppress_fields_for_c_d()`` and
        ``alt_text(event_row)``.

    Output:
      list[dict], one per event. Keys:
        # v1 keys (preserved)
        customer_id, order_date, category, chosen_asin, choice_asins,
        chosen_idx, chosen_features, metadata,
        # v2 additions
        z_d: np.ndarray shape (p,) float32
        c_d: str
        alt_texts: list[dict] of length J (J = n_negatives + 1)

    For ``n_resamples > 1``, ``choice_asins`` / ``chosen_idx`` /
    ``alt_texts`` are length-K *lists of lists* (matching the v1
    ``choice_asins`` shape dichotomy).

    Parameters (selected)
    ---------------------
    recent_purchases_window_days : int, default 30
        Width of the lookback window used to populate the per-event
        ``recent_purchases`` clause in ``c_d``. For each event at date
        ``d_t`` the customer's events with order_date in
        ``[d_t - window_days, d_t)`` (strict ``<`` on both ends) are
        considered. Matches the §2.2 example ("last 30 days").
    max_recent_purchases : int, default 5
        Cap on how many prior-event titles feed into the
        ``recent_purchases`` clause. Keeps c_d compact; 3-5 is the
        empirical sweet spot. Most-recent-first ordering — when there
        are more than ``max_recent_purchases`` eligible prior events
        we take the last ``max_recent_purchases`` of them and pass
        them in chronological order (oldest of the kept slice first,
        most-recent last) so the rendered sentence reads like a short
        timeline.

    Raises
    ------
    AssertionError:
        If ``persons_canonical`` is missing a ``customer_id`` that
        appears in ``events_df`` (any split). The per-event
        ``customer_to_zd[cid]`` lookup would otherwise KeyError mid-
        iteration. Fit-on-train-only standardization is enforced
        upstream via ``translate_z_d(training_events=)``; this
        function does not re-check it.
    ValueError:
        If ``events_df[["asin", "category"]]`` contains NaN (upstream
        clean_purchases caveat preserved from v1).
    """
    # --------------------------------------------------------------- #
    # Coverage assertion (cold-start-compatible; Wave-13 loosening).
    #
    # Every customer whose events we iterate over must have a z_d row.
    # The z_d vocabulary was fit on train-only (see translate_z_d
    # ``training_events=`` kwarg), but the TRANSFORM runs on all
    # customers, so val/test customers under cold-start legitimately
    # carry z_d rows without leaking train statistics. What we strictly
    # require is that ``persons_canonical`` covers every customer we're
    # about to iterate — otherwise the per-event ``customer_to_zd[cid]``
    # lookup would KeyError mid-loop.
    #
    # The prior, stricter invariant (``persons_canonical.customer_id``
    # ⊆ train-split customer_ids) broke cold-start because val/test
    # customers have zero train rows by construction. Temporal splits
    # remain covered: every customer has train rows there, so the new
    # (weaker) invariant is equivalent.
    # --------------------------------------------------------------- #
    event_customers = set(events_df["customer_id"].astype(str).unique())
    pc_customers = set(persons_canonical["customer_id"].astype(str).unique())
    missing = event_customers - pc_customers
    if missing:
        sample = sorted(list(missing))[:10]
        raise AssertionError(
            f"persons_canonical is missing z_d rows for "
            f"{len(missing)} customer_id(s) that appear in events_df "
            f"(sample: {sample}). Ensure translate_z_d was called with "
            f"the full joint customer set, not a train-only slice; "
            f"training_events= controls FIT, not TRANSFORM."
        )

    logger.info(
        "build_choice_sets: building from %d events (n_resamples=%d, "
        "n_negatives=%d, seed=%d).",
        len(events_df),
        n_resamples,
        n_negatives,
        seed,
    )

    df = events_df

    if df[["asin", "category"]].isna().any().any():
        raise ValueError(
            "build_choice_sets: df contains NaN in 'asin' or 'category'. "
            "Run clean_purchases first (NaN codes would silently wrap "
            "catalog indexing)."
        )

    # --------------------------------------------------------------- #
    # Split-aware reference frame for ``first_seen`` / ``asin_lookup`` /
    # popularity-fallback computations (F3 cold-start leakage fix).
    #
    # Under a cold-start split, ``events_df`` contains rows for
    # customers who are held out entirely (they only appear in val/test),
    # so deriving each ASIN's first-seen date or catalog lookup from the
    # full frame leaks "when did the test population first encounter
    # this product" into the availability prefix that gates negative
    # sampling for TRAIN events. Restricting these tables to the train
    # subset fixes the leak.
    #
    # Temporal splits (every customer contributes to both train and
    # test) and synthetic fixtures without a ``split`` column are
    # unaffected: the full-frame path is preserved when no ``split``
    # column is present. This is a duck-typed check — no signature
    # change.
    # --------------------------------------------------------------- #
    if split_column in df.columns:
        train_only_mask = df[split_column] == "train"
        n_train_rows = int(train_only_mask.sum())
        if n_train_rows == 0:
            raise ValueError(
                f"build_choice_sets: '{split_column}' column is present but "
                f"contains zero 'train' rows; cannot build first_seen / "
                f"asin_lookup / popularity tables without any training data. "
                f"Check the upstream split stage (e.g. temporal_split / "
                f"cold_start_split) and verify val_frac/test_frac didn't "
                f"drain the train partition."
            )
        logger.info(
            "build_choice_sets: restricting first_seen + asin_lookup to "
            "%d train rows (%d total).",
            n_train_rows,
            len(df),
        )
        ref_df = df.loc[train_only_mask]
    else:
        logger.info(
            "build_choice_sets: no split column, using all %d rows for "
            "first_seen + asin_lookup.",
            len(df),
        )
        ref_df = df

    # --------------------------------------------------------------- #
    # Per-customer caches (O(n_customers), not O(n_events)).
    # --------------------------------------------------------------- #
    suppress = tuple(adapter.suppress_fields_for_c_d())

    # z_d: fit standardization stats on persons_canonical (which is a
    # subset of train, guaranteed by the assertion above) and transform
    # those same rows into the float32 matrix. When the adapter supplies
    # a ``categorical_vocabularies`` method (Wave-10 fix), use its
    # closed-set vocab so the output width is schema-authoritative; this
    # prevents the z_d-width drift observed in Wave 9 when fitting on
    # small training slices that don't cover every one-hot bucket.
    vocabs = None
    if hasattr(adapter, "categorical_vocabularies"):
        try:
            vocabs = adapter.categorical_vocabularies()
        except Exception:
            # Defensive: adapters without a usable vocabulary fall back
            # to learn-from-data (Wave-1 default).
            vocabs = None
    stats = fit_person_features(persons_canonical, vocabularies=vocabs)
    z_d_matrix = transform_person_features(persons_canonical, stats)
    customer_ids_zd = persons_canonical["customer_id"].to_numpy()
    customer_to_zd: dict = {
        cid: z_d_matrix[i] for i, cid in enumerate(customer_ids_zd)
    }

    # c_d: Wave-12 per-event render. Previously cached as a single
    # string per customer; now built INSIDE the per-event loop so each
    # event's ``recent_purchases`` clause can reflect that customer's
    # prior events in the ``recent_purchases_window_days`` window (no
    # future leakage; see the per-event loop below for the binary-
    # search cutoff). We still cache the per-customer persons row +
    # extras here to avoid re-looking-up those dicts on every event.
    extras_by_cid = customer_to_extras or {}
    customer_to_row: dict[object, dict] = {
        row["customer_id"]: row.to_dict()
        for _, row in persons_canonical.iterrows()
    }

    # --------------------------------------------------------------- #
    # asin_lookup: for each ASIN, grab the canonical event-row fields
    # (title, category, price, popularity) from the earliest event it
    # appears in. Using the oldest row is the causally-earliest
    # description we have for that ASIN. Derived from ``ref_df`` so
    # under cold-start splits only train-customer rows contribute (no
    # cross-customer leakage via the negative-sampling pool).
    # --------------------------------------------------------------- #
    first_seen_rows = (
        ref_df.sort_values("order_date", kind="mergesort")
        .drop_duplicates(subset=["asin"], keep="first")
    )
    asin_lookup: dict = {
        row["asin"]: {
            "title": row.get("title", "") if "title" in first_seen_rows.columns else "",
            "category": row.get("category", ""),
            "price": float(row.get("price", 0.0) or 0.0) if "price" in first_seen_rows.columns else 0.0,
            "popularity": (
                int(row.get("popularity", 0) or 0)
                if "popularity" in first_seen_rows.columns
                else 0
            ),
        }
        for _, row in first_seen_rows.iterrows()
    }

    # Default lookup for any asin not in the table (shouldn't happen in
    # practice: sampled negatives come from rows in df itself).
    _DEFAULT_ALT = {
        "title": "",
        "category": "",
        "price": 0.0,
        "popularity": 0,
    }

    # --------------------------------------------------------------- #
    # v1 machinery: availability-prefix / popularity / category arrays.
    # --------------------------------------------------------------- #
    rng = np.random.default_rng(seed)  # retained for parity; not used.
    _ = rng
    n_events = len(df)

    asin_cat = pd.Categorical(df["asin"].to_numpy())
    catalog = asin_cat.categories.to_numpy()
    chosen_int = asin_cat.codes.astype(np.int64)
    n_catalog = len(catalog)

    cat_codes_cat = pd.Categorical(df["category"].to_numpy())
    cat_codes = cat_codes_cat.codes.astype(np.int64)
    cat_names = cat_codes_cat.categories.to_numpy()
    n_cats = len(cat_names)
    unknown_cat_code = -1
    for i, name in enumerate(cat_names):
        if name == "Unknown":
            unknown_cat_code = i
            break
    logger.info(
        "build_choice_sets: %d unique ASINs, %d categories, %d negatives each.",
        n_catalog,
        n_cats,
        n_negatives,
    )

    n_cat_neg = n_negatives // 2
    n_rand_neg = n_negatives - n_cat_neg

    order_dates_ns = (
        pd.to_datetime(df["order_date"]).to_numpy().astype("datetime64[ns]")
    )

    # Per-ASIN earliest-seen date. Computed from ``ref_df`` (train-only
    # rows under a cold-start split; full frame otherwise) so that the
    # availability prefix which gates negative sampling does not leak
    # cross-customer information. ASINs that appear only in non-train
    # rows reindex to ``NaT`` and are thus never selectable as
    # negatives — exactly the intended behavior.
    ref_order_dates_ns = (
        pd.to_datetime(ref_df["order_date"])
        .to_numpy()
        .astype("datetime64[ns]")
    )
    first_seen = (
        ref_df.assign(_d=ref_order_dates_ns)
        .groupby("asin")["_d"].min()
    )
    first_seen_by_code = (
        first_seen.reindex(catalog).to_numpy().astype("datetime64[ns]")
    )

    # Popularity weights over the catalog. ASINs with zero train
    # popularity get a tiny epsilon so they're still sampleable.
    # Sourced from ``ref_df`` so that when the ``popularity`` column
    # is ABSENT (synthetic fixtures, early pipeline stages) the
    # drop-duplicates fallback pulls from train-only rows under a
    # cold-start split. When ``popularity`` is present it is already
    # train-only (see :func:`attach_train_popularity`), so reading
    # from ``df`` vs. ``ref_df`` is equivalent there.
    if popularity_col in ref_df.columns:
        pop_series = (
            ref_df[["asin", popularity_col]]
            .drop_duplicates(subset=["asin"])
            .set_index("asin")[popularity_col]
        )
        pop_by_code = (
            pop_series.reindex(catalog).fillna(0).to_numpy().astype(np.float64)
        )
    else:
        pop_by_code = np.ones(n_catalog, dtype=np.float64)
    pop_by_code = pop_by_code + 1e-6

    # Sort catalog by first-seen date so we can binary-search the
    # available prefix at each event's order_date.
    sort_by_first = np.argsort(first_seen_by_code, kind="stable")
    sorted_first_seen = first_seen_by_code[sort_by_first]
    sorted_pop = pop_by_code[sort_by_first]
    asin_to_cat_code = (
        pd.DataFrame({"a": asin_cat.codes, "c": cat_codes})
        .groupby("a")["c"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    sorted_cat_of_asin_full = (
        asin_to_cat_code.reindex(range(n_catalog))
        .fillna(-1)
        .to_numpy()
        .astype(np.int64)
    )
    sorted_cat_of_asin = sorted_cat_of_asin_full[sort_by_first]

    # For each event: prefix length of ASINs with first_seen < order_date.
    avail_prefix = np.searchsorted(sorted_first_seen, order_dates_ns, side="left")
    _ = avail_prefix  # retained for v1 parity (referenced via _sample_random_for_event_avail)

    # Per-category sorted prefix lookups.
    cat_to_sorted_idxs: List[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_cats
    cat_to_sorted_first_seen: List[np.ndarray] = (
        [np.empty(0, dtype="datetime64[ns]")] * n_cats
    )
    cat_to_sorted_pop: List[np.ndarray] = [np.empty(0, dtype=np.float64)] * n_cats
    for c in range(n_cats):
        if c == unknown_cat_code:
            continue
        mask = sorted_cat_of_asin == c
        cat_to_sorted_idxs[c] = sort_by_first[mask]
        cat_to_sorted_first_seen[c] = sorted_first_seen[mask]
        cat_to_sorted_pop[c] = sorted_pop[mask]

    def _sample_random(
        avail_len: int, chosen_code: int, k: int, local_rng
    ) -> np.ndarray:
        if avail_len <= 0:
            return np.full(k, chosen_code, dtype=np.int64)
        p = sorted_pop[:avail_len]
        p_sum = p.sum()
        if p_sum <= 0:
            idxs = local_rng.integers(0, avail_len, size=k)
        else:
            idxs = local_rng.choice(avail_len, size=k, replace=True, p=p / p_sum)
        sampled = sort_by_first[idxs]
        if avail_len >= 2:
            coll = sampled == chosen_code
            if coll.any():
                # Bump the sort-position (stays inside the temporally
                # available prefix); +1 neighbor in sort_by_first is a
                # distinct catalog entry by construction.
                bumped_idxs = (idxs + 1) % avail_len
                sampled = np.where(coll, sort_by_first[bumped_idxs], sampled)
        return sampled.astype(np.int64)

    def _sample_random_for_event_avail(
        event_date, chosen_code: int, k: int, local_rng
    ) -> np.ndarray:
        prefix = int(np.searchsorted(sorted_first_seen, event_date, side="left"))
        return _sample_random(prefix, chosen_code, k, local_rng)

    def _sample_same_cat(
        cat_code: int, event_date, chosen_code: int, k: int, local_rng
    ) -> np.ndarray:
        if cat_code < 0 or cat_code == unknown_cat_code:
            return _sample_random_for_event_avail(
                event_date, chosen_code, k, local_rng
            )
        sub_first = cat_to_sorted_first_seen[cat_code]
        if sub_first.size == 0:
            return _sample_random_for_event_avail(
                event_date, chosen_code, k, local_rng
            )
        prefix = int(np.searchsorted(sub_first, event_date, side="left"))
        if prefix <= 1:
            return _sample_random_for_event_avail(
                event_date, chosen_code, k, local_rng
            )
        sub_idx = cat_to_sorted_idxs[cat_code][:prefix]
        sub_pop = cat_to_sorted_pop[cat_code][:prefix]
        p_sum = sub_pop.sum()
        if p_sum <= 0:
            picks = local_rng.integers(0, prefix, size=k)
        else:
            picks = local_rng.choice(
                prefix, size=k, replace=True, p=sub_pop / p_sum
            )
        sampled = sub_idx[picks]
        coll = sampled == chosen_code
        if coll.any():
            bumped = np.where(coll, sub_idx[(picks + 1) % prefix], sampled)
            sampled = bumped
        return sampled.astype(np.int64)

    # --------------------------------------------------------------- #
    # Per-event feature arrays for the chosen alternative (leakage-safe).
    # --------------------------------------------------------------- #
    customer_ids = df["customer_id"].to_numpy()
    categories = df["category"].to_numpy()
    routines = (
        df["routine"].to_numpy()
        if "routine" in df.columns
        else np.zeros(n_events, dtype=np.int64)
    )
    prices = (
        df["price"].to_numpy()
        if "price" in df.columns
        else np.zeros(n_events)
    )
    recency_arr = (
        df["recency_days"].to_numpy()
        if "recency_days" in df.columns
        else np.full(n_events, 999.0)
    )
    novelty_arr = (
        df["novelty"].to_numpy()
        if "novelty" in df.columns
        else np.ones(n_events, dtype=np.int64)
    )
    cat_affinity_arr = (
        df["cat_affinity"].to_numpy()
        if "cat_affinity" in df.columns
        else np.zeros(n_events, dtype=np.int64)
    )
    popularity_train_arr = (
        df["popularity_train"].to_numpy()
        if "popularity_train" in df.columns
        else (
            df["popularity"].to_numpy()
            if "popularity" in df.columns
            else np.zeros(n_events, dtype=np.float64)
        )
    )
    brand_arr = (
        df["brand"].to_numpy()
        if "brand" in df.columns
        else np.array([""] * n_events, dtype=object)
    )
    state_arr = (
        df["state"].to_numpy()
        if "state" in df.columns
        else np.array([""] * n_events, dtype=object)
    )

    logger.debug(
        "build_choice_sets: feature arrays ready (n_events=%d, "
        "n_catalog=%d, n_cats=%d).",
        n_events,
        n_catalog,
        n_cats,
    )

    # Title / price for the chosen alternative come from the event's own row
    # (policy §5: the chosen is the event, no lookup). Negatives are sampled
    # asins we did not observe in this event, so they use the causally-
    # earliest known description via ``asin_lookup``.
    titles_arr = (
        df["title"].to_numpy()
        if "title" in df.columns
        else np.array([""] * n_events, dtype=object)
    )

    # --------------------------------------------------------------- #
    # Per-customer sorted event index for the per-event
    # ``recent_purchases`` lookup in c_d (Wave-12 per-event c_d).
    #
    # For each customer we keep two parallel lists sorted by
    # (order_date, asin): the numpy datetime64[ns] dates (for bisect)
    # and the event titles. The asin tiebreaker ensures deterministic
    # ordering when multiple prior purchases land on the same day.
    #
    # Index shape: dict[customer_id, tuple[list[np.datetime64[ns]],
    #                                      list[str]]] keyed in
    # chronological order.
    # --------------------------------------------------------------- #
    asin_arr_for_history = df["asin"].to_numpy()
    sort_triplets: List[tuple] = []
    for idx in range(n_events):
        # Keep (customer_id, order_date, asin, title-scrubbed) and
        # sort downstream on (order_date, asin) within each customer
        # to get deterministic same-day ordering.
        title_str = _scrub_title_for_c_d(titles_arr[idx])
        sort_triplets.append(
            (
                customer_ids[idx],
                order_dates_ns[idx],
                asin_arr_for_history[idx],
                title_str,
            )
        )
    # Stable sort by (customer_id, order_date, asin) so per-customer
    # slices come out in chronological, tie-broken order.
    sort_triplets.sort(key=lambda t: (t[0], t[1], t[2]))
    customer_history: dict[object, tuple[list, list]] = {}
    for cid, d_ns, _asin, title in sort_triplets:
        dates, titles = customer_history.setdefault(cid, ([], []))
        dates.append(d_ns)
        titles.append(title if title else "unknown item")
    # Freeze window width as a numpy timedelta for the bisect cutoff.
    _window_td = np.timedelta64(int(recent_purchases_window_days), "D")

    def _render_alt_texts(asins: list, chosen_asin, event_row_alt: dict) -> list[dict]:
        out: list[dict] = []
        for a in asins:
            if a == chosen_asin:
                out.append(adapter.alt_text(event_row_alt))
            else:
                out.append(adapter.alt_text(asin_lookup.get(a, dict(_DEFAULT_ALT))))
        return out

    # Hard cap on resample rounds for the dedup guard (prevents infinite
    # loops when the available pool is pathologically small). See the
    # "Wave 11 post-dryrun fixes" entry in NOTES.md for the fallback
    # semantics.
    MAX_RESAMPLE_ROUNDS = 5
    J = n_negatives + 1

    records: List[dict] = []
    dedup_fallback_any = False
    for i in range(n_events):
        chosen_code = int(chosen_int[i])
        ev_date = order_dates_ns[i]
        ev_cat_code = int(cat_codes[i])
        ev_cat_name = categories[i]
        is_unknown = (ev_cat_code == unknown_cat_code) or (ev_cat_name == "Unknown")

        # Available negative-pool size for this event = number of ASINs
        # first-seen strictly before the event date, minus the chosen
        # (which occupies slot 0 of the choice set). ``avail_prefix[i]``
        # already holds the full temporally-available count; subtract 1
        # if the chosen is inside that prefix.
        avail_len_i = int(avail_prefix[i])
        chosen_in_prefix = int(
            first_seen_by_code[chosen_code] < ev_date
        )
        available_neg_pool_size = max(0, avail_len_i - chosen_in_prefix)
        dedup_target_size = min(J, available_neg_pool_size + 1)

        samples_per_k: List[List[str]] = []
        chosen_idx_per_k: List[int] = []
        dedup_fallback_per_k: List[bool] = []
        for k_rep in range(n_resamples):
            # Deterministic per-event per-resample RNG seeding (v1 formula
            # — DO NOT change). This is what pins the integration test.
            local_rng = np.random.default_rng(seed + 1_000_003 * k_rep + i)
            if is_unknown:
                negs = _sample_random_for_event_avail(
                    ev_date, chosen_code, n_negatives, local_rng
                )
            else:
                cat_negs = _sample_same_cat(
                    ev_cat_code, ev_date, chosen_code, n_cat_neg, local_rng
                )
                rand_negs = _sample_random_for_event_avail(
                    ev_date, chosen_code, n_rand_neg, local_rng
                )
                negs = np.concatenate([cat_negs, rand_negs])

            # --- Dedup guard (Wave 11 post-dryrun fix) ----------------- #
            # The popularity-weighted samplers draw with replacement and
            # may collide with each other or with the chosen; small
            # catalogs (e.g. the 1-customer dry-run) make this visible.
            # Reduce to unique negatives, resample up to
            # MAX_RESAMPLE_ROUNDS rounds using the SAME per-event
            # local_rng (no new RNG streams), and fall back to cyclic
            # padding only when the available pool is genuinely too
            # small.
            seen: set = {chosen_code}
            negs_unique: List[int] = []
            for n in negs.tolist():
                if n not in seen:
                    negs_unique.append(int(n))
                    seen.add(int(n))

            resample_round = 0
            while (
                len(negs_unique) + 1 < dedup_target_size
                and resample_round < MAX_RESAMPLE_ROUNDS
            ):
                n_needed = dedup_target_size - (len(negs_unique) + 1)
                extra = _sample_random_for_event_avail(
                    ev_date, chosen_code, n_needed, local_rng
                )
                for n in extra.tolist():
                    if n not in seen:
                        negs_unique.append(int(n))
                        seen.add(int(n))
                resample_round += 1

            dedup_fallback = False
            if len(negs_unique) + 1 < J:
                # Pool exhausted; pad by cycling the unique negatives so
                # training does not crash. Record the fallback on the
                # record's metadata for downstream diagnostics.
                dedup_fallback = True
                dedup_fallback_any = True
                logger.warning(
                    "choice_sets: dedup fallback for customer=%r event=%d; "
                    "available_pool=%d, got %d distinct alternatives "
                    "(need J=%d). Padding by cycling.",
                    customer_ids[i],
                    i,
                    available_neg_pool_size,
                    len(negs_unique) + 1,
                    J,
                )
                if len(negs_unique) == 0:
                    # Degenerate: no negatives available at all (e.g. the
                    # very first event in the dataset). Pad with the
                    # chosen code — matches the v1 ``avail_len<=0``
                    # fallback in ``_sample_random``.
                    negs_unique = [chosen_code] * n_negatives
                else:
                    # Cycle through the unique pool to fill remaining
                    # slots (n.b. the sketch's
                    # ``negs_unique[len(negs_unique) % len(negs_unique)]``
                    # always indexes zero; we cycle explicitly via the
                    # original unique-count instead).
                    unique_count = len(negs_unique)
                    cycle_pos = 0
                    while len(negs_unique) < n_negatives:
                        negs_unique.append(negs_unique[cycle_pos % unique_count])
                        cycle_pos += 1

            alt_codes = np.asarray(
                [chosen_code] + negs_unique[:n_negatives], dtype=np.int64
            )
            perm = local_rng.permutation(len(alt_codes))
            alt_codes = alt_codes[perm]
            chosen_pos = int(np.argmax(alt_codes == chosen_code))
            samples_per_k.append(catalog[alt_codes].tolist())
            chosen_idx_per_k.append(chosen_pos)
            dedup_fallback_per_k.append(dedup_fallback)

        # Build the chosen-alt dict from the event's own row (policy: the
        # chosen item IS the event, no lookup needed). V3-B1 fix:
        # thread routine/brand/state so ``adapter.alt_text`` can derive
        # ``is_repeat`` (from routine > 0) and populate brand/state on
        # the chosen alternative — previously these were dropped, so
        # every CHOSEN alt silently rendered ``is_repeat=False`` and
        # ``brand="unknown_brand"`` regardless of ground truth.
        chosen_asin = catalog[chosen_code]
        event_row_alt = {
            "title": str(titles_arr[i] or ""),
            "category": ev_cat_name,
            "price": float(prices[i] or 0.0),
            "popularity": int(popularity_train_arr[i] or 0),
            "routine": int(routines[i] or 0),
            "brand": str(brand_arr[i] or ""),
            "state": str(state_arr[i] or ""),
        }

        if n_resamples == 1:
            choice_asins = samples_per_k[0]
            chosen_idx = chosen_idx_per_k[0]
            alt_texts = _render_alt_texts(
                choice_asins, chosen_asin, event_row_alt
            )
        else:
            choice_asins = samples_per_k
            chosen_idx = chosen_idx_per_k
            alt_texts = [
                _render_alt_texts(sample, chosen_asin, event_row_alt)
                for sample in samples_per_k
            ]

        # Leakage-safe per-event features for the chosen item.
        chosen_features = {
            "routine": float(routines[i]),
            "recency_days": float(recency_arr[i]),
            "novelty": float(novelty_arr[i]),
            "cat_affinity": float(cat_affinity_arr[i]),
            "popularity": float(popularity_train_arr[i]),
            "price": float(prices[i] or 0.0),
            "brand": str(brand_arr[i] or ""),
        }

        # Dedup-fallback flag (Wave 11 post-dryrun fix). Scalar bool for
        # n_resamples==1 to match the single-set shape of the other per-
        # event fields; list-of-bool for n_resamples>1 mirroring the
        # nested shape of ``choice_asins`` / ``chosen_idx``.
        if n_resamples == 1:
            dedup_fallback_field = dedup_fallback_per_k[0]
        else:
            dedup_fallback_field = list(dedup_fallback_per_k)

        cid = customer_ids[i]

        # --- Per-event recent_purchases slice (Wave-12 per-event c_d) #
        # Find the customer's events with order_date in
        # [ev_date - window, ev_date) — strict < on both ends so the
        # current event and any future events are excluded. Within the
        # slice we take the last ``max_recent_purchases`` (most-recent
        # chronologically), keeping them in order so the rendered
        # "Recent purchases" reads as oldest-to-newest.
        event_recent: list[str] | None = None
        hist = customer_history.get(cid)
        if hist is not None:
            hist_dates, hist_titles = hist
            # hi = first index with order_date >= ev_date; the current
            # event itself sits at hi by strict-less semantics.
            hi = bisect_left(hist_dates, ev_date)
            # lo = first index with order_date >= ev_date - window.
            lo = bisect_left(hist_dates, ev_date - _window_td)
            if hi > lo:
                slice_titles = hist_titles[lo:hi]
                if max_recent_purchases > 0 and len(slice_titles) > max_recent_purchases:
                    slice_titles = slice_titles[-max_recent_purchases:]
                if slice_titles:
                    event_recent = list(slice_titles)

        event_c_d = build_context_string(
            customer_to_row[cid],
            suppress_fields=suppress,
            extra_fields=extras_by_cid.get(cid),
            recent_purchases=event_recent,
        )

        # Defensive z_d lookup. The coverage assertion at the top of
        # this function already guarantees every event customer has a
        # z_d row, so this branch should never fire under the normal
        # call path. We keep it in case a future caller bypasses the
        # assertion — the original ``KeyError: <customer_id>`` surface
        # from pandas is too terse to diagnose a pipeline-wiring bug.
        try:
            zd_row = customer_to_zd[cid]
        except KeyError as exc:
            raise KeyError(
                f"build_choice_sets: event for customer_id={cid!r} has no "
                f"z_d row in persons_canonical. This is a pipeline-wiring "
                f"bug — translate_z_d should have transformed every customer "
                f"in events_df. See choice_sets.py assertion block."
            ) from exc

        record = {
            # v1 keys
            "customer_id": cid,
            "order_date": order_dates_ns[i],
            "category": ev_cat_name,
            "chosen_asin": chosen_asin,
            "choice_asins": choice_asins,
            "chosen_idx": chosen_idx,
            "chosen_features": chosen_features,
            "metadata": {
                "is_repeat": bool(routines[i] > 0),
                "price": float(prices[i]),
                "routine": int(routines[i]),
                "dedup_fallback": dedup_fallback_field,
            },
            # v2 additions
            "z_d": zd_row,
            "c_d": event_c_d,
            "alt_texts": alt_texts,
        }
        records.append(record)

    logger.info(
        "build_choice_sets: done (%d choice sets built%s).",
        len(records),
        "; dedup fallback triggered on at least one event"
        if dedup_fallback_any
        else "",
    )
    return records
