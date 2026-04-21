"""
Data preparation for Amazon e-commerce Structure Descent.

Builds ordered purchase sequences and choice sets per customer.
Each purchase event (s_t, a*) becomes a training instance for the conditional logit.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "amazon_ecom"


def load_data(
    purchases_path: Optional[Path] = None,
    survey_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    purchases_path = purchases_path or DATA_DIR / "amazon-purchases.csv"
    survey_path = survey_path or DATA_DIR / "survey.csv"

    print(f"[load_data] Loading purchases from {purchases_path} ...")
    purchases = pd.read_csv(purchases_path, parse_dates=["Order Date"])
    print(f"[load_data] Loaded {len(purchases):,} rows, {purchases.shape[1]} columns")

    print(f"[load_data] Loading survey from {survey_path} ...")
    survey = pd.read_csv(survey_path)
    print(f"[load_data] Loaded {len(survey):,} survey respondents")
    return purchases, survey


def clean_purchases(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[clean] Starting with {len(df):,} rows ...")
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {
        "Order Date": "order_date",
        "Purchase Price Per Unit": "price",
        "Quantity": "quantity",
        "Shipping Address State": "state",
        "Title": "title",
        "ASIN/ISBN": "asin",
        "ASIN/ISBN (Product Code)": "asin",
        "Category": "category",
        "Survey ResponseID": "customer_id",
    }
    df = df.rename(columns=col_map)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    n_before = len(df)
    df = df.dropna(subset=["customer_id", "order_date", "asin"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[clean] Dropped {n_dropped:,} rows with missing customer_id/order_date/asin")
    df["category"] = df["category"].fillna("Unknown").astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["asin"] = df["asin"].astype(str)
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    print(f"[clean] Done: {len(df):,} rows, {df['customer_id'].nunique():,} customers, {df['asin'].nunique():,} ASINs, {df['category'].nunique():,} categories")
    return df


def join_survey(df: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """
    Merge survey columns onto events by customer_id == Survey ResponseID.
    Light cleaning: drop all-null columns, snake_case column names. No encoding.
    """
    print(f"[survey] Joining survey onto {len(df):,} events ...")
    survey = survey.copy()
    id_col = None
    for c in survey.columns:
        if c.strip() == "Survey ResponseID":
            id_col = c
            break
    if id_col is None:
        raise KeyError("join_survey: no 'Survey ResponseID' column found in survey DataFrame")
    survey = survey.rename(columns={id_col: "customer_id"})
    survey["customer_id"] = survey["customer_id"].astype(str)

    all_null = [c for c in survey.columns if survey[c].isna().all()]
    if all_null:
        survey = survey.drop(columns=all_null)

    def _snake(name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"[^0-9a-z]+", "_", s)
        return s.strip("_")

    survey.columns = [c if c == "customer_id" else _snake(c) for c in survey.columns]
    survey = survey.drop_duplicates(subset=["customer_id"], keep="first")

    before_cols = set(df.columns)
    merged = df.merge(survey, on="customer_id", how="left")
    new_cols = [c for c in merged.columns if c not in before_cols]
    print(f"[survey] Done: added {len(new_cols)} survey columns ({len(all_null)} all-null dropped)")
    return merged


_BRAND_STOPWORDS = {
    "premium", "new", "upgraded", "upgrade", "original", "genuine", "authentic",
    "professional", "pro", "deluxe", "luxury", "ultra", "super", "best",
    "the", "a", "an", "for", "with", "by", "of",
    "2-pack", "3-pack", "4-pack", "6-pack", "10-pack", "pack",
    "set", "kit", "bundle", "case", "box",
}


def _first_brand_token(title: str) -> str:
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


def compute_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each purchase, compute history-derived state features:
      - routine:      how many times this customer bought this item before
      - recency_days: days since last purchase of this item (NaN if never)
      - novelty:      1 if first time buying this item, 0 otherwise
      - cat_count_7d: purchases in same category in past 7 days (time-based)
      - cat_count_30d: purchases in same category in past 30 days (time-based)
      - cat_affinity: total prior purchases in this category (affinity proxy)
      - brand:        mode first-token brand per ASIN, with stopword filter

    Popularity is NOT computed here — see attach_train_popularity, which must
    be called after temporal_split to avoid leaking val/test counts into train.
    """
    print("[features] Computing state features ...")
    df = df.copy().sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    print("[features]   routine (repeat purchase count) ...")
    df["routine"] = df.groupby(["customer_id", "asin"]).cumcount()
    df["novelty"] = (df["routine"] == 0).astype(int)

    print("[features]   recency_days (days since last purchase of item) ...")
    df["last_purchase_date"] = df.groupby(["customer_id", "asin"])["order_date"].shift(1)
    df["recency_days"] = (df["order_date"] - df["last_purchase_date"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(999)

    print("[features]   cat_affinity (category purchase count) ...")
    df["cat_affinity"] = df.groupby(["customer_id", "category"]).cumcount()

    print("[features]   cat_count_7d / cat_count_30d (time-based rolling counts of events) ...")
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

    print("[features]   brand (mode first-token per ASIN) ...")
    brand_map = _build_asin_brand_map(df)
    brand_counts = pd.Series(list(brand_map.values())).value_counts()
    ambiguous = set(brand_counts[brand_counts < 2].index) if len(brand_counts) else set()
    df["brand"] = df["asin"].map(brand_map).fillna("")
    df.loc[df["brand"].isin(ambiguous) | (df["brand"] == ""), "brand"] = "unknown_brand"

    repeat_rate = (df["routine"] > 0).mean()
    print(f"[features] Done: {len(df):,} events, repeat rate={repeat_rate:.1%}, {df['brand'].nunique():,} brands")
    return df


def attach_train_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ASIN popularity from train rows only and broadcast to all rows.

    Popularity must be computed after temporal_split to prevent leakage of
    val/test purchase counts into the features the model sees at train time.
    Val/test rows get the train-derived count (a legitimate "what the model
    knew at training time"); rows whose ASIN never appears in train get 0.
    """
    if "split" not in df.columns:
        raise KeyError("attach_train_popularity requires a 'split' column — call temporal_split first")
    print("[popularity] Computing train-only ASIN popularity ...")
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
    print(f"[popularity] Done: {len(pop):,} ASINs with train counts, {n_unseen:,} rows mapped to popularity=0")
    return df


def build_choice_sets(
    df: pd.DataFrame,
    n_negatives: int = 9,
    seed: int = 42,
    n_resamples: int = 1,
    popularity_col: str = "popularity",
) -> List[dict]:
    # McFadden caveat: sampled conditional logit with stratified (same-category)
    # negatives yields coefficients consistent for *relative* utility between the
    # chosen alternative and the sampled alternatives, not absolute utility over
    # the full catalog. Downstream probability/rank metrics inherit that caveat.
    """
    For each purchase event, build a choice set of size n_negatives + 1:
      - positive: the actual purchased ASIN
      - negatives: half same-category (if category != "Unknown"), half random,
                   drawn from ASINs first seen strictly before the event date,
                   with random-pool draws weighted by (train-derived) popularity.

    Parameters
    ----------
    n_resamples : int
        If >1, each event gets a list of K distinct sampled choice sets (different
        RNG streams) instead of a single set. The returned record then has
        'choice_asins' and 'chosen_idx' as length-K lists so trainers can cycle
        through them to prevent memorization of specific negative ASINs.
        Default 1 preserves the single-set behavior.
    popularity_col : str
        Column in df used as the random-negative sampling weight (popularity-
        weighted sampling approximates McFadden's choice-based correction).
    """
    print(f"[choice_sets] Building from {len(df):,} events (n_resamples={n_resamples}) ...")
    if df[["asin", "category"]].isna().any().any():
        raise ValueError(
            "build_choice_sets: df contains NaN in 'asin' or 'category'. "
            "Run clean_purchases first (NaN codes would silently wrap catalog indexing)."
        )
    rng = np.random.default_rng(seed)
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
    print(f"[choice_sets] {n_catalog:,} unique ASINs, {n_cats:,} categories, {n_negatives} negatives each")

    n_cat_neg = n_negatives // 2
    n_rand_neg = n_negatives - n_cat_neg

    order_dates_ns = pd.to_datetime(df["order_date"]).to_numpy().astype("datetime64[ns]")

    # Build per-ASIN earliest-seen date (over the full df).
    first_seen = (
        df.assign(_d=order_dates_ns)
        .groupby("asin")["_d"].min()
    )
    # Align first_seen to the catalog ordering.
    first_seen_by_code = first_seen.reindex(catalog).to_numpy().astype("datetime64[ns]")

    # Popularity weights over the catalog (train-derived). ASINs with zero
    # train popularity get a tiny epsilon so they're still sampleable.
    if popularity_col in df.columns:
        pop_series = (
            df[["asin", popularity_col]]
            .drop_duplicates(subset=["asin"])
            .set_index("asin")[popularity_col]
        )
        pop_by_code = pop_series.reindex(catalog).fillna(0).to_numpy().astype(np.float64)
    else:
        pop_by_code = np.ones(n_catalog, dtype=np.float64)
    pop_by_code = pop_by_code + 1e-6

    # Sort catalog by first-seen date so we can binary-search the available
    # prefix at each event's order_date.
    sort_by_first = np.argsort(first_seen_by_code, kind="stable")
    sorted_first_seen = first_seen_by_code[sort_by_first]
    sorted_pop = pop_by_code[sort_by_first]
    sorted_cat_of_asin = np.empty(n_catalog, dtype=np.int64)
    # Map each catalog code back to its (majority) category via df groupby.
    asin_to_cat_code = (
        pd.DataFrame({"a": asin_cat.codes, "c": cat_codes})
        .groupby("a")["c"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    sorted_cat_of_asin_full = asin_to_cat_code.reindex(range(n_catalog)).fillna(-1).to_numpy().astype(np.int64)
    sorted_cat_of_asin = sorted_cat_of_asin_full[sort_by_first]

    # For each event, find the prefix length of ASINs with first_seen < order_date.
    avail_prefix = np.searchsorted(sorted_first_seen, order_dates_ns, side="left")

    # Per-category prefix lookups: for each category c, list of indices into the
    # sorted-by-first-seen array (already sorted by first-seen within category
    # since we sliced from a globally sorted array).
    cat_to_sorted_idxs: List[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_cats
    cat_to_sorted_first_seen: List[np.ndarray] = [np.empty(0, dtype="datetime64[ns]")] * n_cats
    cat_to_sorted_pop: List[np.ndarray] = [np.empty(0, dtype=np.float64)] * n_cats
    for c in range(n_cats):
        if c == unknown_cat_code:
            continue
        mask = sorted_cat_of_asin == c
        cat_to_sorted_idxs[c] = sort_by_first[mask]
        cat_to_sorted_first_seen[c] = sorted_first_seen[mask]
        cat_to_sorted_pop[c] = sorted_pop[mask]

    def _sample_random(avail_len: int, chosen_code: int, k: int, local_rng) -> np.ndarray:
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
                # Bump the sort-position (stays inside the temporally-available
                # prefix), not the catalog code. The +1 neighbor in sort_by_first
                # is a distinct catalog entry by construction, so it cannot equal
                # chosen_code.
                bumped_idxs = (idxs + 1) % avail_len
                sampled = np.where(coll, sort_by_first[bumped_idxs], sampled)
        return sampled.astype(np.int64)

    def _sample_same_cat(cat_code: int, event_date, chosen_code: int, k: int, local_rng) -> np.ndarray:
        if cat_code < 0 or cat_code == unknown_cat_code:
            return _sample_random_for_event_avail(event_date, chosen_code, k, local_rng)
        sub_first = cat_to_sorted_first_seen[cat_code]
        if sub_first.size == 0:
            return _sample_random_for_event_avail(event_date, chosen_code, k, local_rng)
        prefix = int(np.searchsorted(sub_first, event_date, side="left"))
        if prefix <= 1:
            return _sample_random_for_event_avail(event_date, chosen_code, k, local_rng)
        sub_idx = cat_to_sorted_idxs[cat_code][:prefix]
        sub_pop = cat_to_sorted_pop[cat_code][:prefix]
        p_sum = sub_pop.sum()
        if p_sum <= 0:
            picks = local_rng.integers(0, prefix, size=k)
        else:
            picks = local_rng.choice(prefix, size=k, replace=True, p=sub_pop / p_sum)
        sampled = sub_idx[picks]
        coll = sampled == chosen_code
        if coll.any():
            bumped = np.where(coll, sub_idx[(picks + 1) % prefix], sampled)
            sampled = bumped
        return sampled.astype(np.int64)

    def _sample_random_for_event_avail(event_date, chosen_code: int, k: int, local_rng) -> np.ndarray:
        prefix = int(np.searchsorted(sorted_first_seen, event_date, side="left"))
        return _sample_random(prefix, chosen_code, k, local_rng)

    customer_ids = df["customer_id"].to_numpy()
    categories = df["category"].to_numpy()
    routines = (
        df["routine"].to_numpy() if "routine" in df.columns else np.zeros(n_events, dtype=np.int64)
    )
    prices = df["price"].to_numpy() if "price" in df.columns else np.zeros(n_events)

    # Per-event features for the chosen item, captured AT the event timestamp
    # from the causal history features computed by compute_state_features.
    # These are the only feature values the model may legitimately see for the
    # positive alternative — using any whole-df lookup would leak future state.
    recency_arr = (
        df["recency_days"].to_numpy() if "recency_days" in df.columns else np.full(n_events, 999.0)
    )
    novelty_arr = (
        df["novelty"].to_numpy() if "novelty" in df.columns else np.ones(n_events, dtype=np.int64)
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
        df["brand"].to_numpy() if "brand" in df.columns else np.array([""] * n_events, dtype=object)
    )

    records: List[dict] = []
    for i in range(n_events):
        chosen_code = int(chosen_int[i])
        ev_date = order_dates_ns[i]
        ev_cat_code = int(cat_codes[i])
        ev_cat_name = categories[i]
        is_unknown = (ev_cat_code == unknown_cat_code) or (ev_cat_name == "Unknown")

        samples_per_k: List[List[str]] = []
        chosen_idx_per_k: List[int] = []
        for k_rep in range(n_resamples):
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

            alt_codes = np.concatenate([[chosen_code], negs])
            perm = local_rng.permutation(len(alt_codes))
            alt_codes = alt_codes[perm]
            chosen_pos = int(np.argmax(alt_codes == chosen_code))
            samples_per_k.append(catalog[alt_codes].tolist())
            chosen_idx_per_k.append(chosen_pos)

        if n_resamples == 1:
            choice_asins = samples_per_k[0]
            chosen_idx = chosen_idx_per_k[0]
        else:
            choice_asins = samples_per_k
            chosen_idx = chosen_idx_per_k

        # Leakage-safe per-event features for the chosen item.
        # All values are the customer's state AT the event's order_date (strictly
        # prior history, via compute_state_features.cumcount/shift), except
        # popularity which uses the train-only broadcast (attach_train_popularity).
        chosen_features = {
            "routine": float(routines[i]),
            "recency_days": float(recency_arr[i]),
            "novelty": float(novelty_arr[i]),
            "cat_affinity": float(cat_affinity_arr[i]),
            "popularity": float(popularity_train_arr[i]),
            "price": float(prices[i] or 0.0),
            "brand": str(brand_arr[i] or ""),
        }

        records.append({
            "customer_id": customer_ids[i],
            "order_date": order_dates_ns[i],
            "category": ev_cat_name,
            "chosen_asin": catalog[chosen_code],
            "choice_asins": choice_asins,
            "chosen_idx": chosen_idx,
            "chosen_features": chosen_features,
            "metadata": {
                "is_repeat": bool(routines[i] > 0),
                "price": float(prices[i]),
                "routine": int(routines[i]),
            },
        })

    print(f"[choice_sets] Done: {len(records):,} choice sets built")
    return records


def temporal_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1) -> pd.DataFrame:
    """
    Per-customer temporal split: train on earliest, val + test on most recent.
    Adds a 'split' column: 'train' | 'val' | 'test'.
    """
    print(f"[split] Temporal split: val={val_frac:.0%}, test={test_frac:.0%} per customer ...")
    df = df.sort_values(["customer_id", "order_date"]).copy().reset_index(drop=True)

    def assign_splits(grp_indices):
        n = len(grp_indices)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val
        return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    split_labels = []
    for _, grp in df.groupby("customer_id", sort=False):
        split_labels.extend(assign_splits(grp.index))

    df["split"] = split_labels
    counts = df["split"].value_counts()
    print(f"[split] Done: train={counts.get('train',0):,}, val={counts.get('val',0):,}, test={counts.get('test',0):,}")
    return df
