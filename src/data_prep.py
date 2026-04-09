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


def compute_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each purchase, compute history-derived state features:
      - routine:      how many times this customer bought this item before
      - recency_days: days since last purchase of this item (NaN if never)
      - novelty:      1 if first time buying this item, 0 otherwise
      - cat_count_7d: purchases in same category in past 7 days
      - cat_count_30d: purchases in same category in past 30 days
      - cat_affinity: total prior purchases in this category (affinity proxy)
      - popularity:   global purchase count of this ASIN across all customers
      - brand:        extracted brand token from title (first word)
    """
    df = df.copy().sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    df["routine"] = df.groupby(["customer_id", "asin"]).cumcount()
    df["novelty"] = (df["routine"] == 0).astype(int)

    df["last_purchase_date"] = df.groupby(["customer_id", "asin"])["order_date"].shift(1)
    df["recency_days"] = (df["order_date"] - df["last_purchase_date"]).dt.days
    df["recency_days"] = df["recency_days"].fillna(999)

    df["cat_affinity"] = df.groupby(["customer_id", "category"]).cumcount()

    df = df.sort_values(["customer_id", "order_date"])
    df["cat_count_7d"] = (
        df.groupby("customer_id")["cat_affinity"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    df["cat_count_30d"] = (
        df.groupby("customer_id")["cat_affinity"]
        .transform(lambda x: x.rolling(30, min_periods=1).sum())
    )

    popularity = df.groupby("asin")["customer_id"].count().rename("popularity")
    df = df.join(popularity, on="asin")

    df["brand"] = df["title"].fillna("").str.split().str[0].str.lower()

    return df


def build_choice_sets(
    df: pd.DataFrame,
    n_negatives: int = 9,
    seed: int = 42,
) -> List[dict]:
    """
    For each purchase event, build a choice set of size n_negatives + 1:
      - positive (chosen_idx=0 before shuffle): the actual purchased ASIN
      - negatives: half same-category, half random from catalog

    Returns list of dicts:
      customer_id, order_date, category, chosen_asin,
      choice_asins [list], chosen_idx [int], metadata [dict]
    """
    rng = np.random.default_rng(seed)
    catalog = df["asin"].unique()
    cat_to_asins = {
        cat: [a for a in grp["asin"].unique()]
        for cat, grp in df.groupby("category")
    }

    records = []
    for _, row in df.iterrows():
        chosen = row["asin"]
        cat = row.get("category", "")

        same_cat = [a for a in cat_to_asins.get(cat, []) if a != chosen]
        n_cat = min(n_negatives // 2, len(same_cat))
        n_rand = n_negatives - n_cat

        neg_cat = list(rng.choice(same_cat, size=n_cat, replace=False)) if same_cat else []
        rand_pool = [a for a in catalog if a != chosen and a not in neg_cat]
        neg_rand = list(rng.choice(rand_pool, size=min(n_rand, len(rand_pool)), replace=False))

        choice_asins = [chosen] + neg_cat + neg_rand
        rng.shuffle(choice_asins)
        chosen_idx = choice_asins.index(chosen)

        records.append({
            "customer_id": row["customer_id"],
            "order_date": row["order_date"],
            "category": cat,
            "chosen_asin": chosen,
            "choice_asins": choice_asins,
            "chosen_idx": chosen_idx,
            "metadata": {
                "is_repeat": row.get("routine", 0) > 0,
                "price": row.get("price", 0.0),
                "routine": row.get("routine", 0),
            },
        })

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
