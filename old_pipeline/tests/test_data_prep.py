"""
End-to-end test of the data-prep pipeline on a hand-built 25-row dataset.

Checks, with explicit manually-computed expected values:
  1. train-only popularity (no test/val leakage)
  2. causal history features: routine, recency_days, cat_affinity
  3. time-based rolling windows (cat_count_7d/30d) — not row-based
  4. negative sampling respects temporal availability (first_seen < event_date)
  5. Unknown-category events still produce valid, availability-respecting sets
  6. brand stopword filtering ("Premium ..." never yields brand="premium")
  7. survey join attaches demographic columns by customer_id
  8. temporal split is monotone per customer (no test row before a train row)

Run:  python tests/test_data_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from old_pipeline.src.data_prep import (
    _BRAND_STOPWORDS,
    attach_train_popularity,
    build_choice_sets,
    clean_purchases,
    compute_state_features,
    join_survey,
    load_data,
    temporal_split,
)

FIXTURES = Path(__file__).parent / "fixtures"

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        msg = f"  FAIL  {label}"
        if detail:
            msg += f"\n        {detail}"
        print(msg)
        _failures.append(label)


def row(df: pd.DataFrame, customer: str, date: str, asin: str) -> pd.Series:
    m = (df["customer_id"] == customer) & (df["order_date"] == pd.Timestamp(date)) & (df["asin"] == asin)
    rows = df[m]
    assert len(rows) == 1, f"expected exactly 1 row for ({customer}, {date}, {asin}), got {len(rows)}"
    return rows.iloc[0]


def main() -> int:
    print("=" * 70)
    print("Data-prep pipeline test on mini fixture")
    print("=" * 70)

    purchases_raw, survey = load_data(
        purchases_path=FIXTURES / "mini_purchases.csv",
        survey_path=FIXTURES / "mini_survey.csv",
    )

    df = clean_purchases(purchases_raw)
    df = join_survey(df, survey)
    df = compute_state_features(df)

    # MIN_PURCHASES=5 filter — all 4 customers qualify, so this is a no-op here,
    # but keep the call pattern identical to notebooks/01.
    counts = df["customer_id"].value_counts()
    valid = counts[counts >= 5].index
    df = df[df["customer_id"].isin(valid)].reset_index(drop=True)

    df = temporal_split(df, val_frac=0.1, test_frac=0.1)
    df = attach_train_popularity(df)

    choices = build_choice_sets(df, n_negatives=9, seed=42)

    print()
    print("-- 1. Train-only popularity (no val/test leakage) --")
    expected_pop = {
        "B001": 2, "B002": 2, "B003": 1,
        "B005": 2, "B006": 2,
        "B008": 1, "B009": 3, "B010": 1,
        "B012": 2, "B013": 1,
        "B004": 0, "B007": 0, "B011": 0, "B014": 0,
    }
    actual_pop = (
        df.drop_duplicates("asin").set_index("asin")["popularity"].to_dict()
    )
    for asin, exp in expected_pop.items():
        got = int(actual_pop.get(asin, -1))
        check(f"popularity[{asin}] == {exp}", got == exp, f"got {got}")

    print()
    print("-- 2. Causal history features (no future information) --")
    # R_A first B001 purchase — nothing prior
    r = row(df, "R_A", "2019-01-01", "B001")
    check("R_A@01-01 B001 routine == 0", int(r["routine"]) == 0)
    check("R_A@01-01 B001 recency_days == 999", int(r["recency_days"]) == 999)
    check("R_A@01-01 B001 cat_affinity == 0", int(r["cat_affinity"]) == 0)

    # 2nd B001 purchase: prior B001 was 01-01; prior FLASH was 01-01 only
    r = row(df, "R_A", "2019-01-10", "B001")
    check("R_A@01-10 B001 routine == 1", int(r["routine"]) == 1)
    check("R_A@01-10 B001 recency_days == 9", int(r["recency_days"]) == 9)
    check("R_A@01-10 B001 cat_affinity == 1", int(r["cat_affinity"]) == 1)

    # 3rd B001 purchase: last B001 was 01-10; prior FLASH: 01-01, 01-10, 01-15
    r = row(df, "R_A", "2019-02-01", "B001")
    check("R_A@02-01 B001 routine == 2", int(r["routine"]) == 2)
    check("R_A@02-01 B001 recency_days == 22", int(r["recency_days"]) == 22)
    check("R_A@02-01 B001 cat_affinity == 3", int(r["cat_affinity"]) == 3)

    print()
    print("-- 3. Time-based rolling windows (cat_count_7d/30d) --")
    # R_C's 2019-03-15 KITCHEN event: prior KITCHEN events (01-15/16/17) are
    # 57+ days before, so they are outside BOTH the 7d and 30d time windows.
    # Only the current event should count. Row-based rolling(30) would give 4.
    r = row(df, "R_C", "2019-03-15", "B009")
    check(
        "R_C@03-15 B009 cat_count_30d == 1 (time-based, not 4)",
        int(r["cat_count_30d"]) == 1,
        f"got {int(r['cat_count_30d'])} — row-based bug would produce 4",
    )
    check("R_C@03-15 B009 cat_count_7d == 1", int(r["cat_count_7d"]) == 1)

    # R_C @ 03-16 B011 KITCHEN: window now includes 03-15 KITCHEN event too.
    r = row(df, "R_C", "2019-03-16", "B011")
    check(
        "R_C@03-16 B011 cat_count_30d == 2 (03-15 + 03-16)",
        int(r["cat_count_30d"]) == 2,
        f"got {int(r['cat_count_30d'])}",
    )

    # R_A @ 01-10 B001 FLASH: 7d window = [01-03, 01-10]; 01-01 is excluded (9d prior).
    # 30d window = [12-11, 01-10] includes 01-01 and 01-10.
    r = row(df, "R_A", "2019-01-10", "B001")
    check(
        "R_A@01-10 B001 cat_count_7d == 1 (01-01 out of window)",
        int(r["cat_count_7d"]) == 1,
        f"got {int(r['cat_count_7d'])}",
    )
    check(
        "R_A@01-10 B001 cat_count_30d == 2 (01-01 + 01-10)",
        int(r["cat_count_30d"]) == 2,
        f"got {int(r['cat_count_30d'])}",
    )

    print()
    print("-- 4. Negative sampling respects temporal availability --")
    # Build a lookup: (customer_id, order_date, asin) -> record
    choice_by_key = {
        (c["customer_id"], pd.Timestamp(c["order_date"]), c["chosen_asin"]): c
        for c in choices
    }

    # First-seen per ASIN (over the whole df) for the availability assertion.
    first_seen = df.groupby("asin")["order_date"].min().to_dict()

    # Universal availability check: every non-chosen alt must have first_seen
    # strictly before the event's order_date.
    leaks = 0
    total_alts = 0
    for c in choices:
        ev_date = pd.Timestamp(c["order_date"])
        for a in c["choice_asins"]:
            if a == c["chosen_asin"]:
                continue
            total_alts += 1
            if first_seen[a] >= ev_date:
                leaks += 1
    check(
        f"no choice_set alt has first_seen >= event date  ({total_alts} alts checked)",
        leaks == 0,
        f"{leaks} leaked alternatives found",
    )

    # Spot check: R_A @ 01-15 B003 — only B001/B002/B005/B008 are available before.
    c = choice_by_key[("R_A", pd.Timestamp("2019-01-15"), "B003")]
    allowed = {"B001", "B002", "B005", "B008", "B003"}
    bad = [a for a in c["choice_asins"] if a not in allowed]
    check(
        "R_A@01-15 B003 choice_asins ⊆ {B001,B002,B005,B008,B003}",
        len(bad) == 0,
        f"disallowed alts: {bad}",
    )

    print()
    print("-- 5. Unknown-category event (R_C @ 01-10 B008) --")
    c = choice_by_key[("R_C", pd.Timestamp("2019-01-10"), "B008")]
    # Available before 01-10: B001 (01-01), B002 (01-05), B005 (01-08).
    allowed_unk = {"B001", "B002", "B005", "B008"}
    bad_unk = [a for a in c["choice_asins"] if a not in allowed_unk]
    check(
        "R_C@01-10 B008 (Unknown) choice_asins ⊆ {B001,B002,B005,B008}",
        len(bad_unk) == 0,
        f"disallowed alts: {bad_unk}",
    )
    check(
        "R_C@01-10 B008 category recorded as 'Unknown'",
        c["category"] == "Unknown",
    )

    print()
    print("-- 6. Brand stopword filter --")
    brands_seen = set(df["brand"].unique())
    overlap = brands_seen & _BRAND_STOPWORDS
    check(
        "no brand value falls inside _BRAND_STOPWORDS",
        len(overlap) == 0,
        f"stopwords leaked as brands: {sorted(overlap)}",
    )

    b013_brand = df.loc[df["asin"] == "B013", "brand"].iloc[0]
    check(
        f"B013 ('Premium Fantasy Novel') brand != 'premium' (got '{b013_brand}')",
        b013_brand != "premium",
    )

    # sony appears for both B002 (all Sony titles) and B007 ("Premium Sony ..."),
    # so its count in the brand map is >= 2 and it survives the singleton filter.
    for asin in ("B002", "B007"):
        got = df.loc[df["asin"] == asin, "brand"].iloc[0]
        check(f"{asin} brand == 'sony'", got == "sony", f"got '{got}'")
    # lodge appears for B010 and B011.
    for asin in ("B010", "B011"):
        got = df.loc[df["asin"] == asin, "brand"].iloc[0]
        check(f"{asin} brand == 'lodge'", got == "lodge", f"got '{got}'")

    print()
    print("-- 7. Survey join --")
    for col in ("q_demos_age", "q_demos_income", "q_demos_state"):
        check(f"survey column '{col}' present after join", col in df.columns)
    r = df[df["customer_id"] == "R_A"].iloc[0]
    check("R_A age == '35 - 44 years'", r["q_demos_age"] == "35 - 44 years")
    r = df[df["customer_id"] == "R_D"].iloc[0]
    check("R_D state == 'Florida'", r["q_demos_state"] == "Florida")

    print()
    print("-- 8. Per-customer temporal split is monotone --")
    for cust, grp in df.groupby("customer_id"):
        train_max = grp.loc[grp["split"] == "train", "order_date"].max()
        val_min = grp.loc[grp["split"] == "val", "order_date"].min()
        test_min = grp.loc[grp["split"] == "test", "order_date"].min()
        ok = True
        detail = ""
        if pd.notna(val_min) and val_min <= train_max:
            ok = False
            detail = f"val({val_min}) <= train_max({train_max})"
        if pd.notna(test_min) and test_min <= train_max:
            ok = False
            detail = f"test({test_min}) <= train_max({train_max})"
        check(f"{cust}: train precedes val/test", ok, detail)

    print()
    print("=" * 70)
    if _failures:
        print(f"FAILED: {len(_failures)} checks")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
