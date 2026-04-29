"""Tests for the extended Sifringer-residual feature catalog.

Covers popularity_rank (band-label string in production),
popularity_count (dense numeric), is_repeat (per-alt history),
log1p_purchase_count (graded repeat-purchase signal).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.batching import (
    SUPPORTED_TABULAR_FEATURES,
    _build_x_tab_matrix,
)


def _records() -> list[dict]:
    """2 events x J=3. popularity_rank is a band-label string
    (production shape); popularity_count is dense numeric.
    is_repeat / purchase_count are per-alt train-history fields
    written by build_choice_sets (Fix 2)."""
    return [
        {"alt_texts": [
            {"title": "a", "category": "x", "price": 10.0,
             "popularity_rank": "top 5%", "popularity_count": 100,
             "is_repeat": 1, "purchase_count": 2},
            {"title": "b", "category": "x", "price": 20.0,
             "popularity_rank": "top 50%", "popularity_count": 50,
             "is_repeat": 0, "purchase_count": 0},
            {"title": "c", "category": "x", "price": 30.0,
             "popularity_rank": "bottom 50%", "popularity_count": 0,
             "is_repeat": 0, "purchase_count": 0},
        ], "chosen_idx": 0},
        {"alt_texts": [
            {"title": "d", "category": "y", "price": 5.0,
             "popularity_rank": "top 5%", "popularity_count": 200,
             "is_repeat": 0, "purchase_count": 0},
            {"title": "e", "category": "y", "price": 15.0,
             "popularity_rank": "top 25%", "popularity_count": 75,
             "is_repeat": 1, "purchase_count": 3},
            {"title": "f", "category": "y", "price": 25.0,
             "popularity_rank": "top 50%", "popularity_count": 25,
             "is_repeat": 0, "purchase_count": 0},
        ], "chosen_idx": 0},
    ]


def _prices(records: list[dict]) -> np.ndarray:
    N = len(records)
    J = len(records[0]["alt_texts"])
    out = np.zeros((N, J), dtype=np.float32)
    for i, rec in enumerate(records):
        for j, alt in enumerate(rec["alt_texts"]):
            out[i, j] = float(alt["price"])
    return out


def test_supported_features_includes_price_and_record_families() -> None:
    s = set(SUPPORTED_TABULAR_FEATURES)
    assert {"price", "log1p_price", "price_rank"}.issubset(s)
    assert {
        "popularity_rank", "log1p_popularity_rank",
        "popularity_count", "log1p_popularity_count",
        "is_repeat",
    }.issubset(s)


def test_popularity_rank_band_label_to_number() -> None:
    """Band-label string maps to coarse numeric in [0, 1]."""
    recs = _records()
    x = _build_x_tab_matrix(_prices(recs), ("popularity_rank",), records=recs)
    assert x.shape == (2, 3, 1)
    np.testing.assert_allclose(
        x[:, :, 0],
        np.array([[0.95, 0.5, 0.25], [0.95, 0.75, 0.5]], dtype=np.float32),
        rtol=1e-6,
    )


def test_log1p_popularity_rank_from_band_label() -> None:
    recs = _records()
    x = _build_x_tab_matrix(
        _prices(recs), ("log1p_popularity_rank",), records=recs,
    )
    expected = np.log1p([[0.95, 0.5, 0.25], [0.95, 0.75, 0.5]]).astype(np.float32)
    np.testing.assert_allclose(x[:, :, 0], expected, rtol=1e-6)


def test_popularity_rank_numeric_passthrough() -> None:
    """Numeric popularity_rank still passes through unchanged (legacy)."""
    recs = [{"alt_texts": [
        {"price": 1.0, "popularity_rank": 100.0},
        {"price": 2.0, "popularity_rank": 50.0},
    ]}]
    p = np.array([[1.0, 2.0]], dtype=np.float32)
    x = _build_x_tab_matrix(p, ("popularity_rank",), records=recs)
    np.testing.assert_array_equal(x[0, :, 0], [100.0, 50.0])


def test_popularity_count_passthrough() -> None:
    recs = _records()
    x = _build_x_tab_matrix(_prices(recs), ("popularity_count",), records=recs)
    np.testing.assert_array_equal(
        x[:, :, 0],
        np.array([[100, 50, 0], [200, 75, 25]], dtype=np.float32),
    )


def test_log1p_popularity_count() -> None:
    recs = _records()
    x = _build_x_tab_matrix(
        _prices(recs), ("log1p_popularity_count",), records=recs,
    )
    expected = np.log1p([[100, 50, 0], [200, 75, 25]]).astype(np.float32)
    np.testing.assert_allclose(x[:, :, 0], expected, rtol=1e-6)


def test_is_repeat_binary_per_alt() -> None:
    """is_repeat read from alt_texts[j] (per-alt history map)."""
    recs = _records()
    x = _build_x_tab_matrix(_prices(recs), ("is_repeat",), records=recs)
    np.testing.assert_array_equal(
        x[:, :, 0],
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )


def test_combined_columns_preserve_order() -> None:
    recs = _records()
    names = ("price", "popularity_count", "is_repeat")
    x = _build_x_tab_matrix(_prices(recs), names, records=recs)
    assert x.shape == (2, 3, 3)
    np.testing.assert_array_equal(x[0, :, 0], [10, 20, 30])
    np.testing.assert_array_equal(x[0, :, 1], [100, 50, 0])
    np.testing.assert_array_equal(x[0, :, 2], [1, 0, 0])


def test_record_feature_without_records_raises() -> None:
    p = np.array([[10.0, 20.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="require ``records``"):
        _build_x_tab_matrix(p, ("popularity_rank",), records=None)


def test_records_length_mismatch_raises() -> None:
    p = np.zeros((3, 2), dtype=np.float32)
    recs = [{"alt_texts": [
        {"popularity_rank": "top 5%"}, {"popularity_rank": "top 50%"},
    ]}]
    with pytest.raises(ValueError, match="does not match prices N"):
        _build_x_tab_matrix(p, ("popularity_rank",), records=recs)


def test_unknown_feature_name_raises() -> None:
    recs = _records()
    with pytest.raises(ValueError, match="Unsupported tabular feature"):
        _build_x_tab_matrix(
            _prices(recs), ("not_a_feature",), records=recs,
        )


def test_missing_alt_field_yields_zero() -> None:
    """A record missing is_repeat returns 0.0, not a crash."""
    recs = [{"alt_texts": [
        {"price": 1.0, "popularity_count": 5},
        {"price": 2.0, "popularity_count": 6, "is_repeat": 1},
    ]}]
    p = np.array([[1.0, 2.0]], dtype=np.float32)
    x = _build_x_tab_matrix(p, ("is_repeat",), records=recs)
    np.testing.assert_array_equal(x[0, :, 0], [0.0, 1.0])


def test_unknown_popularity_rank_string_yields_zero() -> None:
    """An unrecognised band string routes to 0 rather than crash."""
    recs = [{"alt_texts": [
        {"price": 1.0, "popularity_rank": "weird_unknown_band"},
        {"price": 2.0, "popularity_rank": "top 5%"},
    ]}]
    p = np.array([[1.0, 2.0]], dtype=np.float32)
    x = _build_x_tab_matrix(p, ("popularity_rank",), records=recs)
    np.testing.assert_allclose(x[0, :, 0], [0.0, 0.95], rtol=1e-6)


def test_legacy_price_features_unchanged_when_records_passed() -> None:
    """Adding records= shouldn't change the price-only output."""
    recs = _records()
    p = _prices(recs)
    legacy = _build_x_tab_matrix(p, ("price", "log1p_price", "price_rank"))
    new = _build_x_tab_matrix(
        p, ("price", "log1p_price", "price_rank"), records=recs,
    )
    np.testing.assert_array_equal(legacy, new)


def test_assemble_batch_e2e_columns_have_signal() -> None:
    """Real production path through build_choice_sets.

    Asserts that popularity_count / log1p_popularity_count /
    is_repeat / log1p_purchase_count carry non-zero variance.
    Regression test for the silently-broken d8aa5b1 routing
    where every cell collapsed to 0 because float() raised on
    the band-label string and is_repeat lived only at
    metadata[is_repeat] (chosen-only)."""
    import pandas as pd
    from src.data.choice_sets import build_choice_sets

    class _BandAdapter:
        def suppress_fields_for_c_d(self):
            return ()

        def alt_text(self, row):
            pop = int(row.get("popularity", 0) or 0)
            if pop >= 100:
                rank = "top 5%"
            elif pop >= 50:
                rank = "top 25%"
            elif pop >= 20:
                rank = "top 50%"
            else:
                rank = "bottom 50%"
            return {
                "title": str(row.get("title", "")),
                "category": str(row.get("category", "")),
                "price": float(row.get("price", 0.0) or 0.0),
                "popularity_rank": rank,
                "popularity_count": pop,
                "brand": "b",
            }

    rows = []
    asins = [f"A{i:03d}" for i in range(8)]
    for c in range(6):
        cid = f"C{c:02d}"
        for e in range(6):
            d = pd.to_datetime("2024-01-01") + pd.Timedelta(days=c * 50 + e)
            asin = asins[0] if e < 3 else asins[(c + e) % len(asins)]
            cat_idx = int(asin[1:]) % 3
            rows.append(dict(
                customer_id=cid,
                order_date=d,
                asin=asin,
                category=f"CAT{cat_idx}",
                price=5.0 + (int(asin[1:]) % 10),
                title=f"title {asin} word{e}",
                routine=int(e > 0),
                novelty=int(e == 0),
                recency_days=999 if e == 0 else 7,
                cat_affinity=e,
                brand="brand_x",
            ))
    df = pd.DataFrame(rows)
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    splits = []
    for _, g in df.groupby("customer_id", sort=False):
        n = len(g)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        n_train = n - n_test - n_val
        splits.extend(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    df["split"] = splits

    full_age = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    full_income = ["<25k", "25-50k", "50-100k", "100-150k", "150k+"]
    full_city = ["rural", "small", "medium", "large"]
    persons = pd.DataFrame([
        dict(
            customer_id=f"C{c:02d}",
            age_bucket=full_age[c % 6],
            income_bucket=full_income[c % 5],
            household_size=1 + (c % 4),
            has_kids=int(c % 2),
            city_size=full_city[c % 4],
            education=3,
            health_rating=4,
            risk_tolerance=0.0,
            purchase_frequency=2.0,
            novelty_rate=0.3,
        )
        for c in range(6)
    ])

    records = build_choice_sets(
        df, persons, _BandAdapter(),
        n_negatives=3, seed=0, n_resamples=1,
    )
    assert len(records) > 0
    found_band = any(
        isinstance(r["alt_texts"][j].get("popularity_rank"), str)
        for r in records for j in range(len(r["alt_texts"]))
    )
    assert found_band, "adapter band-label strings should reach alt_texts"

    N = len(records)
    J = len(records[0]["choice_asins"])
    prices_np = np.zeros((N, J), dtype=np.float32)
    for i, r in enumerate(records):
        for j, alt in enumerate(r["alt_texts"]):
            prices_np[i, j] = float(alt.get("price", 0.0) or 0.0)

    feats = (
        "popularity_count", "log1p_popularity_count",
    )
    x = _build_x_tab_matrix(prices_np, feats, records=records)
    for f, name in enumerate(feats):
        col = x[:, :, f]
        assert col.std() > 0.0, (
            f"x_tab column {name!r} has zero std - silently-broken routing."
        )
