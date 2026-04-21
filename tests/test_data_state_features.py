"""Tests for :mod:`src.data.state_features` (Wave 8, design doc §5).

Covers the ported-verbatim compute_state_features + attach_train_popularity
path. Most tests construct small hand-built DataFrames so the expected
value for every per-event feature is obvious; one smoke test also runs
the full ``load -> clean -> compute_state_features`` pipeline against
the 100-customer fixture for coverage of the real column layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_events
from src.data.invariants import InvariantError
from src.data.load import load
from src.data.schema_map import load_schema
from src.data.state_features import (
    _BRAND_STOPWORDS,
    _build_asin_brand_map,
    _first_brand_token,
    attach_train_popularity,
    compute_state_features,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fixture_events() -> pd.DataFrame:
    """Full ``load -> clean`` path against the 100-customer fixture."""
    schema = load_schema(AMAZON_YAML)
    events_raw, _ = load(
        schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    return clean_events(events_raw, schema)


def _make_events(rows: list[dict]) -> pd.DataFrame:
    """Build a small canonical-columns events DataFrame for an in-line test.

    ``rows`` entries must already use the cleaned column names
    (``customer_id``, ``order_date``, ``asin``, ``category``, ``title``,
    ``price``). ``order_date`` is coerced to datetime.
    """
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["customer_id"] = df["customer_id"].astype(str)
    df["asin"] = df["asin"].astype(str)
    df["category"] = df["category"].astype(str)
    if "title" not in df.columns:
        df["title"] = ""
    if "price" not in df.columns:
        df["price"] = 0.0
    df["price"] = df["price"].astype(float)
    return df


# --------------------------------------------------------------------------- #
# compute_state_features — column + row-count smoke tests
# --------------------------------------------------------------------------- #


def test_state_features_adds_all_columns(fixture_events):
    out = compute_state_features(fixture_events)
    required = {
        "routine", "novelty", "recency_days", "cat_affinity",
        "cat_count_7d", "cat_count_30d", "brand",
    }
    assert required.issubset(set(out.columns)), (
        f"missing columns: {required - set(out.columns)!r}"
    )


def test_state_features_row_count_preserved(fixture_events):
    out = compute_state_features(fixture_events)
    assert len(out) == len(fixture_events)


# --------------------------------------------------------------------------- #
# Per-feature semantics
# --------------------------------------------------------------------------- #


def test_routine_starts_at_zero():
    """First purchase per (customer, asin) has routine == 0."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-05", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-07", "asin": "b",
         "category": "C2", "title": "Bar"},
    ])
    out = compute_state_features(events)
    # Re-index by (customer_id, order_date) after the stage's final sort.
    out = out.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    assert out["routine"].tolist() == [0, 1, 0]


def test_novelty_matches_routine_eq_zero():
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-05", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-07", "asin": "b",
         "category": "C1", "title": "Bar"},
        {"customer_id": "u2", "order_date": "2022-01-02", "asin": "a",
         "category": "C1", "title": "Foo"},
    ])
    out = compute_state_features(events)
    # novelty == (routine == 0) on every row.
    assert ((out["novelty"] == 1) == (out["routine"] == 0)).all()


def test_recency_sentinel_for_first_purchase():
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u2", "order_date": "2022-02-01", "asin": "a",
         "category": "C1", "title": "Foo"},
    ])
    out = compute_state_features(events)
    assert (out["recency_days"] == 999).all(), (
        "every (customer, asin) pair is a first occurrence → all 999."
    )


def test_recency_days_for_repeat():
    """A 2-purchase (u1, a) pair with a 10-day gap → recency_days == 10."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-11", "asin": "a",
         "category": "C1", "title": "Foo"},
    ])
    out = compute_state_features(events)
    out = out.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    # First row is a first-time purchase (sentinel), second row is the repeat.
    assert out.loc[0, "recency_days"] == 999
    assert out.loc[1, "recency_days"] == 10


def test_cat_affinity_counts_prior_in_category():
    """cat_affinity == cumulative count of prior same-category purchases."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-02", "asin": "b",
         "category": "C1", "title": "Bar"},
        {"customer_id": "u1", "order_date": "2022-01-03", "asin": "c",
         "category": "C2", "title": "Baz"},
        {"customer_id": "u1", "order_date": "2022-01-04", "asin": "d",
         "category": "C1", "title": "Qux"},
    ])
    out = compute_state_features(events)
    out = out.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    # C1: 0, 1, (C2: 0), 2
    assert out["cat_affinity"].tolist() == [0, 1, 0, 2]


def test_cat_count_7d_and_30d():
    """cat_count_7d / _30d exist, are non-negative, and 7d <= 30d pointwise."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-03", "asin": "b",
         "category": "C1", "title": "Bar"},
        {"customer_id": "u1", "order_date": "2022-01-20", "asin": "c",
         "category": "C1", "title": "Baz"},
    ])
    out = compute_state_features(events)
    for col in ["cat_count_7d", "cat_count_30d"]:
        assert col in out.columns
        assert (out[col] >= 0).all()
        # rolling sum of an all-ones column is integral even though dtype
        # may be float after the groupby.rolling result.
        assert np.allclose(out[col], out[col].astype(np.int64))
    assert (out["cat_count_7d"] <= out["cat_count_30d"]).all()


def test_brand_fills_unknown_for_ambiguous():
    """Single-occurrence brand tokens collapse to 'unknown_brand'."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Apple AirPods Pro"},
        {"customer_id": "u1", "order_date": "2022-01-02", "asin": "b",
         "category": "C1", "title": "Sony WH1000XM4"},
        # Ambiguous titles that yield empty tokens.
        {"customer_id": "u1", "order_date": "2022-01-03", "asin": "c",
         "category": "C1", "title": ""},
    ])
    out = compute_state_features(events)
    # 'apple' and 'sony' each appear once across ASINs → both ambiguous.
    # ASIN 'c' has no token at all → also 'unknown_brand'.
    assert set(out["brand"].unique()) == {"unknown_brand"}


# --------------------------------------------------------------------------- #
# attach_train_popularity
# --------------------------------------------------------------------------- #


def test_popularity_requires_split():
    """DataFrame without `split` → InvariantError from the pre-check."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
    ])
    # No `split` column on this frame.
    with pytest.raises(InvariantError) as excinfo:
        attach_train_popularity(events)
    err = excinfo.value
    assert err.invariant_name == "split_required_for_popularity"
    assert err.stage == "state_features"


def test_popularity_train_only():
    """Popularity equals the TRAIN count per ASIN, broadcast to all rows."""
    events = _make_events([
        # ASIN "a" appears 3x in train, 1x in val, 1x in test → popularity=3.
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u2", "order_date": "2022-01-02", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u3", "order_date": "2022-01-03", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u4", "order_date": "2022-01-04", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u5", "order_date": "2022-01-05", "asin": "a",
         "category": "C", "title": "Foo"},
    ])
    events["split"] = ["train", "train", "train", "val", "test"]

    out = attach_train_popularity(events)
    # All five rows reference the same ASIN, so every row sees popularity=3.
    assert (out["popularity"] == 3).all()
    assert out["popularity"].dtype == np.int64


def test_popularity_unseen_asin_is_zero():
    """An ASIN that only appears in val (never in train) → popularity=0."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u2", "order_date": "2022-01-02", "asin": "a",
         "category": "C", "title": "Foo"},
        {"customer_id": "u3", "order_date": "2022-01-03", "asin": "b",
         "category": "C", "title": "Bar"},
    ])
    events["split"] = ["train", "train", "val"]

    out = attach_train_popularity(events)
    # ASIN "a": 2 train rows → popularity=2.
    # ASIN "b": only val → popularity=0.
    out_sorted = out.sort_values(["asin", "order_date"]).reset_index(drop=True)
    asin_to_pop = (
        out_sorted.drop_duplicates("asin").set_index("asin")["popularity"].to_dict()
    )
    assert asin_to_pop["a"] == 2
    assert asin_to_pop["b"] == 0


# --------------------------------------------------------------------------- #
# Invariants integration
# --------------------------------------------------------------------------- #


def test_validate_state_features_runs():
    """A corrupted DataFrame (negative recency) must fail validation."""
    events = _make_events([
        {"customer_id": "u1", "order_date": "2022-01-01", "asin": "a",
         "category": "C1", "title": "Foo"},
        {"customer_id": "u1", "order_date": "2022-01-05", "asin": "a",
         "category": "C1", "title": "Foo"},
    ])
    # Monkey-wrap compute_state_features: compute it, then corrupt a value
    # and re-validate via the invariants directly. Simpler and more direct:
    # construct a DataFrame that already has all the state columns, then
    # inject a negative recency and call the validator.
    out = compute_state_features(events).copy()
    out.loc[0, "recency_days"] = -5.0

    # The module re-runs validate_state_features before returning, so
    # recompute via the direct validator to show the invariant fires.
    from src.data.invariants import validate_state_features

    with pytest.raises(InvariantError) as excinfo:
        validate_state_features(out)
    assert excinfo.value.invariant_name == "recency_days_non_negative"


# --------------------------------------------------------------------------- #
# Helpers (brand stopwords + first-brand-token)
# --------------------------------------------------------------------------- #


def test_first_brand_token_skips_stopwords_and_numerics():
    # Stopword skipped, first real brand token returned.
    assert _first_brand_token("Premium Apple AirPods") == "apple"
    # Numeric / pack codes skipped.
    assert _first_brand_token("2-pack Sony Headphones") == "sony"
    # All stopwords → empty.
    assert _first_brand_token("Premium Pro Deluxe") == ""
    # Non-string → empty.
    assert _first_brand_token(None) == ""  # type: ignore[arg-type]


def test_brand_stopwords_is_the_v1_set():
    # A small spot check that the verbatim port is intact.
    for tok in ["premium", "pack", "the", "kit"]:
        assert tok in _BRAND_STOPWORDS
    # And the same instance is used by the helper.
    assert _first_brand_token("pack Sony") == "sony"


def test_build_asin_brand_map_mode():
    """_build_asin_brand_map returns the mode token per ASIN."""
    df = pd.DataFrame({
        "asin": ["a", "a", "a", "b"],
        "title": [
            "Apple Widget", "Apple Gadget", "Sony Gadget",  # 'apple' is mode for a
            "Sony Widget",
        ],
    })
    m = _build_asin_brand_map(df)
    assert m["a"] == "apple"
    assert m["b"] == "sony"
