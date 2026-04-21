"""Tests for src/data/split.py (Wave 8, design doc §1).

Covers the per-customer temporal split, schema/kwargs contract, and the
v1 edge-case logic for small customers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_events
from src.data.invariants import InvariantError
from src.data.load import load
from src.data.schema_map import load_schema
from src.data.split import temporal_split


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def amazon_schema():
    return load_schema(AMAZON_YAML)


@pytest.fixture(scope="module")
def cleaned_events(amazon_schema) -> pd.DataFrame:
    """Real cleaned events from the 100-customer fixture."""
    events_raw, _ = load(
        amazon_schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    return clean_events(events_raw, amazon_schema)


def _make_events(n_by_customer: dict[str, int]) -> pd.DataFrame:
    """Build a tiny synthetic events frame for edge-case tests.

    ``n_by_customer`` maps ``customer_id -> n_events``. Each customer's
    events are stamped on consecutive days starting 2020-01-01 so that
    temporal ordering is unambiguous.
    """
    rows = []
    base = pd.Timestamp("2020-01-01")
    for cid, n in n_by_customer.items():
        for i in range(n):
            rows.append(
                {
                    "customer_id": cid,
                    "order_date": base + pd.Timedelta(days=i),
                    "price": 1.0 + i,
                    "asin": f"A{i:04d}",
                    "category": "CAT",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_temporal_split_adds_column(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert "split" in out.columns


def test_split_values_subset_of_train_val_test(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert set(out["split"].unique()).issubset({"train", "val", "test"})


def test_every_customer_has_train_row(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    train_customers = set(out.loc[out["split"] == "train", "customer_id"].unique())
    all_customers = set(out["customer_id"].unique())
    assert train_customers == all_customers


def test_large_customer_has_expected_fractions():
    df = _make_events({"C_big": 100})
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    c = out.loc[out["customer_id"] == "C_big", "split"].value_counts()
    # int(100 * 0.1) == 10 exactly, with n_train = 80.
    assert int(c.get("test", 0)) == 10
    assert int(c.get("val", 0)) == 10
    assert int(c.get("train", 0)) == 80


def test_small_customer_gets_at_least_one_train():
    # With n=2, val_frac=test_frac=0.1: int(0.2)=0 -> max(1, 0)=1 each,
    # so n_test + n_val == 2 == n, triggering the collapse to (1, 0).
    # That gives n_train=1, n_val=0, n_test=1.
    df = _make_events({"C_small": 2})
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    c = out.loc[out["customer_id"] == "C_small", "split"].value_counts()
    assert int(c.get("train", 0)) == 1
    assert int(c.get("val", 0)) == 0
    assert int(c.get("test", 0)) == 1


def test_1_event_customer_is_all_train():
    """Document the v1 edge-case interaction with validate_split.

    Under v1's exactly-ported allocation, a 1-event customer yields
    ``n_test=max(1,0)=1``, ``n_val=max(1,0)=1``; the collapse sets
    ``(n_test, n_val) = (1, 0)`` and ``n_train = 1 - 1 - 0 = 0``. That
    single row is labelled ``"test"``, so :func:`validate_split` rejects
    the frame for having no train row for that customer. The v1 logic
    therefore does NOT support standalone 1-event customers — the test
    pins that behaviour so the invariant hook stays loud.
    """
    df = _make_events({"C_one": 1})
    with pytest.raises(InvariantError):
        temporal_split(df, val_frac=0.1, test_frac=0.1)


def test_split_is_temporal():
    df = _make_events({"C_a": 20, "C_b": 30})
    # Shuffle the rows to ensure temporal_split re-sorts.
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    for cid, grp in out.groupby("customer_id"):
        # Within each customer, dates must be non-decreasing.
        dates = grp["order_date"].tolist()
        assert dates == sorted(dates)
        # The earliest row's split must be 'train' and the latest 'test'.
        assert grp.iloc[0]["split"] == "train"
        assert grp.iloc[-1]["split"] == "test"


def test_split_preserves_row_count(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert len(out) == len(cleaned_events)


def test_split_reset_index(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert list(out.index) == list(range(len(out)))


# --------------------------------------------------------------------------- #
# schema-or-kwargs contract
# --------------------------------------------------------------------------- #


def test_split_uses_schema_defaults(amazon_schema, cleaned_events):
    # schema only, no explicit fractions.
    out = temporal_split(cleaned_events, schema=amazon_schema)
    assert "split" in out.columns
    # Compare against an explicit-kwargs run with the schema's values.
    out_explicit = temporal_split(
        cleaned_events,
        val_frac=amazon_schema.val_frac,
        test_frac=amazon_schema.test_frac,
    )
    pd.testing.assert_series_equal(out["split"], out_explicit["split"])


def test_split_explicit_kwargs_override_schema(amazon_schema):
    # Build a frame large enough that 0.1 vs 0.4 produce different counts.
    df = _make_events({"C_big": 100})
    out_schema = temporal_split(df, schema=amazon_schema)
    out_override = temporal_split(
        df, schema=amazon_schema, val_frac=0.4, test_frac=0.4
    )
    # Different fractions must yield different per-split counts.
    cs = out_schema["split"].value_counts()
    co = out_override["split"].value_counts()
    assert int(cs.get("test", 0)) != int(co.get("test", 0))
    # Override uses 0.4 -> int(100 * 0.4) = 40 for each of val/test.
    assert int(co.get("val", 0)) == 40
    assert int(co.get("test", 0)) == 40
    assert int(co.get("train", 0)) == 20


def test_split_without_schema_or_kwargs_raises():
    df = _make_events({"C_a": 10})
    with pytest.raises(ValueError):
        temporal_split(df)
    # Partial-kwargs (only one of the two) with no schema also raises.
    with pytest.raises(ValueError):
        temporal_split(df, val_frac=0.1)
    with pytest.raises(ValueError):
        temporal_split(df, test_frac=0.1)
