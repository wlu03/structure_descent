"""Tests for src/data/clean.py (Wave 8, design doc §1).

Fixture: ``tests/fixtures/amazon_events_100.csv`` (30,484 rows of the real
Amazon-purchases export, reduced to 100 customers).
Schema: ``configs/datasets/amazon.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_events
from src.data.invariants import InvariantError
from src.data.schema_map import load_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"
AMAZON_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"

# Fixture row count, checked manually with `wc -l` on the CSV (header
# excluded). If the fixture is regenerated this constant must be updated.
_FIXTURE_ROWS: int = 30_484


# --------------------------------------------------------------------------- #
# Helpers / shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def amazon_schema():
    return load_schema(AMAZON_YAML)


@pytest.fixture(scope="module")
def amazon_raw() -> pd.DataFrame:
    return pd.read_csv(AMAZON_FIXTURE, parse_dates=["Order Date"])


@pytest.fixture(scope="module")
def amazon_cleaned(amazon_raw: pd.DataFrame, amazon_schema) -> pd.DataFrame:
    return clean_events(amazon_raw, amazon_schema)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_clean_amazon_fixture_shapes(amazon_raw: pd.DataFrame, amazon_cleaned: pd.DataFrame) -> None:
    """Fixture starts at 30,484 rows; after clean the count must be equal
    (no required dropna targets are null in this fixture) or smaller. The
    fixture is also canonical on customer_id / order_date / asin, so the
    equality branch is what we actually expect here."""
    assert len(amazon_raw) == _FIXTURE_ROWS
    assert len(amazon_cleaned) <= _FIXTURE_ROWS
    # Canonical columns must be present.
    for c in ("customer_id", "order_date", "asin", "price", "category"):
        assert c in amazon_cleaned.columns, f"missing canonical column {c!r}"


def test_clean_canonical_columns(amazon_cleaned: pd.DataFrame) -> None:
    """Output columns must be a superset of the canonical events schema."""
    expected = {"customer_id", "order_date", "asin", "price", "category"}
    assert expected.issubset(set(amazon_cleaned.columns)), (
        f"missing canonical columns {expected - set(amazon_cleaned.columns)} "
        f"in cleaned output; actual: {list(amazon_cleaned.columns)}"
    )


def test_clean_sorted_by_customer_date(amazon_cleaned: pd.DataFrame) -> None:
    """Within each customer_id group, order_date must be non-decreasing."""
    grouped = amazon_cleaned.groupby("customer_id", sort=False)["order_date"]
    # For every group, check diff() >= 0 (after the first NaN entry).
    for cid, series in grouped:
        diffs = series.diff().dropna()
        if len(diffs) == 0:
            continue
        assert (diffs >= pd.Timedelta(0)).all(), (
            f"customer {cid!r}: order_date not non-decreasing; "
            f"first violating diff: {diffs[diffs < pd.Timedelta(0)].head(1).tolist()}"
        )


def test_clean_reset_index(amazon_cleaned: pd.DataFrame) -> None:
    """The returned index must be a fresh RangeIndex(0..N-1)."""
    n = len(amazon_cleaned)
    assert isinstance(amazon_cleaned.index, pd.RangeIndex), (
        f"expected RangeIndex, got {type(amazon_cleaned.index).__name__}"
    )
    assert list(amazon_cleaned.index[:3]) == [0, 1, 2]
    assert amazon_cleaned.index[-1] == n - 1


def test_clean_idempotent_wrt_canonical_input(
    amazon_cleaned: pd.DataFrame, amazon_schema
) -> None:
    """Cleaning an already-clean DataFrame is a no-op modulo the sort:
    row count is preserved and columns are preserved. This exercises the
    case where the raw->canonical rename is a no-op because the input is
    already canonical (schema_map treats ``"customer_id" -> "customer_id"``
    etc as a pass-through rename).
    """
    # Build a fresh raw-shaped DataFrame from the already-cleaned one by
    # un-renaming the canonical columns back to the raw keys that the
    # schema's events_column_map expects. We only need the dropna_subset
    # canonicals plus category (Amazon YAML's category_null_fill target).
    canonical_to_raw = {
        "order_date": "Order Date",
        "price": "Purchase Price Per Unit",
        "quantity": "Quantity",
        "state": "Shipping Address State",
        "title": "Title",
        "asin": "ASIN/ISBN (Product Code)",
        "category": "Category",
        "customer_id": "Survey ResponseID",
    }
    raw_shaped = amazon_cleaned.rename(columns=canonical_to_raw)
    again = clean_events(raw_shaped, amazon_schema)
    assert len(again) == len(amazon_cleaned)
    assert list(again.columns) == list(amazon_cleaned.columns)


def test_clean_invariant_raises_on_bad_dtype(
    amazon_raw: pd.DataFrame, amazon_schema
) -> None:
    """Inject a negative price into the raw fixture and confirm the
    post-clean invariant fires (``price_non_negative``)."""
    bad = amazon_raw.copy()
    # Force a negative into the price column before translate_events so
    # that coercion preserves it as a valid float64 (just < 0).
    assert "Purchase Price Per Unit" in bad.columns
    bad.loc[bad.index[0], "Purchase Price Per Unit"] = -1.0
    with pytest.raises(InvariantError) as excinfo:
        clean_events(bad, amazon_schema)
    # The failing invariant should be the non-negativity check on price.
    err = excinfo.value
    assert err.invariant_name == "price_non_negative", (
        f"expected price_non_negative, got {err.invariant_name!r}"
    )
    assert err.stage == "clean"


def test_clean_logs_completion(
    amazon_raw: pd.DataFrame, amazon_schema, caplog: pytest.LogCaptureFixture
) -> None:
    """The clean stage must emit a single INFO (or WARNING) record that
    mentions the cleaned row count."""
    caplog.set_level(logging.INFO, logger="src.data.clean")
    out = clean_events(amazon_raw, amazon_schema)

    relevant = [r for r in caplog.records if r.name == "src.data.clean"]
    assert len(relevant) >= 1, "no log records from src.data.clean emitted."
    # On this fixture, no rows are dropped (< 5%), so the top-level line
    # should be INFO, not WARNING.
    top = relevant[-1]
    assert top.levelno == logging.INFO, (
        f"expected INFO (drop fraction << 5%); got level {top.levelname!r} "
        f"with message {top.getMessage()!r}"
    )
    msg = top.getMessage()
    assert str(len(out)) in msg, (
        f"expected row count {len(out)} in log message; got {msg!r}"
    )
    assert "cleaned" in msg.lower()
