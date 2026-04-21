"""Tests for src/data/load.py (Wave 8, design doc §1).

Exercises the end-to-end ``load_schema`` -> ``load`` path against the
real 100-customer fixture under ``tests/fixtures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.invariants import InvariantError
from src.data.load import load
from src.data.schema_map import load_schema


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


@pytest.fixture(scope="module")
def amazon_schema():
    return load_schema(AMAZON_YAML)


def test_load_amazon_fixture_shapes(amazon_schema):
    """schema + override paths → expected fixture row/column counts."""
    events_raw, persons_raw = load(
        amazon_schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    assert isinstance(events_raw, pd.DataFrame)
    assert isinstance(persons_raw, pd.DataFrame)
    assert events_raw.shape == (30484, 8)
    assert persons_raw.shape == (100, 23)


def test_load_parses_order_date_as_datetime(amazon_schema):
    """events.`Order Date` must be parsed as datetime64[ns] per schema."""
    events_raw, _ = load(
        amazon_schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    assert "Order Date" in events_raw.columns
    assert str(events_raw["Order Date"].dtype) == "datetime64[ns]"


def test_load_validates_raw_schema(amazon_schema, tmp_path):
    """Corrupted events CSV (missing `Order Date`) must raise InvariantError."""
    good = pd.read_csv(EVENTS_FIXTURE)
    corrupt = good.drop(columns=["Order Date"])
    bad_path = tmp_path / "amazon_events_bad.csv"
    corrupt.to_csv(bad_path, index=False)

    with pytest.raises(InvariantError) as excinfo:
        load(
            amazon_schema,
            events_path=bad_path,
            persons_path=PERSONS_FIXTURE,
        )
    # The validate_loaded raises on the canonical-satisfied rule; make sure
    # we did not slip past it by checking the rule name on the exception.
    assert excinfo.value.invariant_name == "events_column_map_canonical_satisfied"


def test_load_missing_file_raises_filenotfound(amazon_schema, tmp_path):
    """Nonexistent path → FileNotFoundError (not pandas's own error)."""
    ghost = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load(
            amazon_schema,
            events_path=ghost,
            persons_path=PERSONS_FIXTURE,
        )
    with pytest.raises(FileNotFoundError):
        load(
            amazon_schema,
            events_path=EVENTS_FIXTURE,
            persons_path=ghost,
        )


def test_load_logs_row_counts(amazon_schema, caplog):
    """An INFO log must mention both n_events and n_persons."""
    with caplog.at_level(logging.INFO, logger="src.data.load"):
        load(
            amazon_schema,
            events_path=EVENTS_FIXTURE,
            persons_path=PERSONS_FIXTURE,
        )
    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        "30484" in msg and "100" in msg and "events" in msg and "persons" in msg
        for msg in info_messages
    ), f"no INFO record with row counts; got: {info_messages!r}"
