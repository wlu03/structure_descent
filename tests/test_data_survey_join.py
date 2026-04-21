"""Tests for :mod:`src.data.survey_join` (Wave 8, design doc §2).

Exercises the ``load -> translate_events -> join_survey`` slice against
the real 100-customer Amazon fixture under ``tests/fixtures/``. A few
unit cases use hand-crafted DataFrames to isolate the drop-all-null,
dedupe, and orphan-warning behaviours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import clean_events
from src.data.invariants import InvariantError
from src.data.load import load
from src.data.schema_map import DatasetSchema, load_schema
from src.data.survey_join import _snake_case, join_survey


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# Canonical events columns. Post-join, anything NOT in this set (other than
# customer_id which is the join key) is a joined survey column.
_CANONICAL_EVENT_COLS = {
    "customer_id",
    "order_date",
    "price",
    "quantity",
    "state",
    "title",
    "asin",
    "category",
}


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def amazon_schema() -> DatasetSchema:
    return load_schema(AMAZON_YAML)


@pytest.fixture(scope="module")
def amazon_raw(amazon_schema: DatasetSchema):
    events_raw, persons_raw = load(
        amazon_schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    return events_raw, persons_raw


@pytest.fixture(scope="module")
def amazon_clean(amazon_schema: DatasetSchema, amazon_raw):
    events_raw, persons_raw = amazon_raw
    events_clean = clean_events(events_raw, amazon_schema)
    return events_clean, persons_raw


# --------------------------------------------------------------------------- #
# Happy-path against the real fixture.
# --------------------------------------------------------------------------- #


def test_join_preserves_events_row_count(amazon_schema, amazon_clean):
    events_clean, persons_raw = amazon_clean
    merged = join_survey(events_clean, persons_raw, amazon_schema)
    assert len(merged) == len(events_clean)


def test_join_adds_survey_columns(amazon_schema, amazon_clean):
    events_clean, persons_raw = amazon_clean
    merged = join_survey(events_clean, persons_raw, amazon_schema)

    # Every pre-existing event column must survive the join.
    assert set(events_clean.columns).issubset(set(merged.columns))

    survey_cols = [c for c in merged.columns if c not in set(events_clean.columns)]
    assert len(survey_cols) >= 10, (
        f"expected at least 10 joined survey columns; got {len(survey_cols)}: "
        f"{survey_cols}"
    )
    # Every joined survey column must be snake_cased (lowercase + digits +
    # underscores, no leading / trailing underscores).
    for col in survey_cols:
        assert col == _snake_case(col), (
            f"joined survey column {col!r} is not snake_cased."
        )


def test_join_id_column_stays_customer_id(amazon_schema, amazon_clean):
    events_clean, persons_raw = amazon_clean
    merged = join_survey(events_clean, persons_raw, amazon_schema)

    assert "customer_id" in merged.columns

    # A snake_cased derivative of "Survey ResponseID" must not appear; the
    # id column is kept as canonical `customer_id`.
    forbidden = {"survey_responseid", "survey_response_id"}
    assert forbidden.isdisjoint(set(merged.columns))


def test_join_snake_case(amazon_schema, amazon_clean):
    """The Amazon survey ships Q-demos-age; it must appear snake_cased."""
    events_clean, persons_raw = amazon_clean
    merged = join_survey(events_clean, persons_raw, amazon_schema)
    assert "q_demos_age" in merged.columns
    # Raw (un-snaked) variants must not leak through.
    for raw in ("Q-demos-age", "Q_demos_age", "q-demos-age"):
        assert raw not in merged.columns


def test_snake_case_helper_shape():
    """Direct unit-test of the v1-verbatim snake_case helper."""
    assert _snake_case("Q-demos-age") == "q_demos_age"
    assert _snake_case("  Foo Bar!!  ") == "foo_bar"
    assert _snake_case("Survey ResponseID") == "survey_responseid"
    assert _snake_case("Q-demos-income") == "q_demos_income"


# --------------------------------------------------------------------------- #
# Behaviour on synthetic persons frames.
# --------------------------------------------------------------------------- #


def _tiny_events() -> pd.DataFrame:
    """Three cleaned events, three distinct customers."""
    return pd.DataFrame(
        {
            "customer_id": pd.Series(["c1", "c2", "c3"], dtype="object"),
            "order_date": pd.to_datetime(
                ["2022-01-01", "2022-02-01", "2022-03-01"]
            ),
            "price": pd.Series([9.99, 1.0, 0.0], dtype="float64"),
            "asin": pd.Series(["A1", "A2", "A3"], dtype="object"),
            "category": pd.Series(["cat1", "cat2", "cat1"], dtype="object"),
        }
    )


def test_join_drops_all_null_columns(amazon_schema):
    """An all-null column on the persons frame must be absent post-join."""
    events = _tiny_events()
    persons = pd.DataFrame(
        {
            "Survey ResponseID": ["c1", "c2", "c3"],
            "Q-demos-age": ["18 - 24 years", "25 - 34 years", "35 - 44 years"],
            "Q-all-null": [None, None, None],
        }
    )
    merged = join_survey(events, persons, amazon_schema)
    # The snake_cased version of Q-all-null must not appear at all.
    assert "q_all_null" not in merged.columns
    # The non-null survey column survives the join.
    assert "q_demos_age" in merged.columns


def test_join_dedupes_persons(amazon_schema):
    """Duplicate customer_id rows in persons → keep first; every event
    carries exactly one joined survey row."""
    events = _tiny_events()
    persons = pd.DataFrame(
        {
            "Survey ResponseID": ["c1", "c1", "c2", "c3"],
            # Two distinct "Q-demos-age" values for c1 — keep_first means the
            # first one ("18 - 24 years") wins.
            "Q-demos-age": [
                "18 - 24 years",
                "25 - 34 years",
                "35 - 44 years",
                "45 - 54 years",
            ],
        }
    )
    merged = join_survey(events, persons, amazon_schema)
    assert len(merged) == len(events)
    # c1 must carry the *first* survey row.
    c1_rows = merged.loc[merged["customer_id"] == "c1"]
    assert len(c1_rows) == 1
    assert c1_rows["q_demos_age"].iloc[0] == "18 - 24 years"


def test_join_logs_orphan_count(amazon_schema, amazon_clean, caplog):
    """When some events have no matching persons row a WARNING must be
    logged with both the count and the percentage."""
    events_clean, persons_raw = amazon_clean

    # Remove a single customer from persons so their events become orphans.
    # The fixture has 100 matched customers; drop the one with the fewest
    # events so the orphan fraction stays well under the 5% InvariantError
    # threshold in validate_joined.
    n_by_cust = events_clean.groupby("customer_id").size().sort_values()
    drop_cid = str(n_by_cust.index[0])
    n_orphan_expected = int(n_by_cust.iloc[0])

    persons_drop = persons_raw.loc[
        persons_raw["Survey ResponseID"].astype(str) != drop_cid
    ].copy()
    assert len(persons_drop) == len(persons_raw) - 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.data.survey_join"):
        merged = join_survey(events_clean, persons_drop, amazon_schema)

    # Row count preserved — left-join keeps orphans.
    assert len(merged) == len(events_clean)

    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "src.data.survey_join"
    ]
    assert warnings, (
        f"expected a WARNING from src.data.survey_join; got records: "
        f"{[(r.name, r.levelno, r.getMessage()) for r in caplog.records]!r}"
    )
    orphan_pct = 100.0 * n_orphan_expected / len(events_clean)
    # At least one warning names the count and the percentage.
    assert any(
        str(n_orphan_expected) in msg
        and f"{orphan_pct:.2f}" in msg
        for msg in warnings
    ), f"no WARNING with count {n_orphan_expected} and pct {orphan_pct:.2f}; got {warnings!r}"


def test_join_invariant_validates(amazon_schema, amazon_clean):
    """A corrupted join (null customer_id in the joined events) must raise
    :class:`InvariantError` via the ``validate_joined`` hook."""
    events_clean, persons_raw = amazon_clean

    # Inject a null customer_id on one event row. translate_events already
    # coerced customer_id to object dtype, so None passes through.
    events_bad = events_clean.copy()
    events_bad.loc[0, "customer_id"] = None

    with pytest.raises(InvariantError) as exc_info:
        join_survey(events_bad, persons_raw, amazon_schema)
    assert exc_info.value.stage == "survey_join"
    assert exc_info.value.invariant_name == "customer_id_non_null"


def test_join_missing_customer_id_column_raises(amazon_schema, amazon_clean):
    """events_df without a ``customer_id`` column is a programmer error."""
    events_clean, persons_raw = amazon_clean
    bad = events_clean.drop(columns=["customer_id"])
    with pytest.raises(KeyError):
        join_survey(bad, persons_raw, amazon_schema)


def test_join_missing_persons_id_column_raises(amazon_schema, amazon_clean):
    """persons_raw without ``schema.persons_id_column`` is a programmer error."""
    events_clean, persons_raw = amazon_clean
    bad = persons_raw.drop(columns=[amazon_schema.persons_id_column])
    with pytest.raises(KeyError):
        join_survey(events_clean, bad, amazon_schema)
