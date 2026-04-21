"""Tests for src/data/invariants.py (Wave 8 design doc §3)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.invariants import (
    InvariantError,
    assert_columns_present,
    assert_dtype,
    assert_no_nan,
    assert_non_negative,
    assert_values_in_set,
    validate_cleaned,
    validate_joined,
    validate_loaded,
    validate_popularity,
    validate_split,
    validate_state_features,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_schema() -> SimpleNamespace:
    """Duck-typed stand-in for ``DatasetSchema`` used by validate_loaded.

    Mirrors the Amazon schema's raw column names (pre-rename) so the
    ``events_column_map_keys_present`` invariant is exercised against
    realistic keys.
    """
    events_column_map = {
        "Order Date": "order_date",
        "Purchase Price Per Unit": "price",
        "Quantity": "quantity",
        "Shipping Address State": "state",
        "Title": "title",
        "ASIN/ISBN (Product Code)": "asin",
        "Category": "category",
        "Survey ResponseID": "customer_id",
    }
    return SimpleNamespace(
        events_column_map=events_column_map,
        persons_id_column="Survey ResponseID",
    )


def _valid_raw_events() -> pd.DataFrame:
    schema = _make_schema()
    row = {raw: "x" for raw in schema.events_column_map.keys()}
    return pd.DataFrame([row, row])


def _valid_raw_persons() -> pd.DataFrame:
    return pd.DataFrame({"Survey ResponseID": ["c1", "c2", "c3"]})


def _valid_cleaned_events() -> pd.DataFrame:
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


def _valid_joined_events() -> pd.DataFrame:
    """Cleaned events with a few joined survey columns (all non-null)."""
    base = _valid_cleaned_events()
    base["Q-demos-age"] = ["25 - 34 years", "35 - 44 years", "45 - 54 years"]
    base["Q-demos-income"] = [
        "$25,000 - $49,999",
        "$50,000 - $74,999",
        "$100,000 - $149,999",
    ]
    return base


def _valid_state_feature_events() -> pd.DataFrame:
    base = _valid_joined_events()
    base["routine"] = [0.1, 0.5, 1.0]
    base["novelty"] = [0.0, 1.0, 0.5]
    base["cat_affinity"] = [0.0, 0.2, 0.3]
    base["recency_days"] = [0.0, 10.0, 999.0]  # sentinel allowed
    return base


def _valid_split_events() -> pd.DataFrame:
    """Three customers, each with at least one train row."""
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2", "c3"],
            "split": ["train", "val", "train", "test", "train"],
        }
    )


def _valid_popularity_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": ["A1", "A2", "A3"],
            "popularity": pd.Series([5, 10, 2], dtype="int64"),
        }
    )


# --------------------------------------------------------------------------- #
# InvariantError basic behavior
# --------------------------------------------------------------------------- #


def test_invariant_error_carries_structured_context() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    err = InvariantError(
        "some_rule",
        "some_stage",
        "something broke",
        column="x",
        offending=df,
    )
    assert err.invariant_name == "some_rule"
    assert err.stage == "some_stage"
    assert err.column == "x"
    assert err.offending is df
    s = str(err)
    assert "invariant=some_rule" in s
    assert "stage=some_stage" in s
    assert "column=x" in s
    assert "offending rows" in s


def test_invariant_error_is_assertion_error() -> None:
    err = InvariantError("r", "s", "m")
    assert isinstance(err, AssertionError)


# --------------------------------------------------------------------------- #
# Shared helpers (spot-checks; validators exercise them in full)
# --------------------------------------------------------------------------- #


def test_assert_columns_present_green() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert_columns_present(df, ["a", "b"], invariant_name="ok", stage="s")


def test_assert_columns_present_red() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(InvariantError) as exc_info:
        assert_columns_present(
            df, ["a", "missing"], invariant_name="ok", stage="s"
        )
    assert exc_info.value.invariant_name == "ok"
    assert exc_info.value.stage == "s"


def test_assert_no_nan_attaches_head5() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    with pytest.raises(InvariantError) as exc_info:
        assert_no_nan(df, "x", invariant_name="nn", stage="s")
    err = exc_info.value
    assert err.offending is not None
    # At most 5 offending rows carried.
    assert len(err.offending) == 5


def test_assert_non_negative_sentinel_allowed() -> None:
    df = pd.DataFrame({"r": [0.0, 5.0, 999.0]})
    assert_non_negative(
        df, "r", invariant_name="rn", stage="s", allow_sentinel=999.0
    )


def test_assert_non_negative_rejects_nan() -> None:
    df = pd.DataFrame({"r": [0.0, float("nan")]})
    with pytest.raises(InvariantError):
        assert_non_negative(df, "r", invariant_name="rn", stage="s")


def test_assert_dtype_red() -> None:
    df = pd.DataFrame({"x": [1, 2]})  # int64
    with pytest.raises(InvariantError):
        assert_dtype(df, "x", "float64", invariant_name="dt", stage="s")


def test_assert_values_in_set_red() -> None:
    df = pd.DataFrame({"split": ["train", "bogus"]})
    with pytest.raises(InvariantError):
        assert_values_in_set(
            df, "split", {"train", "val", "test"},
            invariant_name="vs", stage="s",
        )


# --------------------------------------------------------------------------- #
# validate_loaded
# --------------------------------------------------------------------------- #


def test_validate_loaded_green() -> None:
    schema = _make_schema()
    validate_loaded(_valid_raw_events(), _valid_raw_persons(), schema)


def test_validate_loaded_missing_events_col() -> None:
    schema = _make_schema()
    events = _valid_raw_events().drop(columns=["Order Date"])
    with pytest.raises(InvariantError) as exc_info:
        validate_loaded(events, _valid_raw_persons(), schema)
    assert exc_info.value.invariant_name == "events_column_map_canonical_satisfied"
    assert exc_info.value.stage == "load"


def test_validate_loaded_alias_satisfied() -> None:
    """Multiple raw keys mapping to the same canonical: presence of ANY one satisfies.

    Amazon's YAML maps both ``"ASIN/ISBN (Product Code)"`` (current export) and
    ``"ASIN/ISBN"`` (older export) to canonical ``asin``. Only one spelling
    will ever be present in a given dataset; the invariant must pass.
    """
    base = _make_schema()
    aliased_map = dict(base.events_column_map)
    aliased_map["ASIN/ISBN"] = "asin"   # add the alias; both now map to asin
    schema = SimpleNamespace(
        events_column_map=aliased_map,
        persons_id_column=base.persons_id_column,
    )
    # Fixture only has the new-spelling column — alias not present.
    validate_loaded(_valid_raw_events(), _valid_raw_persons(), schema)


def test_validate_loaded_missing_persons_id_col() -> None:
    schema = _make_schema()
    persons = _valid_raw_persons().drop(columns=["Survey ResponseID"])
    with pytest.raises(InvariantError) as exc_info:
        validate_loaded(_valid_raw_events(), persons, schema)
    assert exc_info.value.invariant_name == "persons_id_column_present"
    assert exc_info.value.stage == "load"


# --------------------------------------------------------------------------- #
# validate_cleaned
# --------------------------------------------------------------------------- #


def test_validate_cleaned_green() -> None:
    schema = _make_schema()
    validate_cleaned(_valid_cleaned_events(), schema)


def test_validate_cleaned_null_customer_id() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df.loc[0, "customer_id"] = None
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "customer_id_non_null"
    assert exc_info.value.stage == "clean"


def test_validate_cleaned_bad_order_date() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df.loc[1, "order_date"] = pd.NaT
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "order_date_no_nat"
    assert exc_info.value.stage == "clean"


def test_validate_cleaned_wrong_order_date_dtype() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df["order_date"] = df["order_date"].astype(str)
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "order_date_dtype"


def test_validate_cleaned_negative_price() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df.loc[2, "price"] = -1.0
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "price_non_negative"
    assert exc_info.value.stage == "clean"


def test_validate_cleaned_null_asin() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df.loc[0, "asin"] = None
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "asin_non_null"


def test_validate_cleaned_null_category() -> None:
    schema = _make_schema()
    df = _valid_cleaned_events()
    df.loc[0, "category"] = None
    with pytest.raises(InvariantError) as exc_info:
        validate_cleaned(df, schema)
    assert exc_info.value.invariant_name == "category_non_null"


# --------------------------------------------------------------------------- #
# validate_joined
# --------------------------------------------------------------------------- #


def test_validate_joined_green() -> None:
    schema = _make_schema()
    validate_joined(_valid_joined_events(), schema)


def test_validate_joined_orphan_under_threshold_warns(caplog) -> None:
    """One orphan row out of 40 (2.5%) is below the 5% threshold — warns only."""
    schema = _make_schema()
    base = _valid_joined_events()
    # Replicate to 40 rows; null out survey cols on just row 0.
    df = pd.concat([base] * 14, ignore_index=True).iloc[:40].reset_index(drop=True)
    df.loc[0, "Q-demos-age"] = None
    df.loc[0, "Q-demos-income"] = None

    with caplog.at_level(logging.WARNING, logger="src.data.invariants"):
        validate_joined(df, schema)

    assert any(
        "survey-join miss rate" in rec.getMessage() for rec in caplog.records
    ), f"expected WARNING on join miss rate; got: {[r.getMessage() for r in caplog.records]!r}"


def test_validate_joined_orphan_over_threshold_raises() -> None:
    schema = _make_schema()
    df = _valid_joined_events()
    # 1 orphan out of 3 == 33% miss rate, well above 5%.
    df.loc[0, "Q-demos-age"] = None
    df.loc[0, "Q-demos-income"] = None
    with pytest.raises(InvariantError) as exc_info:
        validate_joined(df, schema)
    assert exc_info.value.invariant_name == "no_orphan_customers"
    assert exc_info.value.stage == "survey_join"


def test_validate_joined_null_customer_id() -> None:
    schema = _make_schema()
    df = _valid_joined_events()
    df.loc[0, "customer_id"] = None
    with pytest.raises(InvariantError) as exc_info:
        validate_joined(df, schema)
    assert exc_info.value.invariant_name == "customer_id_non_null"


# --------------------------------------------------------------------------- #
# validate_state_features
# --------------------------------------------------------------------------- #


def test_validate_state_features_green() -> None:
    validate_state_features(_valid_state_feature_events())


def test_validate_state_features_missing_routine_column() -> None:
    df = _valid_state_feature_events().drop(columns=["routine"])
    with pytest.raises(InvariantError) as exc_info:
        validate_state_features(df)
    assert exc_info.value.invariant_name == "state_feature_columns_present"
    assert exc_info.value.stage == "state_features"


def test_validate_state_features_negative_recency_raises() -> None:
    df = _valid_state_feature_events()
    df.loc[0, "recency_days"] = -1.0
    with pytest.raises(InvariantError) as exc_info:
        validate_state_features(df)
    assert exc_info.value.invariant_name == "recency_days_non_negative"


def test_validate_state_features_nan_recency_raises() -> None:
    df = _valid_state_feature_events()
    df.loc[0, "recency_days"] = float("nan")
    with pytest.raises(InvariantError) as exc_info:
        validate_state_features(df)
    assert exc_info.value.invariant_name == "recency_days_non_negative"


def test_validate_state_features_999_sentinel_passes() -> None:
    df = _valid_state_feature_events()
    df["recency_days"] = [999.0, 999.0, 999.0]
    validate_state_features(df)


def test_validate_state_features_negative_routine_raises() -> None:
    df = _valid_state_feature_events()
    df.loc[0, "routine"] = -0.1
    with pytest.raises(InvariantError) as exc_info:
        validate_state_features(df)
    assert exc_info.value.invariant_name == "routine_non_negative"


# --------------------------------------------------------------------------- #
# validate_split
# --------------------------------------------------------------------------- #


def test_validate_split_green() -> None:
    validate_split(_valid_split_events())


def test_validate_split_bad_label() -> None:
    df = _valid_split_events()
    df.loc[0, "split"] = "bogus"
    with pytest.raises(InvariantError) as exc_info:
        validate_split(df)
    assert exc_info.value.invariant_name == "split_values_subset"
    assert exc_info.value.stage == "split"


def test_validate_split_missing_train_for_customer() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c3"],
            "split": ["train", "val", "val", "test"],  # c2, c3 have no train row
        }
    )
    with pytest.raises(InvariantError) as exc_info:
        validate_split(df)
    assert exc_info.value.invariant_name == "every_customer_has_train_row"
    assert exc_info.value.stage == "split"


# --------------------------------------------------------------------------- #
# validate_popularity
# --------------------------------------------------------------------------- #


def test_validate_popularity_green() -> None:
    validate_popularity(_valid_popularity_events())


def test_validate_popularity_zero_warns(caplog) -> None:
    df = _valid_popularity_events()
    df.loc[0, "popularity"] = 0
    with caplog.at_level(logging.WARNING, logger="src.data.invariants"):
        validate_popularity(df)
    assert any(
        "popularity == 0" in rec.getMessage() for rec in caplog.records
    ), f"expected unseen-ASIN warning; got: {[r.getMessage() for r in caplog.records]!r}"


def test_validate_popularity_negative() -> None:
    df = _valid_popularity_events()
    df.loc[0, "popularity"] = -1
    with pytest.raises(InvariantError) as exc_info:
        validate_popularity(df)
    assert exc_info.value.invariant_name == "popularity_non_negative"
    assert exc_info.value.stage == "popularity"


def test_validate_popularity_wrong_dtype() -> None:
    df = _valid_popularity_events()
    df["popularity"] = df["popularity"].astype("float64")
    with pytest.raises(InvariantError) as exc_info:
        validate_popularity(df)
    assert exc_info.value.invariant_name == "popularity_dtype"
