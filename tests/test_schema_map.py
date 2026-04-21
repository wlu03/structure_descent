"""Tests for src/data/schema_map.py (Wave 8).

The primary fixture is ``configs/datasets/amazon.yaml``; additional tests use
tiny hand-built schemas or ZDFieldSpec instances to exercise a single kind.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schema_map import (
    CompositeFormulaError,
    DatasetSchema,
    UnknownCategoryError,
    ZDFieldSpec,
    load_schema,
    translate_events,
    translate_persons,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"

Z_D_ORDER = (
    "age_bucket",
    "income_bucket",
    "household_size",
    "has_kids",
    "city_size",
    "education",
    "health_rating",
    "risk_tolerance",
    "purchase_frequency",
    "novelty_rate",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _amazon_schema() -> DatasetSchema:
    return load_schema(AMAZON_YAML)


def _raw_amazon_events() -> pd.DataFrame:
    """A tiny events DataFrame with Amazon raw headers."""
    return pd.DataFrame(
        {
            "Order Date": ["2020-05-01", "2020-05-02", "2020-05-03"],
            "Purchase Price Per Unit": ["10.99", "5.50", None],
            "Quantity": [1, 2, 1],
            "Shipping Address State": ["CA", "NY", "TX"],
            "Title": ["a", "b", "c"],
            "ASIN/ISBN (Product Code)": ["A1", "A2", "A3"],
            "Category": ["Books", None, "Electronics"],
            "Survey ResponseID": ["u1", "u2", "u3"],
        }
    )


def _raw_amazon_persons() -> pd.DataFrame:
    """Tiny persons DF with Amazon raw headers matching amazon.yaml sources."""
    return pd.DataFrame(
        {
            "Survey ResponseID": ["u1", "u2", "u3", "u4", "u5"],
            "Q-demos-age": [
                "18 - 24 years",
                "35 - 44 years",
                "25 - 34 years",
                "65 and older",
                "45 - 54 years",
            ],
            "Q-demos-income": [
                "Less than $25,000",
                "$50,000 - $74,999",
                "$75,000 - $99,999",
                "Prefer not to say",  # will be dropped
                "$150,000 or more",
            ],
            "Q-amazon-use-hh-size": ["1 (just me!)", "2", "3", "4+", "2"],
            "Q-demos-state": ["CA", "NY", "TX", "FL", "WA"],
            "Q-demos-education": [
                "Some high school or less",
                "High school diploma or GED",
                "Bachelor's degree",
                "Bachelor's degree",
                "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS, etc)",
            ],
            "Q-personal-diabetes": ["No", "Yes", "No", "Yes", "No"],
            "Q-personal-wheelchair": ["No", "No", "No", "Yes", "No"],
            "Q-substance-use-cigarettes": ["No", "Yes", "No", "No", "No"],
            "Q-substance-use-marijuana": ["No", "No", "Yes", "Yes", "No"],
            "Q-substance-use-alcohol": ["Yes", "Yes", "Yes", "Yes", "No"],
        }
    )


def _amazon_training_events() -> pd.DataFrame:
    """Simulated training-split events with canonical column names + novelty."""
    return pd.DataFrame(
        {
            "customer_id": [
                "u1", "u1", "u1",         # 3 events
                "u2", "u2",                # 2 events
                "u3",                      # 1 event
                "u5", "u5", "u5", "u5",    # 4 events
                # u4 has none (but u4 is dropped by drop_on_unknown anyway)
            ],
            "order_date": pd.to_datetime(
                ["2020-05-01"] * 3 + ["2020-05-02"] * 2 + ["2020-05-03"]
                + ["2020-05-04"] * 4
            ),
            "novelty": [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],  # means: u1=2/3, u2=1, u3=0, u5=1
        }
    )


# --------------------------------------------------------------------------- #
# YAML load + events translation
# --------------------------------------------------------------------------- #


def test_load_amazon_yaml():
    schema = _amazon_schema()
    assert schema.name == "amazon"
    assert len(schema.z_d_mapping) == 10
    # The YAML shape must yield every canonical z_d column, in order.
    assert tuple(s.canonical_column for s in schema.z_d_mapping) == Z_D_ORDER
    # Every raw Amazon header referenced in events.column_map is recorded.
    raw_keys = set(schema.events_column_map.keys())
    for expected in (
        "Order Date",
        "Purchase Price Per Unit",
        "Quantity",
        "Shipping Address State",
        "Title",
        "ASIN/ISBN (Product Code)",
        "Category",
        "Survey ResponseID",
    ):
        assert expected in raw_keys


def test_events_column_map_rename():
    schema = _amazon_schema()
    df = translate_events(_raw_amazon_events(), schema)
    for col in ("customer_id", "order_date", "asin", "price", "category", "title"):
        assert col in df.columns
    # Category NaN filled with "Unknown".
    assert (df["category"].astype(str) != "nan").all()
    assert "Unknown" in set(df["category"])
    # Price coerced to float.
    assert df["price"].dtype == float
    # dropna_subset removes rows with any of customer_id/order_date/asin NaN.
    # None of the test rows should be dropped.
    assert len(df) == 3


def test_translate_events_strict_on_missing_raw_col():
    schema = _amazon_schema()
    raw = _raw_amazon_events().drop(columns=["Order Date"])
    with pytest.raises(KeyError, match="order_date"):
        translate_events(raw, schema)


# --------------------------------------------------------------------------- #
# drop_on_unknown + UnknownCategoryError
# --------------------------------------------------------------------------- #


def test_drop_on_unknown():
    schema = _amazon_schema()
    out = translate_persons(
        _raw_amazon_persons(),
        schema,
        training_events=_amazon_training_events(),
    )
    # u4 had income_bucket == "Prefer not to say" and must be dropped.
    assert list(out["customer_id"]) == ["u1", "u2", "u3", "u5"]


def test_unknown_categorical_raises():
    schema = _amazon_schema()
    persons = _raw_amazon_persons()
    persons.loc[0, "Q-demos-age"] = "foo"
    with pytest.raises(UnknownCategoryError, match="Q-demos-age"):
        translate_persons(
            persons, schema, training_events=_amazon_training_events()
        )


# --------------------------------------------------------------------------- #
# Kind handlers (one focused test per kind)
# --------------------------------------------------------------------------- #


def _mini_schema_single_field(spec: ZDFieldSpec, *, id_col="uid") -> DatasetSchema:
    """Helper: a schema whose persons block has just this one z_d field."""
    return DatasetSchema(
        name="mini",
        description="",
        events_path=Path("/dev/null"),
        events_parse_dates=(),
        events_column_map={},
        events_dropna_subset=(),
        events_category_null_fill="Unknown",
        events_dtype_coerce={},
        persons_path=Path("/dev/null"),
        persons_id_column=id_col,
        z_d_mapping=(spec,),
        choice_set_size=10,
        n_resamples=1,
        val_frac=0.1,
        test_frac=0.1,
        subsample_enabled=False,
        subsample_n_customers=0,
        subsample_seed=0,
    )


def test_kind_categorical_map():
    spec = ZDFieldSpec(
        canonical_column="age_bucket",
        kind="categorical_map",
        source="raw_age",
        values={"18 - 24 years": "18-24", "25 - 34 years": "25-34"},
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame({"uid": ["a", "b"], "raw_age": ["18 - 24 years", "25 - 34 years"]})
    out = translate_persons(persons, schema)
    assert list(out["age_bucket"]) == ["18-24", "25-34"]


def test_kind_categorical_map_with_collapse():
    spec = ZDFieldSpec(
        canonical_column="income_bucket",
        kind="categorical_map_with_collapse",
        source="raw_inc",
        values={
            "$50,000 - $74,999": "50-100k",
            "$75,000 - $99,999": "50-100k",
        },
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame(
        {
            "uid": ["a", "b"],
            "raw_inc": ["$50,000 - $74,999", "$75,000 - $99,999"],
        }
    )
    out = translate_persons(persons, schema)
    assert list(out["income_bucket"]) == ["50-100k", "50-100k"]


def test_kind_categorical_to_int():
    spec = ZDFieldSpec(
        canonical_column="household_size",
        kind="categorical_to_int",
        source="raw_hh",
        values={"1 (just me!)": 1, "2": 2, "3": 3, "4+": 4},
    )
    schema = _mini_schema_single_field(spec)
    # Note trailing whitespace on "2 " and " 4+" — should strip.
    persons = pd.DataFrame(
        {"uid": ["a", "b", "c"], "raw_hh": ["1 (just me!)", "2 ", " 4+"]}
    )
    out = translate_persons(persons, schema)
    assert list(out["household_size"]) == [1, 2, 4]


def test_kind_ordinal_map():
    """ordinal_map behaves identically to categorical_to_int (semantic alias)."""
    cti_spec = ZDFieldSpec(
        canonical_column="education",
        kind="categorical_to_int",
        source="raw_ed",
        values={"Some high school": 1, "Bachelor's degree": 4},
    )
    ord_spec = replace(cti_spec, kind="ordinal_map")
    schema_cti = _mini_schema_single_field(cti_spec)
    schema_ord = _mini_schema_single_field(ord_spec)
    persons = pd.DataFrame(
        {"uid": ["a", "b"], "raw_ed": ["Some high school", "Bachelor's degree"]}
    )
    out_cti = translate_persons(persons, schema_cti)
    out_ord = translate_persons(persons, schema_ord)
    assert list(out_cti["education"]) == list(out_ord["education"]) == [1, 4]


def test_kind_constant():
    spec = ZDFieldSpec(
        canonical_column="has_kids",
        kind="constant",
        source=None,
        value=0,
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame({"uid": ["a", "b", "c"]})
    out = translate_persons(persons, schema)
    assert list(out["has_kids"]) == [0, 0, 0]


def test_kind_composite_health_rating():
    """health_rating = 5 - (diabetes==Yes) - (wheelchair==Yes); clamp [1,5]."""
    spec = ZDFieldSpec(
        canonical_column="health_rating",
        kind="composite",
        source=None,
        formula="5 - (`Q-personal-diabetes` == 'Yes') - (`Q-personal-wheelchair` == 'Yes')",
        clamp=(1.0, 5.0),
        fallback=3,
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "Q-personal-diabetes": ["No", "Yes", "Yes"],
            "Q-personal-wheelchair": ["No", "No", "Yes"],
        }
    )
    out = translate_persons(persons, schema)
    assert list(out["health_rating"]) == [5.0, 4.0, 3.0]


def test_kind_composite_risk_tolerance():
    """risk_tolerance = cigarettes_yes + marijuana_yes + alcohol_yes (0..3)."""
    spec = ZDFieldSpec(
        canonical_column="risk_tolerance",
        kind="composite",
        source=None,
        formula=(
            "(`Q-substance-use-cigarettes` == 'Yes') + "
            "(`Q-substance-use-marijuana` == 'Yes') + "
            "(`Q-substance-use-alcohol` == 'Yes')"
        ),
        fallback=0,
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d"],
            "Q-substance-use-cigarettes": ["No", "Yes", "No", "Yes"],
            "Q-substance-use-marijuana":   ["No", "No",  "Yes", "Yes"],
            "Q-substance-use-alcohol":     ["No", "No",  "Yes", "Yes"],
        }
    )
    out = translate_persons(persons, schema)
    assert list(out["risk_tolerance"]) == [0.0, 1.0, 2.0, 3.0]


def test_kind_external_lookup_empty_csv_fallback(tmp_path: Path):
    csv = tmp_path / "lookup.csv"
    csv.write_text("key,value\n")  # header-only
    spec = ZDFieldSpec(
        canonical_column="city_size",
        kind="external_lookup",
        source="state",
        lookup_path=csv,
        fallback="medium",
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame({"uid": ["a", "b", "c"], "state": ["CA", "NY", "TX"]})
    out = translate_persons(persons, schema)
    assert list(out["city_size"]) == ["medium", "medium", "medium"]


def test_kind_derived_from_events():
    count_spec = ZDFieldSpec(
        canonical_column="purchase_frequency",
        kind="derived_from_events",
        aggregator="count",
        group_by="customer_id",
    )
    mean_spec = ZDFieldSpec(
        canonical_column="novelty_rate",
        kind="derived_from_events",
        aggregator="mean",
        aggregator_column="novelty",
        group_by="customer_id",
    )
    # Build a mini schema carrying both fields and no others.
    schema = DatasetSchema(
        name="mini",
        description="",
        events_path=Path("/dev/null"),
        events_parse_dates=(),
        events_column_map={},
        events_dropna_subset=(),
        events_category_null_fill="Unknown",
        events_dtype_coerce={},
        persons_path=Path("/dev/null"),
        persons_id_column="uid",
        z_d_mapping=(count_spec, mean_spec),
        choice_set_size=10,
        n_resamples=1,
        val_frac=0.1,
        test_frac=0.1,
        subsample_enabled=False,
        subsample_n_customers=0,
        subsample_seed=0,
    )
    persons = pd.DataFrame({"uid": ["c1", "c2", "c3", "c4", "c5"]})
    events = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2", "c2", "c3", "c4", "c4"],
            "novelty": [1, 0, 1, 1, 0, 1, 0, 0],
        }
    )
    out = translate_persons(persons, schema, training_events=events)
    counts = dict(zip(out["customer_id"], out["purchase_frequency"]))
    assert counts["c1"] == 2
    assert counts["c2"] == 3
    assert counts["c3"] == 1
    assert counts["c4"] == 2
    # c5 has no events -> NaN
    assert pd.isna(counts["c5"])

    means = dict(zip(out["customer_id"], out["novelty_rate"]))
    assert means["c1"] == pytest.approx(0.5)
    assert means["c2"] == pytest.approx(2 / 3)
    assert means["c3"] == pytest.approx(1.0)
    assert means["c4"] == pytest.approx(0.0)
    assert pd.isna(means["c5"])


def test_kind_passthrough():
    spec = ZDFieldSpec(
        canonical_column="novelty_rate",
        kind="passthrough",
        source="raw_novelty",
    )
    schema = _mini_schema_single_field(spec)
    persons = pd.DataFrame(
        {"uid": ["a", "b", "c"], "raw_novelty": [0.1, 0.5, 0.9]}
    )
    out = translate_persons(persons, schema)
    assert list(out["novelty_rate"]) == [0.1, 0.5, 0.9]


# --------------------------------------------------------------------------- #
# Composite parser safety
# --------------------------------------------------------------------------- #


def _composite_spec(formula: str) -> ZDFieldSpec:
    return ZDFieldSpec(
        canonical_column="x",
        kind="composite",
        source=None,
        formula=formula,
    )


def test_composite_rejects_eval_attempt():
    schema = _mini_schema_single_field(_composite_spec("__import__('os').system('echo pwned')"))
    persons = pd.DataFrame({"uid": ["a"]})
    with pytest.raises(CompositeFormulaError):
        translate_persons(persons, schema)


def test_composite_rejects_unknown_function():
    schema_bad = _mini_schema_single_field(_composite_spec("sum(1, 2)"))
    schema_ok = _mini_schema_single_field(_composite_spec("min(1, 2)"))
    persons = pd.DataFrame({"uid": ["a"]})
    with pytest.raises(CompositeFormulaError):
        translate_persons(persons, schema_bad)
    out = translate_persons(persons, schema_ok)
    assert list(out["x"]) == [1.0]


def test_composite_rejects_unknown_column():
    schema = _mini_schema_single_field(_composite_spec("MissingColumn + 1"))
    persons = pd.DataFrame({"uid": ["a"]})
    with pytest.raises(CompositeFormulaError, match="MissingColumn"):
        translate_persons(persons, schema)


# --------------------------------------------------------------------------- #
# derived_from_events training-events requirement
# --------------------------------------------------------------------------- #


def test_derived_from_events_requires_training_events():
    schema = _amazon_schema()
    persons = _raw_amazon_persons()
    with pytest.raises(ValueError, match="training_events"):
        translate_persons(persons, schema, training_events=None)


# --------------------------------------------------------------------------- #
# z_d column ordering contract
# --------------------------------------------------------------------------- #


def test_z_d_column_order_matches_spec():
    schema = _amazon_schema()
    out = translate_persons(
        _raw_amazon_persons(),
        schema,
        training_events=_amazon_training_events(),
    )
    # Expected: customer_id first, then the canonical z_d order.
    expected = ["customer_id", *Z_D_ORDER]
    assert list(out.columns) == expected
