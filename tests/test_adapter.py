"""Unit tests for ``src/data/adapter.py`` (Wave 9, design doc §2-§3).

Exercises ``YamlAdapter`` + ``AmazonAdapter`` against the real Amazon YAML
and the 100-customer fixtures under ``tests/fixtures/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.adapter import AmazonAdapter, DatasetAdapter, YamlAdapter
from src.data.schema_map import DatasetSchema
from src.data import adapter as adapter_module


REPO_ROOT = Path(__file__).resolve().parent.parent
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_fixture_yaml(tmp_path: Path) -> Path:
    """Clone ``configs/datasets/amazon.yaml`` into ``tmp_path`` with the
    events/persons paths redirected to the 100-row fixtures.

    The external_lookup CSV path is rewritten to the real (empty) file under
    ``configs/datasets/amazon/`` so suppress-logic tests stay consistent.
    """
    with AMAZON_YAML.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ds = doc["dataset"]
    ds["events"]["path"] = str(EVENTS_FIXTURE)
    ds["persons"]["path"] = str(PERSONS_FIXTURE)
    # Rebase the external_lookup CSV to its absolute location so CWD doesn't
    # matter for tests running via pytest from the repo root.
    for col_name, col_spec in ds["persons"]["z_d_mapping"].items():
        if isinstance(col_spec, dict) and col_spec.get("kind") == "external_lookup":
            p = col_spec.get("lookup_path")
            if p is not None and not Path(p).is_absolute():
                col_spec["lookup_path"] = str(REPO_ROOT / p)

    out = tmp_path / "amazon_fixture.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return out


# --------------------------------------------------------------------------- #
# Factory + Protocol smoke
# --------------------------------------------------------------------------- #


def test_amazon_adapter_factory():
    adapter = AmazonAdapter()
    assert isinstance(adapter, YamlAdapter)
    assert adapter.name == "amazon"
    assert isinstance(adapter.schema, DatasetSchema)
    assert adapter.schema.name == "amazon"


def test_protocol_runtime_check():
    adapter = AmazonAdapter()
    # runtime_checkable Protocol: attribute/method presence test.
    assert isinstance(adapter, DatasetAdapter)


def test_yaml_adapter_loads_from_arbitrary_path(tmp_path):
    yaml_path = _write_fixture_yaml(tmp_path)
    adapter = YamlAdapter(yaml_path)
    assert adapter.name == "amazon"
    assert adapter.yaml_path == yaml_path
    assert isinstance(adapter.schema, DatasetSchema)


# --------------------------------------------------------------------------- #
# Loading + caching
# --------------------------------------------------------------------------- #


def test_load_events_and_persons(tmp_path):
    """Both frames load at the expected shapes for the 100-row fixture."""
    yaml_path = _write_fixture_yaml(tmp_path)
    adapter = YamlAdapter(yaml_path)
    events = adapter.load_events()
    persons = adapter.load_persons()
    assert isinstance(events, pd.DataFrame)
    assert isinstance(persons, pd.DataFrame)
    # Fixture shapes (matches tests/test_data_load.py::test_load_amazon_fixture_shapes).
    assert events.shape == (30484, 8)
    assert persons.shape == (100, 23)


def test_load_events_caches(tmp_path, monkeypatch):
    """Calling load_events twice hits the underlying ``load`` exactly once.

    Shares its cache with ``load_persons``.
    """
    yaml_path = _write_fixture_yaml(tmp_path)
    adapter = YamlAdapter(yaml_path)

    call_count = {"n": 0}
    real_load = adapter_module.load

    def counting_load(*args, **kwargs):
        call_count["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "load", counting_load)

    a = adapter.load_events()
    b = adapter.load_events()
    # Same object both times.
    assert a is b
    # And load_persons also hits the cache.
    c = adapter.load_persons()
    assert call_count["n"] == 1
    assert isinstance(c, pd.DataFrame)


# --------------------------------------------------------------------------- #
# Static projections
# --------------------------------------------------------------------------- #


def test_event_column_map():
    adapter = AmazonAdapter()
    m = adapter.event_column_map()
    assert isinstance(m, dict)
    assert m["Order Date"] == "order_date"
    assert m["Purchase Price Per Unit"] == "price"
    assert m["Title"] == "title"
    assert m["Category"] == "category"
    # Both ASIN aliases map to the same canonical name.
    assert m["ASIN/ISBN (Product Code)"] == "asin"
    assert m["ASIN/ISBN"] == "asin"
    # Caller mutation must not leak into the schema.
    m["Order Date"] = "CLOBBERED"
    again = adapter.event_column_map()
    assert again["Order Date"] == "order_date"


def test_person_id_column():
    adapter = AmazonAdapter()
    assert adapter.person_id_column() == "Survey ResponseID"


def test_derived_z_d_columns_for_amazon():
    adapter = AmazonAdapter()
    derived = adapter.derived_z_d_columns()
    assert isinstance(derived, tuple)
    # Amazon has exactly two derived fields, in YAML order.
    assert derived == ("purchase_frequency", "novelty_rate")


# --------------------------------------------------------------------------- #
# z_d translation
# --------------------------------------------------------------------------- #


def test_translate_z_d_requires_training_events_for_derived_fields():
    """Amazon has derived_from_events fields -> None training_events must raise."""
    adapter = AmazonAdapter()
    persons_raw = pd.read_csv(PERSONS_FIXTURE)
    with pytest.raises(ValueError, match="derived_from_events"):
        adapter.translate_z_d(persons_raw, training_events=None)


def test_translate_z_d_with_training_events_succeeds():
    """A minimal training_events frame is enough to populate purchase_frequency
    and novelty_rate."""
    adapter = AmazonAdapter()
    persons_raw = pd.read_csv(PERSONS_FIXTURE).head(3).copy()
    ids = persons_raw["Survey ResponseID"].astype(str).tolist()

    # Build a small training_events frame carrying the canonical columns the
    # two derived aggregators reference: 'customer_id' (groupby key) and
    # 'novelty' (aggregator_column for novelty_rate).
    rows = []
    for i, cid in enumerate(ids):
        for k in range(10):
            rows.append(
                {
                    "customer_id": cid,
                    "novelty": int((k + i) % 2),  # ~half novel
                }
            )
    training_events = pd.DataFrame(rows)

    translated = adapter.translate_z_d(persons_raw, training_events=training_events)
    assert "purchase_frequency" in translated.columns
    assert "novelty_rate" in translated.columns
    assert "customer_id" in translated.columns
    # Every of the 3 customers has exactly 10 events -> purchase_frequency == 10.
    assert (translated["purchase_frequency"] == 10).all()
    # Novelty rate is in [0, 1] (half-ish).
    assert ((translated["novelty_rate"] >= 0) & (translated["novelty_rate"] <= 1)).all()


# --------------------------------------------------------------------------- #
# alt_text
# --------------------------------------------------------------------------- #


def test_alt_text_default_fallback_no_percentile_fn():
    """Pre-Wave-10: popularity_rank is the stub 'popularity score N' string."""
    adapter = AmazonAdapter()
    assert adapter._popularity_percentile_fn is None

    event_row = {
        "title": "Widget Deluxe",
        "category": "Home",
        "price": 19.99,
        "popularity": 42,
    }
    out = adapter.alt_text(event_row)
    assert out["title"] == "Widget Deluxe"
    assert out["category"] == "Home"
    assert out["price"] == 19.99
    assert out["popularity_rank"] == "popularity score 42"


def test_alt_text_with_percentile_fn():
    """Wave-10 hookup: monkeypatch the fn and alt_text uses it."""
    adapter = AmazonAdapter()
    adapter._popularity_percentile_fn = lambda n: "top 5%"
    out = adapter.alt_text(
        {"title": "X", "category": "Y", "price": 5.0, "popularity": 99}
    )
    assert out["popularity_rank"] == "top 5%"


# --------------------------------------------------------------------------- #
# suppress_fields_for_c_d
# --------------------------------------------------------------------------- #


def test_suppress_fields_for_c_d_amazon():
    """Amazon YAML: ``has_kids`` (constant) and ``city_size`` (empty lookup)
    are both suppressed."""
    adapter = AmazonAdapter()
    suppress = adapter.suppress_fields_for_c_d()
    assert isinstance(suppress, tuple)
    # has_kids is a constant; city_size is external_lookup with an empty CSV.
    assert "has_kids" in suppress
    assert "city_size" in suppress
    # Non-constant non-empty-lookup fields must not be suppressed.
    assert "age_bucket" not in suppress
    assert "income_bucket" not in suppress
    assert "purchase_frequency" not in suppress
    assert "novelty_rate" not in suppress
    assert "education" not in suppress  # ordinal_map
    # health_rating and risk_tolerance are composite for Amazon, not constant.
    assert "health_rating" not in suppress
    assert "risk_tolerance" not in suppress


def test_suppress_fields_list_is_cached(tmp_path, monkeypatch):
    """The list is computed once at __init__ and memoised thereafter."""
    yaml_path = _write_fixture_yaml(tmp_path)

    # Observe how often the empty-check runs: it's a staticmethod, so patch it
    # on the class via monkeypatch with a counting wrapper.
    call_count = {"n": 0}
    original = YamlAdapter._external_lookup_is_empty

    def counting(lookup_path):
        call_count["n"] += 1
        return original(lookup_path)

    monkeypatch.setattr(YamlAdapter, "_external_lookup_is_empty", staticmethod(counting))

    adapter = YamlAdapter(yaml_path)  # __init__ exercises the check once.
    count_after_init = call_count["n"]

    a = adapter.suppress_fields_for_c_d()
    b = adapter.suppress_fields_for_c_d()
    # Two cache hits must not re-invoke the empty-check helper.
    assert call_count["n"] == count_after_init
    assert a == b
    assert a is b  # cached tuple is the same object


def test_suppress_fields_for_minimal_yaml(tmp_path):
    """YAML with no constants and no external_lookup -> empty suppress tuple."""
    # Build a minimal YAML by hand; the events/persons paths point at dummy
    # CSVs that are never actually read (load_schema is side-effect-free).
    events_csv = tmp_path / "events.csv"
    persons_csv = tmp_path / "persons.csv"
    events_csv.write_text("order_date,customer_id,asin\n")
    persons_csv.write_text("respondent_id,age,income,hh,edu\n")

    doc = {
        "dataset": {
            "name": "minimal",
            "description": "no constants, no external_lookup",
            "events": {
                "path": str(events_csv),
                "parse_dates": ["order_date"],
                "column_map": {
                    "order_date": "order_date",
                    "customer_id": "customer_id",
                    "asin": "asin",
                },
                "dropna_subset": ["customer_id", "order_date", "asin"],
                "category_null_fill": "Unknown",
                "dtype_coerce": {},
            },
            "persons": {
                "path": str(persons_csv),
                "id_column": "respondent_id",
                "z_d_mapping": {
                    "age_bucket": {
                        "source": "age",
                        "kind": "categorical_map",
                        "values": {"a": "a"},
                    },
                    "income_bucket": {
                        "source": "income",
                        "kind": "categorical_map_with_collapse",
                        "values": {"x": "x"},
                    },
                    "household_size": {
                        "source": "hh",
                        "kind": "categorical_to_int",
                        "values": {"1": 1},
                    },
                    "education": {
                        "source": "edu",
                        "kind": "ordinal_map",
                        "values": {"1": 1},
                    },
                },
            },
            "training": {
                "choice_set_size": 10,
                "n_resamples": 1,
                "val_frac": 0.1,
                "test_frac": 0.1,
            },
        }
    }
    yaml_path = tmp_path / "minimal.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)

    adapter = YamlAdapter(yaml_path)
    assert adapter.suppress_fields_for_c_d() == ()
