"""Dataset #2 end-to-end gate for the Wave 9 adapter layer.

Concrete answer to "can a new dataset be added without editing Python code?"
This test stands up a completely synthetic dataset in ``tmp_path`` — a YAML
whose column names, lookup-free z_d_mapping, and CSV fixtures are all
invented here — and runs the full Wave-8 pipeline against it via
:class:`src.data.adapter.YamlAdapter`, with zero import from
``amazon_ecom/`` or ``configs/datasets/amazon.yaml``.

A passing test means the canonical code path does not hard-code anything
Amazon-specific at the adapter-to-schema boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.adapter import YamlAdapter
from src.data.clean import clean_events
from src.data.context_string import build_context_string
from src.data.split import temporal_split
from src.data.state_features import attach_train_popularity, compute_state_features
from src.data.survey_join import join_survey


# --------------------------------------------------------------------------- #
# Synthetic dataset construction
# --------------------------------------------------------------------------- #


def _build_synthetic_yaml(tmp_path: Path) -> Path:
    """Write a synthetic dataset YAML + events/persons CSVs into ``tmp_path``.

    Column names are deliberately non-Amazon (``user_id``, ``dt``, ``prod_code``,
    ``cat``, ``amount``, ``desc``) so any Python code that "knew" the Amazon
    names would fail. The z_d_mapping uses every required kind from the Wave 8
    vocabulary *except* external_lookup (constant short-circuits it) so no
    auxiliary files are needed.
    """
    events_csv = tmp_path / "events.csv"
    persons_csv = tmp_path / "persons.csv"

    # 5 customers. Each gets ~4 events (>= 3 so temporal_split assigns a train row).
    events_rows = []
    persons_rows = []

    # Canonical z_d labels are chosen to match the hard-coded phrasings in
    # src/data/context_string.py (DEFAULT_PHRASINGS).
    customers = [
        ("u01", "18 - 24", "<25k-raw",    "1 (only me)", "grade"),
        ("u02", "25 - 34", "25-50k-raw",  "2",           "high"),
        ("u03", "35 - 44", "50-74k-raw",  "3",           "college"),
        ("u04", "45 - 54", "75-99k-raw",  "4",           "bachelors"),
        ("u05", "55 - 64", "100-150k-raw", "2",          "grad"),
    ]
    for user_id, age_raw, income_raw, hh_raw, edu_raw in customers:
        persons_rows.append(
            {
                "user_id": user_id,
                "age_raw": age_raw,
                "income_raw": income_raw,
                "hh_raw": hh_raw,
                "edu_raw": edu_raw,
            }
        )

    # Four events per customer, spread across four categories.
    categories = ["electronics", "books", "pantry", "outdoors"]
    titles = ["alpha widget", "beta gizmo", "gamma gadget", "delta thing"]
    base_dates = ["2023-01-05", "2023-01-12", "2023-01-20", "2023-02-01"]
    for user_id, *_ in customers:
        for i in range(4):
            events_rows.append(
                {
                    "user_id": user_id,
                    "dt": base_dates[i],
                    "prod_code": f"P-{user_id}-{i}",
                    "cat": categories[i],
                    "amount": 10.0 + i,
                    "desc": titles[i],
                }
            )

    pd.DataFrame(events_rows).to_csv(events_csv, index=False)
    pd.DataFrame(persons_rows).to_csv(persons_csv, index=False)

    # z_d_mapping covers: categorical_map, categorical_map_with_collapse,
    # categorical_to_int, ordinal_map, constant, composite (not needed —
    # risk_tolerance is a constant per brief), derived_from_events.
    # Canonical labels match DEFAULT_PHRASINGS so context_string accepts them.
    doc = {
        "dataset": {
            "name": "synth",
            "description": "Synthetic 5-customer fixture for adapter gate test.",
            "events": {
                "path": str(events_csv),
                "parse_dates": ["dt"],
                "column_map": {
                    "user_id": "customer_id",
                    "dt": "order_date",
                    "prod_code": "asin",
                    "cat": "category",
                    "amount": "price",
                    "desc": "title",
                },
                "dropna_subset": ["customer_id", "order_date", "asin"],
                "category_null_fill": "Unknown",
                "dtype_coerce": {
                    "price": "float",
                    "customer_id": "str",
                    "asin": "str",
                },
            },
            "persons": {
                "path": str(persons_csv),
                "id_column": "user_id",
                "z_d_mapping": {
                    "age_bucket": {
                        "source": "age_raw",
                        "kind": "categorical_map",
                        "values": {
                            "18 - 24": "18-24",
                            "25 - 34": "25-34",
                            "35 - 44": "35-44",
                            "45 - 54": "45-54",
                            "55 - 64": "55-64",
                        },
                    },
                    # 2 raw buckets ("50-74k-raw" + "75-99k-raw") collapse to "50-100k".
                    "income_bucket": {
                        "source": "income_raw",
                        "kind": "categorical_map_with_collapse",
                        "values": {
                            "<25k-raw": "<25k",
                            "25-50k-raw": "25-50k",
                            "50-74k-raw": "50-100k",
                            "75-99k-raw": "50-100k",
                            "100-150k-raw": "100-150k",
                        },
                    },
                    "household_size": {
                        "source": "hh_raw",
                        "kind": "categorical_to_int",
                        "values": {
                            "1 (only me)": 1,
                            "2": 2,
                            "3": 3,
                            "4": 4,
                        },
                    },
                    # Per brief: constants for the four suppressible sentinels.
                    "has_kids": {
                        "source": None,
                        "kind": "constant",
                        "value": 0,
                    },
                    "city_size": {
                        "source": None,
                        "kind": "constant",
                        "value": "medium",
                    },
                    "education": {
                        "source": "edu_raw",
                        "kind": "ordinal_map",
                        "values": {
                            "grade": 1,
                            "high": 2,
                            "college": 3,
                            "bachelors": 4,
                            "grad": 5,
                        },
                    },
                    "health_rating": {
                        "source": None,
                        "kind": "constant",
                        "value": 3,
                    },
                    "risk_tolerance": {
                        "source": None,
                        "kind": "constant",
                        "value": 0,
                    },
                    "purchase_frequency": {
                        "source": "derived_from_events",
                        "aggregator": "count",
                        "group_by": "customer_id",
                    },
                    "novelty_rate": {
                        "source": "derived_from_events",
                        "aggregator": "mean",
                        "aggregator_column": "novelty",
                        "group_by": "customer_id",
                    },
                },
            },
            "training": {
                "choice_set_size": 4,
                "n_resamples": 1,
                "val_frac": 0.25,
                "test_frac": 0.25,
            },
        }
    }
    yaml_path = tmp_path / "synthetic.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return yaml_path


# --------------------------------------------------------------------------- #
# The gate test
# --------------------------------------------------------------------------- #


def test_synthetic_dataset_end_to_end(tmp_path):
    """Full pipeline against a fabricated dataset, driven only by the YAML."""
    yaml_path = _build_synthetic_yaml(tmp_path)

    # Step 1: adapter.
    adapter = YamlAdapter(yaml_path)
    assert adapter.name == "synth"

    # Step 2: load raw frames through the adapter (tests the caching path).
    events_raw = adapter.load_events()
    persons_raw = adapter.load_persons()
    assert len(events_raw) == 20   # 5 customers * 4 events
    assert len(persons_raw) == 5

    # Step 3: full Wave-8 pipeline.
    cleaned = clean_events(events_raw, adapter.schema)
    joined = join_survey(cleaned, persons_raw, adapter.schema)
    with_state = compute_state_features(joined)
    split_df = temporal_split(with_state, adapter.schema)
    final = attach_train_popularity(split_df)

    train_df = final.loc[final["split"] == "train"].copy()
    assert len(train_df) > 0, "split produced no training rows"
    # compute_state_features populated the 'novelty' column novelty_rate needs.
    assert "novelty" in train_df.columns

    # Step 4: z_d translation via the adapter (delegates to schema_map).
    translated = adapter.translate_z_d(persons_raw, training_events=train_df)

    # Assertion 1: all 10 canonical z_d columns present + customer_id.
    expected = {
        "customer_id",
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
    }
    assert expected.issubset(set(translated.columns)), (
        f"missing columns: {expected - set(translated.columns)}"
    )

    # Assertion 2: at least one customer survives the split.
    assert len(translated) > 0

    # Derived columns got real values (not NaN for every customer).
    assert translated["purchase_frequency"].notna().any()
    assert translated["novelty_rate"].notna().any()

    # Assertion 3: suppress_fields_for_c_d contains all four constant sentinels.
    suppress = adapter.suppress_fields_for_c_d()
    assert "has_kids" in suppress
    assert "city_size" in suppress
    assert "health_rating" in suppress
    assert "risk_tolerance" in suppress

    # Assertion 4: build_context_string on a translated row with the suppress
    # list produces >= 2 non-empty lines of rendered text.
    row = translated.iloc[0].to_dict()
    # build_context_string expects a mapping with every canonical column set.
    rendered = build_context_string(row, suppress_fields=suppress)
    non_empty_lines = [ln for ln in rendered.split("\n") if ln.strip()]
    assert len(non_empty_lines) >= 2, f"rendered too short: {rendered!r}"


# --------------------------------------------------------------------------- #
# Guardrail: no import coupling to the Amazon dataset.
# --------------------------------------------------------------------------- #


def test_module_does_not_import_amazon(tmp_path):
    """Reading the module source, no mention of the real amazon.yaml path.

    (Belt-and-suspenders: the synthetic test above already demonstrates this,
    but a direct source-level check catches a future maintainer accidentally
    hard-coding an Amazon path into adapter.py.)
    """
    import src.data.adapter as adapter_module

    src_text = Path(adapter_module.__file__).read_text()
    # The AmazonAdapter factory deliberately references configs/datasets/amazon.yaml;
    # the rest of the module should not reference the amazon_ecom data dir.
    assert "amazon_ecom" not in src_text, (
        "adapter.py must not import/reference amazon_ecom data paths directly."
    )
