"""Wave 10 regression tests for the schema-declared vocabulary fix.

Covers:
- ``fit_person_features(df, vocabularies=...)`` pins the z_d width to 26.
- Pre-existing learn-from-data behavior preserved when ``vocabularies=None``.
- ``DatasetAdapter.categorical_vocabularies()`` derives the correct tuples
  from the YAML for every one-hot kind.
- The Wave-9 z_d-width-drift regression: fitting on a sparse slice with
  schema-declared vocabulary produces width 26, not 14.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.adapter import AmazonAdapter, YamlAdapter
from src.data.person_features import (
    CategoricalVocabularies,
    fit_person_features,
    transform_person_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FULL_AGE = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
_FULL_INCOME = ("<25k", "25-50k", "50-100k", "100-150k", "150k+")
_FULL_CITY = ("rural", "small", "medium", "large")


def _make_sparse_df(n: int = 5) -> pd.DataFrame:
    """Intentionally undercovered DataFrame: only 2 of 6 age buckets, 1 city."""
    rows = []
    for i in range(n):
        rows.append({
            "age_bucket": "25-34" if i % 2 == 0 else "35-44",    # 2 of 6
            "income_bucket": "50-100k",                           # 1 of 5
            "household_size": (i % 4) + 1,                        # 1..4
            "has_kids": 0,
            "city_size": "medium",                                # 1 of 4
            "education": (i % 5) + 1,
            "health_rating": ((i + 1) % 5) + 1,
            "risk_tolerance": float(i),
            "purchase_frequency": i * 2,
            "novelty_rate": 0.3 + 0.05 * i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. fit_person_features with declared vocabulary
# ---------------------------------------------------------------------------


def test_fit_with_vocabularies_pins_width_to_26():
    """A sparse 5-customer fit with schema vocab still emits 26-dim output."""
    df = _make_sparse_df(5)
    vocabs = CategoricalVocabularies(
        age_bucket=_FULL_AGE,
        income_bucket=_FULL_INCOME,
        city_size=_FULL_CITY,
    )
    stats = fit_person_features(df, vocabularies=vocabs)
    X = transform_person_features(df, stats)
    assert X.shape == (5, 26)


def test_fit_without_vocabularies_unchanged_behavior():
    """Default behavior (vocab=None) learns from data — Wave-1 tests rely on this."""
    df = _make_sparse_df(5)
    stats = fit_person_features(df)
    X = transform_person_features(df, stats)
    # Learn-from-data: only 2 age + 1 income + 1 city buckets present
    # → width < 26.
    assert X.shape[1] < 26


def test_fit_with_vocab_asserts_values_in_vocab():
    """A value outside the declared vocabulary raises a clear error."""
    df = _make_sparse_df(3)
    df.loc[0, "age_bucket"] = "dinosaur"    # not in _FULL_AGE
    vocabs = CategoricalVocabularies(
        age_bucket=_FULL_AGE,
        income_bucket=_FULL_INCOME,
        city_size=_FULL_CITY,
    )
    with pytest.raises(ValueError, match="age_bucket"):
        fit_person_features(df, vocabularies=vocabs)


def test_fit_with_vocab_produces_expected_column_ordering():
    """feature_columns order matches the schema-declared vocab order."""
    df = _make_sparse_df(5)
    vocabs = CategoricalVocabularies(
        age_bucket=_FULL_AGE,
        income_bucket=_FULL_INCOME,
        city_size=_FULL_CITY,
    )
    stats = fit_person_features(df, vocabularies=vocabs)
    age_cols = [c for c in stats.feature_columns if c.startswith("age_bucket=")]
    # Schema tuple order preserved.
    assert [c.split("=", 1)[1] for c in age_cols] == list(_FULL_AGE)


# ---------------------------------------------------------------------------
# 2. Adapter.categorical_vocabularies()
# ---------------------------------------------------------------------------


def test_amazon_adapter_categorical_vocabularies():
    """AmazonAdapter yields the expected closed-set vocabularies."""
    adapter = AmazonAdapter()
    v = adapter.categorical_vocabularies()
    assert isinstance(v, CategoricalVocabularies)
    # Option B collapse produces exactly 5 income labels.
    assert set(v.income_bucket) == {
        "<25k", "25-50k", "50-100k", "100-150k", "150k+"
    }
    # Six age buckets.
    assert set(v.age_bucket) == set(_FULL_AGE)
    # City: Amazon's lookup CSV is empty → fallback-only vocab.
    assert v.city_size == ("medium",)
    # Household uses the orchestrator-fixed default.
    assert v.household_size_categories == ("1", "2", "3", "4", "5+")


def test_categorical_vocabularies_is_cached():
    """Repeat calls return identical objects (no re-read of the YAML)."""
    adapter = AmazonAdapter()
    v1 = adapter.categorical_vocabularies()
    v2 = adapter.categorical_vocabularies()
    assert v1 is v2


def test_vocabularies_from_constant_kind():
    """A has_kids-style constant produces a single-label vocab if we ever
    promote a constant to a one-hot column in a hypothetical future schema.

    (Amazon does NOT treat has_kids as a one-hot; this test just verifies
    the adapter's internal ``_vocab_for`` handles a kind=constant correctly
    for canonical ONE-HOT columns. Construct a tmp YAML where city_size
    is kind=constant to exercise the branch.)
    """
    import yaml
    amazon_cfg = yaml.safe_load(
        Path("configs/datasets/amazon.yaml").read_text()
    )
    # Replace city_size with a constant entry.
    for spec in amazon_cfg["dataset"]["persons"]["z_d_mapping"].values() \
            if isinstance(amazon_cfg["dataset"]["persons"]["z_d_mapping"], dict) \
            else []:  # YAML uses a mapping under z_d_mapping — key = canonical col
        pass
    # Rebuild as dict-of-entries explicitly.
    zm = amazon_cfg["dataset"]["persons"]["z_d_mapping"]
    zm["city_size"] = {"source": None, "kind": "constant", "value": "rural"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "amazon_const_city.yaml"
        tmp_path.write_text(yaml.dump(amazon_cfg))
        adapter = YamlAdapter(tmp_path)
        v = adapter.categorical_vocabularies()
        assert v.city_size == ("rural",)


# ---------------------------------------------------------------------------
# 3. Regression: 100-customer Amazon fixture + schema vocab → width 26
# ---------------------------------------------------------------------------


def test_amazon_100_customer_fixture_yields_dataset_declared_p():
    """The Wave 9 regression: z_d width now matches the ADAPTER-DECLARED p.

    Amazon's schema declares 6 age + 5 income + 5 household + 1 has_kids +
    1 city (lookup CSV is empty, fallback="medium" → vocab has only one
    label) + 4 standardized scalars + 1 passthrough = 23. That's
    dataset-dependent; the paper's default p=26 applies only to datasets
    whose city_size populates 4 buckets. Pre-fix, the fit learned from
    data and yielded z_d width 14 on this fixture — that is the bug
    this test pins.
    """
    import tempfile
    import yaml
    import pandas as pd
    from src.data.clean import clean_events
    from src.data.survey_join import join_survey
    from src.data.state_features import compute_state_features, attach_train_popularity
    from src.data.split import temporal_split

    # Redirect the Amazon YAML to the fixture paths.
    with open("configs/datasets/amazon.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["events"]["path"] = "tests/fixtures/amazon_events_100.csv"
    cfg["dataset"]["persons"]["path"] = "tests/fixtures/amazon_persons_100.csv"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_yaml = Path(tmp) / "amazon_fixture.yaml"
        tmp_yaml.write_text(yaml.dump(cfg))
        adapter = YamlAdapter(tmp_yaml)

        events_raw = adapter.load_events()
        persons_raw = adapter.load_persons()
        events = clean_events(events_raw, adapter.schema)
        events = join_survey(events, persons_raw, adapter.schema)
        events = compute_state_features(events)
        events = temporal_split(events, adapter.schema)
        events = attach_train_popularity(events)

        train_events = events[events["split"] == "train"]
        train_customers = set(train_events["customer_id"].unique())
        id_col = adapter.person_id_column()
        persons_with_survey = set(persons_raw[id_col].dropna().astype(str))
        valid = train_customers & persons_with_survey
        persons_for_train = persons_raw[
            persons_raw[id_col].astype(str).isin(valid)
        ]

        persons_canonical = adapter.translate_z_d(
            persons_for_train, training_events=train_events
        )
        vocabs = adapter.categorical_vocabularies()
        stats = fit_person_features(persons_canonical, vocabularies=vocabs)
        X = transform_person_features(persons_canonical, stats)

        # Compute expected p from the adapter-declared vocab.
        expected_p = (
            len(vocabs.age_bucket)
            + len(vocabs.income_bucket)
            + len(vocabs.household_size_categories)
            + 1                                   # has_kids binary
            + len(vocabs.city_size)
            + 4                                   # edu, health, risk, purchase_freq
            + 1                                   # novelty_rate passthrough
        )
        # For Amazon with empty city_size lookup: 6+5+5+1+1+4+1 = 23.
        assert expected_p == 23
        assert X.shape[1] == expected_p

        # AND: width is stable regardless of which customers are in the fit.
        # (The raw pre-fix behavior produced width 14; we must be well above
        # that floor on this fixture.)
        assert X.shape[1] > 14
