"""Tests for :mod:`src.data.person_features` (redesign.md §2.1, orchestrator override).

Covers the 26-dim breakdown under the orchestrator-resolved encoding table
(household_size one-hot over 5 bins, novelty_rate pass-through), the one-hot
structure of the four categorical blocks, has_kids binarity, log1p on
purchase_frequency, novelty_rate pass-through, the 5+ bucket, train-only
standardization, unknown-category errors, JSON round-trip on the stats
artifact, and determinism across fits.

No conftest edits — the synthetic DataFrame is built here as a local fixture.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.person_features import (
    PersonFeatureStats,
    _bucket_household_size,
    fit_person_features,
    fit_transform_person_features,
    transform_person_features,
)


# --------------------------------------------------------------------------- #
# Local synthetic fixture (no conftest edits, per task spec)
# --------------------------------------------------------------------------- #

_AGE_BINS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]       # 6
_INCOME_BINS = ["<25k", "25-50k", "50-100k", "100-200k", ">200k"]      # 5
_CITY_BINS = ["rural", "small", "medium", "large"]                     # 4


def _make_synthetic_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic training-style DataFrame with every category.

    Household-size values span the 5 one-hot bins (1..5 and a sprinkle of 6
    to exercise the "5+" fold-in). Purchase_frequency is a non-negative
    count so ``log1p`` is well-defined.
    """
    rng = np.random.default_rng(seed)

    def _cycle(vals: list[str], n: int) -> list[str]:
        reps = (n + len(vals) - 1) // len(vals)
        return (vals * reps)[:n]

    # Explicit household_size sequence that covers each bucket including 5+.
    hs_cycle = [1, 2, 3, 4, 5, 6, 7]
    household = [hs_cycle[i % len(hs_cycle)] for i in range(n)]

    rows = {
        "age_bucket": _cycle(_AGE_BINS, n),
        "income_bucket": _cycle(_INCOME_BINS, n),
        "household_size": np.asarray(household, dtype=int),
        "has_kids": rng.integers(0, 2, size=n).astype(int),
        "city_size": _cycle(_CITY_BINS, n),
        "education": rng.integers(1, 6, size=n).astype(int),          # 1..5
        "health_rating": rng.integers(1, 6, size=n).astype(int),      # 1..5
        "risk_tolerance": rng.normal(0.0, 1.0, size=n),
        "purchase_frequency": rng.integers(0, 200, size=n).astype(int),
        "novelty_rate": rng.uniform(0.0, 1.0, size=n),
    }
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_df(n=60, seed=0)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_output_shape_and_dtype(synthetic_df):
    X, stats = fit_transform_person_features(synthetic_df)
    assert X.shape == (len(synthetic_df), 26)
    assert X.dtype == np.float32
    assert isinstance(stats, PersonFeatureStats)


def test_feature_column_order(synthetic_df):
    """Matches the orchestrator binding table — one-hots first, then scalars."""
    stats = fit_person_features(synthetic_df)
    cols = stats.feature_columns
    assert len(cols) == 26

    # age(6) + income(5) + household(5) + has_kids(1) + city(4)
    # + education(1) + health(1) + risk(1) + purchase(1) + novelty(1) = 26
    assert all(c.startswith("age_bucket=") for c in cols[0:6])
    assert all(c.startswith("income_bucket=") for c in cols[6:11])
    assert all(c.startswith("household_size=") for c in cols[11:16])
    assert cols[16] == "has_kids"
    assert all(c.startswith("city_size=") for c in cols[17:21])
    assert cols[21] == "education"
    assert cols[22] == "health_rating"
    assert cols[23] == "risk_tolerance"
    assert cols[24] == "purchase_frequency"
    assert cols[25] == "novelty_rate"

    # Cross-check per-block counts.
    n_age = sum(1 for c in cols if c.startswith("age_bucket="))
    n_income = sum(1 for c in cols if c.startswith("income_bucket="))
    n_hs = sum(1 for c in cols if c.startswith("household_size="))
    n_city = sum(1 for c in cols if c.startswith("city_size="))
    assert n_age == 6
    assert n_income == 5
    assert n_hs == 5
    assert n_city == 4
    assert n_age + n_income + n_hs + 1 + n_city + 1 + 1 + 1 + 1 + 1 == 26

    # Household-size vocabulary is the fixed 5-bin set.
    assert stats.household_size_categories == ["1", "2", "3", "4", "5+"]


def test_standardization_fitted_on_train_only(synthetic_df):
    """Stats fit on train != stats fit on val; re-transforming train is stable."""
    train = synthetic_df.iloc[:40].reset_index(drop=True)
    val = synthetic_df.iloc[40:].reset_index(drop=True)

    stats_train = fit_person_features(train)
    X_train_via_train = transform_person_features(train, stats_train)
    X_val_via_train = transform_person_features(val, stats_train)

    # Re-running transform with the same stats must be exactly reproducible.
    X_train_via_train2 = transform_person_features(train, stats_train)
    X_val_via_train2 = transform_person_features(val, stats_train)
    np.testing.assert_array_equal(X_train_via_train, X_train_via_train2)
    np.testing.assert_array_equal(X_val_via_train, X_val_via_train2)

    # Fitting on val alone must produce DIFFERENT stats — i.e. val was NOT
    # used to compute the train-fit means/stds.
    stats_val = fit_person_features(val)
    assert not (
        np.allclose(stats_train.means, stats_val.means)
        and np.allclose(stats_train.stds, stats_val.stds)
    )


def test_one_hot_sums_to_one(synthetic_df):
    """Each of the four one-hot blocks sums to 1 per row with only {0, 1} entries."""
    stats = fit_person_features(synthetic_df)
    X = transform_person_features(synthetic_df, stats)

    age_block = X[:, 0:6]
    income_block = X[:, 6:11]
    household_block = X[:, 11:16]
    city_block = X[:, 17:21]

    np.testing.assert_allclose(age_block.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(income_block.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(household_block.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(city_block.sum(axis=1), 1.0, atol=1e-6)

    for block in (age_block, income_block, household_block, city_block):
        assert np.all((block == 0.0) | (block == 1.0))


def test_household_size_5plus_bucket(synthetic_df):
    """Values 5, 6, 7 all land in the '5+' bucket (same one-hot position)."""
    stats = fit_person_features(synthetic_df)

    # hs=5 and hs=7 should produce the same row-wise one-hot.
    row_template = synthetic_df.iloc[0].copy()

    row_5 = row_template.copy()
    row_5["household_size"] = 5
    row_7 = row_template.copy()
    row_7["household_size"] = 7

    df_two = pd.DataFrame([row_5, row_7])
    X = transform_person_features(df_two, stats)

    hs_block = X[:, 11:16]
    np.testing.assert_array_equal(hs_block[0], hs_block[1])
    # The 5+ bin is the last slot in the household-size vocabulary.
    assert hs_block[0, -1] == 1.0
    assert hs_block[0, :-1].sum() == 0.0

    # Direct helper behaviour: 5 and 7 both map to "5+".
    assert _bucket_household_size(5) == "5+"
    assert _bucket_household_size(7) == "5+"
    assert _bucket_household_size(4) == "4"
    assert _bucket_household_size(1) == "1"

    # Invalid inputs must raise ValueError.
    with pytest.raises(ValueError):
        _bucket_household_size(-1)
    with pytest.raises(ValueError):
        _bucket_household_size(0)
    with pytest.raises(ValueError):
        _bucket_household_size(2.5)
    with pytest.raises(ValueError):
        _bucket_household_size(None)


def test_has_kids_binary(synthetic_df):
    stats = fit_person_features(synthetic_df)
    X = transform_person_features(synthetic_df, stats)
    hk_col = X[:, stats.feature_columns.index("has_kids")]
    assert set(np.unique(hk_col).tolist()).issubset({0.0, 1.0})


def test_novelty_rate_passthrough(synthetic_df):
    """Raw novelty_rate 0.3 must come through as 0.3 — no standardization."""
    stats = fit_person_features(synthetic_df)

    row = synthetic_df.iloc[0].copy()
    row["novelty_rate"] = 0.3
    df_one = pd.DataFrame([row])
    X = transform_person_features(df_one, stats)

    nov_idx = stats.feature_columns.index("novelty_rate")
    np.testing.assert_allclose(X[0, nov_idx], 0.3, atol=1e-6)


def test_log1p_applied_to_purchase_frequency(synthetic_df):
    """Standardized value of raw 0 equals ``-mean_log1p / std_log1p``."""
    stats = fit_person_features(synthetic_df)

    row = synthetic_df.iloc[0].copy()
    row["purchase_frequency"] = 0
    df_one = pd.DataFrame([row])
    X = transform_person_features(df_one, stats)

    pf_idx = stats.feature_columns.index("purchase_frequency")
    # The standardized-columns order inside stats.means/stds is
    # (education, health_rating, risk_tolerance, purchase_frequency).
    std_idx = 3

    expected = (np.log1p(0.0) - stats.means[std_idx]) / stats.stds[std_idx]
    np.testing.assert_allclose(X[0, pf_idx], expected, atol=1e-6)
    # log1p(0) == 0, so expected simplifies to -mean_log1p / std_log1p.
    np.testing.assert_allclose(
        X[0, pf_idx], -stats.means[std_idx] / stats.stds[std_idx], atol=1e-6
    )


def test_unknown_category_raises(synthetic_df):
    stats = fit_person_features(synthetic_df)
    bad = synthetic_df.iloc[:1].copy()
    bad.loc[bad.index[0], "age_bucket"] = "not-a-real-bucket"
    with pytest.raises(ValueError, match="Unknown category"):
        transform_person_features(bad, stats)


def test_stats_roundtrip(synthetic_df):
    stats = fit_person_features(synthetic_df)
    d = stats.to_dict()

    d2 = json.loads(json.dumps(d))
    stats2 = PersonFeatureStats.from_dict(d2)

    np.testing.assert_allclose(stats.means, stats2.means)
    np.testing.assert_allclose(stats.stds, stats2.stds)
    assert stats.age_categories == stats2.age_categories
    assert stats.income_categories == stats2.income_categories
    assert stats.household_size_categories == stats2.household_size_categories
    assert stats.city_size_categories == stats2.city_size_categories
    assert stats.feature_columns == stats2.feature_columns

    # And the transform is byte-identical with the reloaded stats.
    X1 = transform_person_features(synthetic_df, stats)
    X2 = transform_person_features(synthetic_df, stats2)
    np.testing.assert_array_equal(X1, X2)


def test_deterministic(synthetic_df):
    s1 = fit_person_features(synthetic_df)
    s2 = fit_person_features(synthetic_df)
    np.testing.assert_array_equal(s1.means, s2.means)
    np.testing.assert_array_equal(s1.stds, s2.stds)
    assert s1.age_categories == s2.age_categories
    assert s1.income_categories == s2.income_categories
    assert s1.household_size_categories == s2.household_size_categories
    assert s1.city_size_categories == s2.city_size_categories
    assert s1.feature_columns == s2.feature_columns

    X1 = transform_person_features(synthetic_df, s1)
    X2 = transform_person_features(synthetic_df, s2)
    np.testing.assert_array_equal(X1, X2)
