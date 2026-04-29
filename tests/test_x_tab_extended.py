"""Tests for the extended Sifringer-residual feature catalog.

Adds: ``popularity_rank``, ``log1p_popularity_rank``, ``is_repeat`` —
features the encoder cannot read directly (Sifringer L-MNL principle:
only add features the neural branch genuinely cannot see).

The existing 3 price-derived features (``price``, ``log1p_price``,
``price_rank``) keep their tests in :mod:`tests.test_batching`; this
file covers the new record-derived columns and the validation surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.batching import (
    SUPPORTED_TABULAR_FEATURES,
    _build_x_tab_matrix,
)


# ---------------------------------------------------------------------------
# Tiny fixture: 2 events, J=3 alternatives.
# ---------------------------------------------------------------------------


def _records() -> list[dict]:
    """2 events × J=3 alts. Each alt carries popularity_rank + is_repeat."""
    return [
        {
            "alt_texts": [
                {"title": "a", "category": "x", "price": 10.0,
                 "popularity_rank": 100.0, "is_repeat": 1},
                {"title": "b", "category": "x", "price": 20.0,
                 "popularity_rank": 50.0, "is_repeat": 0},
                {"title": "c", "category": "x", "price": 30.0,
                 "popularity_rank": 0.0, "is_repeat": 0},
            ],
        },
        {
            "alt_texts": [
                {"title": "d", "category": "y", "price": 5.0,
                 "popularity_rank": 200.0, "is_repeat": 0},
                {"title": "e", "category": "y", "price": 15.0,
                 "popularity_rank": 75.0, "is_repeat": 1},
                {"title": "f", "category": "y", "price": 25.0,
                 "popularity_rank": 25.0, "is_repeat": 0},
            ],
        },
    ]


def _prices(records: list[dict]) -> np.ndarray:
    N = len(records)
    J = len(records[0]["alt_texts"])
    out = np.zeros((N, J), dtype=np.float32)
    for i, rec in enumerate(records):
        for j, alt in enumerate(rec["alt_texts"]):
            out[i, j] = float(alt["price"])
    return out


# ---------------------------------------------------------------------------
# Feature-catalog content
# ---------------------------------------------------------------------------


def test_supported_features_includes_price_and_record_families() -> None:
    s = set(SUPPORTED_TABULAR_FEATURES)
    # Price-family (legacy)
    assert {"price", "log1p_price", "price_rank"}.issubset(s)
    # Record-family (new)
    assert {"popularity_rank", "log1p_popularity_rank", "is_repeat"}.issubset(s)


# ---------------------------------------------------------------------------
# Record-derived feature values
# ---------------------------------------------------------------------------


def test_popularity_rank_passthrough() -> None:
    recs = _records()
    x = _build_x_tab_matrix(_prices(recs), ("popularity_rank",), records=recs)
    assert x.shape == (2, 3, 1)
    np.testing.assert_array_equal(
        x[:, :, 0],
        np.array([[100, 50, 0], [200, 75, 25]], dtype=np.float32),
    )


def test_log1p_popularity_rank() -> None:
    recs = _records()
    x = _build_x_tab_matrix(
        _prices(recs), ("log1p_popularity_rank",), records=recs,
    )
    expected = np.log1p([[100, 50, 0], [200, 75, 25]]).astype(np.float32)
    np.testing.assert_allclose(x[:, :, 0], expected, rtol=1e-6)


def test_is_repeat_binary() -> None:
    recs = _records()
    x = _build_x_tab_matrix(_prices(recs), ("is_repeat",), records=recs)
    np.testing.assert_array_equal(
        x[:, :, 0],
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )


def test_combined_columns_preserve_order() -> None:
    """Multi-feature pull preserves column order matching feature_names."""
    recs = _records()
    names = ("price", "popularity_rank", "is_repeat")
    x = _build_x_tab_matrix(_prices(recs), names, records=recs)
    assert x.shape == (2, 3, 3)
    np.testing.assert_array_equal(x[0, :, 0], [10, 20, 30])      # price
    np.testing.assert_array_equal(x[0, :, 1], [100, 50, 0])      # popularity
    np.testing.assert_array_equal(x[0, :, 2], [1, 0, 0])         # is_repeat


# ---------------------------------------------------------------------------
# Defensive validation
# ---------------------------------------------------------------------------


def test_record_feature_without_records_raises() -> None:
    """Asking for popularity_rank without records is a config typo, fail loud."""
    p = np.array([[10.0, 20.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="require ``records``"):
        _build_x_tab_matrix(p, ("popularity_rank",), records=None)


def test_records_length_mismatch_raises() -> None:
    p = np.zeros((3, 2), dtype=np.float32)  # N=3
    recs = [{"alt_texts": [{"popularity_rank": 1.0}, {"popularity_rank": 2.0}]}]
    with pytest.raises(ValueError, match="does not match prices N"):
        _build_x_tab_matrix(p, ("popularity_rank",), records=recs)


def test_unknown_feature_name_raises() -> None:
    recs = _records()
    with pytest.raises(ValueError, match="Unsupported tabular feature"):
        _build_x_tab_matrix(
            _prices(recs), ("not_a_feature",), records=recs,
        )


def test_missing_alt_field_yields_zero() -> None:
    """A record missing 'is_repeat' gets 0.0 for that cell, not a crash."""
    recs = [
        {"alt_texts": [
            {"price": 1.0, "popularity_rank": 5.0},  # no is_repeat
            {"price": 2.0, "popularity_rank": 6.0, "is_repeat": 1},
        ]},
    ]
    p = np.array([[1.0, 2.0]], dtype=np.float32)
    x = _build_x_tab_matrix(p, ("is_repeat",), records=recs)
    np.testing.assert_array_equal(x[0, :, 0], [0.0, 1.0])


# ---------------------------------------------------------------------------
# Backward compat: the 3-feature default keeps its exact behaviour
# ---------------------------------------------------------------------------


def test_legacy_price_features_unchanged_when_records_passed() -> None:
    """Adding records= shouldn't change the price-only output."""
    recs = _records()
    p = _prices(recs)
    legacy = _build_x_tab_matrix(p, ("price", "log1p_price", "price_rank"))
    new = _build_x_tab_matrix(
        p, ("price", "log1p_price", "price_rank"), records=recs,
    )
    np.testing.assert_array_equal(legacy, new)
