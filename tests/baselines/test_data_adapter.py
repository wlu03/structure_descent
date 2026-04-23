"""Tests for :mod:`src.baselines.data_adapter`.

Verifies that PO-LEU :func:`build_choice_sets`-shaped records convert
to a valid :class:`BaselineEventBatch` with the documented
built-in feature ordering and that the shape contracts downstream
baselines rely on are preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.base import BaselineEventBatch
from src.baselines.data_adapter import (
    BUILTIN_FEATURE_NAMES,
    records_to_baseline_batch,
)


def _record(
    customer_id: str,
    chosen_idx: int,
    *,
    J: int = 4,
    prices: list[float] | None = None,
    brands: list[str] | None = None,
    is_repeat: list[bool] | None = None,
    popularity_rank: list[object] | None = None,
    category: str = "BOOK",
    is_repeat_meta: bool = False,
) -> dict:
    prices = prices if prices is not None else [10.0 + 5.0 * j for j in range(J)]
    brands = brands if brands is not None else [f"brand{j % 2}" for j in range(J)]
    if is_repeat is None:
        is_repeat = [False] * J
        is_repeat[chosen_idx] = is_repeat_meta
    popularity_rank = (
        popularity_rank
        if popularity_rank is not None
        else [f"popularity score {10 * (j + 1)}" for j in range(J)]
    )
    alt_texts = [
        {
            "title": f"t{j}",
            "category": category,
            "price": prices[j],
            "popularity_rank": popularity_rank[j],
            "brand": brands[j],
            "is_repeat": is_repeat[j],
            "state": "CA",
        }
        for j in range(J)
    ]
    return {
        "customer_id": customer_id,
        "chosen_asin": f"A{chosen_idx:02d}",
        "choice_asins": [f"A{j:02d}" for j in range(J)],
        "chosen_idx": chosen_idx,
        "z_d": np.zeros(23, dtype=np.float32),
        "c_d": f"Person {customer_id}.",
        "alt_texts": alt_texts,
        "chosen_features": {"price": prices[chosen_idx]},
        "order_date": pd.Timestamp("2024-01-01"),
        "category": category,
        "metadata": {"is_repeat": is_repeat_meta, "price": prices[chosen_idx]},
    }


def test_feature_matrix_shape_and_ordering():
    records = [_record("C0", chosen_idx=j, J=4) for j in range(3)]
    batch = records_to_baseline_batch(records)
    assert isinstance(batch, BaselineEventBatch)
    assert batch.n_events == 3
    assert batch.n_alternatives == 4
    assert batch.n_base_terms == len(BUILTIN_FEATURE_NAMES)
    assert tuple(batch.base_feature_names) == BUILTIN_FEATURE_NAMES
    for mat in batch.base_features_list:
        assert mat.shape == (4, len(BUILTIN_FEATURE_NAMES))
        assert mat.dtype == np.float32


def test_price_and_log1p_price_columns_match_alt_prices():
    prices = [9.99, 14.50, 0.0, 199.99]
    rec = _record("C0", chosen_idx=1, J=4, prices=prices)
    batch = records_to_baseline_batch([rec])
    mat = batch.base_features_list[0]
    price_col = BUILTIN_FEATURE_NAMES.index("price")
    log_col = BUILTIN_FEATURE_NAMES.index("log1p_price")
    np.testing.assert_allclose(mat[:, price_col], np.asarray(prices, dtype=np.float32))
    np.testing.assert_allclose(
        mat[:, log_col],
        np.log1p(np.maximum(np.asarray(prices, dtype=np.float32), 0.0)),
        rtol=1e-6,
    )


def test_price_rank_scales_to_unit_interval():
    # Ascending prices -> rank equals j / (J-1).
    rec = _record("C0", chosen_idx=0, J=5, prices=[1.0, 2.0, 3.0, 4.0, 5.0])
    batch = records_to_baseline_batch([rec])
    rk = batch.base_features_list[0][:, BUILTIN_FEATURE_NAMES.index("price_rank")]
    np.testing.assert_allclose(rk, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_popularity_rank_parses_embedded_number():
    rec = _record(
        "C0",
        chosen_idx=0,
        J=3,
        popularity_rank=["popularity score 42", "top 20%", 7],
    )
    batch = records_to_baseline_batch([rec])
    col = batch.base_features_list[0][
        :, BUILTIN_FEATURE_NAMES.index("popularity_rank")
    ]
    # "popularity score 42" -> 42.0; "top 20%" -> 20.0; 7 -> 7.0.
    np.testing.assert_allclose(col, np.asarray([42.0, 20.0, 7.0], dtype=np.float32))


def test_is_repeat_column_per_alt():
    flags = [False, True, False, True]
    rec = _record("C0", chosen_idx=1, J=4, is_repeat=flags)
    batch = records_to_baseline_batch([rec])
    col = batch.base_features_list[0][:, BUILTIN_FEATURE_NAMES.index("is_repeat")]
    np.testing.assert_allclose(col, np.asarray([0.0, 1.0, 0.0, 1.0]))


def test_brand_known_matches_chosen_brand_only():
    # Chosen is alt #2 (brand 'A'); two negatives share brand 'A', one
    # has brand 'B'.
    rec = _record(
        "C0",
        chosen_idx=2,
        J=4,
        brands=["A", "B", "A", "A"],
    )
    batch = records_to_baseline_batch([rec])
    col = batch.base_features_list[0][
        :, BUILTIN_FEATURE_NAMES.index("brand_known")
    ]
    np.testing.assert_allclose(col, np.asarray([1.0, 0.0, 1.0, 1.0]))


def test_brand_known_is_zero_when_chosen_brand_is_unknown():
    # Chosen has the sentinel default; every alt should get brand_known=0
    # regardless of its own brand string.
    rec = _record(
        "C0",
        chosen_idx=0,
        J=3,
        brands=["unknown_brand", "A", "A"],
    )
    batch = records_to_baseline_batch([rec])
    col = batch.base_features_list[0][
        :, BUILTIN_FEATURE_NAMES.index("brand_known")
    ]
    np.testing.assert_allclose(col, np.zeros(3, dtype=np.float32))


def test_non_uniform_J_raises():
    r0 = _record("C0", chosen_idx=0, J=3)
    r1 = _record("C1", chosen_idx=1, J=4)
    with pytest.raises(ValueError, match="uniform J"):
        records_to_baseline_batch([r0, r1])


def test_empty_records_raises():
    with pytest.raises(ValueError, match="empty"):
        records_to_baseline_batch([])


def test_metadata_defaults_is_repeat_false():
    # ``metadata`` without explicit is_repeat shouldn't crash the
    # evaluate.py breakdown code.
    rec = _record("C0", chosen_idx=0)
    rec["metadata"] = {}  # strip everything
    batch = records_to_baseline_batch([rec])
    assert batch.metadata[0]["is_repeat"] is False


def test_extra_feature_fn_appends_named_columns():
    rec = _record("C0", chosen_idx=1, J=3)

    def extra(record, alt_idx):
        return {"fake_affinity": float(alt_idx) * 2.0}

    batch = records_to_baseline_batch(
        [rec],
        extra_feature_fn=extra,
        extra_feature_names=("fake_affinity",),
    )
    assert tuple(batch.base_feature_names) == BUILTIN_FEATURE_NAMES + ("fake_affinity",)
    col = batch.base_features_list[0][:, -1]
    np.testing.assert_allclose(col, np.asarray([0.0, 2.0, 4.0]))
