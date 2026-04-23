"""Tests for src/data/split.py (Wave 8, design doc §1).

Covers the per-customer temporal split, schema/kwargs contract, and the
v1 edge-case logic for small customers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_events
from src.data.invariants import InvariantError
from src.data.load import load
from src.data.schema_map import load_schema
from src.data.split import (
    cold_start_split,
    kfold_customer_cv,
    temporal_split,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def amazon_schema():
    return load_schema(AMAZON_YAML)


@pytest.fixture(scope="module")
def cleaned_events(amazon_schema) -> pd.DataFrame:
    """Real cleaned events from the 100-customer fixture."""
    events_raw, _ = load(
        amazon_schema,
        events_path=EVENTS_FIXTURE,
        persons_path=PERSONS_FIXTURE,
    )
    return clean_events(events_raw, amazon_schema)


def _make_events(n_by_customer: dict[str, int]) -> pd.DataFrame:
    """Build a tiny synthetic events frame for edge-case tests.

    ``n_by_customer`` maps ``customer_id -> n_events``. Each customer's
    events are stamped on consecutive days starting 2020-01-01 so that
    temporal ordering is unambiguous.
    """
    rows = []
    base = pd.Timestamp("2020-01-01")
    for cid, n in n_by_customer.items():
        for i in range(n):
            rows.append(
                {
                    "customer_id": cid,
                    "order_date": base + pd.Timedelta(days=i),
                    "price": 1.0 + i,
                    "asin": f"A{i:04d}",
                    "category": "CAT",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_temporal_split_adds_column(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert "split" in out.columns


def test_split_values_subset_of_train_val_test(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert set(out["split"].unique()).issubset({"train", "val", "test"})


def test_every_customer_has_train_row(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    train_customers = set(out.loc[out["split"] == "train", "customer_id"].unique())
    all_customers = set(out["customer_id"].unique())
    assert train_customers == all_customers


def test_large_customer_has_expected_fractions():
    df = _make_events({"C_big": 100})
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    c = out.loc[out["customer_id"] == "C_big", "split"].value_counts()
    # int(100 * 0.1) == 10 exactly, with n_train = 80.
    assert int(c.get("test", 0)) == 10
    assert int(c.get("val", 0)) == 10
    assert int(c.get("train", 0)) == 80


def test_small_customer_gets_at_least_one_train():
    # With n=2, val_frac=test_frac=0.1: int(0.2)=0 -> max(1, 0)=1 each,
    # so n_test + n_val == 2 == n, triggering the collapse to (1, 0).
    # That gives n_train=1, n_val=0, n_test=1.
    df = _make_events({"C_small": 2})
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    c = out.loc[out["customer_id"] == "C_small", "split"].value_counts()
    assert int(c.get("train", 0)) == 1
    assert int(c.get("val", 0)) == 0
    assert int(c.get("test", 0)) == 1


def test_1_event_customer_is_all_train():
    """Document the v1 edge-case interaction with validate_split.

    Under v1's exactly-ported allocation, a 1-event customer yields
    ``n_test=max(1,0)=1``, ``n_val=max(1,0)=1``; the collapse sets
    ``(n_test, n_val) = (1, 0)`` and ``n_train = 1 - 1 - 0 = 0``. That
    single row is labelled ``"test"``, so :func:`validate_split` rejects
    the frame for having no train row for that customer. The v1 logic
    therefore does NOT support standalone 1-event customers — the test
    pins that behaviour so the invariant hook stays loud.
    """
    df = _make_events({"C_one": 1})
    with pytest.raises(InvariantError):
        temporal_split(df, val_frac=0.1, test_frac=0.1)


def test_split_is_temporal():
    df = _make_events({"C_a": 20, "C_b": 30})
    # Shuffle the rows to ensure temporal_split re-sorts.
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = temporal_split(df, val_frac=0.1, test_frac=0.1)
    for cid, grp in out.groupby("customer_id"):
        # Within each customer, dates must be non-decreasing.
        dates = grp["order_date"].tolist()
        assert dates == sorted(dates)
        # The earliest row's split must be 'train' and the latest 'test'.
        assert grp.iloc[0]["split"] == "train"
        assert grp.iloc[-1]["split"] == "test"


def test_split_preserves_row_count(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert len(out) == len(cleaned_events)


def test_split_reset_index(cleaned_events):
    out = temporal_split(cleaned_events, val_frac=0.1, test_frac=0.1)
    assert list(out.index) == list(range(len(out)))


# --------------------------------------------------------------------------- #
# schema-or-kwargs contract
# --------------------------------------------------------------------------- #


def test_split_uses_schema_defaults(amazon_schema, cleaned_events):
    # schema only, no explicit fractions.
    out = temporal_split(cleaned_events, schema=amazon_schema)
    assert "split" in out.columns
    # Compare against an explicit-kwargs run with the schema's values.
    out_explicit = temporal_split(
        cleaned_events,
        val_frac=amazon_schema.val_frac,
        test_frac=amazon_schema.test_frac,
    )
    pd.testing.assert_series_equal(out["split"], out_explicit["split"])


def test_split_explicit_kwargs_override_schema(amazon_schema):
    # Build a frame large enough that 0.1 vs 0.4 produce different counts.
    df = _make_events({"C_big": 100})
    out_schema = temporal_split(df, schema=amazon_schema)
    out_override = temporal_split(
        df, schema=amazon_schema, val_frac=0.4, test_frac=0.4
    )
    # Different fractions must yield different per-split counts.
    cs = out_schema["split"].value_counts()
    co = out_override["split"].value_counts()
    assert int(cs.get("test", 0)) != int(co.get("test", 0))
    # Override uses 0.4 -> int(100 * 0.4) = 40 for each of val/test.
    assert int(co.get("val", 0)) == 40
    assert int(co.get("test", 0)) == 40
    assert int(co.get("train", 0)) == 20


def test_split_without_schema_or_kwargs_raises():
    df = _make_events({"C_a": 10})
    with pytest.raises(ValueError):
        temporal_split(df)
    # Partial-kwargs (only one of the two) with no schema also raises.
    with pytest.raises(ValueError):
        temporal_split(df, val_frac=0.1)
    with pytest.raises(ValueError):
        temporal_split(df, test_frac=0.1)


# --------------------------------------------------------------------------- #
# Cold-start split (between-customer)
# --------------------------------------------------------------------------- #


def _big_fake_events(n_customers: int, events_per_customer: int = 8) -> pd.DataFrame:
    """Build a synthetic events frame with ``n_customers`` distinct ids.

    Each customer gets ``events_per_customer`` rows on consecutive days
    starting 2020-01-01. Useful for cold-start / k-fold tests where we
    need a non-trivial customer pool without spinning up the real Amazon
    fixture.
    """
    by = {f"C_{i:04d}": events_per_customer for i in range(n_customers)}
    return _make_events(by)


def test_cold_start_split_adds_column():
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    assert "split" in out.columns


def test_cold_start_split_values_subset_of_train_val_test():
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    assert set(out["split"].unique()) <= {"train", "val", "test"}


def test_cold_start_customers_are_disjoint_across_splits():
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    train = set(out.loc[out["split"] == "train", "customer_id"].unique())
    val = set(out.loc[out["split"] == "val", "customer_id"].unique())
    test = set(out.loc[out["split"] == "test", "customer_id"].unique())
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_cold_start_every_customer_fully_in_one_split():
    # A customer never shows up under more than one split label — the
    # core cold-start invariant.
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    per_customer = out.groupby("customer_id")["split"].nunique()
    assert per_customer.max() == 1


def test_cold_start_fractions_roughly_match_requested():
    n = 100
    df = _big_fake_events(n, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    n_val_customers = out.loc[out["split"] == "val", "customer_id"].nunique()
    n_test_customers = out.loc[out["split"] == "test", "customer_id"].nunique()
    n_train_customers = out.loc[out["split"] == "train", "customer_id"].nunique()
    assert n_val_customers == 20
    assert n_test_customers == 20
    assert n_train_customers == 60


def test_cold_start_is_reproducible_with_seed():
    df = _big_fake_events(50, events_per_customer=4)
    out1 = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=42)
    out2 = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=42)
    pd.testing.assert_series_equal(
        out1["split"].reset_index(drop=True),
        out2["split"].reset_index(drop=True),
    )


def test_cold_start_different_seeds_yield_different_partitions():
    df = _big_fake_events(50, events_per_customer=4)
    out1 = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    out2 = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=2)
    # At least SOME customers land in different splits across seeds.
    labels1 = dict(zip(out1["customer_id"], out1["split"]))
    labels2 = dict(zip(out2["customer_id"], out2["split"]))
    disagreements = sum(1 for c in labels1 if labels1[c] != labels2[c])
    assert disagreements > 0


def test_cold_start_preserves_row_count():
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, val_customer_frac=0.2, test_customer_frac=0.2, seed=1)
    assert len(out) == len(df)


def test_cold_start_uses_schema_defaults(amazon_schema):
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(df, schema=amazon_schema, seed=0)
    # amazon.yaml defaults (val_frac=0.1, test_frac=0.1) reinterpreted
    # as customer fractions: 5 val + 5 test + 40 train.
    n_train = out.loc[out["split"] == "train", "customer_id"].nunique()
    n_val = out.loc[out["split"] == "val", "customer_id"].nunique()
    n_test = out.loc[out["split"] == "test", "customer_id"].nunique()
    assert n_val == 5
    assert n_test == 5
    assert n_train == 40


def test_cold_start_explicit_kwargs_override_schema(amazon_schema):
    df = _big_fake_events(50, events_per_customer=4)
    out = cold_start_split(
        df,
        schema=amazon_schema,
        val_customer_frac=0.3,
        test_customer_frac=0.3,
        seed=0,
    )
    n_val = out.loc[out["split"] == "val", "customer_id"].nunique()
    n_test = out.loc[out["split"] == "test", "customer_id"].nunique()
    assert n_val == 15
    assert n_test == 15


def test_cold_start_without_schema_or_kwargs_raises():
    df = _big_fake_events(10, events_per_customer=4)
    with pytest.raises(ValueError):
        cold_start_split(df)
    with pytest.raises(ValueError):
        cold_start_split(df, val_customer_frac=0.1)
    with pytest.raises(ValueError):
        cold_start_split(df, test_customer_frac=0.1)


def test_cold_start_raises_when_fractions_leave_no_train():
    df = _big_fake_events(10, events_per_customer=4)
    with pytest.raises(ValueError):
        cold_start_split(df, val_customer_frac=0.5, test_customer_frac=0.5, seed=0)


# --------------------------------------------------------------------------- #
# k-fold customer CV
# --------------------------------------------------------------------------- #


def test_kfold_yields_n_folds():
    df = _big_fake_events(50, events_per_customer=3)
    folds = list(kfold_customer_cv(df, n_folds=5, seed=7))
    assert len(folds) == 5
    assert [i for i, _ in folds] == [0, 1, 2, 3, 4]


def test_kfold_each_fold_passes_cold_start_invariant():
    df = _big_fake_events(50, events_per_customer=3)
    for _, fold_df in kfold_customer_cv(df, n_folds=5, seed=7):
        per_customer = fold_df.groupby("customer_id")["split"].nunique()
        assert per_customer.max() == 1


def test_kfold_test_customers_are_disjoint_across_folds():
    # The central k-fold guarantee: each customer appears in the
    # ``test`` split in EXACTLY one fold.
    df = _big_fake_events(50, events_per_customer=3)
    appearances: dict = {}
    for fold_idx, fold_df in kfold_customer_cv(df, n_folds=5, seed=7):
        test_ids = set(
            fold_df.loc[fold_df["split"] == "test", "customer_id"].unique()
        )
        for cid in test_ids:
            appearances.setdefault(cid, []).append(fold_idx)
    for cid, folds_seen in appearances.items():
        assert len(folds_seen) == 1, (
            f"customer {cid} appears as test in folds {folds_seen}; expected 1"
        )


def test_kfold_union_of_test_folds_covers_every_customer():
    df = _big_fake_events(50, events_per_customer=3)
    covered = set()
    for _, fold_df in kfold_customer_cv(df, n_folds=5, seed=7):
        covered.update(
            fold_df.loc[fold_df["split"] == "test", "customer_id"].unique()
        )
    assert covered == set(df["customer_id"].unique())


def test_kfold_deterministic_given_seed():
    df = _big_fake_events(50, events_per_customer=3)
    runs = []
    for seed in (7, 7):  # same seed twice
        folds = [
            (i, tuple(sorted(fdf.loc[fdf["split"] == "test", "customer_id"])))
            for i, fdf in kfold_customer_cv(df, n_folds=5, seed=seed)
        ]
        runs.append(folds)
    assert runs[0] == runs[1]


def test_kfold_different_seeds_differ():
    df = _big_fake_events(50, events_per_customer=3)

    def _sig(seed: int) -> list:
        return [
            tuple(sorted(fdf.loc[fdf["split"] == "test", "customer_id"]))
            for _, fdf in kfold_customer_cv(df, n_folds=5, seed=seed)
        ]

    assert _sig(7) != _sig(13)


def test_kfold_rejects_n_folds_lt_2():
    df = _big_fake_events(10, events_per_customer=3)
    with pytest.raises(ValueError):
        list(kfold_customer_cv(df, n_folds=1))


def test_kfold_rejects_when_n_customers_less_than_folds():
    df = _big_fake_events(3, events_per_customer=3)
    with pytest.raises(ValueError):
        list(kfold_customer_cv(df, n_folds=5))


# --------------------------------------------------------------------------- #
# F1 / F6 hardening — reproducibility + row-order invariance
# --------------------------------------------------------------------------- #


def _customer_split_map(df: pd.DataFrame) -> dict[str, str]:
    """Return a ``customer_id -> split`` dict from a split DataFrame.

    Since cold-start guarantees one split label per customer, this
    collapse loses no information and makes cross-run comparison clean.
    """
    pairs = df[["customer_id", "split"]].drop_duplicates()
    return dict(zip(pairs["customer_id"].astype(str), pairs["split"]))


def test_cold_start_partition_invariant_under_row_shuffle():
    """F6 analog for ``cold_start_split``: shuffling input rows must
    not change the seeded partition."""
    df = _big_fake_events(50, events_per_customer=4)

    out_a = cold_start_split(
        df, val_customer_frac=0.2, test_customer_frac=0.2, seed=42
    )
    mapping_a = _customer_split_map(out_a)

    shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
    out_b = cold_start_split(
        shuffled, val_customer_frac=0.2, test_customer_frac=0.2, seed=42
    )
    mapping_b = _customer_split_map(out_b)

    assert mapping_a == mapping_b


def test_cold_start_partition_invariant_under_global_numpy_seed():
    """The function must use a local ``np.random.default_rng`` seeded
    only by ``seed``; the global legacy RNG state must not leak in."""
    df = _big_fake_events(50, events_per_customer=4)

    np.random.seed(1)
    out1 = cold_start_split(
        df, val_customer_frac=0.2, test_customer_frac=0.2, seed=42
    )
    mapping1 = _customer_split_map(out1)

    np.random.seed(2)
    out2 = cold_start_split(
        df, val_customer_frac=0.2, test_customer_frac=0.2, seed=42
    )
    mapping2 = _customer_split_map(out2)

    assert mapping1 == mapping2


def test_cold_start_mixed_type_customer_id_coerced_or_raises():
    """Mixed-type ids (``"C_a"`` and ``1`` for the same logical
    customer) must either be coerced to a single canonical str id or
    raise ``ValueError``. We pin the coerce-to-str behavior documented
    in the F1 fix: both rows collapse onto customer ``"1"`` vs ``"C_a"``
    (they are distinct strings, so the customer count is 2 and every
    row is deterministically labelled)."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    # Two rows with str id, two rows with int id — different str forms,
    # but both should sort/partition reproducibly after coercion.
    for i, cid in enumerate(["C_a", 1, "C_a", 1]):
        rows.append(
            {
                "customer_id": cid,
                "order_date": base + pd.Timedelta(days=i),
                "price": 1.0,
                "asin": f"A{i:04d}",
                "category": "CAT",
            }
        )
    # Add enough additional customers so val/test fractions produce
    # at least one train customer per the _partition_customers floor.
    for i in range(8):
        rows.append(
            {
                "customer_id": f"C_extra_{i}",
                "order_date": base + pd.Timedelta(days=10 + i),
                "price": 1.0,
                "asin": f"B{i:04d}",
                "category": "CAT",
            }
        )
    df = pd.DataFrame(rows)

    out = cold_start_split(
        df, val_customer_frac=0.2, test_customer_frac=0.2, seed=0
    )
    # Coercion behavior: customer_id column on output is str-typed.
    assert out["customer_id"].map(type).eq(str).all()
    # Each logical id (post-coercion) lands in exactly one split.
    per_customer = out.groupby("customer_id")["split"].nunique()
    assert per_customer.max() == 1


def test_cold_start_nan_customer_id_raises():
    """A NaN / null ``customer_id`` has no clean string form — the
    function must reject the frame with a ``ValueError`` mentioning
    NaN / null."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    for i, cid in enumerate(["C_a", "C_b", None, "C_c", "C_d"]):
        rows.append(
            {
                "customer_id": cid,
                "order_date": base + pd.Timedelta(days=i),
                "price": 1.0,
                "asin": f"A{i:04d}",
                "category": "CAT",
            }
        )
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=r"(?i)na[nN]|null"):
        cold_start_split(
            df, val_customer_frac=0.2, test_customer_frac=0.2, seed=0
        )


def test_cold_start_duplicate_customer_ids_all_same_split():
    """Every row for a given customer must share a single split label,
    even when that customer's rows are scattered across the frame."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    # C_a: 20 rows, C_b: 5 rows, plus padding customers so val/test
    # fractions work out cleanly.
    pattern: list[str] = (
        ["C_a"] * 10 + ["C_b"] * 2 + ["C_a"] * 10 + ["C_b"] * 3
    )
    for i, cid in enumerate(pattern):
        rows.append(
            {
                "customer_id": cid,
                "order_date": base + pd.Timedelta(days=i),
                "price": 1.0,
                "asin": f"A{i:04d}",
                "category": "CAT",
            }
        )
    for i in range(8):
        rows.append(
            {
                "customer_id": f"C_pad_{i}",
                "order_date": base + pd.Timedelta(days=100 + i),
                "price": 1.0,
                "asin": f"P{i:04d}",
                "category": "CAT",
            }
        )
    df = pd.DataFrame(rows)

    out = cold_start_split(
        df, val_customer_frac=0.2, test_customer_frac=0.2, seed=0
    )
    a_labels = set(out.loc[out["customer_id"] == "C_a", "split"].unique())
    b_labels = set(out.loc[out["customer_id"] == "C_b", "split"].unique())
    assert len(a_labels) == 1
    assert len(b_labels) == 1
    # Sanity: 20 rows for C_a all present, 5 rows for C_b all present.
    assert (out["customer_id"] == "C_a").sum() == 20
    assert (out["customer_id"] == "C_b").sum() == 5


def test_kfold_row_order_invariance():
    """F6 fix: ``kfold_customer_cv`` must produce identical fold
    partitions for two frames with the same customer set but different
    row orders."""
    df = _big_fake_events(50, events_per_customer=3)

    def _test_customers_per_fold(frame: pd.DataFrame) -> list[tuple[str, ...]]:
        out = []
        for _, fold_df in kfold_customer_cv(frame, n_folds=5, seed=7):
            test_ids = tuple(
                sorted(
                    fold_df.loc[fold_df["split"] == "test", "customer_id"]
                    .unique()
                    .tolist()
                )
            )
            out.append(test_ids)
        return out

    sig_a = _test_customers_per_fold(df)
    shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
    sig_b = _test_customers_per_fold(shuffled)
    assert sig_a == sig_b


def test_kfold_val_and_test_disjoint_within_fold():
    """Within any single fold, val and test customer sets must be
    disjoint — otherwise the fold leaks target rows into tuning."""
    df = _big_fake_events(50, events_per_customer=3)
    for _, fold_df in kfold_customer_cv(
        df, n_folds=5, val_customer_frac=0.2, seed=7
    ):
        val_customers = set(
            fold_df.loc[fold_df["split"] == "val", "customer_id"].unique()
        )
        test_customers = set(
            fold_df.loc[fold_df["split"] == "test", "customer_id"].unique()
        )
        assert val_customers & test_customers == set()


def test_kfold_fold_df_is_independent_copy():
    """Mutating one yielded fold's DataFrame must not affect any
    subsequent fold's ``split`` column — the generator must return
    independent copies."""
    df = _big_fake_events(20, events_per_customer=3)
    folds = list(kfold_customer_cv(df, n_folds=2, seed=7))
    assert len(folds) == 2

    _, df_fold0 = folds[0]
    _, df_fold1 = folds[1]

    fold1_splits_before = df_fold1["split"].copy()
    df_fold0["split"] = "train"

    pd.testing.assert_series_equal(
        df_fold1["split"].reset_index(drop=True),
        fold1_splits_before.reset_index(drop=True),
    )
