"""Per-customer splits for the PO-LEU data layer.

Two shapes are exported:

- :func:`temporal_split` (Wave 8, design doc §1) — per-customer
  *within-customer* split: every customer contributes train + val + test
  rows, assigned earliest→latest. Ports ``old_pipeline/src/data_prep.py::
  temporal_split`` (lines 496–520) onto the canonical schema produced by
  :mod:`src.data.schema_map`. The v1 edge-case logic is preserved
  verbatim — ``max(1, int(n * frac))`` and the ``n_test + n_val >= n``
  collapse to ``(1, 0)`` for small customers.
- :func:`cold_start_split` — *between-customer* split: every event for
  a given customer lands in the same split. Used for the cold-start
  robustness question ("how does the model handle users it has never
  seen?"). Disjoint customer partitions are the central invariant, so
  cold-start output is validated by
  :func:`src.data.invariants.validate_cold_start_split` rather than
  the within-customer ``validate_split``.
- :func:`kfold_customer_cv` — a generator that yields K variants of the
  events DataFrame, each carrying a fresh ``split`` column built by
  rotating which customer partition serves as test. Uses the cold-start
  discipline (no within-customer leakage across folds).

All three are pure and deterministic given ``seed`` where randomisation
applies. No module-level side effects.
"""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from src.data.invariants import validate_cold_start_split, validate_split
from src.data.schema_map import DatasetSchema


__all__ = [
    "cold_start_split",
    "kfold_customer_cv",
    "temporal_split",
]


logger = logging.getLogger(__name__)


def temporal_split(
    events_df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    *,
    val_frac: float | None = None,
    test_frac: float | None = None,
) -> pd.DataFrame:
    """Per-customer temporal split.

    Either ``schema`` or both of ``val_frac``/``test_frac`` must be given.
    When both are present, explicit kwargs win. When only ``schema`` is
    given, uses ``schema.val_frac`` and ``schema.test_frac``.

    Per-customer allocation (v1 logic, preserved exactly)::

        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val

    Labels are assigned in order earliest -> latest: ``n_train`` ``"train"``
    rows, then ``n_val`` ``"val"`` rows, then ``n_test`` ``"test"`` rows.

    Runs :func:`src.data.invariants.validate_split` before returning.

    Parameters
    ----------
    events_df:
        Cleaned events DataFrame carrying ``customer_id`` and
        ``order_date`` canonical columns.
    schema:
        Optional :class:`DatasetSchema`; supplies default fractions via
        ``schema.val_frac`` / ``schema.test_frac``.
    val_frac, test_frac:
        Explicit fractions; when supplied they override the schema
        defaults.

    Returns
    -------
    pd.DataFrame
        A copy of ``events_df`` sorted by ``(customer_id, order_date)``
        with a fresh ``RangeIndex`` and a new ``split`` column whose
        values are in ``{"train", "val", "test"}``.

    Raises
    ------
    ValueError
        If neither ``schema`` nor both of ``val_frac``/``test_frac`` are
        provided.
    InvariantError
        From :func:`src.data.invariants.validate_split` — e.g. if any
        customer ends up with no ``"train"`` row.
    """
    # Resolve (val_frac, test_frac) from kwargs with fallback to schema.
    if val_frac is None:
        if schema is None:
            raise ValueError(
                "temporal_split: either `schema` or explicit "
                "`val_frac`/`test_frac` kwargs must be provided."
            )
        val_frac = float(schema.val_frac)
    else:
        val_frac = float(val_frac)

    if test_frac is None:
        if schema is None:
            raise ValueError(
                "temporal_split: either `schema` or explicit "
                "`val_frac`/`test_frac` kwargs must be provided."
            )
        test_frac = float(schema.test_frac)
    else:
        test_frac = float(test_frac)

    # Deterministic per-customer temporal ordering.
    df = (
        events_df.sort_values(["customer_id", "order_date"], kind="mergesort")
        .copy()
        .reset_index(drop=True)
    )

    def assign_splits(n: int) -> list[str]:
        # v1 edge-case logic — preserved EXACTLY.
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val
        return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    split_labels: list[str] = []
    for _, grp in df.groupby("customer_id", sort=False):
        split_labels.extend(assign_splits(len(grp.index)))

    df["split"] = split_labels

    counts = df["split"].value_counts()
    n_train = int(counts.get("train", 0))
    n_val = int(counts.get("val", 0))
    n_test = int(counts.get("test", 0))

    # Post-stage validation: asserts split values subset of
    # {train, val, test} and every customer has >= 1 train row.
    validate_split(df)

    logger.info(
        "temporal_split: val_frac=%.4f, test_frac=%.4f -> "
        "train=%d, val=%d, test=%d (total=%d).",
        val_frac,
        test_frac,
        n_train,
        n_val,
        n_test,
        len(df),
    )

    return df


# --------------------------------------------------------------------------- #
# Cold-start split (between-customer partition)
# --------------------------------------------------------------------------- #


def _partition_customers(
    customers: list[object],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[set, set, set]:
    """Partition an iterable of customer ids into disjoint
    ``(train, val, test)`` sets by a seeded shuffle.

    The partition uses ``np.floor`` so fractions round down; any residue
    lands in the train bucket. This matches the usual "train gets the
    remainder" convention in k-fold libraries and keeps train ≥ val +
    test for reasonable fractions.
    """
    if val_frac < 0 or test_frac < 0:
        raise ValueError(
            f"cold_start_split: val_frac ({val_frac}) and test_frac "
            f"({test_frac}) must be non-negative."
        )
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"cold_start_split: val_frac + test_frac must be < 1.0 so "
            f"there is at least one train customer; got "
            f"{val_frac} + {test_frac} = {val_frac + test_frac}."
        )

    n = len(customers)
    if n == 0:
        return set(), set(), set()

    rng = np.random.default_rng(int(seed))
    # Shuffle a stable copy so the same (customers, seed) always yields
    # the same partition regardless of the caller's dict/set iteration
    # order.
    shuffled = list(customers)
    rng.shuffle(shuffled)

    n_test = int(np.floor(n * float(test_frac)))
    n_val = int(np.floor(n * float(val_frac)))
    # Guard against the degenerate tiny-N case: if the floor produced
    # zeros but the caller asked for a positive fraction, promote to
    # at least one so every requested split is represented (unless the
    # customer pool is too small to support it, in which case the
    # final assertion below fires and the caller sees a clean error).
    if n_val == 0 and val_frac > 0 and n > 2:
        n_val = 1
    if n_test == 0 and test_frac > 0 and n > 2:
        n_test = 1
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"cold_start_split: n_train={n_train} after partitioning "
            f"{n} customers with val_frac={val_frac}, test_frac={test_frac}. "
            "Need at least one train customer."
        )

    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    test = set(shuffled[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def cold_start_split(
    events_df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    *,
    val_customer_frac: float | None = None,
    test_customer_frac: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Cold-start (between-customer) split.

    Partition **customers** into disjoint ``train / val / test`` groups;
    every event for a given customer lands in that customer's group.
    Used to measure how a model generalises to users it has never seen
    during training — the complement of :func:`temporal_split`, which
    slices time within each customer.

    Either ``schema`` or both of ``val_customer_frac`` /
    ``test_customer_frac`` must be given. When both are present, explicit
    kwargs win. When only ``schema`` is given, uses
    ``schema.val_frac`` / ``schema.test_frac`` — the same schema fields
    ``temporal_split`` reads, deliberately reused because a paper that
    reports temporal 90/5/5 and cold-start 80/10/10 on the same dataset
    is confusing; callers who want different fractions should pass the
    kwargs explicitly.

    Parameters
    ----------
    events_df:
        Cleaned events DataFrame carrying the ``customer_id`` canonical
        column.
    schema:
        Optional :class:`DatasetSchema`; supplies default fractions via
        ``schema.val_frac`` / ``schema.test_frac`` (reinterpreted as
        *customer* fractions in cold-start mode).
    val_customer_frac, test_customer_frac:
        Explicit customer-level fractions; override the schema
        defaults. ``val_customer_frac + test_customer_frac`` must be
        strictly less than 1.0.
    seed:
        RNG seed for the customer shuffle. Reproducible across runs.

    Returns
    -------
    pd.DataFrame
        A copy of ``events_df`` with a fresh ``RangeIndex`` and a new
        ``split`` column whose values are in ``{"train", "val", "test"}``.
        Row order within each customer's events is preserved.

    Notes
    -----
    The returned partition is **row-order invariant**: two input frames
    with identical customer sets but different row orders produce
    identical per-customer split assignments for the same ``seed``.

    ``customer_id`` is coerced to ``str`` in a local copy before sorting
    so that mixed-type ids (e.g. some rows holding ``"42"`` and others
    holding ``42``) collapse onto a single canonical id rather than
    raising a pandas ``TypeError`` or producing an order-sensitive sort.
    A NaN / null ``customer_id`` has no clean string representation and
    raises :class:`ValueError`.

    Raises
    ------
    ValueError
        If neither ``schema`` nor both of ``val_customer_frac`` /
        ``test_customer_frac`` are provided, the requested fractions
        leave zero train customers, or any ``customer_id`` is NaN /
        null.
    InvariantError
        From :func:`src.data.invariants.validate_cold_start_split` if
        a customer somehow ends up under more than one split label.
    """
    # Resolve (val_frac, test_frac) from kwargs with fallback to schema.
    if val_customer_frac is None:
        if schema is None:
            raise ValueError(
                "cold_start_split: either `schema` or explicit "
                "`val_customer_frac`/`test_customer_frac` kwargs must "
                "be provided."
            )
        val_customer_frac = float(schema.val_frac)
    else:
        val_customer_frac = float(val_customer_frac)

    if test_customer_frac is None:
        if schema is None:
            raise ValueError(
                "cold_start_split: either `schema` or explicit "
                "`val_customer_frac`/`test_customer_frac` kwargs must "
                "be provided."
            )
        test_customer_frac = float(schema.test_frac)
    else:
        test_customer_frac = float(test_customer_frac)

    if "customer_id" not in events_df.columns:
        raise KeyError(
            "cold_start_split: events_df is missing required column "
            "'customer_id'."
        )

    # Coerce customer_id to str BEFORE sorting so mixed-type ids
    # (e.g. "42" vs 42) produce a stable, reproducible partition
    # independent of caller row order. A NaN id has no clean string
    # form, so reject the frame loudly rather than silently map it to
    # the literal string "nan".
    if events_df["customer_id"].isna().any():
        raise ValueError(
            "cold_start_split: events_df contains NaN / null "
            "customer_id values; every row must carry a concrete "
            "customer id for a reproducible partition."
        )
    df = events_df.copy()
    df["customer_id"] = df["customer_id"].astype(str)

    # Deterministic ordering: sort by (customer_id, order_date) when
    # order_date is available so the returned DataFrame has a
    # consistent shape regardless of input ordering. Falls back to
    # customer_id-only when order_date is missing (e.g. synthetic
    # fixtures).
    sort_cols = ["customer_id"]
    if "order_date" in events_df.columns:
        sort_cols.append("order_date")
    df = (
        df.sort_values(sort_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    customers = df["customer_id"].drop_duplicates().tolist()
    train_ids, val_ids, test_ids = _partition_customers(
        customers,
        val_frac=val_customer_frac,
        test_frac=test_customer_frac,
        seed=seed,
    )

    def _label(cid: object) -> str:
        if cid in train_ids:
            return "train"
        if cid in val_ids:
            return "val"
        return "test"  # every customer is in exactly one of the three sets

    df["split"] = df["customer_id"].map(_label).astype(object)

    counts = df["split"].value_counts()
    n_train = int(counts.get("train", 0))
    n_val = int(counts.get("val", 0))
    n_test = int(counts.get("test", 0))

    # Post-stage invariant check: customer-level disjointness.
    validate_cold_start_split(df)

    logger.info(
        "cold_start_split: val_frac=%.4f, test_frac=%.4f, seed=%d -> "
        "%d train customers (%d events), %d val (%d events), "
        "%d test (%d events).",
        val_customer_frac,
        test_customer_frac,
        int(seed),
        len(train_ids),
        n_train,
        len(val_ids),
        n_val,
        len(test_ids),
        n_test,
    )

    return df


# --------------------------------------------------------------------------- #
# k-fold customer CV (cold-start discipline)
# --------------------------------------------------------------------------- #


def kfold_customer_cv(
    events_df: pd.DataFrame,
    n_folds: int = 5,
    *,
    val_customer_frac: float = 0.1,
    seed: int = 0,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """Yield ``n_folds`` cold-start customer-level CV folds.

    Customers are shuffled once (seeded), then partitioned into
    ``n_folds`` roughly-equal test buckets. At fold ``k``:

    - **test customers**  = the k-th bucket.
    - **val customers**   = ``val_customer_frac`` of the non-test
      customers, drawn deterministically from the shuffled order.
    - **train customers** = the remaining non-test, non-val customers.

    Every event for a customer lands in that customer's assigned split
    for that fold. Across the ``n_folds`` yielded DataFrames, each
    customer appears in ``test`` in **exactly one** fold (the disjoint-
    test guarantee that makes k-fold mean what it usually means).

    Parameters
    ----------
    events_df:
        Cleaned events DataFrame with ``customer_id`` (and optionally
        ``order_date``).
    n_folds:
        Number of CV folds. Typical: 5.
    val_customer_frac:
        Fraction of the *non-test* customer pool reserved for
        validation at each fold. Default 0.1.
    seed:
        RNG seed for the global customer shuffle. The same ``seed``
        always yields the same ``n_folds`` partitions.

    Yields
    ------
    (fold_idx, events_df_with_split) : tuple
        ``fold_idx`` in ``[0, n_folds)``. ``events_df_with_split`` is
        a fresh copy with the ``split`` column populated per the rules
        above; passes :func:`validate_cold_start_split`.

    Typical usage
    -------------

    ::

        from src.baselines.run_all import run_all_baselines
        from src.baselines.data_adapter import records_to_baseline_batch
        from src.data.split import kfold_customer_cv

        per_fold_rows = []
        for fold_idx, df in kfold_customer_cv(events, n_folds=5, seed=7):
            records = build_choice_sets(df, ...)  # downstream as usual
            train_b = records_to_baseline_batch([r for r in records
                                                 if split_of[r] == "train"])
            val_b   = records_to_baseline_batch([...])
            test_b  = records_to_baseline_batch([...])
            rows = run_all_baselines(train_b, val_b, test_b)
            per_fold_rows.append(rows)

        # Aggregate: mean ± std across per_fold_rows per baseline.

    Notes
    -----
    - This function does NOT touch ``build_choice_sets`` / the baseline
      runner — it only produces the ``split`` column. The caller is
      responsible for threading each fold's DataFrame through the rest
      of the pipeline and aggregating the per-fold leaderboards.
    - For "repeat the same sampling 3 times with different seeds" (the
      Wave-12 pattern), call this with a single fold count and rotate
      ``seed`` across invocations instead. True k-fold guarantees
      disjoint test sets; seed repetition does not.
    - **Row-order invariance**: the customer list is derived from a
      ``(customer_id, order_date)``-sorted copy of the input, so two
      frames with identical customer sets but different row orders
      produce identical fold partitions for the same ``seed``. This
      mirrors the discipline of :func:`cold_start_split`.
    - ``customer_id`` is coerced to ``str`` in a local copy before
      sorting so mixed-type ids (e.g. ``"42"`` vs ``42``) collapse
      onto a single canonical id. A NaN / null id raises.

    Raises
    ------
    ValueError
        If ``n_folds < 2``, the customer pool is too small to
        support ``n_folds`` non-empty test buckets plus a train set,
        or any ``customer_id`` is NaN / null.
    """
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(
            f"kfold_customer_cv: n_folds must be >= 2, got {n_folds}."
        )
    if "customer_id" not in events_df.columns:
        raise KeyError(
            "kfold_customer_cv: events_df is missing required column "
            "'customer_id'."
        )
    if not (0.0 <= val_customer_frac < 1.0):
        raise ValueError(
            f"kfold_customer_cv: val_customer_frac must be in [0, 1), "
            f"got {val_customer_frac}."
        )

    # Row-order invariance: coerce customer_id to str and sort by
    # (customer_id, order_date) BEFORE extracting the unique customer
    # list, so the seeded shuffle operates on an order deterministic
    # in the customer set rather than in caller-supplied row order.
    # Mirrors cold_start_split's F1/F6 discipline.
    if events_df["customer_id"].isna().any():
        raise ValueError(
            "kfold_customer_cv: events_df contains NaN / null "
            "customer_id values; every row must carry a concrete "
            "customer id for a reproducible partition."
        )
    base_df = events_df.copy()
    base_df["customer_id"] = base_df["customer_id"].astype(str)

    sort_cols = ["customer_id"]
    if "order_date" in events_df.columns:
        sort_cols.append("order_date")
    base_df = (
        base_df.sort_values(sort_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    customers = base_df["customer_id"].drop_duplicates().tolist()
    n_customers = len(customers)
    if n_customers < n_folds:
        raise ValueError(
            f"kfold_customer_cv: n_customers={n_customers} is smaller "
            f"than n_folds={n_folds}; cannot form non-empty test "
            "buckets."
        )

    rng = np.random.default_rng(int(seed))
    shuffled = list(customers)
    rng.shuffle(shuffled)

    # np.array_split handles the "not evenly divisible" case by
    # distributing the remainder across the first few folds (fold 0
    # gets ceil(n/K), last folds get floor(n/K)). This is the standard
    # k-fold chunking.
    test_buckets: list[list[object]] = [
        list(chunk) for chunk in np.array_split(shuffled, n_folds)
    ]

    for fold_idx in range(n_folds):
        test_ids = set(test_buckets[fold_idx])
        non_test = [c for c in shuffled if c not in test_ids]

        n_val = int(np.floor(len(non_test) * float(val_customer_frac)))
        # Promote the val bucket to at least 1 if the user asked for
        # any val at all and there's room.
        if n_val == 0 and val_customer_frac > 0 and len(non_test) > 1:
            n_val = 1
        val_ids = set(non_test[:n_val])
        train_ids = set(non_test[n_val:])

        if not train_ids:
            raise ValueError(
                f"kfold_customer_cv: fold {fold_idx} has 0 train "
                f"customers (n_customers={n_customers}, "
                f"n_folds={n_folds}, val_customer_frac={val_customer_frac})."
            )

        # Each yielded fold must be an independent copy so the caller
        # can mutate one fold's DataFrame without bleeding into the
        # next fold's yield.
        df = base_df.copy()

        def _label_fold(cid: object, _train=train_ids, _val=val_ids) -> str:
            if cid in _train:
                return "train"
            if cid in _val:
                return "val"
            return "test"

        df["split"] = df["customer_id"].map(_label_fold).astype(object)

        validate_cold_start_split(df)

        logger.info(
            "kfold_customer_cv: fold=%d/%d seed=%d -> "
            "%d train customers, %d val, %d test.",
            fold_idx + 1,
            n_folds,
            int(seed),
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )

        yield fold_idx, df
