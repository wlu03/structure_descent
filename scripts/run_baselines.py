"""Unified baseline-vs-PO-LEU leaderboard runner.

Reproduces the PO-LEU data split, converts ``build_choice_sets`` records
into :class:`BaselineEventBatch`, fits every Phase-1 baseline in
:mod:`src.baselines.run_all`, and prints a single table with the same
metric formulas and tie-breaking as :mod:`src.eval.metrics`.

This script is PO-LEU-free on purpose: it only consumes the *records*
produced by :func:`src.data.choice_sets.build_choice_sets`, so it can
run with or without the LLM-generation stage having been completed.
When a PO-LEU run artifact is supplied via ``--poleu-logits``, its
scores are folded into the leaderboard as an additional row.

CLI
---

    python -m scripts.run_baselines \\
        --dataset-yaml configs/datasets/amazon.yaml \\
        --n-customers 20 --seed 7 \\
        [--poleu-logits path/to/poleu_test_logits.npz]

    # smoke test
    python -m scripts.run_baselines --synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from src.baselines.data_adapter import records_to_baseline_batch
from src.baselines.run_all import BASELINE_REGISTRY, format_table, run_all_baselines

logger = logging.getLogger("run_baselines")


def _build_records_from_dataset(
    dataset_yaml: str,
    n_customers: int,
    seed: int,
    min_events_per_customer: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load + clean + split + subsample + build_choice_sets.

    Mirrors the Wave-10 driver (``scripts/run_dataset.py``) up to the
    point where per-event records are emitted, then splits those
    records into (train, val, test) by the temporal ``split`` column.
    """
    import pandas as pd

    from src.data.adapter import YamlAdapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import extract_extra_fields_from_row
    from src.data.split import temporal_split
    from src.data.state_features import attach_train_popularity, compute_state_features
    from src.data.survey_join import join_survey
    from src.train.loop import try_import_subsample_weights

    adapter = YamlAdapter(dataset_yaml)

    events_raw = adapter.load_events()
    persons_raw = adapter.load_persons()

    events = clean_events(events_raw, adapter.schema)
    events = join_survey(events, persons_raw, adapter.schema)
    events = compute_state_features(events)

    counts = events.groupby("customer_id").size()
    keep = set(counts[counts >= int(min_events_per_customer)].index)
    events = events[events["customer_id"].isin(keep)].reset_index(drop=True)

    events = temporal_split(events, adapter.schema)
    events = attach_train_popularity(events)

    train_events = events[events["split"] == "train"].copy()

    selected_ids, _weights = try_import_subsample_weights(
        train_events, n_customers=int(n_customers), seed=int(seed)
    )
    if selected_ids is not None:
        events = events[events["customer_id"].isin(set(selected_ids))].reset_index(
            drop=True
        )

    # Orphan filter: keep only customers that appear in BOTH train events
    # and the survey (mirrors run_dataset.py stage 8).
    persons_id_col = adapter.schema.persons_id_column
    if persons_id_col not in persons_raw.columns:
        raise SystemExit(
            f"persons_raw missing id column {persons_id_col!r} "
            f"(schema: {dataset_yaml})."
        )
    surveyed_customers = set(
        persons_raw[persons_id_col].dropna().astype(str).unique().tolist()
    )
    train_customers = set(
        events.loc[events["split"] == "train", "customer_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    joint_customers = train_customers & surveyed_customers
    if not joint_customers:
        raise SystemExit(
            "orphan filter: no customers in both train events and survey "
            f"(train={len(train_customers)}, survey={len(surveyed_customers)})."
        )
    events = events[
        events["customer_id"].astype(str).isin(joint_customers)
    ].reset_index(drop=True)

    # translate_z_d fit-on-train (mirrors run_dataset.py stage 9).
    train_events_subset = events[events["split"] == "train"].copy()
    persons_raw_keep = persons_raw[
        persons_raw[persons_id_col].astype(str).isin(joint_customers)
    ].copy()
    persons_canonical = adapter.translate_z_d(
        persons_raw_keep,
        training_events=train_events_subset,
    )

    # Wave-11 parity: normalize purchase_frequency count -> events/week.
    if len(train_events_subset) > 0 and "purchase_frequency" in persons_canonical.columns:
        _dates = pd.to_datetime(train_events_subset["order_date"])
        _window_weeks = max((_dates.max() - _dates.min()).days / 7.0, 1.0)
        persons_canonical = persons_canonical.copy()
        persons_canonical["purchase_frequency"] = (
            persons_canonical["purchase_frequency"].astype(float) / _window_weeks
        )

    # Post-translate orphan filter: translate_z_d may drop rows via
    # drop_on_unknown. Cascade-filter events to surviving customers.
    surviving = set(persons_canonical["customer_id"].astype(str))
    events = events[
        events["customer_id"].astype(str).isin(surviving)
    ].reset_index(drop=True)

    # c_d enrichment: load ``persons.c_d_extra_fields`` extras block.
    import yaml as _yaml

    with open(dataset_yaml, "r", encoding="utf-8") as _fh:
        dataset_yaml_dict = _yaml.safe_load(_fh) or {}
    extras_block = (
        dataset_yaml_dict.get("dataset", {})
        .get("persons", {})
        .get("c_d_extra_fields", {})
    ) or {}
    customer_to_extras: dict[str, dict] = {}
    if extras_block:
        for _, raw_row in persons_raw.iterrows():
            cid = str(raw_row.get(persons_id_col, ""))
            if cid not in surviving:
                continue
            customer_to_extras[cid] = extract_extra_fields_from_row(
                raw_row.to_dict(), extras_block
            )

    records = build_choice_sets(
        events,
        persons_canonical,
        adapter,
        seed=int(seed),
        n_resamples=int(adapter.schema.n_resamples),
        n_negatives=int(adapter.schema.choice_set_size) - 1,
        customer_to_extras=customer_to_extras or None,
    )

    # Split records by the ``split`` column of each record's source event.
    # ``build_choice_sets`` carries ``order_date`` but not ``split``; we
    # re-derive it from the events DataFrame keyed on (customer_id,
    # chosen_asin, order_date).
    split_map: dict[tuple, str] = {
        (str(row["customer_id"]), str(row["asin"]), row["order_date"]): str(
            row["split"]
        )
        for _, row in events.iterrows()
    }

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []
    for r in records:
        key = (str(r["customer_id"]), str(r["chosen_asin"]), r["order_date"])
        bucket = split_map.get(key, "train")
        if bucket == "train":
            train_recs.append(r)
        elif bucket == "val":
            val_recs.append(r)
        elif bucket == "test":
            test_recs.append(r)
        else:
            train_recs.append(r)

    return train_recs, val_recs, test_recs


def _poleu_row(
    logits_path: str,
    test_batch,
    train_n_events: int,
) -> dict[str, object]:
    """Compute the PO-LEU leaderboard row from a saved logits artifact.

    Expects a ``.npz`` with keys ``logits`` (shape ``(N, J)``),
    ``c_star`` (shape ``(N,)``), and optionally ``n_params``.
    """
    from src.eval.metrics import compute_all

    data = np.load(logits_path, allow_pickle=False)
    logits = data["logits"]
    c_star = data["c_star"]
    n_params = int(data["n_params"]) if "n_params" in data else 0

    em = compute_all(
        logits, c_star, n_params=n_params, n_train=int(train_n_events)
    )
    return {
        "name": "PO-LEU",
        "status": "ok",
        "top1": em.top1,
        "top5": em.top5,
        "mrr": em.mrr_val,
        "test_nll": em.nll_val,
        "aic": em.aic_val,
        "bic": em.bic_val,
        "n_params": em.n_params,
        "fit_seconds": 0.0,
        "description": f"PO-LEU artifact {Path(logits_path).name}",
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-yaml", type=str, default=None)
    parser.add_argument("--n-customers", type=int, default=20)
    parser.add_argument("--min-events-per-customer", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Smoke-test against synthetic BaselineEventBatch (no dataset read).",
    )
    parser.add_argument(
        "--poleu-logits",
        type=str,
        default=None,
        help="Path to a .npz carrying PO-LEU's test-set logits / c_star / n_params.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.synthetic:
        from src.baselines._synthetic import make_synthetic_batch

        train = make_synthetic_batch(n_events=400, seed=1001)
        val = make_synthetic_batch(n_events=200, seed=1002)
        test = make_synthetic_batch(n_events=200, seed=1003)
        logger.info("synthetic mode: %d train / %d val / %d test events",
                    train.n_events, val.n_events, test.n_events)
    else:
        if not args.dataset_yaml:
            parser.error(
                "--dataset-yaml is required unless --synthetic is passed"
            )
        logger.info("loading dataset %s", args.dataset_yaml)
        train_recs, val_recs, test_recs = _build_records_from_dataset(
            dataset_yaml=args.dataset_yaml,
            n_customers=int(args.n_customers),
            seed=int(args.seed),
            min_events_per_customer=int(args.min_events_per_customer),
        )
        logger.info(
            "records: %d train / %d val / %d test",
            len(train_recs), len(val_recs), len(test_recs),
        )
        if not train_recs or not test_recs:
            logger.error(
                "empty train or test split (train=%d test=%d); cannot run",
                len(train_recs), len(test_recs),
            )
            return 2

        train = records_to_baseline_batch(train_recs)
        val = (
            records_to_baseline_batch(val_recs)
            if val_recs
            else records_to_baseline_batch(train_recs[: max(1, len(train_recs) // 5)])
        )
        test = records_to_baseline_batch(test_recs)

    rows: List[dict[str, object]] = run_all_baselines(
        train, val, test, verbose=True
    )

    if args.poleu_logits:
        rows.append(_poleu_row(args.poleu_logits, test, train.n_events))

    print()
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
