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
from dotenv import load_dotenv

# Load .env before any LLM client is constructed so ANTHROPIC_API_KEY,
# OPENAI_API_KEY, GOOGLE_CLOUD_PROJECT, etc. are visible to os.environ.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.baselines.data_adapter import records_to_baseline_batch
from src.baselines.run_all import (
    BASELINE_REGISTRY,
    format_table,
    run_all_baselines,
    save_rows_to_data_dir,
)

logger = logging.getLogger("run_baselines")


def _build_records_from_dataset(
    dataset_yaml: str,
    n_customers: int,
    seed: int,
    min_events_per_customer: int,
    split_mode: str = "temporal",
    split_seed: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load + clean + split + subsample + build_choice_sets.

    Mirrors the Wave-10 driver (``scripts/run_dataset.py``) up to the
    point where per-event records are emitted, then splits those
    records into (train, val, test) by the ``split`` column.

    ``split_mode`` selects between the per-customer temporal split (the
    default; every customer appears in train+val+test) and the
    between-customer cold-start split (val/test customers have no
    training events). Under cold-start, ``split_seed`` seeds the
    customer-level partition; when ``None`` it falls back to ``seed``.
    """
    import pandas as pd

    from src.data.adapter import YamlAdapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import extract_extra_fields_from_row
    from src.data.split import cold_start_split, temporal_split
    from src.data.state_features import (
        attach_train_brand_map,
        attach_train_popularity,
        compute_state_features,
    )
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

    if split_mode == "cold_start":
        effective_split_seed = int(
            split_seed if split_seed is not None else seed
        )
        logger.info(
            "split_mode=cold_start (between-customer; split_seed=%d)",
            effective_split_seed,
        )
        events = cold_start_split(
            events, schema=adapter.schema, seed=effective_split_seed
        )
    else:
        logger.info("split_mode=temporal (per-customer within-customer)")
        events = temporal_split(events, adapter.schema)
    events = attach_train_popularity(events)
    events = attach_train_brand_map(events)

    train_events = events[events["split"] == "train"].copy()

    selected_ids, _weights = try_import_subsample_weights(
        train_events, n_customers=int(n_customers), seed=int(seed)
    )
    if selected_ids is not None:
        # Subsample is a TRAIN-customer selection (Appendix-C leverage
        # scores are computed on train events only). Under cold-start,
        # val/test customers never appear in ``selected_ids``; a blanket
        # ``.isin`` filter would therefore drop every val/test event.
        #
        # Generalized rule: keep an event iff its customer is in
        # ``selected_set`` OR its customer has no train rows at all.
        # Under temporal split every customer has train rows, so the
        # second clause is empty and the rule collapses to the pre-fix
        # ``.isin(selected_set)`` behavior exactly. Under cold-start,
        # val/test customers are precisely "customers with no train
        # rows", so all their events survive.
        selected_set = set(
            selected_ids.tolist() if hasattr(selected_ids, "tolist")
            else list(selected_ids)
        )
        train_customer_set = set(
            events.loc[events["split"] == "train", "customer_id"]
            .unique()
            .tolist()
        )
        val_test_only_customers = (
            set(events["customer_id"].unique().tolist()) - train_customer_set
        )
        keep_customers = selected_set | val_test_only_customers
        events = events[
            events["customer_id"].isin(keep_customers)
        ].reset_index(drop=True)

    # Orphan filter: keep only customers that appear in the survey
    # (mirrors run_dataset.py stage 8). ``joint_customers`` must span ALL
    # customers in ``events`` (not just train-split customers) — under
    # cold-start, train_customers is a STRICT SUBSET of all customers,
    # and restricting the filter to that subset would drop every val/test
    # customer's events. The downstream ``translate_z_d`` call already
    # fits on a train-only subset (see ``train_events_subset`` below),
    # so widening here does not introduce leakage.
    persons_id_col = adapter.schema.persons_id_column
    if persons_id_col not in persons_raw.columns:
        raise SystemExit(
            f"persons_raw missing id column {persons_id_col!r} "
            f"(schema: {dataset_yaml})."
        )
    surveyed_customers = set(
        persons_raw[persons_id_col].dropna().astype(str).unique().tolist()
    )
    all_customers = set(
        events["customer_id"].astype(str).unique().tolist()
    )
    joint_customers = all_customers & surveyed_customers
    if not joint_customers:
        raise SystemExit(
            "orphan filter: no customers in both events and survey "
            f"(events={len(all_customers)}, survey={len(surveyed_customers)})."
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
    parser.add_argument(
        "--split-mode",
        choices=["temporal", "cold_start"],
        default="temporal",
        help=(
            "Which split to use. 'temporal' = per-customer time split "
            "(default; every customer appears in train+val+test). "
            "'cold_start' = between-customer split (val/test customers "
            "have no train events — measures generalization to unseen users)."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for the cold-start partition. Defaults to --seed.",
    )
    parser.add_argument(
        "--allow-empty-val-fallback",
        action="store_true",
        help=(
            "Legacy paranoid-safe behavior: if the val split is empty, "
            "fill it with a slice of the train records instead of "
            "failing. Default: disabled (fail loud)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write the leaderboard artifacts (JSON + CSV "
            "+ TXT). When unset, results only print to stdout."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Optional suffix for the leaderboard filename stem, e.g. "
            "--tag=main_seed7 yields baselines_leaderboard_main_seed7.json. "
            "Ignored when --output-dir is unset."
        ),
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
            split_mode=str(args.split_mode),
            split_seed=(
                int(args.split_seed) if args.split_seed is not None else None
            ),
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
        if val_recs:
            val = records_to_baseline_batch(val_recs)
        elif args.allow_empty_val_fallback:
            logger.warning(
                "val records empty; --allow-empty-val-fallback is set, "
                "filling val with a slice of train (legacy behavior)."
            )
            val = records_to_baseline_batch(
                train_recs[: max(1, len(train_recs) // 5)]
            )
        else:
            logger.error(
                "val set empty — check dataset has enough per-customer "
                "events (split_mode=%s, train=%d, test=%d). Pass "
                "--allow-empty-val-fallback to restore the legacy "
                "fill-from-train behavior.",
                args.split_mode, len(train_recs), len(test_recs),
            )
            raise SystemExit(2)
        test = records_to_baseline_batch(test_recs)

    # Paper-grade evaluation sidecar (see
    # ``docs/paper_evaluation_additions.md`` §3): pin the canonical
    # test-event ordering on disk so downstream significance/segment
    # scripts can cross-reference by event index across baselines.
    # Written BEFORE the baseline fits so a mid-run crash still produces
    # the sidecar for the rows that did complete.
    if args.output_dir:
        _write_events_sidecar(
            test_batch=test,
            output_dir=args.output_dir,
            tag=args.tag,
            seed=int(args.seed),
            split_mode=str(args.split_mode),
            n_customers=int(args.n_customers),
        )

    rows: List[dict[str, object]] = run_all_baselines(
        train, val, test, verbose=True
    )

    if args.poleu_logits:
        rows.append(_poleu_row(args.poleu_logits, test, train.n_events))

    print()
    print(format_table(rows))

    if args.output_dir:
        paths = save_rows_to_data_dir(rows, args.output_dir, tag=args.tag)
        print()
        print("wrote leaderboard to:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")

    return 0


def _write_events_sidecar(
    *,
    test_batch,
    output_dir: str,
    tag: str | None,
    seed: int,
    split_mode: str,
    n_customers: int,
) -> str:
    """Write ``events_<tag>.json`` pinning the test-event ordering.

    The filename stem replaces the ``baselines_leaderboard`` prefix
    with ``events``; e.g. ``--tag=main_seed7`` yields
    ``events_main_seed7.json`` (matches the contract in
    ``docs/paper_evaluation_additions.md`` §3). When ``tag`` is unset,
    the file is written as ``events.json``.

    Event schema per design doc:

        {
            "event_idx": int,
            "customer_id": str,
            "category": str,
            "chosen_idx": int,
            "n_alternatives": int,
            "is_repeat": bool,
            "order_date": str,
        }
    """
    import json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "events"
    if tag:
        stem = f"{stem}_{tag}"
    sidecar_path = out_dir / f"{stem}.json"

    n_alt = int(test_batch.n_alternatives)
    events: list[dict] = []
    for i in range(test_batch.n_events):
        meta = test_batch.metadata[i] if i < len(test_batch.metadata) else {}
        order_date_raw = meta.get("order_date", "")
        # Normalise non-string order_date values (pd.Timestamp, datetime)
        # into ISO strings so the JSON is stable across run environments.
        order_date = "" if order_date_raw is None else str(order_date_raw)
        events.append(
            {
                "event_idx": int(i),
                "customer_id": str(test_batch.customer_ids[i]),
                "category": str(test_batch.categories[i]),
                "chosen_idx": int(test_batch.chosen_indices[i]),
                "n_alternatives": n_alt,
                "is_repeat": bool(meta.get("is_repeat", False)),
                "order_date": order_date,
            }
        )

    # n_customers: honour the CLI arg (matches design doc field) but
    # also surface the *observed* customer count in the sidecar via the
    # events list — downstream scripts can recompute if needed.
    payload = {
        "split_mode": str(split_mode),
        "seed": int(seed),
        "n_customers": int(n_customers),
        "n_events": int(test_batch.n_events),
        "events": events,
    }
    sidecar_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    logger.info("wrote events sidecar: %s", sidecar_path.resolve())
    return str(sidecar_path.resolve())


if __name__ == "__main__":
    sys.exit(main())
