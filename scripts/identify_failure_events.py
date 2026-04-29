"""Pick the worst PO-LEU validation events for curriculum refinement.

Reads ``val_per_event.json`` (written by :mod:`scripts.run_dataset`) and
``records.pkl`` (the round-1 PO-LEU run's pickled splits), sorts events by
NLL descending, and writes ``failure_events.json`` containing the top-K
worst events with everything :mod:`scripts.refine_outcomes` needs to
critique-and-revise their outcomes.

We **must** read the val sidecar, not test_per_event.json — picking
refinement targets from test events is leakage.

Output schema (``failure_events.json``):

    {
      "selected_from": "val",
      "policy": "top_k_by_nll",
      "k_selected": 20,
      "events": [
        {
          "event_idx": 17,
          "customer_id": "...",
          "asin_chosen": "...",
          "c_star": 4,
          "p_chosen": 0.013,
          "nll": 4.34,
          "c_d": "...",
          "choice_asins": ["...", ..., "..."],
          "alt_texts":   [{"title": ..., "category": ..., ...}, ...]
        },
        ...
      ]
    }

Usage::

    python -m scripts.identify_failure_events \\
        --run-dir results_data/poleu_50cust_seed7_no_residual \\
        --top-k 20 \\
        --output results_data/poleu_50cust_seed7_no_residual/failure_events.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

logger = logging.getLogger("identify_failure_events")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="identify_failure_events.py",
        description=__doc__,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="PO-LEU output directory containing val_per_event.json + records.pkl.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of worst events to select. Default 20.",
    )
    p.add_argument(
        "--threshold-p-chosen",
        type=float,
        default=None,
        help=(
            "Alternative selection: pick events where p(chosen) < THRESHOLD "
            "instead of top-K. Mutually exclusive with --top-k semantics; "
            "if set, overrides top-K."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <run-dir>/failure_events.json.",
    )
    p.add_argument(
        "--per-event",
        type=Path,
        default=None,
        help=(
            "Override path to the per-event sidecar. Defaults to "
            "<run-dir>/val_per_event.json — never use test_per_event.json "
            "(test-set leakage)."
        ),
    )
    p.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Override path to records.pkl. Defaults to <run-dir>/records.pkl.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        logger.error("--run-dir does not exist or is not a directory: %s", run_dir)
        return 2

    per_event_path: Path = args.per_event or (run_dir / "val_per_event.json")
    records_path: Path = args.records or (run_dir / "records.pkl")
    output_path: Path = args.output or (run_dir / "failure_events.json")

    if not per_event_path.exists():
        logger.error("per-event sidecar not found: %s", per_event_path)
        return 2
    if not records_path.exists():
        logger.error("records.pkl not found: %s", records_path)
        return 2

    payload = json.loads(per_event_path.read_text())
    rows = payload.get("per_event", [])
    if not rows:
        logger.warning("per-event sidecar empty; nothing to refine.")
        output_path.write_text(json.dumps({
            "selected_from": "val", "policy": "empty",
            "k_selected": 0, "events": [],
        }, indent=2))
        return 0

    with records_path.open("rb") as fh:
        bundle = pickle.load(fh)
    # The val per-event sidecar is built from records_val (ordered by event_idx).
    val_records = bundle.get("val") or []
    train_records = bundle.get("train") or []
    # Fallback: PO-LEU's run_dataset reuses train batch when val is empty.
    records_for_lookup = val_records if val_records else train_records

    if args.threshold_p_chosen is not None:
        thr = float(args.threshold_p_chosen)
        candidate = [r for r in rows if r.get("p_chosen", 1.0) < thr]
        candidate.sort(key=lambda r: r.get("nll", 0.0), reverse=True)
        policy = f"p_chosen<{thr}"
        selected = candidate
    else:
        sorted_rows = sorted(rows, key=lambda r: r.get("nll", 0.0), reverse=True)
        selected = sorted_rows[: max(0, int(args.top_k))]
        policy = f"top_{int(args.top_k)}_by_nll"

    events: list[dict] = []
    skipped = 0
    for r in selected:
        idx = int(r["event_idx"])
        if idx < 0 or idx >= len(records_for_lookup):
            skipped += 1
            continue
        rec = records_for_lookup[idx]
        events.append({
            "event_idx": idx,
            "customer_id": str(rec.get("customer_id", r.get("customer_id", ""))),
            "asin_chosen": str(rec.get("chosen_asin", r.get("asin_chosen", ""))),
            "c_star": int(r.get("c_star", rec.get("chosen_idx", 0))),
            "p_chosen": float(r.get("p_chosen", 0.0)),
            "nll": float(r.get("nll", 0.0)),
            "c_d": str(rec.get("c_d", "")),
            "choice_asins": [str(a) for a in rec.get("choice_asins", [])],
            "alt_texts": list(rec.get("alt_texts", [])),
        })

    if skipped:
        logger.warning(
            "%d selected events had event_idx out of range for records bundle; "
            "skipped.",
            skipped,
        )

    out = {
        "selected_from": "val",
        "policy": policy,
        "k_selected": len(events),
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info(
        "wrote %d failure events to %s (policy=%s)",
        len(events), output_path, policy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
