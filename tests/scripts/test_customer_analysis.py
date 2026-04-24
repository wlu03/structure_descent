"""Tests for ``scripts.customer_analysis``.

All fixtures are synthetic — they only need to match the JSON schema
documented in ``docs/paper_evaluation_additions.md`` §Target artifact schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from scripts import customer_analysis as ca


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _events_sidecar(seed: int, events: List[Dict]) -> Dict:
    return {
        "split_mode": "temporal",
        "seed": seed,
        "n_customers": len({e["customer_id"] for e in events}),
        "n_events": len(events),
        "events": [dict(e, event_idx=i) for i, e in enumerate(events)],
    }


def _baseline_row(name: str, per_customer: Dict[str, Dict], status: str = "ok") -> Dict:
    if status != "ok":
        return {"name": name, "status": status, "test_nll": None, "top1": None}
    n_events = sum(v["n_events"] for v in per_customer.values())
    nll = (
        sum(v["nll"] * v["n_events"] for v in per_customer.values()) / n_events
        if n_events
        else 0.0
    )
    return {
        "name": name,
        "status": "ok",
        "test_nll": nll,
        "top1": 0.5,
        "n_events": n_events,
        "per_customer_nll": per_customer,
    }


def _write_pair(
    tmp: Path, seed: int, events: List[Dict], rows: List[Dict]
) -> None:
    (tmp / f"baselines_leaderboard_main_seed{seed}.json").write_text(json.dumps(rows))
    (tmp / f"events_main_seed{seed}.json").write_text(
        json.dumps(_events_sidecar(seed, events))
    )


def _evt(cid: str, cat: str = "BOOK", is_repeat: bool = False) -> Dict:
    return {
        "customer_id": cid,
        "category": cat,
        "chosen_idx": 0,
        "n_alternatives": 10,
        "is_repeat": is_repeat,
        "order_date": "2024-01-01",
    }


# ---------------------------------------------------------------------------
# Bucket unit tests
# ---------------------------------------------------------------------------


def test_trajectory_length_buckets_assigned_correctly(tmp_path: Path) -> None:
    # C0: 1 event (short), C1: 5 events (medium), C2: 10 events (long)
    events = (
        [_evt("C0")]
        + [_evt("C1") for _ in range(5)]
        + [_evt("C2") for _ in range(10)]
    )
    per_cust = {
        "C0": {"nll": 0.5, "n_events": 1, "top1": 1.0},
        "C1": {"nll": 0.6, "n_events": 5, "top1": 0.6},
        "C2": {"nll": 0.7, "n_events": 10, "top1": 0.7},
    }
    _write_pair(tmp_path, 7, events, [_baseline_row("PO-LEU", per_cust)])
    seeds = ca.load_seed_pair(tmp_path, "main_seed*")
    segs = ca.build_customer_segments(seeds)
    assert segs["C0"]["trajectory_length"] == "short"
    assert segs["C1"]["trajectory_length"] == "medium"
    assert segs["C2"]["trajectory_length"] == "long"


def test_category_concentration_buckets(tmp_path: Path) -> None:
    # C0 -> 1 cat (focused), C1 -> 2 cats (moderate), C2 -> 4 cats (diverse)
    events = [
        _evt("C0", "BOOK"),
        _evt("C1", "BOOK"),
        _evt("C1", "ELEC"),
        _evt("C2", "BOOK"),
        _evt("C2", "ELEC"),
        _evt("C2", "TOY"),
        _evt("C2", "FOOD"),
    ]
    per_cust = {
        "C0": {"nll": 0.5, "n_events": 1, "top1": 1.0},
        "C1": {"nll": 0.6, "n_events": 2, "top1": 0.5},
        "C2": {"nll": 0.7, "n_events": 4, "top1": 0.25},
    }
    _write_pair(tmp_path, 7, events, [_baseline_row("PO-LEU", per_cust)])
    seeds = ca.load_seed_pair(tmp_path, "main_seed*")
    segs = ca.build_customer_segments(seeds)
    assert segs["C0"]["category_concentration"] == "focused"
    assert segs["C1"]["category_concentration"] == "moderate"
    assert segs["C2"]["category_concentration"] == "diverse"


def test_novelty_rate_buckets(tmp_path: Path) -> None:
    # 10%: 1 novel of 10 -> mostly_repeat. 50%: 5/10 -> mixed. 90%: 9/10 -> mostly_novel.
    def events_for(cid: str, novel_count: int, total: int) -> List[Dict]:
        return [
            _evt(cid, is_repeat=(i >= novel_count)) for i in range(total)
        ]
    events = events_for("C10", 1, 10) + events_for("C50", 5, 10) + events_for("C90", 9, 10)
    per_cust = {
        "C10": {"nll": 0.5, "n_events": 10, "top1": 0.5},
        "C50": {"nll": 0.5, "n_events": 10, "top1": 0.5},
        "C90": {"nll": 0.5, "n_events": 10, "top1": 0.5},
    }
    _write_pair(tmp_path, 7, events, [_baseline_row("PO-LEU", per_cust)])
    seeds = ca.load_seed_pair(tmp_path, "main_seed*")
    segs = ca.build_customer_segments(seeds)
    assert segs["C10"]["novelty_rate"] == "mostly_repeat"
    assert segs["C50"]["novelty_rate"] == "mixed"
    assert segs["C90"]["novelty_rate"] == "mostly_novel"


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


def test_baseline_not_ok_dropped(tmp_path: Path) -> None:
    events = [_evt("C0"), _evt("C1"), _evt("C1")]
    ok_row = _baseline_row(
        "PO-LEU",
        {
            "C0": {"nll": 0.5, "n_events": 1, "top1": 1.0},
            "C1": {"nll": 0.7, "n_events": 2, "top1": 0.5},
        },
    )
    bad_row = _baseline_row("BROKEN", per_customer={}, status="errored")
    bad_row["status"] = "errored"
    _write_pair(tmp_path, 7, events, [ok_row, bad_row])
    seeds = ca.load_seed_pair(tmp_path, "main_seed*")
    segments = ca.build_customer_segments(seeds)
    metrics = ca.aggregate_customer_metrics(seeds, set(segments.keys()))
    assert "PO-LEU" in metrics
    assert "BROKEN" not in metrics
    assert metrics["PO-LEU"]["C0"]["nll"] == pytest.approx(0.5)


def test_missing_events_sidecar_fails_loud(tmp_path: Path) -> None:
    rows = [
        _baseline_row(
            "PO-LEU", {"C0": {"nll": 0.5, "n_events": 1, "top1": 1.0}}
        )
    ]
    (tmp_path / "baselines_leaderboard_main_seed7.json").write_text(json.dumps(rows))
    # deliberately do NOT write events_main_seed7.json
    with pytest.raises(FileNotFoundError, match="Data inconsistency"):
        ca.load_seed_pair(tmp_path, "main_seed*")


def test_customer_mean_over_seeds_correct(tmp_path: Path) -> None:
    # Customer C0 has per-seed NLLs [0.5, 0.7, 0.9] -> mean 0.7
    for seed, nll in zip((1, 2, 3), (0.5, 0.7, 0.9)):
        events = [_evt("C0")]
        per_cust = {"C0": {"nll": nll, "n_events": 1, "top1": 0.0}}
        _write_pair(tmp_path, seed, events, [_baseline_row("PO-LEU", per_cust)])
    seeds = ca.load_seed_pair(tmp_path, "main_seed*")
    segments = ca.build_customer_segments(seeds)
    metrics = ca.aggregate_customer_metrics(seeds, set(segments.keys()))
    assert metrics["PO-LEU"]["C0"]["nll"] == pytest.approx(0.7)
    assert metrics["PO-LEU"]["C0"]["n_seeds"] == 3


def test_small_segment_flag(tmp_path: Path) -> None:
    # One customer total -> short segment with n=1 -> expect "⚠ small segment" flag
    events = [_evt("C0")]
    per_cust = {"C0": {"nll": 0.5, "n_events": 1, "top1": 1.0}}
    _write_pair(tmp_path, 7, events, [_baseline_row("PO-LEU", per_cust)])
    out_dir = tmp_path / "out"
    ca.run(
        input_dir=tmp_path,
        tag_pattern="main_seed*",
        output_dir=out_dir,
        baseline_of_interest="PO-LEU",
        min_segment_size=3,
    )
    md = (out_dir / "by_trajectory_length.md").read_text()
    assert "⚠ small segment" in md


def test_summary_identifies_largest_effect_size(tmp_path: Path) -> None:
    # Craft data so PO-LEU's gap over others is maximised in the `long` traj segment.
    # short (n=3 customers, 1 event each): PO-LEU ~ 0.60, LASSO-MNL ~ 0.62 -> gap 0.02
    # long (n=3 customers, 10 events each): PO-LEU ~ 0.30, LASSO-MNL ~ 0.80 -> gap 0.50
    events: List[Dict] = []
    po_per_cust: Dict[str, Dict] = {}
    lasso_per_cust: Dict[str, Dict] = {}
    for i in range(3):
        cid = f"S{i}"
        events.append(_evt(cid))
        po_per_cust[cid] = {"nll": 0.60, "n_events": 1, "top1": 0.5}
        lasso_per_cust[cid] = {"nll": 0.62, "n_events": 1, "top1": 0.5}
    for i in range(3):
        cid = f"L{i}"
        for _ in range(10):
            events.append(_evt(cid))
        po_per_cust[cid] = {"nll": 0.30, "n_events": 10, "top1": 0.7}
        lasso_per_cust[cid] = {"nll": 0.80, "n_events": 10, "top1": 0.3}
    rows = [
        _baseline_row("PO-LEU", po_per_cust),
        _baseline_row("LASSO-MNL", lasso_per_cust),
    ]
    _write_pair(tmp_path, 7, events, rows)
    out_dir = tmp_path / "out"
    ca.run(
        input_dir=tmp_path,
        tag_pattern="main_seed*",
        output_dir=out_dir,
        baseline_of_interest="PO-LEU",
        min_segment_size=3,
    )
    summary = (out_dir / "summary.md").read_text()
    # Largest win should be in the `long` trajectory segment
    assert "long" in summary
    # The top ranked line should reference the long segment
    top5_block = summary.split("Top 5")[-1]
    first_long = top5_block.find("long")
    first_short = top5_block.find("short")
    # long must appear before short (or short may not appear at all) in ranking
    assert first_long != -1
    assert first_short == -1 or first_long < first_short
