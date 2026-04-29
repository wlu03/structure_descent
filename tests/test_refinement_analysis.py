"""Smoke test for scripts/refinement_analysis.py.

Builds two fake run dirs (baseline + refined) with synthetic
``test_per_event.json`` and a ``refined_outcomes.json`` bookkeeping file,
runs the analysis script's ``main()``, and verifies the output JSON has
the expected structure + that the refined-ASIN stratum's mean Δ NLL is
indeed negative when the synthetic data says it should be.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _write_per_event(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_event": rows, "n_events": len(rows)}))


def test_refinement_analysis_smoke(tmp_path: Path) -> None:
    """End-to-end on synthetic data — refined wins on touched ASINs only."""
    # 6 events. Two seeds × 3 events; refined ASIN is "asin_R" (event 0 of
    # each seed), the rest are "asin_X". Synthetic NLLs:
    #   event 0 (touched): baseline 4.0 → refined 1.0  (Δ = -3.0)
    #   event 1: baseline 0.5 → refined 0.5            (Δ = 0.0)
    #   event 2: baseline 0.7 → refined 0.7            (Δ = 0.0)
    def _rows(refined: bool) -> list[dict]:
        nll0 = 1.0 if refined else 4.0
        return [
            {"event_idx": 0, "asin_chosen": "asin_R", "nll": nll0,
             "p_chosen": 0.9 if refined else 0.02, "top1_correct": refined,
             "customer_id": "c1", "c_star": 0},
            {"event_idx": 1, "asin_chosen": "asin_X", "nll": 0.5,
             "p_chosen": 0.6, "top1_correct": True,
             "customer_id": "c2", "c_star": 0},
            {"event_idx": 2, "asin_chosen": "asin_X", "nll": 0.7,
             "p_chosen": 0.5, "top1_correct": True,
             "customer_id": "c3", "c_star": 0},
        ]

    for seed in (7, 11):
        b_dir = tmp_path / f"poleu_5cust_seed{seed}_no_residual"
        r_dir = tmp_path / f"poleu_5cust_seed{seed}_refined"
        _write_per_event(b_dir / "test_per_event.json", _rows(refined=False))
        _write_per_event(r_dir / "test_per_event.json", _rows(refined=True))
        # Bookkeeping lives in the BASELINE run dir per the shell script.
        bookkeeping = {
            "summary": {
                "n_revised": 1,
                "n_skipped_above_threshold": 0,
                "n_v1_cache_miss": 0,
                "n_revise_failures": 0,
            },
            "items": [
                {"event_idx": 0, "customer_id": "c1", "asin": "asin_R",
                 "alt_idx": 0, "skipped": False,
                 "v1_outcomes": ["a", "b", "c"],
                 "v2_outcomes": ["x", "y", "z"]},
            ],
        }
        (b_dir / "refined_outcomes.json").write_text(json.dumps(bookkeeping))

    out_dir = tmp_path / "analysis_out"

    from scripts.refinement_analysis import main as analysis_main
    saved = sys.argv
    sys.argv = [
        "refinement_analysis",
        "--baseline-glob", str(tmp_path / "poleu_*_no_residual"),
        "--refined-glob", str(tmp_path / "poleu_*_refined"),
        "--output-dir", str(out_dir),
        "--bootstrap-n", "200",  # fast for test
    ]
    try:
        rc = analysis_main()
    finally:
        sys.argv = saved
    assert rc == 0

    payload = json.loads((out_dir / "refinement_analysis.json").read_text())
    assert payload["seeds"] == [7, 11]
    assert payload["n_events_total"] == 6

    headline = payload["headline"]
    # Mean of [-3, 0, 0, -3, 0, 0] = -1.0
    assert headline["n"] == 6
    assert headline["mean_delta_nll"] == pytest.approx(-1.0)
    # Refined wins twice, loses zero, ties four times.
    assert headline["n_refined_wins"] == 2
    assert headline["n_refined_losses"] == 0

    refined_strat = payload["stratified"]["refined_asin"]
    unrefined_strat = payload["stratified"]["unrefined_asin"]
    # Touched ASIN events: Δ = -3 each, n = 2.
    assert refined_strat["n"] == 2
    assert refined_strat["mean_delta_nll"] == pytest.approx(-3.0)
    # Untouched ASIN events: Δ = 0 each, n = 4.
    assert unrefined_strat["n"] == 4
    assert unrefined_strat["mean_delta_nll"] == pytest.approx(0.0)

    # Cost-effectiveness: 2 seeds × (1 revised × 2 calls + 0 skipped) = 4 calls.
    cost = payload["cost_effectiveness"]
    assert cost is not None
    assert cost["total_llm_calls_refinement"] == 4

    # Markdown side-by-side exists too.
    md = (out_dir / "refinement_analysis.md").read_text()
    assert "Headline" in md
    assert "Stratified" in md
    assert "Difficulty buckets" in md
