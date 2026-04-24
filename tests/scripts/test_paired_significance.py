"""Tests for scripts/paired_significance.py.

Fixtures are hand-built in-test to match the schema in
``docs/paper_evaluation_additions.md`` so we don't depend on the producer
agent's output.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import paired_significance as ps


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_row(
    name: str,
    per_event_nll: list[float],
    per_event_topk_correct: list[bool],
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mean_nll = float(np.mean(per_event_nll)) if per_event_nll else 0.0
    top1 = float(np.mean(per_event_topk_correct)) if per_event_topk_correct else 0.0
    row: dict[str, Any] = {
        "name": name,
        "status": status,
        "top1": top1,
        "test_nll": mean_nll,
        "aic": None,
        "bic": None,
        "n_events": len(per_event_nll),
        "ece": None,
        "pseudo_r2": None,
        "nll_uplift_vs_popularity": None,
        "fit_time_seconds": 0.0,
        "description": f"synthetic {name}",
        "traceback": None,
        "per_event_nll": list(per_event_nll),
        "per_event_topk_correct": list(per_event_topk_correct),
        "per_customer_nll": {},
        "extra_artifacts": None,
    }
    if extra:
        row.update(extra)
    return row


def _write_seed(
    path: Path,
    seed: int,
    rows: list[dict[str, Any]],
) -> Path:
    fp = path / f"baselines_leaderboard_main_seed{seed}.json"
    fp.write_text(json.dumps(rows))
    return fp


# ---------------------------------------------------------------------------
# 1. Bootstrap CI coverage on a known difference
# ---------------------------------------------------------------------------


def test_bootstrap_ci_coverage_on_known_difference() -> None:
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(1.0, 0.5, size=n)
    nll_a = base  # A is lower
    nll_b = base + 0.1  # B is consistently 0.1 higher
    out = ps.paired_bootstrap(nll_a, nll_b, iterations=1000,
                              confidence_level=0.95, rng=np.random.default_rng(1))
    assert out["delta_ci_lo"] > 0.0, out  # excludes zero
    assert out["delta_ci_hi"] > 0.0, out
    assert abs(out["delta_mean"] - 0.1) < 0.02, out
    assert out["p_delta_gt_zero"] > 0.99


# ---------------------------------------------------------------------------
# 2. Bootstrap CI contains zero under no signal
# ---------------------------------------------------------------------------


def test_bootstrap_ci_width_under_no_signal() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(1.0, 0.5, size=300)
    out = ps.paired_bootstrap(base.copy(), base.copy(), iterations=1000,
                              confidence_level=0.95, rng=np.random.default_rng(2))
    # Identical arrays -> zero delta under every resample.
    assert out["delta_mean"] == 0.0
    assert out["delta_ci_lo"] == 0.0
    assert out["delta_ci_hi"] == 0.0
    assert out["p_delta_gt_zero"] == 0.0  # strictly > 0, not >=


def test_bootstrap_ci_contains_zero_under_noise() -> None:
    rng = np.random.default_rng(7)
    nll_a = rng.normal(1.0, 0.5, size=200)
    nll_b = rng.normal(1.0, 0.5, size=200)  # independent draws, same mean
    out = ps.paired_bootstrap(nll_a, nll_b, iterations=1000,
                              confidence_level=0.95, rng=np.random.default_rng(3))
    assert out["delta_ci_lo"] < 0.0 < out["delta_ci_hi"], out


# ---------------------------------------------------------------------------
# 3. Wilcoxon detects consistent shift
# ---------------------------------------------------------------------------


def test_wilcoxon_detects_consistent_shift() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(1.0, 0.3, size=80)
    nll_a = base
    nll_b = base + 0.2
    out = ps.wilcoxon_paired(nll_a, nll_b)
    assert out["wilcoxon_p_value"] is not None
    assert out["wilcoxon_p_value"] < 0.01, out


def test_wilcoxon_all_zero_skipped() -> None:
    v = np.array([1.0, 2.0, 3.0])
    out = ps.wilcoxon_paired(v, v.copy())
    assert out["wilcoxon_statistic"] is None
    assert out["wilcoxon_p_value"] is None
    assert "all deltas zero" in out["wilcoxon_note"]


# ---------------------------------------------------------------------------
# 4. McNemar graceful under no disagreement
# ---------------------------------------------------------------------------


def test_mcnemar_no_disagreement_graceful() -> None:
    top1 = np.array([True, False, True, True, False])
    out = ps.mcnemar_paired(top1, top1.copy())
    assert out["mcnemar_statistic"] is None
    assert out["mcnemar_p_value"] is None
    assert out["mcnemar_note"] == "no disagreement"


def test_mcnemar_detects_disagreement() -> None:
    # A correct on first 40, B correct on last 40, 20 both, 20 neither.
    a = np.array([True] * 40 + [False] * 40 + [True] * 20 + [False] * 20)
    b = np.array([False] * 40 + [True] * 40 + [True] * 20 + [False] * 20)
    out = ps.mcnemar_paired(a, b)
    assert out["mcnemar_p_value"] is not None
    # Symmetric disagreement (40 off-diagonal each side) -> large p-value.
    assert out["mcnemar_p_value"] > 0.5


# ---------------------------------------------------------------------------
# 5. ASO skipped when deepsig missing
# ---------------------------------------------------------------------------


def test_aso_skipped_when_deepsig_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    import importlib as _im

    def fake_import_module(name: str, *args: Any, **kwargs: Any):  # noqa: ANN401
        if name == "deepsig":
            raise ImportError("simulated missing deepsig")
        return _im_orig_import(name, *args, **kwargs)

    _im_orig_import = _im.import_module
    monkeypatch.setattr(ps.importlib, "import_module", fake_import_module)

    scores_a = np.array([1.0, 2.0, 3.0])
    scores_b = np.array([1.5, 2.5, 3.5])
    out = ps.aso_if_available(scores_a, scores_b, confidence_level=0.95)
    assert out["aso_epsilon_min"] is None
    assert "deepsig not installed" in out["aso_note"]


def test_aso_skip_note_reaches_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Full pipeline: schema-valid fixtures, deepsig import forced to fail,
    # markdown should say "ASO skipped".
    import importlib as _im
    real_fn = _im.import_module

    def fake_import_module(name: str, *args: Any, **kwargs: Any):  # noqa: ANN401
        if name == "deepsig":
            raise ImportError("simulated missing deepsig")
        return real_fn(name, *args, **kwargs)

    monkeypatch.setattr(ps.importlib, "import_module", fake_import_module)

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    rng = np.random.default_rng(0)
    for seed in (0, 1):
        base = rng.normal(1.0, 0.3, size=30)
        rows = [
            _make_row("PO-LEU", base.tolist(), [True, False] * 15),
            _make_row("Popularity", (base + 0.2).tolist(), [False, True] * 15),
        ]
        _write_seed(in_dir, seed, rows)

    ps.main([
        "--input-dir", str(in_dir),
        "--tag-pattern", "main_seed*",
        "--baseline-of-interest", "PO-LEU",
        "--output-dir", str(out_dir),
        "--bootstrap-iterations", "100",
        "--seed", "0",
    ])
    md = (out_dir / "pairwise_vs_PO-LEU.md").read_text()
    assert "ASO skipped: deepsig not installed" in md


# ---------------------------------------------------------------------------
# 6. Baseline-of-interest missing raises with clear list
# ---------------------------------------------------------------------------


def test_baseline_of_interest_missing_raises(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rows = [
        _make_row("Popularity", [0.5] * 10, [True] * 10),
        _make_row("Recency", [0.6] * 10, [False] * 10),
    ]
    _write_seed(in_dir, 0, rows)

    with pytest.raises(SystemExit) as excinfo:
        ps.main([
            "--input-dir", str(in_dir),
            "--tag-pattern", "main_seed*",
            "--baseline-of-interest", "PO-LEU",
            "--output-dir", str(tmp_path / "out"),
            "--bootstrap-iterations", "100",
        ])
    msg = str(excinfo.value)
    assert "PO-LEU" in msg
    assert "Popularity" in msg  # actual present baselines listed
    assert "Recency" in msg


# ---------------------------------------------------------------------------
# 7. Old-format schema error references the redesign doc
# ---------------------------------------------------------------------------


def test_schema_mismatch_hints_at_redesign_doc(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    # Old-format rows: no per_event_nll / per_event_topk_correct.
    old_rows = [
        {"name": "PO-LEU", "status": "ok", "top1": 0.5, "test_nll": 1.0},
        {"name": "Popularity", "status": "ok", "top1": 0.3, "test_nll": 1.2},
    ]
    fp = in_dir / "baselines_leaderboard_main_seed0.json"
    fp.write_text(json.dumps(old_rows))

    with pytest.raises(SystemExit) as excinfo:
        ps.main([
            "--input-dir", str(in_dir),
            "--tag-pattern", "main_seed*",
            "--baseline-of-interest", "PO-LEU",
            "--output-dir", str(tmp_path / "out"),
            "--bootstrap-iterations", "100",
        ])
    assert "docs/paper_evaluation_additions.md" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 8. No matching files raises
# ---------------------------------------------------------------------------


def test_no_matching_files_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        ps.main([
            "--input-dir", str(tmp_path),
            "--tag-pattern", "main_seed*",
            "--baseline-of-interest", "PO-LEU",
            "--output-dir", str(tmp_path / "out"),
        ])
    assert "No files matching" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 9. Seed length mismatch within seed raises
# ---------------------------------------------------------------------------


def test_within_seed_length_mismatch_raises(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rows = [
        _make_row("PO-LEU", [0.5] * 10, [True] * 10),
        _make_row("Popularity", [0.5] * 9, [True] * 9),  # length 9 != 10
    ]
    _write_seed(in_dir, 0, rows)
    with pytest.raises(SystemExit) as excinfo:
        ps.main([
            "--input-dir", str(in_dir),
            "--tag-pattern", "main_seed*",
            "--baseline-of-interest", "PO-LEU",
            "--output-dir", str(tmp_path / "out"),
            "--bootstrap-iterations", "100",
        ])
    assert "disagree on n_events" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 10. BOI with status != ok in one seed: that seed is dropped
# ---------------------------------------------------------------------------


def test_boi_bad_status_drops_seed(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    # seed 0: PO-LEU errored
    rows0 = [
        _make_row("PO-LEU", [], [], status="error"),
        _make_row("Popularity", [0.5] * 10, [True] * 10),
    ]
    # seed 1: both ok
    rows1 = [
        _make_row("PO-LEU", [0.4] * 10, [True] * 10),
        _make_row("Popularity", [0.6] * 10, [False] * 10),
    ]
    _write_seed(in_dir, 0, rows0)
    _write_seed(in_dir, 1, rows1)

    out_dir = tmp_path / "out"
    ps.main([
        "--input-dir", str(in_dir),
        "--tag-pattern", "main_seed*",
        "--baseline-of-interest", "PO-LEU",
        "--output-dir", str(out_dir),
        "--bootstrap-iterations", "100",
    ])
    payload = json.loads((out_dir / "pairwise_vs_PO-LEU.json").read_text())
    assert payload["n_seeds"] == 1  # only seed 1 kept
    assert payload["n_events"] == 10
    assert payload["seeds_used"] == [1]


# ---------------------------------------------------------------------------
# 11. Outputs have the documented shape
# ---------------------------------------------------------------------------


def test_outputs_have_documented_shape(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rng = np.random.default_rng(123)
    for seed in (0, 1):
        base = rng.normal(1.0, 0.4, size=40)
        rows = [
            _make_row("PO-LEU", base.tolist(),
                      (rng.random(40) < 0.6).tolist()),
            _make_row("Popularity", (base + 0.15).tolist(),
                      (rng.random(40) < 0.4).tolist()),
            _make_row("Recency", (base + 0.05).tolist(),
                      (rng.random(40) < 0.5).tolist()),
        ]
        _write_seed(in_dir, seed, rows)

    out_dir = tmp_path / "out"
    ps.main([
        "--input-dir", str(in_dir),
        "--tag-pattern", "main_seed*",
        "--baseline-of-interest", "PO-LEU",
        "--output-dir", str(out_dir),
        "--bootstrap-iterations", "500",
        "--seed", "0",
    ])

    payload = json.loads((out_dir / "pairwise_vs_PO-LEU.json").read_text())
    assert payload["baseline_of_interest"] == "PO-LEU"
    assert payload["n_events"] == 80
    assert payload["n_seeds"] == 2
    assert payload["bootstrap_iterations"] == 500
    assert "deepsig_available" in payload
    assert set(payload["pairs"].keys()) == {"Popularity", "Recency"}
    for name, stats in payload["pairs"].items():
        for k in ("delta_mean", "delta_ci_lo", "delta_ci_hi",
                  "p_delta_gt_zero", "wilcoxon_p_value", "mcnemar_p_value"):
            assert k in stats, (name, k)

    md = (out_dir / "pairwise_vs_PO-LEU.md").read_text()
    assert md.startswith("# Pairwise Significance vs PO-LEU")
    assert "n_events = 80" in md
    assert "n_seeds = 2" in md
    assert "bootstrap_iterations = 500" in md
    assert "| Popularity |" in md
    assert "| Recency |" in md
    # Sort-order assertion: Popularity (+0.15) should come before Recency (+0.05)
    assert md.index("| Popularity |") < md.index("| Recency |")
