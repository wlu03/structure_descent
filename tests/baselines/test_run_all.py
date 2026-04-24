"""Registry end-to-end tests for :func:`run_all_baselines`.

Focus: the paper-grade row schema (``docs/paper_evaluation_additions.md``)
threads cleanly through every baseline in the registry, including the
tabular rows that have no extra artifacts.

We run a tiny subset of baselines (not the full registry — LLM baselines
need real clients) and verify:

* Every ``status == "ok"`` row carries ``per_event_nll``,
  ``per_event_topk_correct``, ``per_customer_nll``, ``extra_artifacts``.
* Lengths match ``n_events``; mean(per_event_nll) ≈ ``test_nll``.
* Tabular baselines (no ``extra_artifacts_for_json`` hook) report
  ``extra_artifacts is None``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.baselines._synthetic import make_synthetic_batch
from src.baselines.base import BaselineEventBatch, BaselineReport
from src.baselines.evaluate import evaluate_baseline
from src.baselines.run_all import (
    annotate_uplift_vs_popularity,
    save_rows_to_data_dir,
)


class _DummyFitted:
    """All-zero fitted baseline; no extra_artifacts_for_json hook."""

    name = "Dummy"

    def __init__(self, n_alts: int):
        self._n_alts = n_alts

    def score_events(self, batch: BaselineEventBatch):
        return [np.zeros(self._n_alts) for _ in range(batch.n_events)]

    @property
    def n_params(self) -> int:
        return 0

    @property
    def description(self) -> str:
        return "dummy"


class _DummyWithArtifacts:
    """Fitted baseline that implements ``extra_artifacts_for_json``."""

    name = "DummyX"

    def __init__(self, n_alts: int):
        self._n_alts = n_alts

    def score_events(self, batch: BaselineEventBatch):
        return [np.zeros(self._n_alts) for _ in range(batch.n_events)]

    @property
    def n_params(self) -> int:
        return 0

    @property
    def description(self) -> str:
        return "dummy+artifacts"

    def extra_artifacts_for_json(self):
        return {"note": "present"}


def _build_row_from(fitted, batch: BaselineEventBatch) -> dict:
    """Assemble a row the same way :func:`run_all_baselines` does.

    We mirror the row-assembly code path (including the duck-typed
    ``extra_artifacts`` hook) so the test pins the exact schema that
    gets serialised to disk.
    """
    report = evaluate_baseline(fitted, batch, train_n_events=batch.n_events)
    extra_fn = getattr(fitted, "extra_artifacts_for_json", None)
    extra_artifacts = extra_fn() if callable(extra_fn) else None
    m = report.metrics
    return {
        "name": fitted.name,
        "status": "ok",
        "top1": m["top1"],
        "top5": m["top5"],
        "mrr": m["mrr"],
        "test_nll": m["test_nll"],
        "aic": m["aic"],
        "bic": m["bic"],
        "n_params": report.n_params,
        "fit_seconds": 0.0,
        "description": fitted.description,
        "error": None,
        "per_event_nll": list(report.per_event_nll),
        "per_event_topk_correct": list(report.per_event_topk_correct),
        "per_customer_nll": dict(report.per_customer_nll),
        "extra_artifacts": extra_artifacts,
    }


def test_tabular_row_has_per_event_and_null_extras():
    """A fitted with no ``extra_artifacts_for_json`` → row.extra_artifacts is None."""
    batch = make_synthetic_batch(n_events=30, n_alts=5, seed=11)
    row = _build_row_from(_DummyFitted(n_alts=5), batch)

    assert row["status"] == "ok"
    assert row["extra_artifacts"] is None
    assert len(row["per_event_nll"]) == batch.n_events
    assert len(row["per_event_topk_correct"]) == batch.n_events
    # Invariant: mean of per-event NLL == aggregate test_nll.
    assert abs(float(np.mean(row["per_event_nll"])) - float(row["test_nll"])) < 1e-6


def test_row_with_hook_carries_extra_artifacts():
    """A fitted implementing the hook → row.extra_artifacts is the returned dict."""
    batch = make_synthetic_batch(n_events=12, n_alts=4, seed=22)
    row = _build_row_from(_DummyWithArtifacts(n_alts=4), batch)
    assert row["extra_artifacts"] == {"note": "present"}


def test_per_customer_keys_subset_of_batch_customer_ids_in_row():
    batch = make_synthetic_batch(
        n_events=40, n_alts=4, n_customers=5, seed=9
    )
    row = _build_row_from(_DummyFitted(n_alts=4), batch)
    batch_cids = set(str(c) for c in batch.customer_ids)
    assert set(row["per_customer_nll"].keys()).issubset(batch_cids)


# ---------------------------------------------------------------------------
# Serialization round-trip: save_rows_to_data_dir -> JSON -> parse
# ---------------------------------------------------------------------------


def test_save_rows_json_roundtrip_preserves_per_event_and_extras(tmp_path):
    """Persist the row and verify JSON parse preserves all paper-grade fields."""
    import json

    batch = make_synthetic_batch(n_events=15, n_alts=4, seed=2)
    rows = [
        _build_row_from(_DummyFitted(n_alts=4), batch),
        _build_row_from(_DummyWithArtifacts(n_alts=4), batch),
    ]
    paths = save_rows_to_data_dir(rows, str(tmp_path), tag="test")
    loaded = json.loads(open(paths["json"], "r", encoding="utf-8").read())

    assert len(loaded) == 2
    by_name = {r["name"]: r for r in loaded}
    assert by_name["Dummy"]["extra_artifacts"] is None
    assert by_name["DummyX"]["extra_artifacts"] == {"note": "present"}
    assert len(by_name["Dummy"]["per_event_nll"]) == batch.n_events
    assert len(by_name["Dummy"]["per_event_topk_correct"]) == batch.n_events
    assert isinstance(by_name["Dummy"]["per_customer_nll"], dict)


def test_save_rows_csv_flattens_list_fields(tmp_path):
    """CSV writer must JSON-encode list-valued cells without crashing."""
    import csv

    batch = make_synthetic_batch(n_events=8, n_alts=4, seed=5)
    rows = [_build_row_from(_DummyFitted(n_alts=4), batch)]
    paths = save_rows_to_data_dir(rows, str(tmp_path), tag="csvcheck")

    with open(paths["csv"], "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = list(reader)
    assert len(out) == 1
    # The per-event columns are JSON strings in the CSV; json-decode roundtrip.
    import json
    decoded = json.loads(out[0]["per_event_nll"])
    assert isinstance(decoded, list)
    assert len(decoded) == batch.n_events


# ---------------------------------------------------------------------------
# Smoke: popularity end-to-end via run_all_baselines (minimal registry)
# ---------------------------------------------------------------------------


def test_run_all_baselines_popularity_row_has_new_fields():
    """Run the real orchestrator on popularity only; verify schema.

    We monkey-patch ``BASELINE_REGISTRY`` to contain just Popularity so
    the test is fast and avoids touching LLM clients or optional deps.
    """
    import src.baselines.run_all as run_all_mod

    original = run_all_mod.BASELINE_REGISTRY
    try:
        run_all_mod.BASELINE_REGISTRY = [
            ("Popularity", "src.baselines.popularity", "PopularityBaseline"),
        ]
        train = make_synthetic_batch(n_events=120, n_alts=4, seed=77)
        val = make_synthetic_batch(n_events=30, n_alts=4, seed=78)
        test = make_synthetic_batch(n_events=60, n_alts=4, seed=79)
        rows = run_all_mod.run_all_baselines(train, val, test, verbose=False)
    finally:
        run_all_mod.BASELINE_REGISTRY = original

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["name"] == "Popularity"
    assert "per_event_nll" in row
    assert "per_event_topk_correct" in row
    assert "per_customer_nll" in row
    assert "extra_artifacts" in row
    assert row["extra_artifacts"] is None  # no hook on PopularityFitted
    assert len(row["per_event_nll"]) == test.n_events
    assert len(row["per_event_topk_correct"]) == test.n_events
    assert abs(
        float(np.mean(row["per_event_nll"])) - float(row["test_nll"])
    ) < 1e-6
