"""Tests for the ``events_<tag>.json`` sidecar emitted by
``scripts/run_baselines.py`` (paper-grade evaluation addition 3).

We don't run the full CLI (it imports optional LLM clients). Instead we
call the sidecar helper directly with a hand-built
:class:`BaselineEventBatch` and verify:

* Filename convention: ``--tag=main_seed7`` → ``events_main_seed7.json``.
* Required top-level fields per
  ``docs/paper_evaluation_additions.md`` §3.
* Each event record carries the documented keys with correct types.
* Canonical ordering: ``event_idx`` matches list index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.baselines.base import BaselineEventBatch


def _tiny_batch() -> BaselineEventBatch:
    feats = [np.zeros((4, 6)) for _ in range(5)]
    return BaselineEventBatch(
        base_features_list=feats,
        base_feature_names=[f"f{i}" for i in range(6)],
        chosen_indices=[0, 1, 2, 3, 0],
        customer_ids=["CUST_A", "CUST_B", "CUST_A", "CUST_C", "CUST_B"],
        categories=["BOOK", "ELECTRONICS", "BOOK", "FOOD", "BOOK"],
        metadata=[
            {"is_repeat": False, "order_date": "2024-03-12"},
            {"is_repeat": True, "order_date": "2024-03-14"},
            {"is_repeat": False, "order_date": "2024-04-01"},
            {"is_repeat": False, "order_date": "2024-04-05"},
            {"is_repeat": True, "order_date": "2024-04-07"},
        ],
    )


def test_sidecar_filename_and_schema(tmp_path: Path):
    from scripts.run_baselines import _write_events_sidecar

    batch = _tiny_batch()
    path_str = _write_events_sidecar(
        test_batch=batch,
        output_dir=str(tmp_path),
        tag="main_seed7",
        seed=7,
        split_mode="temporal",
        n_customers=50,
    )
    out = Path(path_str)
    # Filename convention pinned in the doc.
    assert out.name == "events_main_seed7.json"
    assert out.parent == tmp_path.resolve() or out.parent == tmp_path

    payload = json.loads(out.read_text(encoding="utf-8"))
    # Top-level schema.
    assert payload["split_mode"] == "temporal"
    assert payload["seed"] == 7
    assert payload["n_customers"] == 50
    assert payload["n_events"] == batch.n_events == 5
    events = payload["events"]
    assert isinstance(events, list)
    assert len(events) == 5

    # Per-event schema.
    required = {
        "event_idx",
        "customer_id",
        "category",
        "chosen_idx",
        "n_alternatives",
        "is_repeat",
        "order_date",
    }
    for i, ev in enumerate(events):
        assert set(ev.keys()) == required
        assert ev["event_idx"] == i
        assert ev["n_alternatives"] == 4
        assert isinstance(ev["customer_id"], str)
        assert isinstance(ev["category"], str)
        assert isinstance(ev["chosen_idx"], int)
        assert isinstance(ev["is_repeat"], bool)
        assert isinstance(ev["order_date"], str)

    # Canonical ordering: same customer / category / chosen_idx / repeat
    # as the source batch at every index.
    for i in range(5):
        assert events[i]["customer_id"] == batch.customer_ids[i]
        assert events[i]["category"] == batch.categories[i]
        assert events[i]["chosen_idx"] == batch.chosen_indices[i]
        assert events[i]["is_repeat"] == bool(
            batch.metadata[i].get("is_repeat", False)
        )
        assert events[i]["order_date"] == str(
            batch.metadata[i].get("order_date", "")
        )


def test_sidecar_without_tag_uses_default_name(tmp_path: Path):
    """An unset tag writes to ``events.json``."""
    from scripts.run_baselines import _write_events_sidecar

    batch = _tiny_batch()
    path_str = _write_events_sidecar(
        test_batch=batch,
        output_dir=str(tmp_path),
        tag=None,
        seed=3,
        split_mode="cold_start",
        n_customers=10,
    )
    assert Path(path_str).name == "events.json"
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    assert payload["split_mode"] == "cold_start"
    assert payload["seed"] == 3
