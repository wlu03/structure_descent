"""Tests for src/eval/interpret.py (redesign.md §12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.eval.interpret import (
    DEFAULT_HEAD_NAMES,
    counterfactual_report,
    dominant_attribute_report,
    head_naming_report,
    per_decision_report,
    run_all_reports,
)
from src.model.po_leu import POLEU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model_and_batch(synthetic_batch):
    """Return (model, z_d, E, c_star, outcomes, logits, intermediates).

    Single-seeded ``POLEU``; the ``outcomes`` list is synthesized as
    ``f"outcome b{b} j{j} k{k}"`` strings so scoring-related assertions
    can still check string identity.
    """
    torch.manual_seed(0)
    model = POLEU()
    model.eval()

    z_d = synthetic_batch.z_d
    E = synthetic_batch.E
    c_star = synthetic_batch.c_star
    B, J, K = synthetic_batch.B, synthetic_batch.J, synthetic_batch.K

    outcomes = [
        [
            [f"outcome b{b} j{j} k{k}" for k in range(K)]
            for j in range(J)
        ]
        for b in range(B)
    ]

    with torch.no_grad():
        logits, interm = model(z_d, E)

    return model, z_d, E, c_star, outcomes, logits, interm


# ---------------------------------------------------------------------------
# §12.1 — head_naming_report
# ---------------------------------------------------------------------------


def test_head_naming_report_keys(synthetic_batch):
    _, _, _, _, outcomes, _, interm = _build_model_and_batch(synthetic_batch)
    report = head_naming_report(outcomes, interm, top_n=5)
    assert "head_names" in report
    assert "top_outcomes_per_head" in report
    assert len(report["head_names"]) == 5
    assert report["head_names"] == DEFAULT_HEAD_NAMES
    assert len(report["top_outcomes_per_head"]) == 5
    assert set(report["top_outcomes_per_head"].keys()) == {f"m{i}" for i in range(5)}


def test_head_naming_report_scores_sorted(synthetic_batch):
    _, _, _, _, outcomes, _, interm = _build_model_and_batch(synthetic_batch)
    report = head_naming_report(outcomes, interm, top_n=10)
    for head_key, items in report["top_outcomes_per_head"].items():
        scores = [item["score"] for item in items]
        # Strictly non-increasing (descending).
        for a, b in zip(scores, scores[1:]):
            assert a >= b, (head_key, scores)


def test_head_naming_report_custom_names(synthetic_batch):
    _, _, _, _, outcomes, _, interm = _build_model_and_batch(synthetic_batch)
    custom = ["a", "b", "c", "d", "e"]
    report = head_naming_report(outcomes, interm, top_n=3, head_names=custom)
    assert report["head_names"] == custom


# ---------------------------------------------------------------------------
# §12.2 — per_decision_report
# ---------------------------------------------------------------------------


def test_per_decision_report_shape(synthetic_batch):
    _, _, _, c_star, outcomes, logits, interm = _build_model_and_batch(synthetic_batch)
    J, K, M = synthetic_batch.J, synthetic_batch.K, 5
    report = per_decision_report(
        event_idx=0,
        outcomes=outcomes,
        intermediates=interm,
        logits=logits,
        c_star=c_star,
    )

    # Outcomes: J rows, K strings each.
    assert len(report["outcomes"]) == J
    assert all(len(row) == K for row in report["outcomes"])

    # attribute_scores: (J, K, M)
    assert len(report["attribute_scores"]) == J
    for per_alt in report["attribute_scores"]:
        assert len(per_alt) == K
        for per_outcome in per_alt:
            assert len(per_outcome) == M

    # Salience (J, K).
    assert len(report["salience"]) == J
    assert all(len(row) == K for row in report["salience"])

    # Weights (M,).
    assert len(report["weights"]) == M

    # Values (J,), probabilities (J,).
    assert len(report["values"]) == J
    assert len(report["probabilities"]) == J

    assert report["event_idx"] == 0
    assert report["chosen"] == int(c_star[0].item())


def test_per_decision_report_probs_sum_to_one(synthetic_batch):
    _, _, _, c_star, outcomes, logits, interm = _build_model_and_batch(synthetic_batch)
    report = per_decision_report(
        event_idx=1,
        outcomes=outcomes,
        intermediates=interm,
        logits=logits,
        c_star=c_star,
    )
    s = sum(report["probabilities"])
    assert abs(s - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# §12.3 — dominant_attribute_report
# ---------------------------------------------------------------------------


def test_dominant_attribute_report_nonempty(synthetic_batch):
    _, _, _, c_star, _, logits, interm = _build_model_and_batch(synthetic_batch)
    report = dominant_attribute_report(
        logits=logits, c_star=c_star, intermediates=interm
    )
    assert "by_dominant_attribute" in report
    assert "n_by_attribute" in report
    # At least one bucket should be present given a full batch.
    assert len(report["by_dominant_attribute"]) >= 1
    # Counts should sum to B.
    assert sum(report["n_by_attribute"].values()) == synthetic_batch.B
    # Keys share between the two sub-dicts.
    assert set(report["by_dominant_attribute"].keys()) == set(
        report["n_by_attribute"].keys()
    )


# ---------------------------------------------------------------------------
# §12.4 — counterfactual_report
# ---------------------------------------------------------------------------


def test_counterfactual_report_fields(synthetic_batch):
    model, z_d, E, c_star, _, _, _ = _build_model_and_batch(synthetic_batch)

    def perturb(row: torch.Tensor) -> torch.Tensor:
        out = row.clone()
        out[0] = out[0] + 0.5
        return out

    report = counterfactual_report(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        perturbation_fn=perturb,
        event_idx=0,
        label="test",
    )
    for key in (
        "label",
        "event_idx",
        "delta_weights",
        "delta_salience",
        "delta_P_chosen",
        "P_chosen_baseline",
        "P_chosen_counterfactual",
    ):
        assert key in report

    assert report["label"] == "test"
    assert report["event_idx"] == 0
    assert len(report["delta_weights"]) == 5  # M=5
    assert len(report["delta_salience"]) == synthetic_batch.J
    assert all(len(row) == synthetic_batch.K for row in report["delta_salience"])


def test_counterfactual_nonzero_effect_when_perturbation_is_large(synthetic_batch):
    model, z_d, E, c_star, _, _, _ = _build_model_and_batch(synthetic_batch)

    def big_perturb(row: torch.Tensor) -> torch.Tensor:
        return torch.full_like(row, 100.0)

    report = counterfactual_report(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        perturbation_fn=big_perturb,
        event_idx=0,
        label="massive",
    )
    # Under a large perturbation, weights should shift non-trivially, so the
    # chosen-class probability should move meaningfully.
    assert abs(report["delta_P_chosen"]) > 1e-6
    # And the weight delta should not be all zeros.
    assert any(abs(x) > 1e-6 for x in report["delta_weights"])


# ---------------------------------------------------------------------------
# run_all_reports
# ---------------------------------------------------------------------------


def test_run_all_reports_writes_jsons_when_out_dir_given(synthetic_batch, tmp_path):
    model, z_d, E, c_star, outcomes, _, _ = _build_model_and_batch(synthetic_batch)
    out_dir = tmp_path / "reports"
    bundle = run_all_reports(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        outcomes=outcomes,
        out_dir=out_dir,
        event_idx=0,
    )

    expected_files = [
        "head_naming.json",
        "per_decision.json",
        "dominant_attribute.json",
        "counterfactual.json",
    ]
    for fname in expected_files:
        p = out_dir / fname
        assert p.exists(), f"missing {fname}"
        # Round-trips as a dict.
        loaded = json.loads(p.read_text())
        assert isinstance(loaded, dict)

    # Bundle keys also match the four sub-reports.
    assert set(bundle.keys()) == {
        "head_naming",
        "per_decision",
        "dominant_attribute",
        "counterfactual",
    }


def test_run_all_reports_without_out_dir_does_not_write(synthetic_batch, tmp_path):
    model, z_d, E, c_star, outcomes, _, _ = _build_model_and_batch(synthetic_batch)
    # Confirm tmp_path is untouched after the call.
    before = list(tmp_path.iterdir())
    bundle = run_all_reports(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        outcomes=outcomes,
        out_dir=None,
        event_idx=0,
    )
    after = list(tmp_path.iterdir())
    assert after == before, f"tmp_path was written to: {after}"
    assert isinstance(bundle, dict)


def test_all_returned_dicts_json_serializable(synthetic_batch):
    model, z_d, E, c_star, outcomes, _, _ = _build_model_and_batch(synthetic_batch)
    bundle = run_all_reports(
        model=model,
        z_d=z_d,
        E=E,
        c_star=c_star,
        outcomes=outcomes,
        out_dir=None,
        event_idx=0,
    )
    # Whole bundle must dumps cleanly.
    dumped = json.dumps(bundle)
    assert isinstance(dumped, str) and len(dumped) > 0
    # And re-loads to the same structure.
    loaded = json.loads(dumped)
    assert set(loaded.keys()) == set(bundle.keys())
