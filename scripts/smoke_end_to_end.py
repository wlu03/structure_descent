"""End-to-end smoke test on synthetic data.

Generates B=32 synthetic events, runs one epoch of training on the default
PO-LEU model, produces all four §12 interpretability reports, and asserts
the artifacts exist and are non-empty.

Usage
-----

    python scripts/smoke_end_to_end.py [--reports-dir reports/smoke]

No external data, no network calls. All stubs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.eval.interpret import run_all_reports
from src.eval.metrics import compute_all
from src.eval.strata import dominant_attribute_breakdown
from src.model.po_leu import POLEU
from src.train.loop import TrainConfig, fit, iter_batches
from src.train.regularizers import RegularizerConfig


def _make_synthetic_batch(B: int, J: int, K: int, d_e: int, p: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    E = torch.nn.functional.normalize(
        torch.randn(B, J, K, d_e, generator=g), p=2, dim=-1
    )
    z_d = torch.randn(B, p, generator=g)
    c_star = torch.randint(0, J, (B,), generator=g, dtype=torch.int64)
    outcomes = [
        [
            [
                f"I imagine consequence-{b}-{j}-{k} unfolding over the next few weeks with measurable effect."
                for k in range(K)
            ]
            for j in range(J)
        ]
        for b in range(B)
    ]
    return z_d, E, c_star, outcomes


def main(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    B, J, K, d_e, p = 32, 10, 3, 768, 26
    z_d_tr, E_tr, c_tr, outcomes_tr = _make_synthetic_batch(B, J, K, d_e, p, seed=0)
    z_d_va, E_va, c_va, _ = _make_synthetic_batch(B // 2, J, K, d_e, p, seed=1)

    model = POLEU()
    tcfg = TrainConfig(batch_size=8, max_epochs=1, early_stopping_patience=5, lr=1e-3)
    rcfg = RegularizerConfig.from_default()

    def train_fn():
        return iter_batches(z_d_tr, E_tr, c_tr, None,
                            batch_size=tcfg.batch_size, shuffle=True)

    def val_fn():
        return iter_batches(z_d_va, E_va, c_va, None,
                            batch_size=tcfg.batch_size, shuffle=False)

    total_steps = math.ceil(B / tcfg.batch_size) * tcfg.max_epochs
    state = fit(model, train_fn, val_fn,
                train_cfg=tcfg, reg_cfg=rcfg,
                total_steps=total_steps, seed=0)

    model.eval()
    with torch.no_grad():
        logits_va, interm_va = model(z_d_va, E_va)

    metrics = compute_all(logits_va, c_va,
                          n_params=model.num_params(), n_train=B)
    strata = dominant_attribute_breakdown(logits_va, c_va, interm_va)

    with torch.no_grad():
        logits_tr, interm_tr = model(z_d_tr, E_tr)
    report = run_all_reports(model, z_d_tr, E_tr, c_tr, outcomes_tr,
                             out_dir=reports_dir, event_idx=0)

    summary = {
        "config": {"B": B, "J": J, "K": K, "d_e": d_e, "p": p,
                   "epochs": tcfg.max_epochs, "batch_size": tcfg.batch_size,
                   "total_steps": total_steps},
        "train_state": {
            "epoch": state.epoch,
            "train_loss": state.train_loss,
            "val_nll": state.val_nll,
            "stopped_early": state.stopped_early,
        },
        "metrics": metrics.to_dict(),
        "dominant_attribute_groups": sorted(strata.keys()) if strata else [],
        "reports_written": sorted(f.name for f in reports_dir.iterdir()
                                   if f.suffix == ".json"),
        "param_count": model.num_params(),
    }
    (reports_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))

    expected_files = {"head_naming.json", "per_decision.json",
                      "dominant_attribute.json", "counterfactual.json"}
    written = set(summary["reports_written"])
    missing = expected_files - written
    assert not missing, f"missing interpret reports: {missing}"

    for fname in expected_files:
        content = (reports_dir / fname).read_text()
        assert len(content) > 10, f"{fname} is empty"

    assert model.num_params() == 544_779, (
        f"param count drift: {model.num_params()} ≠ 544,779 (see NOTES.md)"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nsmoke OK  →  {reports_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path,
                    default=REPO_ROOT / "reports" / "smoke")
    args = ap.parse_args()
    main(args.reports_dir)
