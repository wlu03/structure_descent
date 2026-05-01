"""Aggregate the mobility residual_lr_multiplier ablation outputs.

Reads:
  - results_data/abl_lrmult_mobility/x{M}/metrics_test.json       — top1/NLL/...
  - results_data/abl_lrmult_mobility/x{M}/test_logits.npz         — V+R logits
  - results_data/abl_lrmult_mobility/x0_no_residual/test_logits.npz — V baseline

Produces:
  - per-multiplier metrics table  (top1, top3, top5, NLL, ECE, R²)
  - V-vs-R decomposition table    (|R|, argmax(V) right, argmax(V+R) right,
                                   flipped→right, flipped→wrong, |R|/|V|)

Mirrors the Amazon ablation analysis methodology (V is the no-residual
model's logits used as a proxy for the V branch — same architecture
without β; R ≈ V+R − V de-meaned across J).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ABL_ROOT = REPO_ROOT / "results_data" / "abl_lrmult_mobility"
MULTIPLIERS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30)


def _load_logits(p: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(p)
    return z["logits"].astype(np.float64), z["c_star"].astype(np.int64)


def _de_mean_alts(x: np.ndarray) -> np.ndarray:
    """Subtract per-row mean across the J alternatives; cross-alt
    discrimination is what affects softmax decisions, constant offsets
    don't."""
    return x - x.mean(axis=1, keepdims=True)


def _v_baseline_path() -> Path:
    """Prefer the retrain-flow no-residual cell (matched hyperparams)
    when present; fall back to the original v4 no-residual run."""
    p1 = ABL_ROOT / "x0_no_residual" / "test_logits.npz"
    if p1.exists():
        return p1
    return REPO_ROOT / "reports" / "mobility_boston_real_v4_no_residual" / "test_logits.npz"


def main() -> int:
    v_path = _v_baseline_path()
    print(f"V baseline: {v_path}")
    V, c_star_v = _load_logits(v_path)

    print("\n## Per-multiplier metrics (test set)\n")
    print(
        "| ×β | top-1 | top-3 | top-5 | NLL | ECE | pseudo R² |"
    )
    print("|---|---|---|---|---|---|---|")

    rows = []
    for m in MULTIPLIERS:
        cell_dir = ABL_ROOT / f"x{m}"
        mt = cell_dir / "metrics_test.json"
        if not mt.exists():
            print(f"| {m} | (missing) | | | | | |")
            continue
        d = json.loads(mt.read_text())
        rows.append((m, d))
        print(
            f"| {m} | {d['top1'] * 100:.2f} % | "
            f"{d['top3'] * 100:.2f} % | "
            f"{d['top5'] * 100:.2f} % | "
            f"{d['nll_val']:.4f} | "
            f"{d['ece_val']:.4f} | "
            f"{d['pseudo_r2']:.4f} |"
        )

    print("\n## V-vs-R decomposition\n")
    print(
        "| ×β | argmax(V) right | argmax(V+R) right | "
        "R flipped V→right | R flipped V→wrong | "
        "mean\\|R\\| | mean\\|V\\| | \\|R\\|/\\|V\\| |"
    )
    print("|---|---|---|---|---|---|---|---|")

    for m, _ in rows:
        cell_dir = ABL_ROOT / f"x{m}"
        lp = cell_dir / "test_logits.npz"
        if not lp.exists():
            continue
        VR, c_star_vr = _load_logits(lp)
        if not np.array_equal(c_star_v, c_star_vr):
            print(f"| {m} | (c_star mismatch — skipping) | | | | | | |")
            continue

        # Cross-alt discrimination only; absolute offsets are softmax-invariant.
        V_dm = _de_mean_alts(V)
        VR_dm = _de_mean_alts(VR)
        R = VR_dm - V_dm

        argV = V_dm.argmax(axis=1)
        argVR = VR_dm.argmax(axis=1)
        right_V = (argV == c_star_v).sum()
        right_VR = (argVR == c_star_vr).sum()

        v_wrong_vr_right = (
            (argV != c_star_v) & (argVR == c_star_vr)
        ).sum()
        v_right_vr_wrong = (
            (argV == c_star_v) & (argVR != c_star_vr)
        ).sum()

        mean_R = float(np.abs(R).mean())
        mean_V = float(np.abs(V_dm).mean())
        ratio = mean_R / mean_V if mean_V > 0 else float("nan")

        print(
            f"| {m} | {right_V} | {right_VR} | "
            f"{int(v_wrong_vr_right)} | {int(v_right_vr_wrong)} | "
            f"{mean_R:.2f} | {mean_V:.2f} | {ratio:.2f} |"
        )

    print("\n## V's share of correct picks at each multiplier\n")
    print(
        "| ×β | V+R correct | V already right (R not load-bearing) | "
        "R flipped V→right (R load-bearing) |"
    )
    print("|---|---|---|---|")
    for m, _ in rows:
        cell_dir = ABL_ROOT / f"x{m}"
        lp = cell_dir / "test_logits.npz"
        if not lp.exists():
            continue
        VR, _ = _load_logits(lp)
        V_dm = _de_mean_alts(V)
        VR_dm = _de_mean_alts(VR)
        argV = V_dm.argmax(axis=1)
        argVR = VR_dm.argmax(axis=1)
        right_VR = int((argVR == c_star_v).sum())
        v_already_right = int(((argV == c_star_v) & (argVR == c_star_v)).sum())
        flipped = int(((argV != c_star_v) & (argVR == c_star_v)).sum())
        n = len(c_star_v)
        if right_VR == 0:
            print(f"| {m} | 0 (0.0 %) | 0 (n/a) | 0 (n/a) |")
            continue
        print(
            f"| {m} | {right_VR} ({right_VR / n * 100:.1f} %) | "
            f"{v_already_right} = **{v_already_right / right_VR * 100:.1f} %** | "
            f"{flipped} = **{flipped / right_VR * 100:.1f} %** |"
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
