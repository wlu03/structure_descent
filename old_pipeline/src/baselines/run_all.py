"""
Orchestrator that fits and evaluates every available baseline.

Discovers baselines via optional imports — if a baseline module hasn't been
implemented yet, or if its dependency stack isn't installed, it is silently
skipped and the corresponding row appears as 'unavailable' in the report.

Usage from a notebook or script:

    from src.baselines.data import load_from_checkpoints
    from src.baselines.run_all import run_all_baselines

    train, val, test = load_from_checkpoints("data")
    rows = run_all_baselines(train, val, test, verbose=True)
    # rows is a list of dicts; convert to a pandas DataFrame for display.

Usage from the command line (against synthetic data, for smoke testing):

    venv/bin/python -m src.baselines.run_all --synthetic
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

from .base import BaselineEventBatch
from .evaluate import evaluate_baseline


# Each entry: (display name, dotted module path, class name).
# Add new baselines here as they land. Order is the order they will run in.
#
# Note: RUMBoost is excluded — its numpy<2 / cythonbiogeme dependency
# conflict (see src/baselines/rumboost_baseline.py) cannot be resolved in
# this venv. It is replaced by three classical ML baselines
# (RandomForest, GradientBoosting, MLP) that serve as flexible
# predictive-ceiling comparators rather than as an RUM-with-trees hybrid.
BASELINE_REGISTRY: List[Tuple[str, str, str]] = [
    ("LASSO-MNL",         "src.baselines.lasso_mnl",     "LassoMnl"),
    ("RandomForest",      "src.baselines.classical_ml",  "RandomForestChoice"),
    ("GradientBoosting",  "src.baselines.classical_ml",  "GradientBoostingChoice"),
    ("MLP",               "src.baselines.classical_ml",  "MLPChoice"),
    ("Paz-VNS",           "src.baselines.paz_vns",       "PazVNS"),
    ("DUET-GA",           "src.baselines.duet_ga",       "DuetGA"),
    ("Bayesian-ARD",      "src.baselines.bayesian_ard",  "BayesianARD"),
    ("Delphos",           "src.baselines.delphos",       "Delphos"),
]


def _try_load(module_path: str, class_name: str):
    """Return (cls, None) on success or (None, error_string) on failure."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        return None, f"import failed: {e}"
    except Exception as e:
        return None, f"unexpected import error: {e}"
    cls = getattr(mod, class_name, None)
    if cls is None:
        return None, f"class {class_name} not found in {module_path}"
    return cls, None


def run_all_baselines(
    train: BaselineEventBatch,
    val: BaselineEventBatch,
    test: BaselineEventBatch,
    baseline_kwargs: Optional[Dict[str, dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """
    Fit and evaluate every available baseline. Skips ones that aren't
    importable (not yet implemented or missing optional deps).

    Parameters
    ----------
    train, val, test : BaselineEventBatch
        Splits produced by src.baselines.data.
    baseline_kwargs : dict
        Optional per-baseline kwargs to pass to the constructor. Keys are
        the display names from BASELINE_REGISTRY.
    verbose : bool
        Print a one-line status per baseline as it runs.

    Returns
    -------
    list of dict
        One row per baseline (available or unavailable), with keys:
        name, status, top1, top5, mrr, test_nll, aic, bic, n_params,
        fit_seconds, description, error.
    """
    baseline_kwargs = baseline_kwargs or {}
    rows: List[Dict[str, object]] = []

    for display_name, module_path, class_name in BASELINE_REGISTRY:
        if verbose:
            print(f"[{display_name}] loading...", flush=True)

        cls, err = _try_load(module_path, class_name)
        if cls is None:
            rows.append({
                "name": display_name,
                "status": "unavailable",
                "error": err,
            })
            if verbose:
                print(f"[{display_name}] unavailable — {err}")
            continue

        kwargs = baseline_kwargs.get(display_name, {})

        try:
            baseline = cls(**kwargs)
            t0 = time.perf_counter()
            fitted = baseline.fit(train, val)
            fit_seconds = time.perf_counter() - t0

            report = evaluate_baseline(
                fitted,
                test,
                train_n_events=train.n_events,
                fit_time_seconds=fit_seconds,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            rows.append({
                "name": display_name,
                "status": "errored",
                "error": str(e),
                "traceback": tb,
            })
            if verbose:
                print(f"[{display_name}] errored — {e}")
            continue

        m = report.metrics
        row = {
            "name": display_name,
            "status": "ok",
            "top1": m["top1"],
            "top5": m["top5"],
            "mrr": m["mrr"],
            "test_nll": m["test_nll"],
            "aic": m["aic"],
            "bic": m["bic"],
            "n_params": report.n_params,
            "fit_seconds": fit_seconds,
            "description": getattr(fitted, "description", ""),
            "error": None,
        }
        rows.append(row)

        if verbose:
            print(report.summary())
            print(f"  fit_time={fit_seconds:.1f}s  {row['description']}")

    return rows


def format_table(rows: List[Dict[str, object]]) -> str:
    """Render a comparison table as a fixed-width string."""
    cols = [
        ("name", 14),
        ("status", 11),
        ("top1", 8),
        ("top5", 8),
        ("mrr", 8),
        ("test_nll", 9),
        ("aic", 11),
        ("bic", 11),
        ("n_params", 9),
        ("fit_seconds", 8),
    ]
    lines: List[str] = []
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        cells = []
        for name, w in cols:
            v = row.get(name)
            if v is None or row.get("status") != "ok":
                if name == "name":
                    cells.append(f"{str(row.get('name', '?')):<{w}}")
                elif name == "status":
                    cells.append(f"{str(row.get('status', '?')):<{w}}")
                else:
                    cells.append(f"{'-':<{w}}")
                continue
            if name in ("top1", "top5"):
                cells.append(f"{v:<{w}.1%}")
            elif name in ("mrr", "test_nll"):
                cells.append(f"{v:<{w}.4f}")
            elif name in ("aic", "bic", "fit_seconds"):
                cells.append(f"{v:<{w}.1f}")
            elif name == "n_params":
                cells.append(f"{int(v):<{w}d}")
            elif name == "name":
                cells.append(f"{str(v):<{w}}")
            else:
                cells.append(f"{str(v):<{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Smoke-test against synthetic batches instead of real checkpoints.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the project's checkpoint directory (default: data).",
    )
    args = parser.parse_args()

    if args.synthetic:
        from ._synthetic import make_synthetic_batch
        train = make_synthetic_batch(n_events=400, seed=1001)
        val = make_synthetic_batch(n_events=200, seed=1002)
        test = make_synthetic_batch(n_events=200, seed=1003)
    else:
        from .data import load_from_checkpoints
        train, val, test = load_from_checkpoints(args.data_dir)

    rows = run_all_baselines(train, val, test, verbose=True)
    print()
    print(format_table(rows))


if __name__ == "__main__":
    _main()
