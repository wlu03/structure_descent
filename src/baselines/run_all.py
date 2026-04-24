"""Orchestrator: fit + evaluate every Phase-1 baseline in one call.

Phase-1 registry covers 6 baselines that have been ported into
``src/baselines/`` and verified against the PO-LEU data contract via
:mod:`src.baselines.data_adapter`:

* LASSO-MNL, Bayesian-ARD — conditional-logit MNL families over the
  expanded feature pool.
* RandomForest, GradientBoosting, MLP — classical predictive-ceiling
  rankers (score alternatives by ``log P(chosen=1 | alt_features)``;
  see :mod:`src.baselines.classical_ml` for the documented caveat).
* DUET — parametric linear-plus-MLP branch with a soft monotonicity
  penalty. Module filename is legacy (``duet_ga.py``); the GA variant
  was superseded by the ANN of Han et al. (2024). ``DuetGA`` remains as
  a back-compat alias.

Not yet ported (Phase 2):

* Delphos, Paz-VNS — require porting ``old_pipeline/src/dsl.py`` and
  ``old_pipeline/src/inner_loop.py`` (~1.5k lines of DSL + hierarchical
  MNL machinery) into ``src/``. Out of scope for Phase 1.

Dropped:

* RUMBoost — unresolved ``numpy<2`` / ``cythonbiogeme`` ABI conflict.

Importable baselines that are unavailable at runtime (e.g. NumPyro
missing for Bayesian-ARD) are reported as ``unavailable`` rather than
aborting the whole run.

Usage from a script:

    from src.baselines.data_adapter import records_to_baseline_batch
    from src.baselines.run_all import run_all_baselines

    train = records_to_baseline_batch(train_records)
    val = records_to_baseline_batch(val_records)
    test = records_to_baseline_batch(test_records)
    rows = run_all_baselines(train, val, test, verbose=True)

CLI smoke test on synthetic data:

    venv/bin/python -m src.baselines.run_all --synthetic
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaselineEventBatch, BaselineReport
from .evaluate import evaluate_baseline


# Each entry: (display name, dotted module path, class name).
# Order is the order they will run in.
#
# Popularity runs first so its ``test_nll`` is available as the reference
# point for the ``nll_uplift_vs_popularity`` post-processing pass below.
# ----- Multi-provider LLM sweep ------------------------------------------- #
#
# Each entry is ``(display_suffix, module_path, class_name, client_factory)``.
# The factory is a zero-arg callable that returns an ``LLMClient`` instance
# at baseline-fit time. We defer instantiation so (a) missing SDKs / missing
# API keys mark only the affected row ``unavailable`` rather than aborting
# the whole suite, and (b) we don't pay Anthropic/Gemini/OpenAI import cost
# unless the baseline is actually running.
#
# To add a model: append one entry. To drop a provider for a given run:
# comment out the entry or pass an override via ``baseline_kwargs``.

def _anthropic_factory(model_id: str) -> Callable[[], Any]:
    """Lazy factory for an ``AnthropicLLMClient`` bound to ``model_id``."""
    def _make() -> Any:
        from src.outcomes.generate import AnthropicLLMClient
        return AnthropicLLMClient(model_id=model_id)
    return _make


def _gemini_factory(model_id: str) -> Callable[[], Any]:
    """Lazy factory for a ``GeminiLLMClient`` (Vertex AI + ADC)."""
    def _make() -> Any:
        from src.outcomes._gemini_client import GeminiLLMClient
        return GeminiLLMClient(model_id=model_id)
    return _make


def _openai_factory(model_id: str) -> Callable[[], Any]:
    """Lazy factory for an ``OpenAILLMClient``."""
    def _make() -> Any:
        from src.outcomes._openai_client import OpenAILLMClient
        return OpenAILLMClient(model_id=model_id)
    return _make


# Each row: (display_suffix, factory).
# display_suffix becomes the "-<model>" tail on the baseline's leaderboard
# name, e.g. ``ZeroShot`` + ``Claude-Sonnet-4.6`` -> ``ZeroShot-Claude-Sonnet-4.6``.
LLM_MODEL_SWEEP: List[Tuple[str, Callable[[], Any]]] = [
    ("Claude-Sonnet-4.6",  _anthropic_factory("claude-sonnet-4-6")),
    ("Claude-Opus-4.6",    _anthropic_factory("claude-opus-4-6")),
    ("Gemini-2.5-Pro",     _gemini_factory("gemini-2.5-pro")),
    ("Gemini-2.5-Flash",   _gemini_factory("gemini-2.5-flash")),
    ("GPT-5",              _openai_factory("gpt-5")),
]

# LLM-backed baselines in the base registry. For each entry below, the
# expansion machinery appends one ``(<prefix>-<model>, module, class)`` row
# per entry in :data:`LLM_MODEL_SWEEP`, and records the corresponding
# factory in :data:`LLM_CLIENT_FACTORIES`.
_LLM_BASELINE_BASES: List[Tuple[str, str, str]] = [
    ("ZeroShot",    "src.baselines.zero_shot_claude_ranker", "ZeroShotClaudeRanker"),
    ("FewShot-ICL", "src.baselines.few_shot_icl_ranker",     "FewShotICLRanker"),
    ("LLM-SR",      "src.baselines.llm_sr",                  "LLMSR"),
    ("LaSR",        "src.baselines.lasr",                    "LaSR"),
]

# Populated below via :func:`_build_registry`. Maps expanded display_name
# (e.g. ``"ZeroShot-Claude-Sonnet-4.6"``) to the zero-arg client factory
# that :func:`run_all_baselines` calls to obtain the LLMClient instance.
LLM_CLIENT_FACTORIES: Dict[str, Callable[[], Any]] = {}


def _build_registry() -> List[Tuple[str, str, str]]:
    """Build the per-run baseline registry.

    Non-LLM baselines appear once. Each LLM baseline appears once per
    entry in :data:`LLM_MODEL_SWEEP` so a single run surfaces a
    head-to-head leaderboard across all providers.
    """
    registry: List[Tuple[str, str, str]] = [
        # Reference / frequency baseline — runs first so its test_nll
        # can anchor the nll_uplift_vs_popularity column.
        ("Popularity",       "src.baselines.popularity",    "PopularityBaseline"),
        # Tabular baselines (per-alt 6-feature matrix via data_adapter).
        ("LASSO-MNL",        "src.baselines.lasso_mnl",     "LassoMnl"),
        ("RandomForest",     "src.baselines.classical_ml",  "RandomForestChoice"),
        ("GradientBoosting", "src.baselines.classical_ml",  "GradientBoostingChoice"),
        ("MLP",              "src.baselines.classical_ml",  "MLPChoice"),
        ("Bayesian-ARD",     "src.baselines.bayesian_ard",  "BayesianARD"),
        ("DUET",             "src.baselines.duet_ga",       "DUET"),
        ("Delphos",          "src.baselines.delphos",       "Delphos"),
        # LLM-free ablation — embeds raw alt-metadata with the same
        # sentence encoder PO-LEU uses and trains a small MLP.
        ("ST-MLP",           "src.baselines.st_mlp_ablation", "STMLPChoice"),
    ]
    # Expand every LLM baseline across all models in LLM_MODEL_SWEEP.
    # Register each expansion's factory so run_all_baselines can inject
    # the right llm_client at construction time.
    for prefix, module_path, class_name in _LLM_BASELINE_BASES:
        for model_suffix, factory in LLM_MODEL_SWEEP:
            name = f"{prefix}-{model_suffix}"
            registry.append((name, module_path, class_name))
            LLM_CLIENT_FACTORIES[name] = factory
    return registry


BASELINE_REGISTRY: List[Tuple[str, str, str]] = _build_registry()


# Key into ``BaselineReport.extra`` under which the uplift-vs-popularity
# nats are written by :func:`annotate_uplift_vs_popularity` and surfaced
# as a column in :func:`format_table`.
UPLIFT_EXTRA_KEY = "nll_uplift_vs_popularity"


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
    """Fit + evaluate every Phase-1 baseline.

    Skips baselines that can't be imported (missing optional deps). Each
    baseline is given ``baseline_kwargs[display_name]`` (default ``{}``)
    at construction time.

    Returns
    -------
    list of dict
        One row per baseline with keys: ``name``, ``status``, and —
        when ``status == "ok"`` — ``top1``, ``top5``, ``mrr``,
        ``test_nll``, ``aic``, ``bic``, ``n_params``, ``fit_seconds``,
        ``description``, ``error``.
    """
    baseline_kwargs = baseline_kwargs or {}
    rows: List[Dict[str, object]] = []
    # Parallel map from display_name -> BaselineReport, kept locally so we
    # can run the uplift-vs-popularity post-processing pass below without
    # re-running evaluate_baseline.
    reports: Dict[str, BaselineReport] = {}

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

        # Start from caller-supplied kwargs; inject a concrete
        # ``llm_client`` from the sweep factory if the baseline is an
        # LLM baseline and the caller didn't already pass one. Factory
        # failures (missing SDK / unset API key / bad ADC config) mark
        # just this row unavailable rather than aborting the suite.
        kwargs = dict(baseline_kwargs.get(display_name, {}))
        if display_name in LLM_CLIENT_FACTORIES and "llm_client" not in kwargs:
            factory = LLM_CLIENT_FACTORIES[display_name]
            try:
                kwargs["llm_client"] = factory()
            except Exception as e:
                rows.append({
                    "name": display_name,
                    "status": "unavailable",
                    "error": f"LLM client factory failed: {e}",
                })
                if verbose:
                    print(
                        f"[{display_name}] unavailable — LLM client "
                        f"factory failed: {e}"
                    )
                continue

        try:
            baseline = cls(**kwargs)
            # Production stub-safety tripwire: an LLM baseline in the
            # registry MUST hold a real client at this point. The
            # factory branch above installs one; the only path to a
            # stub here is a caller passing ``llm_client=None`` via
            # ``baseline_kwargs`` (which falls back to StubLLMClient
            # inside the baseline's __init__). Refuse to run under
            # stub for any registered LLM row so paid real-LLM
            # leaderboards never contain a silently-fake row.
            if display_name in LLM_CLIENT_FACTORIES:
                injected = getattr(baseline, "llm_client", None)
                if injected is None:
                    injected = getattr(baseline, "client", None)
                if getattr(injected, "_is_stub", False):
                    rows.append({
                        "name": display_name,
                        "status": "errored",
                        "error": (
                            f"PRODUCTION SAFETY: {display_name} resolved "
                            "to a StubLLMClient. Registered LLM baselines "
                            "must hold a real provider client — check "
                            "that the LLM_CLIENT_FACTORIES entry fires "
                            "and that baseline_kwargs does not override "
                            "llm_client with a stub."
                        ),
                    })
                    if verbose:
                        print(
                            f"[{display_name}] errored — stub client "
                            "rejected by production safety tripwire"
                        )
                    continue
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

        reports[display_name] = report

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

    # Post-processing pass: compute ``nll_uplift_vs_popularity`` (nats
    # gained over the Popularity baseline) and surface it on every
    # successful report's ``extra`` dict + the returned row. We write
    # ``extra``, NOT ``metrics``, to stay out of the way of other
    # post-processing passes that own the ``metrics`` dict.
    annotate_uplift_vs_popularity(reports, rows)

    return rows


def annotate_uplift_vs_popularity(
    reports: Dict[str, BaselineReport],
    rows: List[Dict[str, object]],
) -> None:
    """Attach ``nll_uplift_vs_popularity`` to each report's ``extra`` dict.

    Given parallel ``reports`` (keyed by display name) and ``rows``
    (the flattened result list returned by :func:`run_all_baselines`),
    this function looks up the Popularity baseline's ``test_nll`` and
    writes ``popularity_nll - report.metrics['test_nll']`` into every
    report's ``extra[UPLIFT_EXTRA_KEY]``. The same value is mirrored on
    the matching row under :data:`UPLIFT_EXTRA_KEY` so
    :func:`format_table` can render it.

    If the Popularity baseline did not run successfully (unavailable /
    errored), this function is a no-op — no rows are annotated and a
    note is *not* raised, so callers that swap Popularity out entirely
    still get a working pipeline.
    """
    pop_report = reports.get("Popularity")
    if pop_report is None:
        return
    popularity_nll = float(pop_report.metrics["test_nll"])

    # Index rows by name for direct mutation.
    rows_by_name = {row.get("name"): row for row in rows}

    for name, report in reports.items():
        uplift = popularity_nll - float(report.metrics["test_nll"])
        report.extra[UPLIFT_EXTRA_KEY] = float(uplift)
        row = rows_by_name.get(name)
        if row is not None:
            row[UPLIFT_EXTRA_KEY] = float(uplift)


def format_table(rows: List[Dict[str, object]]) -> str:
    """Render a comparison table as a fixed-width string."""
    # ``name`` is wide enough for the longest expanded LLM baseline
    # display name (e.g. "FewShot-ICL-Gemini-2.5-Flash").
    cols = [
        ("name", 32),
        ("status", 11),
        ("top1", 8),
        ("top5", 8),
        ("mrr", 8),
        ("test_nll", 9),
        ("aic", 11),
        ("bic", 11),
        ("n_params", 9),
        ("fit_seconds", 8),
        (UPLIFT_EXTRA_KEY, 14),
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
            elif name == UPLIFT_EXTRA_KEY:
                cells.append(f"{v:<{w}.4f}")
            elif name == "n_params":
                cells.append(f"{int(v):<{w}d}")
            elif name == "name":
                cells.append(f"{str(v):<{w}}")
            else:
                cells.append(f"{str(v):<{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def save_rows_to_data_dir(
    rows: List[Dict[str, object]],
    output_dir: str | "os.PathLike[str]" = "data",
    *,
    tag: str | None = None,
) -> dict[str, str]:
    """Persist baseline leaderboard rows to ``output_dir``.

    Writes three artifacts side-by-side so downstream tooling (notebooks,
    leaderboard diff scripts, the PO-LEU paper plotting code) can pick
    whichever format is convenient:

    * ``baselines_leaderboard.json`` — canonical machine-readable form.
    * ``baselines_leaderboard.csv``  — convenient for spreadsheets /
      ``pandas.read_csv``; nested fields (``traceback``) are JSON-encoded
      into a single string column.
    * ``baselines_leaderboard.txt``  — the human-readable table from
      :func:`format_table`, kept alongside for quick inspection.

    ``tag`` (optional) is appended to the stem — e.g. ``tag="20cust"``
    yields ``baselines_leaderboard_20cust.json`` — so multiple runs in
    the same directory don't clobber each other.

    Returns a dict ``{ "json": path, "csv": path, "txt": path }`` with
    the absolute paths written.
    """
    import csv
    import json
    import os

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "baselines_leaderboard"
    if tag:
        stem = f"{stem}_{tag}"

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    txt_path = out_dir / f"{stem}.txt"

    # JSON — preserve nested types (tracebacks, error strings, etc.).
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)

    # CSV — flatten any non-scalar value to a JSON string.
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                flat = {
                    k: (
                        v
                        if v is None or isinstance(v, (str, int, float, bool))
                        else json.dumps(v, default=str)
                    )
                    for k, v in row.items()
                }
                writer.writerow(flat)
    else:
        csv_path.write_text("", encoding="utf-8")

    # TXT — the human-readable table.
    txt_path.write_text(format_table(rows) + "\n", encoding="utf-8")

    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "txt": str(txt_path.resolve()),
    }


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
    parser.add_argument(
        "--output-dir",
        default="data",
        help=(
            "Directory to write the leaderboard artifacts "
            "(JSON + CSV + TXT). Default: ``data``."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for output filenames, e.g. --tag=20cust.",
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

    paths = save_rows_to_data_dir(rows, args.output_dir, tag=args.tag)
    print()
    print(f"wrote leaderboard to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")


if __name__ == "__main__":
    _main()
