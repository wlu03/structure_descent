"""Paired significance tests for baseline leaderboards (Addition 5).

Pure post-hoc analysis. Consumes the extended
``baselines_leaderboard_main_seed{SEED}.json`` files documented in
``docs/paper_evaluation_additions.md`` Section 5.

For every other baseline B and a fixed baseline-of-interest A, we compute:

* **Paired bootstrap CI** (B=1000 by default) over ``mean(NLL_B) - mean(NLL_A)``
  with paired resampling (same event indices for both baselines).
* **Wilcoxon signed-rank** on ``delta_i = nll_B[i] - nll_A[i]`` (two-sided).
* **McNemar's test** on the 2x2 top-1 correctness contingency table
  (``statsmodels.stats.contingency_tables.mcnemar``, ``exact=False, correction=True``).
* **Almost Stochastic Order (ASO)** -- if ``deepsig`` is importable; else skipped.

Seeds are aggregated by concatenating per-event NLL arrays across seeds
(equivalent to treating the combined test set as one larger test set); this
is valid because every baseline within a seed sees the same test-event
ordering, and the paired resampling is performed on the concatenated vectors.

CLI
---

    python -m scripts.paired_significance \\
        --input-dir results_data/ \\
        --tag-pattern "main_seed*" \\
        --baseline-of-interest "PO-LEU" \\
        --output-dir results_data/significance/ \\
        [--bootstrap-iterations 1000] \\
        [--confidence-level 0.95] \\
        [--seed 0]
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

logger = logging.getLogger("paired_significance")

_SCHEMA_HINT = (
    "Missing `per_event_nll` / `per_event_topk_correct` on a baseline row. "
    "See docs/paper_evaluation_additions.md section 'Target artifact schema' "
    "for the extended row fields."
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_seed_files(input_dir: Path, tag_pattern: str) -> list[tuple[int, Path, list[dict]]]:
    """Return list of (seed, path, rows) for every matching leaderboard JSON."""
    pattern = f"baselines_leaderboard_{tag_pattern}.json"
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise SystemExit(
            f"No files matching '{pattern}' under {input_dir}. "
            f"Did Agent A's producer write the extended leaderboard JSONs?"
        )
    out: list[tuple[int, Path, list[dict]]] = []
    for p in paths:
        with p.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"{p}: expected list of baseline rows, got {type(data).__name__}")
        # parse seed out of stem, tail token after last 'seed'
        stem = p.stem  # e.g. baselines_leaderboard_main_seed7
        seed_token = stem.rsplit("seed", 1)[-1]
        try:
            seed = int(seed_token)
        except ValueError:
            seed = -1
        out.append((seed, p, data))
    return out


def extract_paired_vectors(
    seed_files: list[tuple[int, Path, list[dict]]],
    baseline_of_interest: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    """Concatenate per-event vectors across seeds.

    Returns:
        nll_by_name: baseline -> concatenated float array over seeds
        top1_by_name: baseline -> concatenated bool array over seeds
        kept_seeds: list of seed ints that were used
    """
    # First, collect the set of baselines that are ok in baseline_of_interest's
    # seeds. We only use a seed if baseline_of_interest has status == ok.
    nll_chunks: dict[str, list[np.ndarray]] = {}
    top1_chunks: dict[str, list[np.ndarray]] = {}
    kept_seeds: list[int] = []
    found_names: set[str] = set()

    for seed, path, rows in seed_files:
        by_name: dict[str, dict] = {}
        for row in rows:
            name = row.get("name")
            if name is None:
                raise SystemExit(f"{path}: row missing 'name' field")
            by_name[name] = row
            found_names.add(name)

        if baseline_of_interest not in by_name:
            raise SystemExit(
                f"baseline_of_interest '{baseline_of_interest}' not found in {path}. "
                f"Baselines present across all loaded files so far: "
                f"{sorted(found_names)}"
            )

        boi_row = by_name[baseline_of_interest]
        if boi_row.get("status") != "ok":
            logger.warning(
                "Dropping seed %s (%s): baseline_of_interest '%s' has status=%r",
                seed, path.name, baseline_of_interest, boi_row.get("status"),
            )
            continue

        # Validate all ok rows have equal-length per_event arrays for this seed.
        expected_len: int | None = None
        for name, row in by_name.items():
            if row.get("status") != "ok":
                continue
            if "per_event_nll" not in row or "per_event_topk_correct" not in row:
                raise SystemExit(f"{path} baseline '{name}': {_SCHEMA_HINT}")
            n = len(row["per_event_nll"])
            if len(row["per_event_topk_correct"]) != n:
                raise SystemExit(
                    f"{path} baseline '{name}': per_event_nll and "
                    f"per_event_topk_correct lengths disagree ({n} vs "
                    f"{len(row['per_event_topk_correct'])})"
                )
            if expected_len is None:
                expected_len = n
            elif n != expected_len:
                raise SystemExit(
                    f"{path}: baselines disagree on n_events within seed "
                    f"({name} has {n}, earlier ok-row has {expected_len})"
                )

        for name, row in by_name.items():
            if row.get("status") != "ok":
                continue
            nll_chunks.setdefault(name, []).append(
                np.asarray(row["per_event_nll"], dtype=float)
            )
            top1_chunks.setdefault(name, []).append(
                np.asarray(row["per_event_topk_correct"], dtype=bool)
            )
        kept_seeds.append(seed)

    if not kept_seeds:
        raise SystemExit(
            f"No seeds usable: baseline_of_interest '{baseline_of_interest}' "
            f"had status != 'ok' in every seed."
        )

    # Only baselines that were ok in every kept seed get a full-length vector
    # (one chunk per seed). Otherwise drop with warning.
    nll_by_name: dict[str, np.ndarray] = {}
    top1_by_name: dict[str, np.ndarray] = {}
    want_chunks = len(kept_seeds)
    for name, chunks in nll_chunks.items():
        if len(chunks) != want_chunks:
            logger.warning(
                "Dropping baseline '%s': ok in only %d/%d seeds",
                name, len(chunks), want_chunks,
            )
            continue
        nll_by_name[name] = np.concatenate(chunks)
        top1_by_name[name] = np.concatenate(top1_chunks[name])

    if baseline_of_interest not in nll_by_name:
        raise SystemExit(
            f"baseline_of_interest '{baseline_of_interest}' was dropped "
            f"(ok in only some seeds). Cannot proceed."
        )
    return nll_by_name, top1_by_name, kept_seeds


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def paired_bootstrap(
    nll_a: np.ndarray,
    nll_b: np.ndarray,
    iterations: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired bootstrap CI on mean(nll_b) - mean(nll_a)."""
    n = len(nll_a)
    assert len(nll_b) == n
    # Pre-sample all indices at once for speed.
    idx = rng.integers(0, n, size=(iterations, n))
    # mean over axis=1 for each bootstrap row.
    mean_a = nll_a[idx].mean(axis=1)
    mean_b = nll_b[idx].mean(axis=1)
    deltas = mean_b - mean_a
    alpha = 1.0 - confidence_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    ci_lo, ci_hi = np.percentile(deltas, [lo_q, hi_q])
    return {
        "delta_mean": float(deltas.mean()),
        "delta_ci_lo": float(ci_lo),
        "delta_ci_hi": float(ci_hi),
        "p_delta_gt_zero": float((deltas > 0.0).mean()),
    }


def wilcoxon_paired(nll_a: np.ndarray, nll_b: np.ndarray) -> dict[str, Any]:
    deltas = nll_b - nll_a
    if np.all(deltas == 0):
        return {"wilcoxon_statistic": None, "wilcoxon_p_value": None,
                "wilcoxon_note": "all deltas zero"}
    res = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    return {
        "wilcoxon_statistic": float(res.statistic),
        "wilcoxon_p_value": float(res.pvalue),
    }


def mcnemar_paired(top1_a: np.ndarray, top1_b: np.ndarray) -> dict[str, Any]:
    # b_ij: A is i-correct, B is j-correct (i, j in {0, 1}).
    a1 = top1_a.astype(bool)
    b1 = top1_b.astype(bool)
    b00 = int(np.sum((~a1) & (~b1)))
    b01 = int(np.sum((~a1) & b1))
    b10 = int(np.sum(a1 & (~b1)))
    b11 = int(np.sum(a1 & b1))
    table = [[b00, b01], [b10, b11]]
    if b01 + b10 == 0:
        return {"mcnemar_statistic": None, "mcnemar_p_value": None,
                "mcnemar_table": table, "mcnemar_note": "no disagreement"}
    res = mcnemar(np.asarray(table), exact=False, correction=True)
    return {
        "mcnemar_statistic": float(res.statistic),
        "mcnemar_p_value": float(res.pvalue),
        "mcnemar_table": table,
    }


def aso_if_available(
    scores_a: np.ndarray, scores_b: np.ndarray, confidence_level: float
) -> dict[str, Any]:
    try:
        deepsig = importlib.import_module("deepsig")
    except ImportError:
        return {"aso_epsilon_min": None, "aso_note": "ASO skipped: deepsig not installed"}
    try:
        eps = deepsig.aso(scores_a, scores_b, confidence_level=confidence_level)
    except Exception as e:  # noqa: BLE001 -- deepsig can raise various things
        return {"aso_epsilon_min": None, "aso_note": f"ASO errored: {e}"}
    return {"aso_epsilon_min": float(eps)}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_num(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    if signed:
        return f"{x:+.3f}"
    return f"{x:.3f}"


def _fmt_pvalue(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 1e-6:
        return "<1e-6"
    if p < 1e-2:
        return f"{p:.2e}"
    return f"{p:.3f}"


def render_markdown(
    baseline_of_interest: str,
    n_events: int,
    n_seeds: int,
    bootstrap_iterations: int,
    deepsig_available: bool,
    pairs: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# Pairwise Significance vs {baseline_of_interest}")
    lines.append("")
    lines.append(f"n_events = {n_events}")
    lines.append(f"n_seeds = {n_seeds}")
    lines.append(f"bootstrap_iterations = {bootstrap_iterations}")
    if not deepsig_available:
        lines.append("")
        lines.append("ASO skipped: deepsig not installed")
    lines.append("")
    header = (
        f"| Baseline | ΔNLL (B − {baseline_of_interest}) | "
        f"95% CI | P(Δ>0) | Wilcoxon p | McNemar p | ASO ε_min |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    # Sort: biggest gap vs baseline_of_interest first (highest delta_mean).
    ordered = sorted(pairs.items(), key=lambda kv: -kv[1]["delta_mean"])
    for name, stats in ordered:
        ci = f"[{_fmt_num(stats['delta_ci_lo'], signed=True)}, {_fmt_num(stats['delta_ci_hi'], signed=True)}]"
        aso_val = stats.get("aso_epsilon_min")
        aso_str = _fmt_num(aso_val) if aso_val is not None else "n/a"
        lines.append(
            f"| {name} "
            f"| {_fmt_num(stats['delta_mean'], signed=True)} "
            f"| {ci} "
            f"| {stats['p_delta_gt_zero']:.3f} "
            f"| {_fmt_pvalue(stats['wilcoxon_p_value'])} "
            f"| {_fmt_pvalue(stats['mcnemar_p_value'])} "
            f"| {aso_str} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_files = load_seed_files(input_dir, args.tag_pattern)
    nll_by_name, top1_by_name, kept_seeds = extract_paired_vectors(
        seed_files, args.baseline_of_interest,
    )

    nll_a = nll_by_name[args.baseline_of_interest]
    top1_a = top1_by_name[args.baseline_of_interest]
    n_events = int(len(nll_a))

    rng = np.random.default_rng(args.seed)

    # Check deepsig availability once for the output metadata.
    try:
        importlib.import_module("deepsig")
        deepsig_available = True
    except ImportError:
        deepsig_available = False

    pairs: dict[str, dict[str, Any]] = {}
    for name, nll_b in nll_by_name.items():
        if name == args.baseline_of_interest:
            continue
        top1_b = top1_by_name[name]
        stats: dict[str, Any] = {}
        stats.update(paired_bootstrap(
            nll_a, nll_b, args.bootstrap_iterations, args.confidence_level, rng,
        ))
        stats.update(wilcoxon_paired(nll_a, nll_b))
        stats.update(mcnemar_paired(top1_a, top1_b))
        stats.update(aso_if_available(nll_a, nll_b, args.confidence_level))
        pairs[name] = stats

    payload = {
        "baseline_of_interest": args.baseline_of_interest,
        "n_events": n_events,
        "n_seeds": len(kept_seeds),
        "bootstrap_iterations": args.bootstrap_iterations,
        "confidence_level": args.confidence_level,
        "seeds_used": kept_seeds,
        "deepsig_available": deepsig_available,
        "pairs": pairs,
    }

    boi_slug = args.baseline_of_interest
    json_path = output_dir / f"pairwise_vs_{boi_slug}.json"
    md_path = output_dir / f"pairwise_vs_{boi_slug}.md"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    md_path.write_text(render_markdown(
        args.baseline_of_interest, n_events, len(kept_seeds),
        args.bootstrap_iterations, deepsig_available, pairs,
    ))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paired significance tests for baseline leaderboards.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--tag-pattern", default="main_seed*")
    p.add_argument("--baseline-of-interest", default="PO-LEU")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
