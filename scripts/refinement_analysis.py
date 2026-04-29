"""Stratified A/B analysis: PO-LEU vs PO-LEU-REFINED.

Joins per-event NLL sidecars across all seeds for both runs, then computes
diagnostics that the leaderboard's aggregate row cannot show:

  1. **Headline paired ΔNLL** — mean per-event ``nll_refined - nll_baseline``
     with bootstrap 95% CI and Wilcoxon signed-rank p-value. Negative
     means refinement won.
  2. **Stratified gain** — same ΔNLL split by whether the test event's
     chosen ASIN was touched by the refinement loop (read from each
     seed's ``refined_outcomes.json``). Tells you whether refinement
     generalizes or only fixes the events whose alternatives were
     directly rewritten.
  3. **Difficulty-bucketed ΔNLL** — events bucketed by baseline-NLL
     quartile. Catches the "fixes hard events but hurts easy ones"
     failure mode that aggregate stats hide.
  4. **Cost-effectiveness** — ΔNLL per 100 LLM calls spent on the
     refinement loop (read from ``refined_outcomes.json:summary``).

Output goes to ``<output-dir>/refinement_analysis.{md,json}``.

Usage::

    python -m scripts.refinement_analysis \\
        --baseline-glob "results_data/poleu_*cust_seed*_no_residual" \\
        --refined-glob  "results_data/poleu_*cust_seed*_refined" \\
        --output-dir    results_data/refinement_analysis

Both globs are matched independently; pairs are joined by the seed token
extracted from each run-dir name (``..._seedN_...``).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

logger = logging.getLogger("refinement_analysis")


_SEED_RE = re.compile(r"_seed(\d+)")


def _seed_from_path(path: Path) -> int | None:
    m = _SEED_RE.search(path.name)
    return int(m.group(1)) if m else None


def _load_per_event(run_dir: Path) -> tuple[list[dict], list[str]]:
    """Return (per_event_rows, refined_asins) for one run_dir."""
    sidecar = run_dir / "test_per_event.json"
    if not sidecar.exists():
        raise SystemExit(f"missing test_per_event.json in {run_dir}")
    payload = json.loads(sidecar.read_text())
    rows = list(payload.get("per_event", []))
    refined_asins: list[str] = []
    refined_path = run_dir / "refined_outcomes.json"
    if refined_path.exists():
        refined_payload = json.loads(refined_path.read_text())
        for item in refined_payload.get("items", []):
            if not bool(item.get("skipped", False)):
                refined_asins.append(str(item.get("asin", "")))
    return rows, refined_asins


def _bootstrap_ci(
    values: list[float], n_boot: int = 10_000, seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean."""
    import random
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(int(seed))
    n = len(values)
    means = []
    for _ in range(int(n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (lo, hi)


def _wilcoxon_signed_rank_p(deltas: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approximation).

    Hand-rolled — no scipy dep. ``deltas`` are paired diffs; nulls
    (delta == 0) are dropped. Returns NaN if fewer than 6 non-zero pairs
    (the normal approximation is unreliable below that).
    """
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    if n < 6:
        return float("nan")
    abs_vals = sorted([(abs(d), 1.0 if d > 0 else -1.0) for d in nz])
    # Average rank for ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_pos = sum(r for r, (_, sign) in zip(ranks, abs_vals) if sign > 0)
    w_neg = sum(r for r, (_, sign) in zip(ranks, abs_vals) if sign < 0)
    w = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_w == 0:
        return 1.0
    z = (w - mean_w) / sd_w
    # Two-sided normal tail.
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _summarize(label: str, deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {"label": label, "n": 0}
    lo, hi = _bootstrap_ci(deltas)
    return {
        "label": label,
        "n": len(deltas),
        "mean_delta_nll": float(mean(deltas)),
        "median_delta_nll": float(median(deltas)),
        "std_delta_nll": float(stdev(deltas)) if len(deltas) > 1 else 0.0,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "wilcoxon_p_two_sided": float(_wilcoxon_signed_rank_p(deltas)),
        "n_refined_wins": int(sum(1 for d in deltas if d < 0)),
        "n_refined_losses": int(sum(1 for d in deltas if d > 0)),
        "n_ties": int(sum(1 for d in deltas if d == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="refinement_analysis.py", description=__doc__)
    parser.add_argument("--baseline-glob", required=True,
                        help="Glob for the no-residual PO-LEU run dirs.")
    parser.add_argument("--refined-glob", required=True,
                        help="Glob for the PO-LEU-REFINED run dirs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    baseline_dirs = sorted(Path(p) for p in glob.glob(args.baseline_glob))
    refined_dirs = sorted(Path(p) for p in glob.glob(args.refined_glob))
    if not baseline_dirs:
        logger.error("baseline glob matched nothing: %s", args.baseline_glob)
        return 2
    if not refined_dirs:
        logger.error("refined glob matched nothing: %s", args.refined_glob)
        return 2

    baseline_by_seed = {
        s: d for s, d in ((_seed_from_path(p), p) for p in baseline_dirs)
        if s is not None
    }
    refined_by_seed = {
        s: d for s, d in ((_seed_from_path(p), p) for p in refined_dirs)
        if s is not None
    }
    common_seeds = sorted(set(baseline_by_seed) & set(refined_by_seed))
    if not common_seeds:
        logger.error("no shared seeds between baseline + refined globs")
        return 2
    logger.info("seeds matched on both sides: %s", common_seeds)

    # Aggregate per-event deltas across all matched seeds.
    all_deltas: list[float] = []
    deltas_refined_asin: list[float] = []
    deltas_unrefined_asin: list[float] = []
    baseline_nll_by_event: list[float] = []
    n_llm_calls_total = 0

    for seed in common_seeds:
        b_dir = baseline_by_seed[seed]
        r_dir = refined_by_seed[seed]
        b_rows, _ = _load_per_event(b_dir)
        r_rows, refined_asins = _load_per_event(r_dir)
        # Refined ASINs come from the no-residual baseline's bookkeeping.
        # Re-load to be sure we use the right side (refine_outcomes.json
        # is written into the baseline run-dir by run_full_evaluation.sh).
        baseline_refined_asins = _load_per_event(b_dir)[1]
        refined_asin_set = set(baseline_refined_asins or refined_asins)

        if len(b_rows) != len(r_rows):
            logger.warning(
                "seed %d: baseline n=%d != refined n=%d; truncating to min",
                seed, len(b_rows), len(r_rows),
            )
        n = min(len(b_rows), len(r_rows))
        for i in range(n):
            b = b_rows[i]
            r = r_rows[i]
            if b.get("event_idx") != r.get("event_idx"):
                logger.warning(
                    "seed %d event %d: baseline event_idx=%s != refined %s; "
                    "skipping",
                    seed, i, b.get("event_idx"), r.get("event_idx"),
                )
                continue
            b_nll = float(b.get("nll", 0.0))
            r_nll = float(r.get("nll", 0.0))
            delta = r_nll - b_nll
            all_deltas.append(delta)
            baseline_nll_by_event.append(b_nll)
            chosen_asin = str(b.get("asin_chosen", ""))
            if chosen_asin in refined_asin_set:
                deltas_refined_asin.append(delta)
            else:
                deltas_unrefined_asin.append(delta)

        # Tally LLM calls from refined_outcomes.json summary if present.
        refined_path = b_dir / "refined_outcomes.json"
        if refined_path.exists():
            ref_payload = json.loads(refined_path.read_text())
            summary = ref_payload.get("summary", {}) or {}
            # Each non-skipped pair = 1 critic + 1 writer; each skipped = 1 critic.
            n_llm_calls_total += int(summary.get("n_revised", 0)) * 2
            n_llm_calls_total += int(summary.get("n_skipped_above_threshold", 0))

    headline = _summarize("PO-LEU-REFINED − PO-LEU (test events, all)", all_deltas)
    refined_strat = _summarize(
        "events whose chosen ASIN was refined", deltas_refined_asin,
    )
    unrefined_strat = _summarize(
        "events whose chosen ASIN was NOT refined", deltas_unrefined_asin,
    )

    # Difficulty buckets by baseline-NLL quartile.
    buckets: list[dict[str, Any]] = []
    if all_deltas:
        paired = sorted(zip(baseline_nll_by_event, all_deltas), key=lambda t: t[0])
        n = len(paired)
        edges = [int(round(q * n / 4)) for q in (0, 1, 2, 3, 4)]
        for q in range(4):
            lo_i, hi_i = edges[q], edges[q + 1]
            sub = paired[lo_i:hi_i]
            sub_deltas = [d for _, d in sub]
            sub_baseline = [b for b, _ in sub]
            buckets.append({
                "quartile": q + 1,
                "n_events": len(sub),
                "baseline_nll_range": [
                    min(sub_baseline) if sub_baseline else None,
                    max(sub_baseline) if sub_baseline else None,
                ],
                **{k: v for k, v in _summarize(f"Q{q+1}", sub_deltas).items() if k != "label"},
            })

    cost_effectiveness = None
    if n_llm_calls_total > 0 and all_deltas:
        cost_effectiveness = {
            "total_llm_calls_refinement": n_llm_calls_total,
            "delta_nll_per_100_calls": (
                100.0 * float(mean(all_deltas)) / n_llm_calls_total
            ),
        }

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "refinement_analysis.json"
    json_path.write_text(json.dumps({
        "seeds": common_seeds,
        "n_events_total": len(all_deltas),
        "headline": headline,
        "stratified": {
            "refined_asin": refined_strat,
            "unrefined_asin": unrefined_strat,
        },
        "difficulty_buckets": buckets,
        "cost_effectiveness": cost_effectiveness,
    }, indent=2))

    md = _render_markdown(
        common_seeds, headline, refined_strat, unrefined_strat,
        buckets, cost_effectiveness,
    )
    (out_dir / "refinement_analysis.md").write_text(md)
    logger.info("wrote %s and refinement_analysis.md", json_path)
    return 0


def _render_markdown(
    seeds: list[int],
    headline: dict,
    refined_strat: dict,
    unrefined_strat: dict,
    buckets: list[dict],
    cost: dict | None,
) -> str:
    def _fmt(val: float | None, prec: int = 4) -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        return f"{val:.{prec}f}"

    lines: list[str] = []
    lines.append("# Refinement A/B analysis (PO-LEU vs PO-LEU-REFINED)\n")
    lines.append(f"Seeds: {seeds}\n")
    lines.append(f"n_events: {headline.get('n', 0)}\n")
    lines.append("")
    lines.append("## Headline (paired Δ NLL on TEST events; lower is better)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| mean Δ NLL | {_fmt(headline.get('mean_delta_nll'))} |")
    lines.append(f"| 95% bootstrap CI | [{_fmt(headline.get('ci95_lo'))}, {_fmt(headline.get('ci95_hi'))}] |")
    lines.append(f"| Wilcoxon p (two-sided) | {_fmt(headline.get('wilcoxon_p_two_sided'), 4)} |")
    lines.append(f"| refined wins / losses / ties | "
                 f"{headline.get('n_refined_wins', 0)} / "
                 f"{headline.get('n_refined_losses', 0)} / "
                 f"{headline.get('n_ties', 0)} |")
    lines.append("")
    lines.append("**Reading**: a 95% CI strictly below 0 with p < 0.05 is the "
                 "minimum bar for declaring refinement helps. If the CI straddles "
                 "0, the leaderboard's aggregate gain is within seed-noise.\n")

    lines.append("## Stratified gain (does it generalize?)\n")
    lines.append("| Stratum | n | mean Δ NLL | 95% CI |")
    lines.append("|---|---|---|---|")
    for s in (refined_strat, unrefined_strat):
        lines.append(
            f"| {s.get('label','?')} | {s.get('n',0)} | "
            f"{_fmt(s.get('mean_delta_nll'))} | "
            f"[{_fmt(s.get('ci95_lo'))}, {_fmt(s.get('ci95_hi'))}] |"
        )
    lines.append("")
    lines.append("**Reading**: if only `refined ASIN` is negative, refinement "
                 "is essentially memorizing the events touched. The interesting "
                 "case is `unrefined ASIN` also negative — refinement "
                 "generalized.\n")

    lines.append("## Difficulty buckets (does it hurt easy events?)\n")
    lines.append("| Quartile | n | baseline NLL range | mean Δ NLL | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for b in buckets:
        rng = b.get("baseline_nll_range") or [None, None]
        lines.append(
            f"| Q{b.get('quartile','?')} | {b.get('n_events',0)} | "
            f"[{_fmt(rng[0])}, {_fmt(rng[1])}] | "
            f"{_fmt(b.get('mean_delta_nll'))} | "
            f"[{_fmt(b.get('ci95_lo'))}, {_fmt(b.get('ci95_hi'))}] |"
        )
    lines.append("")
    lines.append("**Reading**: Q1 is easiest (lowest baseline NLL). The "
                 "failure mode is Δ negative on Q4 (hard events) but Δ "
                 "positive on Q1 (easy events) — net wash that aggregate "
                 "metrics hide.\n")

    if cost:
        lines.append("## Cost-effectiveness\n")
        lines.append(f"- Total LLM calls (critic + writer): "
                     f"{cost['total_llm_calls_refinement']}")
        lines.append(f"- Δ NLL per 100 calls: "
                     f"{_fmt(cost['delta_nll_per_100_calls'])}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
