"""Customer-segment post-hoc analysis for baseline leaderboards.

Consumes the extended ``baselines_leaderboard_main_seed{SEED}.json`` (per-
customer NLL / top-1) and the companion ``events_main_seed{SEED}.json`` sidecar
(customer/category/is_repeat metadata) produced by the evaluation pipeline
(see ``docs/paper_evaluation_additions.md`` §6).

Three independent segmentation schemes are produced:
  * trajectory length (by total test events)
  * category concentration (by distinct test categories)
  * novelty rate (fraction of test events that are not ``is_repeat``)

For each (baseline, segment) pair we report customer-weighted mean NLL, top-1
accuracy, and customer counts, and emit a paper-friendly ``summary.md``.

CLI
---

    python -m scripts.customer_analysis \\
        --input-dir results_data/ \\
        --tag-pattern "main_seed*" \\
        --output-dir results_data/segmentation/

See the task spec for full argument semantics. The script is pure Python +
NumPy/``statistics``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("customer_analysis")


# ---------------------------------------------------------------------------
# Segmentation rules
# ---------------------------------------------------------------------------

TRAJ_ORDER = ["short", "medium", "long"]
TRAJ_LABEL = {
    "short": "short (1-2 events)",
    "medium": "medium (3-6 events)",
    "long": "long (7+ events)",
}

CAT_ORDER = ["focused", "moderate", "diverse"]
CAT_LABEL = {
    "focused": "focused (1 category)",
    "moderate": "moderate (2 categories)",
    "diverse": "diverse (3+ categories)",
}

NOV_ORDER = ["mostly_repeat", "mixed", "mostly_novel"]
NOV_LABEL = {
    "mostly_repeat": "mostly_repeat (<30% novel)",
    "mixed": "mixed (30-70% novel)",
    "mostly_novel": "mostly_novel (>70% novel)",
}


def bucket_trajectory(n_events: int) -> str:
    if n_events <= 2:
        return "short"
    if n_events <= 6:
        return "medium"
    return "long"


def bucket_category(n_categories: int) -> str:
    if n_categories <= 1:
        return "focused"
    if n_categories == 2:
        return "moderate"
    return "diverse"


def bucket_novelty(novel_fraction: float) -> str:
    if novel_fraction < 0.30:
        return "mostly_repeat"
    if novel_fraction <= 0.70:
        return "mixed"
    return "mostly_novel"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SEED_RE = re.compile(r"main_seed(\d+)")


def _seed_from_filename(path: str) -> str:
    m = _SEED_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def load_seed_pair(input_dir: Path, tag_pattern: str) -> List[Tuple[str, dict, dict]]:
    """Return ``[(seed, leaderboard_json, events_json), ...]``.

    Raises a clear error if any baselines leaderboard lacks a matching events
    sidecar (data inconsistency).
    """
    lb_glob = str(input_dir / f"baselines_leaderboard_{tag_pattern}.json")
    ev_glob = str(input_dir / f"events_{tag_pattern}.json")
    lb_paths = sorted(glob.glob(lb_glob))
    ev_paths = sorted(glob.glob(ev_glob))
    if not lb_paths:
        raise FileNotFoundError(f"No leaderboard files matched {lb_glob}")
    ev_by_seed = {_seed_from_filename(p): p for p in ev_paths}
    out: List[Tuple[str, dict, dict]] = []
    for lb_path in lb_paths:
        seed = _seed_from_filename(lb_path)
        if seed not in ev_by_seed:
            raise FileNotFoundError(
                f"Data inconsistency: leaderboard {lb_path} has no matching "
                f"events_main_seed{seed}.json (looked in {ev_glob})"
            )
        with open(lb_path) as f:
            lb = json.load(f)
        with open(ev_by_seed[seed]) as f:
            ev = json.load(f)
        out.append((seed, lb, ev))
    return out


# ---------------------------------------------------------------------------
# Segmentation + aggregation
# ---------------------------------------------------------------------------


def build_customer_segments(
    seeds: List[Tuple[str, dict, dict]],
) -> Dict[str, Dict[str, str]]:
    """Compute (trajectory/category/novelty) segments per customer.

    Event counts, category sets, and novelty are summed across seeds for the
    same ``customer_id`` (see task spec).
    """
    total_events: Dict[str, int] = defaultdict(int)
    categories: Dict[str, set] = defaultdict(set)
    novel_counts: Dict[str, int] = defaultdict(int)

    for _seed, _lb, ev in seeds:
        for e in ev.get("events", []):
            cid = e["customer_id"]
            total_events[cid] += 1
            categories[cid].add(e.get("category"))
            if not bool(e.get("is_repeat", False)):
                novel_counts[cid] += 1

    segments: Dict[str, Dict[str, str]] = {}
    for cid, n_events in total_events.items():
        if n_events <= 0:
            logger.warning("customer %s has 0 test events; excluding", cid)
            continue
        novel_frac = novel_counts[cid] / n_events
        segments[cid] = {
            "trajectory_length": bucket_trajectory(n_events),
            "category_concentration": bucket_category(len(categories[cid])),
            "novelty_rate": bucket_novelty(novel_frac),
        }
    return segments


def aggregate_customer_metrics(
    seeds: List[Tuple[str, dict, dict]],
    known_customers: set,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns ``{baseline: {customer: {"nll": mean_across_seeds, "top1": ...}}}``.

    A (baseline, seed, customer) cell is dropped if ``status != "ok"`` for
    that row, or if the baseline lacks ``per_customer_nll`` for the customer.
    """
    # baseline -> customer -> [list of per-seed dicts]
    accum: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sidecar_cids_by_seed: Dict[str, set] = {}
    for seed, lb, ev in seeds:
        sidecar_cids_by_seed[seed] = {e["customer_id"] for e in ev.get("events", [])}
        for row in lb:
            if row.get("status") != "ok":
                continue
            pcn = row.get("per_customer_nll") or {}
            bname = row["name"]
            for cid, rec in pcn.items():
                if cid not in sidecar_cids_by_seed[seed]:
                    raise ValueError(
                        f"Data inconsistency: baseline {bname} (seed {seed}) "
                        f"reports customer {cid} not present in "
                        f"events_main_seed{seed}.json"
                    )
                accum[bname][cid].append(
                    {"nll": float(rec["nll"]), "top1": float(rec.get("top1", 0.0))}
                )
            # warn if baselines report unknown customers relative to global set
            for cid in pcn:
                if cid not in known_customers:
                    logger.warning(
                        "baseline %s seed %s reports customer %s not in "
                        "aggregated events; skipping",
                        bname,
                        seed,
                        cid,
                    )

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for bname, per_cust in accum.items():
        out[bname] = {}
        for cid, per_seed in per_cust.items():
            if not per_seed:
                continue
            out[bname][cid] = {
                "nll": statistics.fmean(r["nll"] for r in per_seed),
                "top1": statistics.fmean(r["top1"] for r in per_seed),
                "n_seeds": len(per_seed),
            }
    return out


def aggregate_by_segment(
    customer_metrics: Dict[str, Dict[str, Dict[str, float]]],
    segments: Dict[str, Dict[str, str]],
    scheme: str,
    order: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return ``{segment_key: {"n_customers": int, "baselines": {...}}}``."""
    # group customers by segment value
    cust_by_seg: Dict[str, List[str]] = defaultdict(list)
    for cid, seg in segments.items():
        cust_by_seg[seg[scheme]].append(cid)

    result: Dict[str, Dict[str, Any]] = {}
    for seg in order:
        cids = cust_by_seg.get(seg, [])
        if not cids:
            logger.warning("segment %s/%s has 0 customers; omitting", scheme, seg)
            continue
        seg_entry: Dict[str, Any] = {"n_customers": len(cids), "baselines": {}}
        for bname, per_cust in customer_metrics.items():
            nlls = [per_cust[c]["nll"] for c in cids if c in per_cust]
            t1s = [per_cust[c]["top1"] for c in cids if c in per_cust]
            if not nlls:
                continue
            seg_entry["baselines"][bname] = {
                "nll_mean": statistics.fmean(nlls),
                "nll_std": statistics.stdev(nlls) if len(nlls) > 1 else 0.0,
                "top1": statistics.fmean(t1s),
                "n_customers_with_data": len(nlls),
            }
        result[seg] = seg_entry
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown_table(
    title: str,
    segments: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
    order: List[str],
    min_segment_size: int,
) -> str:
    lines = [f"# {title}", "", "| Segment | n_customers | Baseline | mean_NLL | top1 |",
             "|---|---|---|---|---|"]
    for seg_key in order:
        if seg_key not in segments:
            continue
        seg = segments[seg_key]
        label = labels.get(seg_key, seg_key)
        flag = " ⚠ small segment" if seg["n_customers"] < min_segment_size else ""
        # sort baselines by ascending mean NLL
        ranked = sorted(seg["baselines"].items(), key=lambda kv: kv[1]["nll_mean"])
        if not ranked:
            lines.append(f"| {label}{flag} | {seg['n_customers']} | — | — | — |")
            continue
        for bname, stats in ranked:
            lines.append(
                f"| {label}{flag} | {seg['n_customers']} | {bname} | "
                f"{stats['nll_mean']:.3f} ± {stats['nll_std']:.3f} | "
                f"{stats['top1']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_summary(
    all_schemes: Dict[str, Dict[str, Dict[str, Any]]],
    labels_by_scheme: Dict[str, Dict[str, str]],
    baseline_of_interest: str,
    min_segment_size: int,
) -> str:
    """Summarize effect sizes for the baseline of interest across all schemes."""
    # Per scheme: list of (scheme, seg_key, delta_vs_next_best, runner_up, boi_nll, n)
    rows: List[Dict[str, Any]] = []
    lost_segments: List[Dict[str, Any]] = []
    for scheme, segs in all_schemes.items():
        for seg_key, seg in segs.items():
            bl = seg["baselines"]
            if baseline_of_interest not in bl:
                continue
            boi_nll = bl[baseline_of_interest]["nll_mean"]
            others = [(b, s["nll_mean"]) for b, s in bl.items() if b != baseline_of_interest]
            if not others:
                continue
            best_other_name, best_other_nll = min(others, key=lambda kv: kv[1])
            delta = best_other_nll - boi_nll  # positive means boi wins
            rec = {
                "scheme": scheme,
                "segment_key": seg_key,
                "segment_label": labels_by_scheme[scheme][seg_key],
                "delta": delta,
                "runner_up": best_other_name,
                "boi_nll": boi_nll,
                "runner_up_nll": best_other_nll,
                "n_customers": seg["n_customers"],
            }
            rows.append(rec)
            if delta < 0:
                lost_segments.append(rec)

    wins = [r for r in rows if r["delta"] > 0]
    wins_sorted = sorted(wins, key=lambda r: r["delta"], reverse=True)

    lines = [f"# Segmentation Summary — baseline of interest: {baseline_of_interest}", ""]

    if not rows:
        lines.append(f"*No segments found with {baseline_of_interest} present.*")
        return "\n".join(lines) + "\n"

    # Per-scheme digest
    for scheme in all_schemes:
        scheme_rows = [r for r in rows if r["scheme"] == scheme]
        if not scheme_rows:
            continue
        lines.append(f"## Scheme: {scheme}")
        wins_here = sorted(
            [r for r in scheme_rows if r["delta"] > 0], key=lambda r: r["delta"]
        )
        losses_here = [r for r in scheme_rows if r["delta"] < 0]
        if wins_here:
            biggest = wins_here[-1]
            smallest = wins_here[0]
            lines.append(
                f"- Largest win: {baseline_of_interest} beats {biggest['runner_up']} "
                f"by ΔNLL={biggest['delta']:.3f} in the {biggest['segment_label']} "
                f"segment (n={biggest['n_customers']} customers)."
            )
            if smallest is not biggest:
                lines.append(
                    f"- Smallest win: {baseline_of_interest} beats "
                    f"{smallest['runner_up']} by ΔNLL={smallest['delta']:.3f} in "
                    f"the {smallest['segment_label']} segment "
                    f"(n={smallest['n_customers']} customers)."
                )
        if losses_here:
            for r in losses_here:
                flag = " ⚠ small segment" if r["n_customers"] < min_segment_size else ""
                lines.append(
                    f"- LOSS: {baseline_of_interest} loses to {r['runner_up']} by "
                    f"ΔNLL={-r['delta']:.3f} in the {r['segment_label']} segment "
                    f"(n={r['n_customers']} customers){flag}."
                )
        lines.append("")

    # Top-5 striking wins
    lines.append("## Top 5 strongest segment wins (ranked by ΔNLL)")
    if not wins_sorted:
        lines.append(f"- (no winning segments for {baseline_of_interest})")
    else:
        for r in wins_sorted[:5]:
            flag = " ⚠ small segment" if r["n_customers"] < min_segment_size else ""
            lines.append(
                f"1. {baseline_of_interest} beats the next-best baseline "
                f"({r['runner_up']}) by ΔNLL={r['delta']:.3f} in the "
                f"{r['segment_label']} segment "
                f"(scheme={r['scheme']}, n={r['n_customers']} customers){flag}."
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    input_dir: Path,
    tag_pattern: str,
    output_dir: Path,
    baseline_of_interest: str,
    min_segment_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seed_pair(input_dir, tag_pattern)
    segments = build_customer_segments(seeds)
    if not segments:
        raise RuntimeError("No customers found across inputs; nothing to aggregate")
    customer_metrics = aggregate_customer_metrics(seeds, set(segments.keys()))

    schemes = [
        ("trajectory_length", TRAJ_ORDER, TRAJ_LABEL, "by_trajectory_length",
         "Customer Segmentation — Trajectory Length"),
        ("category_concentration", CAT_ORDER, CAT_LABEL, "by_category_concentration",
         "Customer Segmentation — Category Concentration"),
        ("novelty_rate", NOV_ORDER, NOV_LABEL, "by_novelty_rate",
         "Customer Segmentation — Novelty Rate"),
    ]

    all_schemes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    labels_by_scheme: Dict[str, Dict[str, str]] = {}
    for scheme, order, labels, basename, title in schemes:
        seg_data = aggregate_by_segment(customer_metrics, segments, scheme, order)
        all_schemes[scheme] = seg_data
        labels_by_scheme[scheme] = labels
        md = render_markdown_table(title, seg_data, labels, order, min_segment_size)
        (output_dir / f"{basename}.md").write_text(md)
        js = {"scheme": scheme, "segments": seg_data}
        (output_dir / f"{basename}.json").write_text(json.dumps(js, indent=2))

    summary_md = build_summary(all_schemes, labels_by_scheme, baseline_of_interest,
                               min_segment_size)
    (output_dir / "summary.md").write_text(summary_md)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--tag-pattern", default="main_seed*")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-of-interest", default="PO-LEU")
    p.add_argument("--min-segment-size", type=int, default=3)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        input_dir=args.input_dir,
        tag_pattern=args.tag_pattern,
        output_dir=args.output_dir,
        baseline_of_interest=args.baseline_of_interest,
        min_segment_size=args.min_segment_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
