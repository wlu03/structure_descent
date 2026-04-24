# Paper-Grade Evaluation Additions (Redesign v1)

**Date:** 2026-04-23
**Status:** design frozen; implementation in flight across 3 agents
**Supersedes nothing** — extends `run_full_evaluation.sh`'s output. Core PO-LEU spec (`docs/redesign.md`) is untouched.

## Motivation

The current `run_full_evaluation.sh` produces `leaderboard_summary.json` with `mean ± std` per baseline across seeds. That's enough for an internal dashboard but insufficient for a submission:

1. **No statistical significance** — `mean ± std` over 3 seeds can't tell reviewers whether PO-LEU's NLL advantage is real or noise.
2. **No subgroup analysis** — we can't say "PO-LEU wins most on customers with ≥10 events" or "LaSR helps more in BOOK than ELECTRONICS". Reviewers ask this every time.
3. **No qualitative artifact** — numeric metrics only; no inspection of the equations LLM-SR/LaSR actually propose, or the concept library LaSR accumulates.

This redesign adds five items that unlock (1)-(3) without any extra LLM spend — every number already flows through memory at eval time; we just weren't saving it.

## Target artifact schema

Each seed's existing `baselines_leaderboard_main_seed{SEED}.json` grows from a list of summary rows into a richer list where every row carries per-event and per-customer detail. A new sidecar `events_main_seed{SEED}.json` pins the canonical event ordering so downstream scripts can join across baselines.

### `baselines_leaderboard_main_seed{SEED}.json` (extended row schema)

Every row keeps today's fields (`name`, `top1`, `test_nll`, `aic`, `bic`, `n_events`, `ece`, `pseudo_r2`, `nll_uplift_vs_popularity`, `status`, `fit_time_seconds`, `description`, `traceback`) and gains:

```json
{
  ...existing fields...
  "per_event_nll": [0.81, 2.04, 0.33, ...],
  "per_event_topk_correct": [true, false, true, ...],
  "per_customer_nll": {
    "CUST_0042": {"nll": 0.72, "n_events": 8, "top1": 0.625},
    "CUST_0117": {"nll": 1.18, "n_events": 3, "top1": 0.333}
  },
  "extra_artifacts": {
    "llm_sr_top_equations": [
      {"source": "c[0]*x[0] + c[1]*log1p(x[2])", "nll_train": 0.78, "nll_val": 0.81, "coefficients": [0.42, -1.7]}
    ],
    "lasr_final_concept_library": [
      {"name": "price_sensitivity", "source": "...", "nl_summary": "...", "usage_count": 17, "discovered_at": 3}
    ]
  }
}
```

**Invariants** (must be pinned by tests):
- `len(per_event_nll) == len(per_event_topk_correct) == n_events`
- `abs(mean(per_event_nll) - test_nll) < 1e-6`
- `per_customer_nll` keys ⊆ `events_main_seed{SEED}.json` customer_ids
- `mean(per_event_nll[i] where customer_id == k) == per_customer_nll[k].nll` for every k
- `extra_artifacts == null` for every non-LLM-SR/non-LaSR row
- `per_event_nll`, `per_event_topk_correct`, `per_customer_nll` are populated iff `status == "ok"`; omitted/null otherwise

### `events_main_seed{SEED}.json` (new sidecar)

Pins the test-split event ordering so downstream scripts can cross-reference by position. Written once per seed, shared by all baselines for that seed:

```json
{
  "split_mode": "temporal",
  "seed": 7,
  "n_customers": 50,
  "n_events": 312,
  "events": [
    {
      "event_idx": 0,
      "customer_id": "CUST_0042",
      "category": "BOOK",
      "chosen_idx": 3,
      "n_alternatives": 10,
      "is_repeat": false,
      "order_date": "2024-03-12"
    }
  ]
}
```

Canonical order = the order `evaluate_baseline` iterates over the test batch. All baselines use the same order because they all consume the same `BaselineEventBatch`.

## The five additions

### 1. Per-event NLL arrays

`evaluate_baseline` already computes `-log_softmax(logits)[chosen_idx]` in the aggregate `test_nll`. Stop discarding it:

- Extend `BaselineReport` with `per_event_nll: list[float]` and `per_event_topk_correct: list[bool]`
- Thread through `run_all_baselines` into the row dict
- `save_rows_to_data_dir` serializes verbatim; CSV flattens to JSON-encoded column

~30 LOC. Touches `src/baselines/evaluate.py`, `src/baselines/run_all.py`, `tests/baselines/test_evaluate.py`.

### 2. Per-customer NLL breakdown

Grouped from (1) by `batch.customer_ids`:

- In `evaluate_baseline`, build `per_customer_nll: dict[str, dict]` by grouping per-event NLL and top-1 flag
- Include on `BaselineReport`; serialize alongside per-event arrays

~50 LOC. Same files as (1).

### 3. `events_main_seed{SEED}.json` sidecar

Written once per seed in `scripts/run_baselines.py` main, after the test batch is constructed and before baselines run:

```python
events_sidecar = {
    "split_mode": args.split_mode,
    "seed": args.seed,
    "n_customers": args.n_customers,
    "n_events": test_batch.n_events,
    "events": [
        {
            "event_idx": i,
            "customer_id": test_batch.customer_ids[i],
            "category": test_batch.categories[i],
            "chosen_idx": int(test_batch.chosen_indices[i]),
            "n_alternatives": test_batch.n_alternatives,
            "is_repeat": bool(test_batch.metadata[i].get("is_repeat", False)),
            "order_date": str(test_batch.metadata[i].get("order_date", "")),
        }
        for i in range(test_batch.n_events)
    ],
}
```

~40 LOC. Touches `scripts/run_baselines.py` only.

### 4. LLM-SR / LaSR equation export

Protocol hook: optional `extra_artifacts_for_json(self) -> dict | None` on fitted objects.

- `LLMSRFitted.extra_artifacts_for_json()` returns `{"llm_sr_top_equations": [{"source", "nll_train", "nll_val", "coefficients"}, ...]}` (top-10 by `nll_train + nll_val`)
- `LaSRFitted.extra_artifacts_for_json()` returns both `llm_sr_top_equations` and `lasr_final_concept_library`
- `run_all_baselines` after `evaluate`: `row["extra_artifacts"] = getattr(fitted, "extra_artifacts_for_json", lambda: None)()`

~40 LOC. Touches `src/baselines/llm_sr.py`, `src/baselines/lasr.py`, `src/baselines/run_all.py`, `tests/baselines/test_llm_sr.py`, `tests/baselines/test_lasr.py`.

### 5. Paired-significance script

New script `scripts/paired_significance.py`. Consumes the extended JSONs from all seeds:

```
python -m scripts.paired_significance \
    --input-dir results_data/ \
    --tag-pattern "main_seed*" \
    --baseline-of-interest "PO-LEU" \
    --output-dir results_data/significance/
```

For each other baseline $B$, against PO-LEU:
- **Paired bootstrap CI (B=1000)**: resample test events with replacement, paired across baselines, compute $\text{mean}(\text{NLL}_B - \text{NLL}_{PO-LEU})$. Report 95% percentile CI + $P(\Delta > 0)$.
- **Wilcoxon signed-rank** on per-event NLL differences. Two-sided, report p-value.
- **McNemar's test** on the 2×2 top-1 contingency. Chi-squared, report p-value.
- **ASO** (Dror et al. 2019) if `deepsig` is installed; else skip with a note.

Outputs:
- `results_data/significance/pairwise_vs_PO-LEU.md` — paper-ready table
- `results_data/significance/pairwise_vs_PO-LEU.json` — machine-readable

~200 LOC. Pure post-hoc; zero LLM spend.

### 6. Customer-segment analysis script

New script `scripts/customer_analysis.py`. Consumes `per_customer_nll` + `events_main_seed*.json`:

```
python -m scripts.customer_analysis \
    --input-dir results_data/ \
    --tag-pattern "main_seed*" \
    --output-dir results_data/segmentation/
```

Segments customers by:
- **Trajectory length**: `short` (<5 train events), `medium` (5-15), `long` (>15)
- **Category concentration**: `focused` (single-category), `diverse` (≥3 categories)
- **Novelty rate**: fraction of test events that are novel (vs repeat purchases)

Per (baseline, segment) cell:
- Mean NLL across customers in segment (averaged over seeds)
- Top-1 accuracy across customers in segment
- Count of customers

Outputs:
- `results_data/segmentation/by_trajectory_length.{md,json}`
- `results_data/segmentation/by_category_concentration.{md,json}`
- `results_data/segmentation/by_novelty_rate.{md,json}`
- `results_data/segmentation/summary.md` — top 5 "PO-LEU wins most in segment X" claims ranked by effect size

~150 LOC.

## Implementation plan

Three parallel sub-agents, each self-contained:

| Agent | Scope | New files | Edited files |
|---|---|---|---|
| **A** | (1)-(4): schema + save-path + equation export | — | `src/baselines/evaluate.py`, `src/baselines/base.py`, `src/baselines/run_all.py`, `src/baselines/llm_sr.py`, `src/baselines/lasr.py`, `scripts/run_baselines.py`, tests |
| **B** | (5) paired significance | `scripts/paired_significance.py`, `tests/scripts/test_paired_significance.py` | — |
| **C** | (6) customer segments | `scripts/customer_analysis.py`, `tests/scripts/test_customer_analysis.py` | — |

Agents B and C depend on Agent A's schema only *statically* (the JSON field names in this doc). They develop against synthetic fixtures that match the schema, so they run in parallel with A.

## Downstream paper section mapping

| Paper section | Sources |
|---|---|
| Main leaderboard table | `leaderboard_summary.json` (unchanged) |
| Statistical significance paragraph | `results_data/significance/pairwise_vs_PO-LEU.md` (addition 5) |
| Subgroup-analysis subsection | `results_data/segmentation/*.md` (addition 6) |
| Qualitative-examples appendix | `extra_artifacts.llm_sr_top_equations` + `extra_artifacts.lasr_final_concept_library` (addition 4) |
| "Where does PO-LEU fail" appendix | `per_event_nll` + `events_main_seed*.json` → post-hoc notebook |

## Success criteria

1. `./run_full_evaluation.sh` produces `baselines_leaderboard_main_seed*.json` with all new fields populated and invariants satisfied
2. `python -m scripts.paired_significance` runs against the output dir and emits paper-ready markdown
3. `python -m scripts.customer_analysis` runs against the output dir and emits segment tables
4. Existing 155 baseline tests + all new tests pass
5. Re-running on existing cached data (outcomes_cache + embeddings_cache hot) costs zero additional LLM spend

## Non-goals

- Changing PO-LEU itself
- Changing what rows appear in the leaderboard
- Reducing sweep runtime
- Human read-through of LLM outputs (deferred; raw material lands in addition 4)
