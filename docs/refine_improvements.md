# Refine improvements: per-outcome surgical rewrite

## Background

Curriculum refinement (`scripts/refine_outcomes.py` + `src/outcomes/refine.py`) runs a critic-and-revise loop on the worst PO-LEU val events: a critic LLM scores K candidate outcomes for plausibility and diversity, and if either axis falls below `accept_threshold`, a writer LLM rewrites all K outcomes from scratch.

After running this on the Boston mobility dataset (n=30 customers, 30 worst val events, all 10 alts per event = 300 (event, alt) pairs, ~95% rewritten by the critic), we observed a split-direction outcome:

| Metric | Δ vs base PO-LEU |
|---|---|
| top1 | +0.3 pp |
| top3 | +2.4 pp |
| top5 | +1.5 pp |
| MRR | +0.003 |
| **NLL** | **+0.028** *(worse)* |
| **Brier** | **+0.002** *(worse)* |
| **ECE** | **+0.018** *(worse)* |

Ranking metrics improved, calibration regressed. The most likely cause: the monolithic rewrite was churning *strong* outcomes alongside the weak ones. When the critic flags one or two of K=5 outcomes as bad, rewriting all 5 means losing whatever discriminative signal the strong outcomes carried — replaced with new outcomes that are more decisive in some directions and overconfident in others.

## What's implemented (commit `c205dfb`)

A per-outcome surgical-rewrite path that lets the reviser fix only the flagged positions while preserving strong outcomes byte-identical.

### Critic contract change

`CRITIC_SYSTEM_PROMPT` (`src/outcomes/prompts.py`) now asks the critic to also return `weak_outcome_indices: list[int]` (0-indexed positions). The critic is instructed:

> Aim to flag only outcomes that materially degrade the set; leave the others alone so their signal carries through.

JSON schema:
```json
{
  "plausibility": <int 1-5>,
  "diversity":    <int 1-5>,
  "weak_outcome_indices": [<0-indexed ints>],
  "notes": "<one short sentence per weak outcome, naming the issue>"
}
```

`CritiqueResult` (in `refine.py`) gains a matching `weak_outcome_indices: list[int]` field. The parser is tolerant: missing field, non-list, or mixed types all degrade to `[]`, which routes the reviser to the monolithic fallback so older critics still work.

### Reviser contract change

A new prompt pair, `REVISER_PER_OUTCOME_SYSTEM_PROMPT` + `REVISER_PER_OUTCOME_USER_TEMPLATE`, asks for **exactly N = len(weak_indices) sentences** in ascending position order. The strong outcomes are still rendered to the reviser (with a `*** WEAK — REWRITE ***` tag on the flagged ones) so it can avoid generating paraphrases of consequence types already covered.

### Splice in the orchestrator

`revise()` in `refine.py` now branches:

```
if per_outcome and weak_indices:
    # Writer returns N sentences (one per weak slot).
    parsed_n = parse_completion(writer_output, K=N)
    merged = list(originals)
    for slot_idx, new_text in zip(weak_indices, parsed_n):
        merged[slot_idx] = new_text
    parsed = merged                    # K outcomes, strong slots verbatim
else:
    parsed = parse_completion(writer_output, K=K)   # monolithic fallback
```

The diversity_filter still runs on the final K-sentence merged set, so cross-slot diversity coherence holds.

### Skip rule tightened

The old `refine_outcomes()` skipped the revise call whenever both scores ≥ `accept_threshold`. The new rule:

> Skip ONLY when scores pass AND no weak positions were flagged.

If scores pass but the critic singled out a position as weak, still surgically fix that one slot. This is the cheap, high-precision path the new contract is designed for — fix what's broken, leave everything else alone.

### Cache versioning

`REFINED_PROMPT_VERSION` bumped from `v2_refined` → `v3_refined`. Existing v2_refined entries (from prior runs) stay valid; the cascade `--prompt-version-cascade v3_refined v2_refined v4_mobility_anchored` will prefer v3 entries when they exist, fall through to v2, then to the base v4 cache.

### Telemetry

`scripts/refine_outcomes.py` summary now reports:

- `n_per_outcome_path` — pairs that took the surgical path
- `n_monolithic_path` — pairs that fell back to the old K-sentence rewrite
- `avg_positions_rewritten_per_outcome` — average # positions fixed when per-outcome fired

The end-of-run log line looks like:

> `refinement complete: 240 refined (165 via per-outcome splice, 75 via monolithic rewrite), 60 skipped (passed threshold), avg_positions_rewritten_per_outcome=2.3.`

## Verified paths (stub-LLM smoke)

Three paths covered by deterministic stub-driven tests:

1. **Skipped**: both scores ≥ 4 AND `weak_outcome_indices=[]` → no writer call, originals returned with `skipped=True`.
2. **Per-outcome surgical**: critic flags positions `[1, 3]` of 5 → writer returns 2 sentences, orchestrator splices into slots 1 and 3, slots 0/2/4 byte-identical to originals.
3. **Monolithic fallback**: critic returns `weak_outcome_indices=[]` with low scores → writer rewrites all K, behaves like the old contract.

`tests/test_refine.py`, `tests/test_refine_pipeline.py`, `tests/test_prompts.py`, `tests/test_prompts_anchored.py`, `tests/test_generate.py` all pass after the change.

## How to run

### One-shot integrated pipeline (recommended)

`run_dataset.py` now accepts `--refine` which chains identify → refine → retrain inside one invocation:

```bash
venv/bin/python scripts/run_dataset.py \
  --adapter mobility_boston \
  --n-customers 30 --seed 7 --K 5 \
  --prompt-version-cascade v4_mobility_anchored \
  --add-event-time-to-c-d --add-event-origin-to-c-d \
  --tabular-residual false \
  --refine \
  --refine-top-k 30 \
  --refine-accept-threshold 4 \
  --refine-per-outcome on \
  --refine-critic writer \
  --output-dir reports/mobility_boston_v4_refined
```

Layout produced:

```
reports/mobility_boston_v4_refined/
├── round1/                          # round-1 artifacts (initial train)
│   ├── records.pkl
│   ├── metrics.json
│   ├── metrics_test.json
│   ├── val_per_event.json
│   ├── test_logits.npz
│   └── …
├── failure_events.json              # round-1's worst val events
├── refined_outcomes.json            # critic + reviser bookkeeping
├── records.pkl                      # round-2 records (rebuilt; cache-warm)
├── metrics.json                     # final round-2 metrics
├── metrics_test.json                # final round-2 test metrics
├── test_logits.npz                  # leaderboard-ready logits
└── …
```

`--refine-critic writer` reuses the writer LLM client (cheapest); switch to `gemini` or `openai` for a cross-family critic. The cascade is automatically prepended with `v3_refined`, so the round-2 cache lookups prefer refined entries before falling through to the base prompt.

### Manual 3-step path (still supported)

```bash
# 1. Identify worst val events
venv/bin/python scripts/identify_failure_events.py \
  --run-dir reports/mobility_boston_real_v4_no_residual \
  --top-k 30

# 2. Refine — per-outcome on by default
venv/bin/python scripts/refine_outcomes.py \
  --failure-events reports/mobility_boston_real_v4_no_residual/failure_events.json \
  --outcomes-cache outcomes_cache/mobility_v4/outcomes.sqlite \
  --K 5 --seed 7 \
  --v1-prompt-version v4_mobility_anchored \
  --writer anthropic --critic writer \
  --per-outcome on \
  --output reports/mobility_boston_real_v4_no_residual/refined_outcomes_v3.json

# 3. Retrain — cascade prefers v3, falls back to v2, then base
venv/bin/python scripts/run_dataset.py \
  --adapter mobility_boston \
  --n-customers 30 --seed 7 --K 5 \
  --prompt-version-cascade v3_refined v2_refined v4_mobility_anchored \
  --add-event-time-to-c-d --add-event-origin-to-c-d \
  --tabular-residual false \
  --output-dir reports/mobility_boston_real_v3refined
```

Add `--per-outcome off` (step 2) or `--refine-per-outcome off` (one-shot) for an ablation against the monolithic-only rewrite.

## Expected impact

The per-outcome path directly attacks the source of the calibration regression: strong-outcome churn. Predicted deltas vs the v2_refined run we already have:

- **Calibration**: NLL recovers 0.02–0.04 toward base PO-LEU (back to neutral or slightly better). ECE recovers 0.01–0.03.
- **Ranking**: top3, top5, MRR stay roughly the same as v2_refined or improve marginally — the surgical path doesn't degrade signal that was already working.
- **Cost**: typically lower per pair (one writer call but with N < K sentences requested), maybe ~0.7× the v2_refined LLM spend.

It does **not** close the gap to ST-MLP. The structural finding from the broader mobility analysis still holds: deliberation-style outcome narratives are the wrong inductive bias for trip choice when the actual mechanism is spatial recall + habit. Refine sharpens narratives but can't make narratives the right abstraction.

## Roadmap (other refine improvements from the audit)

The audit found 10 candidate improvements. The ones already implemented are bolded; the rest are future work, ranked by ROI for the calibration regression:

| | Improvement | Implemented? | Effort | Expected impact on NLL |
|---|---|---|---|---|
| 1 | Per-customer cap on failure selection | TODO | S (~20 lines) | High |
| 2 | Lower critic temperature (split writer / critic defaults) | TODO | S (~10 lines) | High |
| 3 | Add grounding axis to critic (3rd score) | TODO | M (~50 lines) | High |
| **4** | **Per-outcome rewrite with splice** | **DONE** (`c205dfb`) | L (~70 lines) | High |
| 5 | Multi-round refine (critic → revise → critic → revise) | TODO | M (~30 lines) | Medium |
| 6 | Domain-specific reviser context (more `optional_fields`) | TODO | M (~40 lines) | Medium |
| 7 | Iterative refinement with retraining (round 1 → train → identify new failures → round 2) | TODO | L (~150 lines, new orchestrator) | High |
| 8 | Hardness-weighted training loss | TODO | M (~50 lines) | High |
| 9 | Refine the negatives' outcomes too | **N/A** (already done in `refine_outcomes.py:224` — iterates all J alts; audit was wrong) | — | — |
| 10 | Cross-family critic by default (`--critic gemini`) | Already supported, just not the default | trivial | Low-Medium |

Quickest follow-ups to implement (S effort, high impact): #1 (per-customer cap) + #2 (lower critic temp). Together these are ~30 lines and should land another 0.01–0.02 of NLL improvement.

## Files touched

- `src/outcomes/prompts.py` — critic schema + per-outcome reviser prompt + builder
- `src/outcomes/refine.py` — `CritiqueResult.weak_outcome_indices` + parser + `revise()` per-outcome path + `refine_outcomes()` skip rule
- `scripts/refine_outcomes.py` — `--per-outcome` flag + bookkeeping + summary telemetry
