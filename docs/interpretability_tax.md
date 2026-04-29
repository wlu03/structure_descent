# The Interpretability Tax

A two-run experiment that measures the predictive cost of PO-LEU's
attribute decomposition. Branch: `experiment/interpretability-tax`.

## The question

PO-LEU's central claim is that the structural decomposition
(`u_m` attribute heads × `w_m(z_d)` weights × `s_k` salience) yields
**readable utility at no predictive cost**. That claim collapses to
a single comparison: PO-LEU (A0) vs. ConcatUtility (A7), with
everything else held fixed.

If A7 ≈ A0 on NLL / top-1, the decomposition is free and the paper's
thesis holds. If A7 substantially beats A0, the decomposition is paying
a real predictive tax — interpretability is not free at this data scale.

## What's varied vs. what's held fixed

| Component | A0 (PO-LEU) | A7 (ConcatUtility) |
|---|---|---|
| Records (train/val/test) | identical (`records.pkl`) | identical |
| Frozen encoder | `all-mpnet-base-v2` (768) | same |
| LLM-generated narratives (K=3 per alt) | served from outcomes cache | same — cache hit |
| Salience layer (softmax over K) | unchanged | unchanged |
| Cross-entropy loss + regularizers | unchanged | unchanged |
| Optimizer / LR / schedule / early-stop | identical | identical |
| **Backbone (the only difference)** | **M=5 attribute heads + weight net → `Σ_m w_m·u_m(e_k)`** | **single MLP `[e_k; z_d] → ℝ`** |

A7 keeps the narratives and the salience layer; only the per-outcome
utility computation changes. This is the only ablation that isolates
the decomposition's contribution from the LLM-narrative stage.

## How to read the result

Let A0 = PO-LEU's test top-1 / NLL on the held-out 418-event split,
A7 = same for ConcatUtility, ST-MLP = the no-decomposition no-narrative
ceiling (already on the leaderboard).

| Outcome | Interpretation |
|---|---|
| `A0 ≈ A7` (within 1–2 pts top-1, 0.05 NLL) | Decomposition is free. Paper's interpretability claim holds. |
| `A7 ≫ A0` (5+ pts) | Decomposition is the bottleneck. Either heads are dead (we've already seen m0 unused) or M=5 is too tight. |
| `A7 ≈ ST-MLP` (both ≈70%) | Narratives carry the same signal as raw metadata. The whole 40-point gap is structural — fix or drop the decomposition. |
| `A7 ∈ (A0, ST-MLP)` | Both narratives and decomposition cost ~half each. Two-front problem. |

`A0` and `A7` are the load-bearing comparison; `ST-MLP` is the
predictive ceiling and clarifies how much of any A0–A7 gap is
"narratives vs. metadata" vs. "decomposition vs. flat MLP".

## Run protocol

Both runs share the same records and the same outcomes cache, so the
LLM stage is a 100% cache hit on the second run. No new API calls.

```bash
# 1. A0 — vanilla PO-LEU. Writes records.pkl, test_logits.npz.
python scripts/run_dataset.py \
    --adapter amazon \
    --n-customers 100 --seed 7 \
    --n-epochs 30 \
    --output-dir reports/interp_tax_A0_n100_s7

# 2. A7 — ConcatUtility backbone. Same data, same narratives, same training.
#    The outcomes cache is hit for every (c_d, alt_text) pair.
python scripts/run_dataset.py \
    --adapter amazon \
    --n-customers 100 --seed 7 \
    --n-epochs 30 \
    --config configs/ablation_concat_utility.yaml \
    --output-dir reports/interp_tax_A7_n100_s7

# 3. Side-by-side leaderboard on the IDENTICAL test set.
#    Either run's records.pkl works since the data CLI args are identical.
python scripts/run_baselines.py \
    --records-from reports/interp_tax_A0_n100_s7/records.pkl \
    --poleu-logits  reports/interp_tax_A0_n100_s7/test_logits.npz \
    --output-dir    reports/interp_tax_A0_n100_s7
# Repeat for A7 to fold its row in alongside the same baselines.
```

**Cache reuse note.** The outcomes cache is keyed on
`(prompt_version, model_id, c_d, alt_text, seed, K)`. None of those
depend on the model backbone, so A0's narratives serve A7 verbatim.
A 10-customer dry run (`smoke10`) already populated the cache for that
slice; an A0/A7 sweep at the same `--n-customers / --seed` reuses it
end-to-end with zero LLM cost.

## Required precondition

The leaderboard rows must use **the same `n_train` for AIC / BIC
penalty**, otherwise the information criteria are not directly
comparable. Since both A0 and A7 train on the same `records.pkl`
train split, this is automatic. The records.pkl roundtrip (commit
`6743faf`) is the precondition that makes this experiment honest.

## Suggested run tag

`interpretability_tax_n{N}_s{SEED}` — e.g.
`interpretability_tax_n100_s7`. Use it as `--tag` for the
`run_baselines.py` output so the leaderboard JSON / CSV / TXT
filenames carry the experiment identity.

## What this experiment does NOT settle

- **Narrative quality.** A7 still consumes LLM narratives; if the
  narratives are bad, A0 and A7 are both bad. The narrative-vs-metadata
  question is settled only by `A7 vs. ST-MLP`.
- **Other ablations.** A1–A8 each test a different component (attribute
  count, salience, weight normalization). The interpretability-tax
  experiment is the *central* ablation, not the *only* ablation.
- **Customer-scale generalization.** Run at `--n-customers ∈ {10, 50,
  100, 200}` if you want to see whether the tax shrinks with data.
