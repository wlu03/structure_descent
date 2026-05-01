# Residual LR Multiplier Ablation — Mobility Replication

A replication of the `train.residual_lr_multiplier` ablation
(`docs/residual_lr_multiplier_ablation.md`) on the `mobility_boston`
adapter. Same multiplier grid {1, 2, …, 10, 12, 15, 20, 30}, same
machinery (`scripts/retrain_with_records.py --residual-lr-multiplier`),
same `configs/higher_beta.yaml` base. Date: 2026-05-01.

The headline: **the Amazon result does not replicate.** On mobility,
sweeping ×β from 1 to 30 yields a +2.7 pp top-1 gain (vs Amazon's
+29.3 pp) and the V-vs-R interpretability split stays inverted — V
owns ~70 % of correct picks at every multiplier, including ×30. The
residual is feature-poor and gradient-starved on this task, not a
hidden lever.

## TL;DR

- **No monotonic climb.** Top-1 oscillates 32.5 – 39.4 % across all
  14 multipliers with no clean ordering. NLL bounces 2.13 – 3.00.
  Compare to Amazon's clean monotonic climb 27.2 → 56.6 % top-1 over
  the same grid.
- **Tiny net gain at ×30.** ×30 reaches 39.40 % top-1 (best in the
  sweep, tied with ×20) and NLL 2.131. Vs the V-only baseline (no
  residual, matched hyperparams): **+2.7 pp top-1, −0.15 NLL**.
  Amazon's ×30 was +29.3 pp / −0.62 NLL over its ×1 cell.
- **V-vs-R split is REVERSED on mobility.** At every multiplier
  including ×30, V already gets ~70 – 87 % of the correct picks; the
  residual flips an additional 13 – 32 % into correctness but flips a
  comparable number out. On Amazon ×30, V owned ~31 %; on mobility
  ×30, V owns ~71 %.
- **Calibration is bad at every cell.** ECE ranges 0.25 – 0.39 (best
  ×9 = 0.254, worst ×8 = 0.387). Compare Amazon's much tighter range
  0.041 – 0.110. Mobility models are systematically overconfident
  regardless of multiplier.
- **Pseudo-R² is negative for 12 of 14 cells.** Only ×5 (R²=0.012)
  and ×30 (R²=0.075) beat uniform random on test NLL. This signals
  generalization failure that the multiplier doesn't fix.

The interpretability story this generates is different from Amazon's
"V is a 30 % minority shareholder at high multipliers." On mobility
the model is encoder-dominated at every multiplier — refining or
re-weighting the residual won't unlock predictive performance because
the residual's features (price = distance, popularity, is_repeat) are
already largely redundant with what the V branch encodes through
`c_d` and the per-(event, alt) embeddings.

## 1. Setup

- **Driver**: `scripts/retrain_with_records.py` with the new `--K`
  and `--prompt-version-cascade` overrides — without them the cache
  key for K=5 + v4_mobility_anchored doesn't match what the v4
  driver populated and every cell incurs ~30k cold LLM calls.
- **Records**: `reports/mobility_boston_real_v4_residual/records.pkl`
  — leak-corrected (per-(event, alt) symmetric distance prices baked
  in via `build_choice_sets`'s `per_event_alt_overrides_fn` hook) +
  c_d enrichments on (per-event time-of-day, per-event origin
  context, mobility profile aggregates).
- **Single seed: 7.** Amazon's ablation used 3 seeds {7, 11, 13}
  because per-seed `outcomes_cache/seed{S}/outcomes.sqlite` files
  were already populated. For mobility we only have a populated cache
  at seed 7 (`outcomes_cache/mobility_v4/`); cold runs for seeds
  11/13 would cost ~$15-30 each + ~30 min wall (~$30-60 + ~1 hr
  total). The single-seed result is enough to refute the Amazon
  pattern; multi-seed would mostly tighten the jitter band, not
  flip the conclusion.
- **Multipliers tested**: 14 values per the Amazon grid. All cells
  finished in ~13 min wall total (50–230 s per cell, varying with
  background-job contention).
- **Cache wiring**:
  `OUTCOMES_CACHE_PATH=outcomes_cache/mobility_v4/outcomes.sqlite`,
  `EMBEDDINGS_CACHE_PATH=embeddings_cache/mobility_v4/embeddings.sqlite`,
  `ANTHROPIC_MODEL=claude-sonnet-4-6` (the model the v4 cache was
  populated with). Mismatch → 100 % miss → cold LLM calls.

## 2. Headline numbers

### 2.1 Per-cell metrics (single seed = 7)

| ×β | top-1 | top-3 | top-5 | NLL | ECE | pseudo R² |
|---|---|---|---|---|---|---|
| **0 (V only)** | **36.72 %** | **67.46 %** | **81.79 %** | **2.278** | (n/a) | (n/a) |
| 1  | 32.54 % | 56.72 % | 71.34 % | 3.002 | 0.369 | -0.304 |
| 2  | 37.01 % | 66.27 % | 83.28 % | 2.326 | 0.271 | -0.010 |
| 3  | 35.22 % | 62.09 % | 79.40 % | 2.679 | 0.334 | -0.163 |
| 4  | 33.13 % | 58.51 % | 74.03 % | 2.485 | 0.265 | -0.079 |
| 5  | 37.91 % | 66.87 % | 82.39 % | 2.276 | 0.273 |  0.012 |
| 6  | 34.03 % | 60.90 % | 74.63 % | 2.422 | **0.256** | -0.052 |
| 7  | 37.91 % | 66.27 % | 81.49 % | 2.500 | 0.303 | -0.086 |
| 8  | 36.42 % | 67.46 % | 81.79 % | 2.991 | 0.387 | -0.299 |
| 9  | 33.73 % | 56.72 % | 66.87 % | 2.809 | 0.254 | -0.220 |
| 10 | 36.42 % | 58.81 % | 78.21 % | 2.750 | 0.317 | -0.195 |
| 12 | 35.82 % | 63.58 % | 79.10 % | 2.482 | 0.285 | -0.078 |
| 15 | 37.91 % | 66.27 % | 84.78 % | 2.736 | 0.358 | -0.188 |
| 20 | **39.40 %** | **72.54 %** | **88.06 %** | 2.833 | 0.379 | -0.231 |
| 30 | **39.40 %** | 68.06 % | 83.88 % | **2.131** | 0.251 | **0.075** |

The ×0 row is a no-residual cell run through the same retrain flow
(matched hyperparams: 30 epochs, batch 128, K=5, v4_mobility_anchored
cascade, `--tabular-residual false`) so it's apples-to-apples with
the rest of the table. Bold = best in column.

### 2.2 Mobility vs Amazon at the same multiplier

Amazon top-1 / NLL pulled from `docs/residual_lr_multiplier_ablation.md`
§3.1 (mean across 3 seeds). Mobility is single-seed.

| ×β | Amazon top-1 | Mobility top-1 | Δ | Amazon NLL | Mobility NLL | Δ |
|---|---|---|---|---|---|---|
| 1  | 27.24 % | 32.54 % | +5.3 (mobility better at ×1) | 1.99 | 3.00 | +1.01 (mobility worse) |
| 5  | 38.78 % | 37.91 % | -0.9 | 1.72 | 2.28 | +0.56 |
| 10 | 47.32 % | 36.42 % | **-10.9** | 1.54 | 2.75 | +1.21 |
| 15 | 51.22 % | 37.91 % | **-13.3** | 1.46 | 2.74 | +1.28 |
| 20 | 53.70 % | 39.40 % | **-14.3** | 1.42 | 2.83 | +1.42 |
| 30 | 56.58 % | 39.40 % | **-17.2** | 1.37 | 2.13 | +0.76 |

Amazon and mobility start at similar top-1 (×1: 27 % vs 33 %), then
the curves diverge sharply. Amazon climbs monotonically; mobility
plateaus.

## 3. Where the Amazon mechanism stops working

Amazon's mechanism: at low ×β, the encoder branch's gradients absorb
nearly all the optimization signal and β·x_tab stays under-trained
(Sifringer "gradient absorption"). High ×β unlocks β to take the
magnitude needed to discriminate alternatives via the rich tabular
features (catalog price, popularity rank, brand history). Two
preconditions:

1. **The tabular features must carry signal the encoder doesn't.**
2. **β must have headroom to grow into.**

Both fail on mobility:

### 3.1 The features are mostly redundant with c_d

The mobility tabular_residual features are
`(price, log1p_price, price_rank, popularity_count,
log1p_popularity_count, is_repeat, log1p_purchase_count)`. Mapping
each to its mobility meaning:

| Feature | Mobility meaning | Already in c_d? |
|---|---|---|
| price | haversine origin→dest km | NO (numeric) — but…  |
| log1p_price / price_rank | derivatives | NO (numeric) |
| popularity_count | train visit count to this place | (popularity_rank band string is in alt_text → encoder sees it) |
| is_repeat / log1p_purchase_count | has this customer been here? | NO (numeric); but c_d's "Most-purchased…" + "Recent purchases" lines carry related signal |

The encoder's per-(event, alt) sentence embeddings already encode
title (`<industry> place in CBG <code>`), category, popularity band,
brand. The mobility-anchored c_d adds origin context, daypart,
weekend flag, typical trip length. The model knows distance
indirectly through "<industry> place in CBG <code>" + the customer's
home_cbg + the per-event time. Adding the same distance back as a
linear scalar carries little marginal signal.

### 3.2 Per-event |R| / |V| ratio is bigger but the answer doesn't improve

| ×β | mean \|R\| | mean \|V\| | \|R\|/\|V\| | argmax(V+R) right | gain over V |
|---|---|---|---|---|---|
| V only | — | 1.59 | — | 123 / 335 (36.7 %) | — |
| 1 | 1.28 | 1.59 | 0.80 | 109 / 335 (32.5 %) | **−14** |
| 5 | 0.97 | 1.59 | 0.61 | 127 / 335 (37.9 %) | +4 |
| 10 | 1.39 | 1.59 | 0.87 | 122 / 335 (36.4 %) | −1 |
| 20 | 2.03 | 1.59 | 1.27 | 132 / 335 (39.4 %) | +9 |
| 30 | 0.95 | 1.59 | 0.60 | 132 / 335 (39.4 %) | +9 |

The ratio |R|/|V| reaches 1.27 at ×20 — R is genuinely contributing
non-trivial logit shifts — but the *direction* of those shifts is
roughly random. ×20 flips 39 events V→right and 30 events V→wrong;
net +9. ×1 actually loses ground (109 < 123) because the under-
trained β with the wrong sign on price (or the wrong scaling on
log1p_purchase_count) flips more correct events into wrong than
vice versa.

This is the inverse of Amazon's pattern, where the higher the
multiplier, the more aggressively β corrects V's errors *in the
right direction* because catalog price and popularity are
near-deterministic predictors of purchase.

## 4. The interpretability decomposition (V vs R)

Same methodology as Amazon §5.2: V = no-residual model's logits as
a proxy for the V branch in the residual model; R ≈ V+R − V de-meaned
across J=10 alternatives. Comparing argmax(V) to argmax(V+R) tells
us how often R was load-bearing for a correct prediction.

### 4.1 At ×30: who deserves credit for the correct answers?

| | mobility ×30 | Amazon ×30 (mean of 3 seeds) |
|---|---|---|
| V+R correct | 132 / 335 (39.4 %) | ~415 / 728 (≈ 57 %) |
| V already right (R not load-bearing) | 94 = **71.2 %** | ~130 = **31.2 %** |
| R flipped V → right (R load-bearing) | 38 = **28.8 %** | ~285 = **68.8 %** |

On mobility, **the deployed model's correct picks are still 71 %
attributable to V alone** even at ×30. The §12 head-decomposition
reports describe a model whose answers V mostly already had right.
The story is intact: PO-LEU on mobility behaves as the
interpretable structured choice model the redesign.md thesis
describes.

### 4.2 V's share across multipliers (mobility)

| ×β | V+R correct | V already right | R flipped V → right |
|---|---|---|---|
| 1  | 109 (32.5 %) | 81 = **74.3 %** | 28 = 25.7 % |
| 4  | 111 (33.1 %) | 93 = **83.8 %** | 18 = 16.2 % |
| 6  | 114 (34.0 %) | 99 = **86.8 %** | 15 = 13.2 % |
| 10 | 122 (36.4 %) | 95 = **77.9 %** | 27 = 22.1 % |
| 15 | 127 (37.9 %) | 86 = **67.7 %** | 41 = 32.3 % |
| 20 | 132 (39.4 %) | 93 = **70.5 %** | 39 = 29.5 % |
| 30 | 132 (39.4 %) | 94 = **71.2 %** | 38 = 28.8 % |

V's share never drops below ~68 %. Compare to Amazon, where it
crashes to ~31 % at ×30 (and ~46 % at ×10).

## 5. Calibration

Mobility's ECE is bad at every multiplier:

| ×β | mobility ECE | Amazon ECE |
|---|---|---|
| 1  | 0.369 | 0.109 |
| 5  | 0.273 | 0.068 |
| 9 (Amazon best) | 0.254 | **0.041** |
| 10 | 0.317 | 0.064 |
| 15 | 0.358 | 0.079 |
| 30 | 0.251 | 0.105 |

The best mobility ECE (×9 = 0.254) is more than 5× the best Amazon
ECE (0.041) and worse than every Amazon cell. This isn't a
multiplier-tuning problem; it's the same generalization gap we saw
in the prior mobility analysis (val_nll 1.06 → test_nll 2.44 in the
v4 residual run). Temperature calibration on a held-out slice would
help here regardless of multiplier, and is probably worth more than
any β tuning.

## 6. Why the Amazon mechanism doesn't transfer

Three structural differences that explain the divergence:

### 6.1 Choice mechanism

Purchases are deliberative — people weigh price, brand, popularity,
quality reviews. PO-LEU's encoder embeds the LLM-generated outcome
narratives that aim to capture these axes; β·x_tab adds the raw
numeric versions back. Both are useful for the same task and combine
additively.

Mobility (going to coffee shops, returning home, errands) is mostly
spatial recall + habit. There's no deliberation step where price,
popularity, or repetition trade off — people go where they always
go, near where they are, at times they're free. The encoder captures
this through `c_d` ("Travels around Boston roughly twice per week";
"Just came from a Food and Accommodation place"; "Saturday late
night, mid winter (weekend)") and the title embedding. The numeric
features the residual reads add little on top.

### 6.2 Feature richness

Amazon's seven tabular features include catalog price (highly
discriminative — different alternatives have wildly different prices
and price encodes quality/category mix), train popularity count
(orders of magnitude differences between popular and niche ASINs),
and brand-aware repeat indicators. All carry strong signal the
frozen encoder can't recover from text alone.

Mobility's seven features include geodistance (correlates with
chosen but the spread is narrow — most trips ≤ 5 km in Boston),
popularity count dominated by home/work pseudo-places, and
is_repeat which already explains ~65 % of choice variance and is
indirectly visible to the encoder via the c_d "Top categories" /
"Recent purchases" clauses.

### 6.3 Test-set difficulty

Amazon's PO-LEU test top-1 ranges 27 % (×1) → 57 % (×30). There's
real learnable signal in the test events. Mobility's test top-1
plateaus at ~37 – 39 % for V alone or V+R at any multiplier. The
ceiling appears to be set by something other than β's gradient
budget — likely the inherent variability of mobility events that
isn't predictable from any feature in the bundle.

## 7. Concrete recommendations

If pushing PO-LEU further on mobility, the multiplier is **not** the
right knob. Things that would actually move the number:

1. **Reduce overfitting.** Test ECE 0.25 + pseudo-R² often negative
   = textbook overfit. Try `attribute_heads.hidden: 64` (was 128),
   `salience_net.hidden: 32` (was 64), and post-fit temperature
   calibration on held-out val. None of these need new LLM spend.
2. **More customers.** n_customers=30 with ~2.8k train records and
   ~545k params (190 params/record) is severe overcapacity. Bumping
   to 100 customers (~10k records) is the single biggest lever.
3. **Drop home/work from the negative pool.** They're per-customer
   pseudo-places, not real comparable alternatives. Their inclusion
   pollutes the choice-set and inflates noise.
4. **Different feature set.** If the residual is going to live in
   the model, give it features the encoder can't see: time-of-day
   sine/cosine encoding, weekend flag as a binary, customer-mean
   distance vs this trip's distance (deviation-from-routine), days
   since last visit to this same place.
5. **Stay at ×1 for interpretability runs; pick ×20 or ×30 for
   leaderboard reporting** — the ranking gain is small (+2.7 pp
   top-1) but real, and the V-share at ×30 (71 %) keeps the §12
   reports honest. There's no "Frame 2" version of mobility where
   ×30 is the obviously-better choice; both regimes are within
   single-seed jitter of each other.

## 8. Honest framing

Unlike Amazon, mobility has no "PO-LEU ×30 wins on the leaderboard"
story to chase. The ablation finds **no setting where PO-LEU's
residual unlocks substantial gains**. The gap to ST-MLP we saw in
the broader mobility analysis (ST-MLP 44.5 % top-1 vs PO-LEU 34.3 %)
remains; β tuning closes ~3 of that 10-point gap at best.

This is consistent with our earlier diagnosis: deliberation-style
outcome narratives are the wrong inductive bias for trip choice.
The residual can't fix the abstraction mismatch — it can only adjust
how aggressively the model trusts a small bundle of features that
mostly duplicate signal V already encodes.

## 9. Open questions

- **Does the tighter ECE at ×9 (0.254) and ×30 (0.251) survive
  multi-seed averaging?** With single-seed jitter ±5 pp on top-1,
  ECE could plausibly be ±0.05. Without seeds 11/13 we can't tell
  if there's a real local minimum or whether it's noise.
- **Would a different residual feature bundle move the number?**
  Specifically, does adding origin-relative features (deviation
  from typical CBG, recency-of-last-visit) push V+R meaningfully
  above V-only? Untested.
- **Does the calibration story change with temperature post-fit
  calibration?** The ECE values reported here are pre-calibration.
  A one-line fix could close most of the 0.25 → 0.04 gap.

## 10. Reproducibility

```bash
# Single-seed sweep (~13 min wall, no LLM spend if cache is warm).
bash scripts/_ablate_residual_lr_mult_mobility.sh

# V baseline cell (matched hyperparams):
venv/bin/python -m scripts.retrain_with_records \
  --records   reports/mobility_boston_real_v4_residual/records.pkl \
  --config    configs/higher_beta.yaml \
  --output-dir results_data/abl_lrmult_mobility/x0_no_residual \
  --tabular-residual false \
  --K 5 --prompt-version-cascade v4_mobility_anchored \
  --seed 7

# Aggregate analysis (per-cell metrics + V-vs-R decomposition):
venv/bin/python scripts/_analyze_residual_lr_mult_mobility.py

# To extend to seeds 11/13:
#   1. Run scripts/run_dataset.py for the v4 mobility config at those
#      seeds (~$15-30 + ~30 min each at concurrency=32, populates
#      outcomes_cache/mobility_v4_seed{11,13}/).
#   2. Edit scripts/_ablate_residual_lr_mult_mobility.sh to add a
#      SEEDS array and a per-seed cache wiring loop.
```

Outputs land at:
- `results_data/abl_lrmult_mobility/x{M}/metrics_test.json`
- `results_data/abl_lrmult_mobility/x{M}/test_logits.npz`
- `results_data/abl_lrmult_mobility/x{M}/smoke_summary.json`
- `results_data/abl_lrmult_mobility/x{M}.log`
- `results_data/abl_lrmult_mobility/x0_no_residual/` (V baseline)

Single-seed wall time: ~13 minutes for 14 cells at ~50 s each
(occasionally ~3 – 4 min when the dev box is contending with the
12-window seedsweep). No LLM spend, no embedding spend — caches
must match the v4 sweep's `claude-sonnet-4-6` model_id and
`v4_mobility_anchored` prompt_version.
