# PO-LEU Evaluation Runbook

Operational guide for `run_full_evaluation.sh` — the one-command sweep that
produces the paper-grade leaderboard comparing PO-LEU against every
baseline, ablation, and LLM competitor on the Amazon dataset.

---

## 1. What the sweep produces

One command writes a seed-aggregated leaderboard with **19 rows** per seed:

| Category | Rows |
|---|---|
| Reference anchor | `Popularity` |
| Tabular baselines | `LASSO-MNL`, `RandomForest`, `GradientBoosting`, `MLP`, `Bayesian-ARD`, `DUET` |
| Ablation | `ST-MLP` ("PO-LEU minus narratives") |
| Zero-shot LLM | `ZeroShot-Claude-Sonnet-4.6`, `ZeroShot-Claude-Opus-4.6`, `ZeroShot-Gemini-2.5-Pro`, `ZeroShot-Gemini-2.5-Flash`, `ZeroShot-GPT-5` |
| Few-shot ICL LLM | `FewShot-ICL-Claude-Sonnet-4.6`, `FewShot-ICL-Claude-Opus-4.6`, `FewShot-ICL-Gemini-2.5-Pro`, `FewShot-ICL-Gemini-2.5-Flash`, `FewShot-ICL-GPT-5` |
| The protagonist | `PO-LEU` |

Every row is scored with identical tie-break, NLL formula, AIC/BIC, ECE,
and pseudo-R² via `src/eval/metrics.py` so rows are directly comparable.
Columns: `top1`, `top5`, `mrr`, `test_nll`, `aic`, `bic`, `n_params`,
`fit_seconds`, `nll_uplift_vs_popularity`.

Three seeds yield three such leaderboards plus an aggregated summary
(mean ± std per row) for the paper's main table.

---

## 2. Prerequisites (one-time setup)

### 2.1 Credentials — `.env` at project root

```bash
# Anthropic — Claude Sonnet 4.6 + Opus 4.6
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI — GPT-5 (client routes max_completion_tokens for reasoning models)
OPENAI_API_KEY=sk-proj-...
# OPENAI_ORGANIZATION=org-...  # optional

# Google Vertex AI (Gemini) — ADC, no API key
GOOGLE_CLOUD_PROJECT=en-decision-modeling-bffc
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
```

`.env` must be in project root. Add to `.gitignore` if not already.

### 2.2 Google ADC (one command, one time per machine)

```bash
gcloud auth application-default login
gcloud config set project en-decision-modeling-bffc
gcloud services enable aiplatform.googleapis.com
```

Verify: `gcloud auth application-default print-access-token` prints a token.

### 2.3 Python deps

```bash
pip install -r requirements.txt
```

Installs `anthropic`, `openai>=1.0`, `google-genai>=1.51.0` plus the
research stack (torch, sentence-transformers, numpy, pandas, etc.).

### 2.4 Dataset

`configs/datasets/amazon.yaml` points at the local Amazon CSVs. If
they're missing, downstream runs fail at the load stage — not something
this script can recover from.

---

## 3. Quick start

```bash
chmod +x run_full_evaluation.sh
./run_full_evaluation.sh
```

Defaults:

| Knob | Default |
|---|---|
| `N_CUSTOMERS` | 50 |
| `SEEDS` | `"7 11 13"` |
| `N_EPOCHS` | 30 (PO-LEU; early stop patience=5) |
| `SPLIT_MODE` | `temporal` |
| `OUTPUT_DIR` | `results_data` |
| `DATASET_YAML` | `configs/datasets/amazon.yaml` |

### Common variants

```bash
# Pilot — 1 seed, full 50 customers, ~8 h, ~$160
SEEDS="7" ./run_full_evaluation.sh

# Tiny smoke — catches the most common failures in ~1 h for ~$30
N_CUSTOMERS=20 SEEDS="7" ./run_full_evaluation.sh

# Cold-start robustness — same compute, "test on unseen users"
SPLIT_MODE=cold_start OUTPUT_DIR=results_cold_start ./run_full_evaluation.sh

# Smaller Opus-free sweep (edit LLM_MODEL_SWEEP in src/baselines/run_all.py
# to drop Claude-Opus-4.6 first — saves ~40% of the LLM cost)
./run_full_evaluation.sh
```

### Long-running safety — always use tmux

```bash
tmux new -s sweep
./run_full_evaluation.sh
# Ctrl-B D to detach; `tmux attach -t sweep` to return.
```

Without tmux, closing the terminal kills the Python process mid-sweep.

---

## 4. Pre-flight — what the script checks before spending

The script fails fast with exit codes 2–4 if any of the following are wrong:

| Check | Fail mode | Fix |
|---|---|---|
| `.env` present at project root | exit 2 | `cp .env.example .env` and fill in |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_CLOUD_PROJECT` in env | python SystemExit | `source .env` (the script does this but verifies) |
| ADC token valid | exit 3 | `gcloud auth application-default login` |
| `anthropic`, `openai`, `google.genai` importable | python SystemExit | `pip install -r requirements.txt` |
| `choice_set_size` in YAML == 4 | interactive warn | see §8 "J mismatch" below |

`results_data/` (or whatever `OUTPUT_DIR` is set to) is auto-created.

---

## 5. What happens per seed

For each seed in `SEEDS`, two commands run sequentially:

### Step 1 — `scripts/run_dataset.py` (train PO-LEU)

```
python -m scripts.run_dataset \
    --adapter amazon \
    --n-customers ${N_CUSTOMERS} --seed ${SEED} \
    --split-mode ${SPLIT_MODE} --n-epochs ${N_EPOCHS} \
    --output-dir ${OUTPUT_DIR}/poleu_${N_CUSTOMERS}cust_seed${SEED}
```

Loads → cleans → joins survey → state_features → split → Appendix-C
subsample → build_choice_sets → LLM outcome generation (this is the paid
step) → encode → train three heads up to 30 epochs with early stop
patience=5 → write `test_logits.npz` + checkpoint + metadata.

Full stdout/stderr is teed to `${OUTPUT_DIR}/poleu_seed${SEED}.log`.

### Step 2 — `scripts/run_baselines.py` (leaderboard + PO-LEU fusion)

```
python -m scripts.run_baselines \
    --dataset-yaml configs/datasets/amazon.yaml \
    --n-customers ${N_CUSTOMERS} --seed ${SEED} \
    --split-mode ${SPLIT_MODE} \
    --output-dir ${OUTPUT_DIR} --tag main_seed${SEED} \
    --poleu-logits ${OUTPUT_DIR}/poleu_${N_CUSTOMERS}cust_seed${SEED}/test_logits.npz
```

Rebuilds the same `(train, val, test)` batches with the same seed (so
splits match PO-LEU's), runs every registered baseline, folds the
PO-LEU logits in as one more row, writes the three leaderboard
artifacts.

Logs land in `${OUTPUT_DIR}/baselines_seed${SEED}.log`.

**Per-seed error isolation**: if PO-LEU training fails, the script logs
the error and skips to the next seed rather than aborting.

---

## 6. Aggregation across seeds

After all seeds finish, the script computes mean ± std per row across
all produced `baselines_leaderboard_main_seed*.json` files and writes
`${OUTPUT_DIR}/leaderboard_summary.json`:

```json
{
  "PO-LEU": {
    "top1_mean": 0.412,
    "top1_std":  0.008,
    "nll_mean":  0.823,
    "nll_std":   0.012,
    "uplift_mean": 0.625,
    "uplift_std":  0.014,
    "n_seeds": 3
  },
  "ZeroShot-Claude-Opus-4.6": { ... },
  ...
}
```

It also prints a sorted-by-uplift human-readable table to stdout so you
can eyeball the ranking without touching the JSON.

---

## 7. Output layout

```
results_data/
├── poleu_50cust_seed7/
│   ├── test_logits.npz             ← consumed by --poleu-logits
│   ├── checkpoint.pt               ← PO-LEU weights
│   └── metadata.json               ← training config + best_val_nll
├── poleu_50cust_seed11/
├── poleu_50cust_seed13/
├── baselines_leaderboard_main_seed7.json
├── baselines_leaderboard_main_seed7.csv
├── baselines_leaderboard_main_seed7.txt       ← human-readable table
├── baselines_leaderboard_main_seed11.{json,csv,txt}
├── baselines_leaderboard_main_seed13.{json,csv,txt}
├── leaderboard_summary.json                   ← mean ± std across seeds
├── poleu_seed7.log                            ← per-seed stdout+stderr
├── poleu_seed11.log
├── poleu_seed13.log
├── baselines_seed7.log
├── baselines_seed11.log
└── baselines_seed13.log
```

### Per-row columns in the leaderboard JSON/CSV

| Key | Meaning |
|---|---|
| `name` | Display name — e.g. `ZeroShot-Claude-Sonnet-4.6`, `PO-LEU` |
| `status` | `ok` / `unavailable` / `errored` |
| `top1` | Top-1 accuracy. Chance = 1/J (=0.25 at J=4) |
| `top5` | Top-5 accuracy (≡1.0 if J=4) |
| `mrr` | Mean reciprocal rank |
| `test_nll` | Mean per-event NLL (natural log) |
| `aic` / `bic` | Information criteria, natural-log |
| `n_params` | Effective param count — 0 for frozen-LLM baselines |
| `fit_seconds` | Wall-clock fit time |
| `nll_uplift_vs_popularity` | `popularity_nll − row_nll` in nats. **The bottom line.** |
| `description` | Row-specific state — e.g. `FewShot-ICL n_shots=3 K=4 cold_start=42/500` |
| `error` / `traceback` | Populated when `status != ok` |

---

## 8. Known gotcha — J=4 vs J=10

`configs/datasets/amazon.yaml` controls `choice_set_size`. The 10 LLM
baselines are **hard-coded to J=4** (the A/B/C/D letter set + K=4 Latin
square permutation debiasing). If the YAML says `choice_set_size: 10`,
every LLM row will show:

```
errored — requires n_alternatives=4 (got 10)
```

The script warns about this before starting and asks for confirmation.
Tabular + ablation + PO-LEU rows still populate at any J. See
`docs/llm_baselines/j_size_decision.md` for the full methodological
write-up (bias scaling, power trade-off, literature survey).

**Recommendation for the paper**: set `choice_set_size: 4` in
`amazon.yaml` so all 19 rows populate. Tabular uplift numbers will be
smaller than at J=10 (easier negatives), but that's a fair trade for
getting the LLM comparison working in the same leaderboard.

---

## 9. Cost + time budget

Estimates for 50 customers per seed (roughly 10k choice sets, ~1k test events):

| Step | Cost | Wall-clock |
|---|---|---|
| PO-LEU training (Sonnet outcome gen + head fit) | ~$25 | ~30 min |
| Popularity + 6 tabular + ST-MLP | $0 | ~1 min |
| `ZeroShot-Claude-Sonnet-4.6` | ~$3 | ~15 min |
| `ZeroShot-Claude-Opus-4.6` | ~$18 | ~30 min |
| `ZeroShot-Gemini-2.5-Pro` | ~$2 | ~15 min |
| `ZeroShot-Gemini-2.5-Flash` | ~$0.50 | ~10 min |
| `ZeroShot-GPT-5` | ~$6 | ~15 min |
| 5× `FewShot-ICL` (slightly more input tokens than Zero-Shot) | ~$35 | ~1.5 h |
| **Per-seed total** | **~$90–160** | **~4–8 h** |

×3 seeds → **~$400, ~24 h wall-clock**. Budget up to $500 for safety.

Drop Opus to save ~$60 per seed. Drop Few-Shot to save ~$35 per seed.

---

## 10. Troubleshooting

### `unavailable — LLM client factory failed`

Factory failed to produce a real client. Matrix:

| Provider | Most common cause | Fix |
|---|---|---|
| `Gemini-*` | ADC not logged in | `gcloud auth application-default login` |
| `Claude-*` | `ANTHROPIC_API_KEY` not exported | `source .env` in the same shell |
| `GPT-5` | `OPENAI_API_KEY` not exported | `source .env` in the same shell |
| All LLMs | Wrong shell (env vars not loaded) | re-source `.env`, re-run |

### `errored — requires n_alternatives=4 (got 10)`

See §8. Set `choice_set_size: 4` in `configs/datasets/amazon.yaml`.

### `errored — PRODUCTION SAFETY: ... StubLLMClient`

Should never happen via the script — `LLM_CLIENT_FACTORIES` always
injects a real client. If you see this, the `baseline_kwargs` override
path may have been misused somewhere. File a bug.

### PO-LEU training dies mid-epoch with OOM

`N_EPOCHS` doesn't help. Reduce `batch.batch_size` via `--batch-size 32`
(or lower) in the `run_dataset` call. Edit `run_full_evaluation.sh` to
pass the flag.

### One seed fails; others should continue

The script's per-seed error handling logs the failure and advances. The
aggregated summary at the end drops failed seeds and reports `n_seeds`
< 3 for any row that wasn't successfully run all 3 times.

### Full sweep was killed by closing the terminal

Use `tmux`. Every long run should be inside a detachable session.

### Dedup fallback warnings cluttering the baseline log

Expected. `build_choice_sets` cycles when a customer's available ASIN
pool is smaller than J-1 (common for late events in a customer's
timeline). Informational, not a bug. The leaderboard numbers are still
valid.

### `Popularity` row shows top1 < 1/J

Expected artifact. Negative sampling in `build_choice_sets` is
popularity-weighted; the Popularity baseline ranks by popularity; the
sampler therefore systematically produces hard negatives for the
Popularity predictor. That's why Popularity is a **reference anchor**,
not a competitive baseline — the `nll_uplift_vs_popularity` column is
the meaningful comparison point.

---

## 11. Viewing results

```bash
# Quick table (main seed 7)
cat results_data/baselines_leaderboard_main_seed7.txt

# Sorted by uplift, all seeds (from Python)
python - <<'PY'
import json
summary = json.load(open("results_data/leaderboard_summary.json"))
for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["uplift_mean"]):
    print(f"{name:<30s}  uplift={s['uplift_mean']:+.3f}±{s['uplift_std']:.3f}  "
          f"top1={s['top1_mean']:.3f}±{s['top1_std']:.3f}  "
          f"n={s['n_seeds']}")
PY

# Tail live progress of a running seed
tail -f results_data/poleu_seed7.log
tail -f results_data/baselines_seed7.log
```

---

## 12. Interpretation decision tree

| Observation | What it means |
|---|---|
| PO-LEU's `nll_uplift_vs_popularity` near 0 | Pipeline broken or dataset too small — investigate |
| LASSO-MNL uplift ≈ PO-LEU uplift | The 6 tabular features explain everything; the LLM stack is decorative |
| **ST-MLP uplift ≈ PO-LEU uplift** | **Narratives are decorative — save the LLM generation cost** (most diagnostic single comparison) |
| ZeroShot-\* uplift ≈ PO-LEU uplift | PO-LEU's training is decorative; direct prompting wins |
| FewShot-ICL uplift ≈ PO-LEU uplift | In-context personalization substitutes for the learned head |
| PO-LEU beats all 18 rows | All four components (narratives + encoder + heads + training) pull weight |
| Claude wins, Gemini/GPT weaker | Conclusion is Anthropic-specific — weaken the paper's claim |
| Claude ≈ Gemini ≈ GPT | Robust across providers — strong paper |

Cross-reference with the cold-start run (`SPLIT_MODE=cold_start`) for the
"generalizes to unseen users" story.

---

## 13. What's *not* covered by this script

- **Wave-12 sampling-sensitivity experiment** (uniform / popularity-weighted / log-q-corrected negative sampling). Separate 3×3×3 matrix; see `docs/pipeline_memo.tex` §6.
- **K-fold cross-validation**. `src/data/split.py::kfold_customer_cv` is available but doesn't have a CLI driver yet.
- **Hyperparameter search**. Fixed values in `configs/default.yaml`.
- **Interpretability / attention-head inspection**. Separate reporting pipeline under `src/eval/`.
- **Delphos / Paz-VNS** (Phase-2 baselines). Require porting `old_pipeline/src/dsl.py` + `inner_loop.py` first.

Each is a separate workstream — see the relevant task IDs in the
project task tracker.

---

## 14. Canonical command recap

The single command that kicks off the publication-grade main result:

```bash
tmux new -s sweep
./run_full_evaluation.sh
# Ctrl-B D to detach, go home, come back 24h later to
# results_data/leaderboard_summary.json
```

Everything else is a knob on top of that.
