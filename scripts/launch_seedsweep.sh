#!/bin/bash
# Launch a 12-window tmux session running PO-LEU across:
#   3 seeds × 2 LLM providers (Gemini-3-Flash-Preview + o4-mini)
#                          × 2 datasets (mobility_boston + amazon)
#
# Each window writes to its own per-(seed, provider, dataset) output dir
# + cache so SQLite WAL contention can't deadlock parallel runs. Per-
# dataset c_d enrichments and prompt cascade are baked in:
#
#   mobility_boston : --K 5 v4_mobility_anchored, event-time + event-
#                     origin enrichments on (matches the leak-corrected
#                     Anthropic baseline)
#   amazon          : --K 5 v3_anchored (financial / health /
#                     convenience / emotional / social purchase axes;
#                     no event-time / event-origin — Amazon doesn't
#                     have those signals)
#
# Usage:
#   bash scripts/launch_seedsweep.sh
#   tmux attach -t poleu_seedsweep
#
# Inside tmux:
#   Ctrl-b w     list windows
#   Ctrl-b n/p   next / prev window
#   Ctrl-b d     detach (runs keep going)
#
# Each window's command is also written to the run dir as run.cmd so it
# can be re-launched manually if a window dies.

set -e

SESSION=poleu_seedsweep
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Single source of truth for the (seed, provider, model, dataset) sweep.
# Format:  seed | provider | model_id | adapter
SWEEP=(
  # Mobility ─────────────────────────────────────
  "0|gemini|gemini-3-flash-preview|mobility_boston"
  "1|gemini|gemini-3-flash-preview|mobility_boston"
  "2|gemini|gemini-3-flash-preview|mobility_boston"
  "0|openai|o4-mini|mobility_boston"
  "1|openai|o4-mini|mobility_boston"
  "2|openai|o4-mini|mobility_boston"
  # Amazon ───────────────────────────────────────
  "0|gemini|gemini-3-flash-preview|amazon"
  "1|gemini|gemini-3-flash-preview|amazon"
  "2|gemini|gemini-3-flash-preview|amazon"
  "0|openai|o4-mini|amazon"
  "1|openai|o4-mini|amazon"
  "2|openai|o4-mini|amazon"
)

# Knobs shared across all windows. n_customers=30 matches the existing
# Anthropic baseline so leaderboard rows are apples-to-apples.
N_CUSTOMERS=30
N_EPOCHS=20
BATCH_SIZE=32
K=5
# Per-window concurrency. With 12 windows × 16 = 192 concurrent calls
# total (96 per provider given the 6/6 split) — a deliberate downshift
# from the 32-per-window default that worked for a single-provider run
# on Anthropic. Bump back to 32 if you're not seeing rate-limit retries
# in the logs and want faster turnaround.
MAX_CONCURRENCY=16
# Reasoning-capable models burn 1-2k tokens on internal thinking before
# any visible output. 4000 covers o4-mini and gemini-3 comfortably.
MAX_TOKENS=4000

# Refine pipeline knobs (run_dataset.py --refine). When ENABLE_REFINE is
# 'on', every window does round1 (initial train) → identify worst val
# events → critic+reviser refine → round2 (retrain with v3_refined
# cascade prepended). Round-1 artifacts land in <output-dir>/round1/;
# final artifacts overwrite <output-dir>/ as if a normal run.
#
# Refine adds REFINE_TOP_K × J=10 critic calls + a smaller number of
# reviser calls per window. Cost overhead is small compared to round
# 1's outcome generation.
ENABLE_REFINE=on
REFINE_TOP_K=30
REFINE_ACCEPT_THRESHOLD=4
REFINE_PER_OUTCOME=on
# REFINE_CRITIC: writer (default, reuses the LLM_PROVIDER client — cheap
# but family-correlated) | anthropic | gemini | openai. Cross-family
# critic costs more API calls but gives independent scoring.
REFINE_CRITIC=writer

# Pre-create the session detached, with a no-op control window we'll
# leave alone (tmux requires at least one window to exist).
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "control" \
  "echo 'poleu_seedsweep control window — Ctrl-b w to list workers'; bash"

n_launched=0
for entry in "${SWEEP[@]}"; do
  IFS='|' read -r SEED PROVIDER MODEL DATASET <<<"$entry"
  TAG="seed_${SEED}_${MODEL}_${DATASET}"
  WIN="${PROVIDER:0:3}${SEED}_${DATASET:0:3}_${MODEL:0:14}"
  OUT_DIR="reports/${TAG}"
  CACHE_DIR_O="outcomes_cache/${TAG}"
  CACHE_DIR_E="embeddings_cache/${TAG}"
  LOG_PATH="${OUT_DIR}/run.log"
  CMD_PATH="${OUT_DIR}/run.cmd"

  mkdir -p "$OUT_DIR" "$CACHE_DIR_O" "$CACHE_DIR_E"

  if [ "$PROVIDER" = "gemini" ]; then
    PROVIDER_ENV="GEMINI_MODEL=${MODEL}"
  else
    PROVIDER_ENV="OPENAI_MODEL=${MODEL}"
  fi

  # Per-dataset prompt cascade and c_d enrichments.
  if [ "$DATASET" = "mobility_boston" ]; then
    PROMPT_CASCADE="v4_mobility_anchored"
    DATASET_FLAGS="--add-event-time-to-c-d --add-event-origin-to-c-d"
  else
    PROMPT_CASCADE="v3_anchored"
    DATASET_FLAGS=""
  fi

  # Refine flags — applied uniformly across all windows when enabled.
  if [ "$ENABLE_REFINE" = "on" ]; then
    REFINE_FLAGS="--refine \
  --refine-top-k ${REFINE_TOP_K} \
  --refine-accept-threshold ${REFINE_ACCEPT_THRESHOLD} \
  --refine-per-outcome ${REFINE_PER_OUTCOME} \
  --refine-critic ${REFINE_CRITIC}"
  else
    REFINE_FLAGS=""
  fi

  read -r -d '' CMD <<EOF || true
set -o allexport; source .env; set +o allexport
export LLM_PROVIDER=${PROVIDER}
export ${PROVIDER_ENV}
export OUTCOMES_CACHE_PATH=${CACHE_DIR_O}/outcomes.sqlite
export EMBEDDINGS_CACHE_PATH=${CACHE_DIR_E}/embeddings.sqlite
export MAX_CONCURRENT_LLM_CALLS=${MAX_CONCURRENCY}
export OUTCOMES_MAX_TOKENS=${MAX_TOKENS}
export PYTHONUNBUFFERED=1

# Drop any stray Vertex env vars so the api_key path stays unambiguous
# (otherwise google-genai routes api-key calls to Vertex's aiplatform
# endpoint, which 401s on api-key auth).
unset GOOGLE_GENAI_USE_VERTEXAI
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_CLOUD_LOCATION

echo "[\$(date '+%H:%M:%S')] starting ${TAG}"
echo "  provider=${PROVIDER} model=${MODEL} seed=${SEED} dataset=${DATASET}"
echo "  prompt_cascade=${PROMPT_CASCADE}"
echo "  refine=${ENABLE_REFINE}"
venv/bin/python scripts/run_dataset.py \\
  --adapter ${DATASET} \\
  --n-customers ${N_CUSTOMERS} --seed ${SEED} \\
  --K ${K} --prompt-version-cascade ${PROMPT_CASCADE} \\
  ${DATASET_FLAGS} \\
  --tabular-residual false \\
  ${REFINE_FLAGS} \\
  --n-epochs ${N_EPOCHS} --batch-size ${BATCH_SIZE} \\
  --config configs/default.yaml \\
  --output-dir ${OUT_DIR} \\
  > ${LOG_PATH} 2>&1
RC=\$?
echo "[\$(date '+%H:%M:%S')] ${TAG} exit=\$RC"
echo "tail of log:"
tail -10 ${LOG_PATH}
echo "Press any key to close this window..."
read -n 1
EOF

  printf "%s\n" "$CMD" > "$CMD_PATH"

  tmux new-window -t "$SESSION" -n "$WIN" \
    "bash -c '$(echo "$CMD" | sed "s/'/'\\\\''/g")'"

  n_launched=$((n_launched + 1))
done

cat <<EOF

Launched ${n_launched}-window tmux session: $SESSION
  6× mobility_boston runs (3 Gemini seeds + 3 o4-mini seeds)
  6× amazon runs           (3 Gemini seeds + 3 o4-mini seeds)
  refine pipeline: ${ENABLE_REFINE}  (top-K=${REFINE_TOP_K},
                                      per_outcome=${REFINE_PER_OUTCOME},
                                      critic=${REFINE_CRITIC})

Per-window outputs (with --refine on):
  reports/seed_<seed>_<model>_<dataset>/
    ├─ run.log               live stdout from run_dataset.py
    ├─ run.cmd               exact command (for re-launch if window dies)
    ├─ round1/               round-1 (initial-train) artifacts
    │   ├─ records.pkl
    │   ├─ metrics.json
    │   ├─ metrics_test.json
    │   ├─ val_per_event.json
    │   └─ test_logits.npz
    ├─ failure_events.json   round-1's worst val events
    ├─ refined_outcomes.json critic + reviser bookkeeping
    ├─ records.pkl           round-2 records (cache-warm rebuild)
    ├─ test_logits.npz       FINAL leaderboard-ready logits (round 2)
    ├─ metrics.json          FINAL val metrics
    └─ metrics_test.json     FINAL test metrics

Attach:
  tmux attach -t $SESSION

Inside tmux:
  Ctrl-b w     list windows
  Ctrl-b n/p   next / prev window
  Ctrl-b d     detach (runs keep going in background)

Cost ballpark (n_customers=${N_CUSTOMERS}, refine=${ENABLE_REFINE}):
  Gemini-3-Flash × 6 runs       ≈ \$30-60     (cheap; flash model)
  o4-mini × 6 runs              ≈ \$400-800   (reasoning model)
EOF
if [ "$ENABLE_REFINE" = "on" ]; then
  cat <<EOF
  refine overhead per run:      ≈ +5-15%      (top-K=${REFINE_TOP_K} events
                                                × J=10 alts × 2 critic+
                                                reviser calls; round 2 is
                                                cache-warm, no new outcomes)
EOF
fi
cat <<EOF

Wall time per window with concurrency=${MAX_CONCURRENCY}:
  round 1 (LLM-bound):  60-180 min  (Amazon ~4-5x mobility events)
EOF
if [ "$ENABLE_REFINE" = "on" ]; then
  cat <<EOF
  refine + round 2:     5-15 min    (cache-warm, ~1k LLM calls)
EOF
fi
cat <<EOF

To switch off refine for a single sweep, edit ENABLE_REFINE=off at the
top of this script. To trim cost: reduce N_CUSTOMERS, or drop o4-mini
rows from the SWEEP array.

Kill the whole sweep:
  tmux kill-session -t $SESSION
EOF
