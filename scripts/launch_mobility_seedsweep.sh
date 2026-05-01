#!/bin/bash
# Launch a 6-window tmux session running PO-LEU on mobility_boston across
# 3 seeds × 2 LLM providers (Gemini 3 Flash Preview + o4-mini).
#
# Each window writes to its own per-seed-per-provider output dir + cache
# so SQLite WAL contention can't deadlock parallel runs. Provider env
# vars (LLM_PROVIDER + GEMINI_MODEL / OPENAI_MODEL) are set per-window
# before run_dataset.py starts.
#
# Usage:
#   bash scripts/launch_mobility_seedsweep.sh
#   tmux attach -t mobility_seedsweep
#
# Inside tmux:
#   Ctrl-b n         next window
#   Ctrl-b p         prev window
#   Ctrl-b w         list windows
#   Ctrl-b d         detach (runs keep going)
#
# Each window's command is also written to reports/seed_<seed>_<model>/run.cmd
# so it can be re-launched manually if a window dies.

set -e

SESSION=mobility_seedsweep
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Single source of truth for the (seed, provider, model) sweep.
# To extend: add rows; window names auto-derive from the model name.
SWEEP=(
  "0|gemini|gemini-3-flash-preview"
  "1|gemini|gemini-3-flash-preview"
  "2|gemini|gemini-3-flash-preview"
  "0|openai|o4-mini"
  "1|openai|o4-mini"
  "2|openai|o4-mini"
)

# Knobs shared across all windows. Match the Anthropic baseline so the
# leaderboard apples-to-apples slot is identical except for the LLM.
N_CUSTOMERS=30
N_EPOCHS=20
BATCH_SIZE=32
K=5
PROMPT_CASCADE="v4_mobility_anchored"
MAX_CONCURRENCY=32
MAX_TOKENS=4000   # reasoning-capable models burn 1-2k thinking tokens

# Pre-create the session detached, with a no-op control window we'll
# leave alone (tmux requires at least one window to exist).
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "control" "echo 'mobility_seedsweep control window — Ctrl-b w to list workers'; bash"

for entry in "${SWEEP[@]}"; do
  IFS='|' read -r SEED PROVIDER MODEL <<<"$entry"
  TAG="seed_${SEED}_${MODEL}"
  WIN="${PROVIDER:0:3}${SEED}_${MODEL}"
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

  read -r -d '' CMD <<EOF || true
set -o allexport; source .env; set +o allexport
export LLM_PROVIDER=${PROVIDER}
export ${PROVIDER_ENV}
export OUTCOMES_CACHE_PATH=${CACHE_DIR_O}/outcomes.sqlite
export EMBEDDINGS_CACHE_PATH=${CACHE_DIR_E}/embeddings.sqlite
export MAX_CONCURRENT_LLM_CALLS=${MAX_CONCURRENCY}
export OUTCOMES_MAX_TOKENS=${MAX_TOKENS}
export PYTHONUNBUFFERED=1

# Drop any stray Vertex env vars so the api_key path stays unambiguous.
unset GOOGLE_GENAI_USE_VERTEXAI
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_CLOUD_LOCATION

echo "[\$(date '+%H:%M:%S')] starting ${TAG} (provider=${PROVIDER} model=${MODEL} seed=${SEED})"
venv/bin/python scripts/run_dataset.py \\
  --adapter mobility_boston \\
  --n-customers ${N_CUSTOMERS} --seed ${SEED} \\
  --K ${K} --prompt-version-cascade ${PROMPT_CASCADE} \\
  --add-event-time-to-c-d --add-event-origin-to-c-d \\
  --tabular-residual false \\
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

  tmux new-window -t "$SESSION" -n "$WIN" "bash -c '$(echo "$CMD" | sed "s/'/'\\\\''/g")'"
done

cat <<EOF

Launched 6-window tmux session: $SESSION
  - 3× Gemini-3-Flash-Preview seeds (0, 1, 2)
  - 3× o4-mini seeds (0, 1, 2)

Each window's logs:    reports/seed_<seed>_<model>/run.log
Each window's command: reports/seed_<seed>_<model>/run.cmd
Each window's outputs: reports/seed_<seed>_<model>/

Attach with:
  tmux attach -t $SESSION

Inside tmux:
  Ctrl-b w     list windows
  Ctrl-b n/p   next / prev window
  Ctrl-b d     detach (runs keep going in background)

Cost ballpark for the full sweep (n_customers=30):
  Gemini × 3   ≈ \$15-30 (gemini-3-flash-preview is cheap)
  o4-mini × 3  ≈ \$200-400 (reasoning model, 4-8x the tokens)

Wall time per run with concurrency=$MAX_CONCURRENCY: ~60-120 min
(o4-mini may be slower; reasoning latency)

Kill the whole sweep:
  tmux kill-session -t $SESSION
EOF
