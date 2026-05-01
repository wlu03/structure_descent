#!/usr/bin/env bash
# Mobility replication of the Amazon residual_lr_multiplier ablation.
# Sweeps train.residual_lr_multiplier ∈ {1..10, 12, 15, 20, 30} on the
# 30-customer mobility_boston seed-7 records, using the same machinery
# (configs/higher_beta.yaml + --residual-lr-multiplier override).
#
# Single-seed for now: only seed 7 has a fully populated outcomes cache
# under outcomes_cache/mobility_v4/. To extend to 3 seeds we'd need
# cold runs for seeds 11/13 (~$15-30 + ~30 min each at concurrency 32).
#
# Outputs land at results_data/abl_lrmult_mobility/x{M}/ — same shape
# as the Amazon ablation outputs so the analysis-script layout
# carries over.

set -u
set -o pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate
set -a
source .env
set +a

# v4 records were generated with claude-sonnet-4-6; mismatch produces
# 100% cache miss + ~hours of LLM credit per cell.
export ANTHROPIC_MODEL=claude-sonnet-4-6
export PYTHONUNBUFFERED=1
export OUTCOMES_CACHE_PATH="outcomes_cache/mobility_v4/outcomes.sqlite"
export EMBEDDINGS_CACHE_PATH="embeddings_cache/mobility_v4/embeddings.sqlite"

# Drop stray Vertex env vars (irrelevant for this run, but the v4
# launcher pattern always unsets them).
unset GOOGLE_GENAI_USE_VERTEXAI
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_CLOUD_LOCATION

OUT_ROOT="results_data/abl_lrmult_mobility"
mkdir -p "${OUT_ROOT}"

# Records source: leak-corrected v4 residual run on mobility seed 7.
# Has the per-(event, alt) symmetric distance prices baked into
# alt_texts via build_choice_sets, so the residual reads non-leaky x_tab.
SEED=7
RECORDS="reports/mobility_boston_real_v4_residual/records.pkl"
if [ ! -f "${RECORDS}" ]; then
  echo "FATAL: ${RECORDS} not found — run the v4 residual driver first."
  exit 2
fi

run_one() {
  local mult="$1"
  local out="${OUT_ROOT}/x${mult}"
  local log="${OUT_ROOT}/x${mult}.log"
  echo "[$(date -Iseconds)] >>> seed=${SEED} multiplier=${mult} -> ${out}"
  # K=5 + v4_mobility_anchored MUST match what the v4 cache was
  # populated with — different K or prompt_version → different
  # cache_prompt_version → 100% cache miss → cold LLM calls.
  python -m scripts.retrain_with_records \
      --records   "${RECORDS}" \
      --config    configs/higher_beta.yaml \
      --output-dir "${out}" \
      --tabular-residual true \
      --residual-lr-multiplier "${mult}" \
      --K 5 \
      --prompt-version-cascade v4_mobility_anchored \
      --seed "${SEED}" \
      >"${log}" 2>&1
  local rc=$?
  echo "[$(date -Iseconds)] <<< seed=${SEED} multiplier=${mult} exit=${rc}"
  return $rc
}

# 14 multipliers — same set as Amazon: 1..10 + 12, 15, 20, 30.
for mult in 1 2 3 4 5 6 7 8 9 10 12 15 20 30; do
  run_one "${mult}" || echo "[warn] mult=${mult} failed; continuing"
done

echo "[$(date -Iseconds)] mobility ablation done — 14 runs"
