#!/usr/bin/env bash
# Measure all three localization axes across the full model ladder.
#
#   bash scripts/run_ladder.sh [run_name] [n_prompts]
#
# Produces one runs/<run_name>-<model>/results.json per model and an aggregated
# runs/table.md whose columns match Table 1 of the paper. Start small: the
# defaults below are a real run, not a smoke test, and the 1.5B models will take
# a while on a single GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${1:-v1}"
N_PROMPTS="${2:-100}"

MODELS=(
  "HuggingFaceTB/SmolLM2-135M-Instruct"
  "HuggingFaceTB/SmolLM2-360M-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen3-0.6B"
  "meta-llama/Llama-3.2-1B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

for model in "${MODELS[@]}"; do
  echo ""
  echo "=============================================================="
  echo " $model"
  echo "=============================================================="
  # Keep going if one model is gated or out of memory; a partial ladder is
  # still a usable result, and you can fill the gap later.
  python -m src.evaluate \
    --model "$model" \
    --run-name "$RUN_NAME" \
    --n-harmful "$N_PROMPTS" \
    --n-benign "$N_PROMPTS" || echo "[ladder] SKIPPED $model"
done

echo ""
echo "Aggregated table:"
python -m src.evaluate --table
