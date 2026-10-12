#!/usr/bin/env bash
# The intervention test: three ways of buying safety, measured the same way.
#
#   bash scripts/run_intervention.sh [model] [n_prompts]
#
# Trains full SFT, LoRA and the localized edit on the same data, then evaluates
# all three on the same probes. The output rows are Table 2 of the paper. The
# claim under test: at 1.5B the localized edit matches full SFT on safety at a
# fraction of the capability cost, and that advantage shrinks or inverts below
# 1B.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
N_PROMPTS="${2:-100}"
# Must match the slug built by src/config.py:run_dir, which replaces "/" with "__".
SLUG="${MODEL//\//__}"

echo "[intervention] baseline measurement for $MODEL"
python -m src.evaluate --model "$MODEL" --run-name "intervention-base" \
  --n-harmful "$N_PROMPTS" --n-benign "$N_PROMPTS"

for method in full_sft lora localized; do
  echo ""
  echo "=============================================================="
  echo " $method"
  echo "=============================================================="
  python -m src.train --model "$MODEL" --method "$method"

  CKPT="runs/baseline-${SLUG}-${method}"
  if [ -d "$CKPT" ]; then
    python -m src.evaluate --model "$CKPT" --run-name "intervention-${method}" \
      --n-harmful "$N_PROMPTS" --n-benign "$N_PROMPTS"
  else
    echo "[intervention] checkpoint not found at $CKPT; check src/train.py output path"
  fi
done

echo ""
echo "Aggregated table:"
python -m src.evaluate --table
