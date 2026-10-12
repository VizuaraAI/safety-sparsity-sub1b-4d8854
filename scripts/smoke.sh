#!/usr/bin/env bash
# The first thing to run. Downloads the smallest model in the ladder and runs
# all three localization axes on a handful of prompts. A few minutes on CPU.
#
#   bash scripts/smoke.sh
#
# If this prints a table at the end, every hook in the repo works. Expect the
# baseline refusal rate to be low and SLI to possibly come back undefined:
# SmolLM2-135M barely refuses anything, which is itself the first real
# observation of the project, not a failure.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.evaluate \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --run-name smoke \
  --n-harmful 6 \
  --n-benign 6
