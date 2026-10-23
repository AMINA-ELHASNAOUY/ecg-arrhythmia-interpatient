#!/bin/bash
# Full pipeline; everything is logged to results/run_log.txt
#   features -> feature baselines -> automatic R-peak pipeline -> CNNs (lead A, two leads) ->
#   AF detector + ensemble + external tests -> plots
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results
{
  python -m src.data \
  && python baseline.py \
  && python detection.py \
  && python train.py --leads 1 \
  && python train.py --leads 2 \
  && python evaluate.py \
  && python scripts/compare_plot.py
  echo "exit code $?"
} 2>&1 | tee results/run_log.txt
