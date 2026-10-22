#!/bin/bash
# Full pipeline; everything is logged to results/run_log.txt
#   features -> feature baselines -> automatic R-peak pipeline -> CNN (3 seeds) -> ensemble + external test -> plots
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results
{
  python -m src.data \
  && python baseline.py \
  && python detection.py \
  && python train.py \
  && python evaluate.py \
  && python scripts/compare_plot.py
  echo "exit code $?"
} 2>&1 | tee results/run_log.txt
