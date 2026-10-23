#!/bin/bash
# Re-run only the evaluation steps (no retraining): uses the trained CNN weights in results/.
#   ensembles + AF rule + fusion gate -> recording-level analysis -> plots   (about 1-2 minutes)
cd "$(dirname "$0")/.."
source .venv/bin/activate
{
  python evaluate.py \
  && python analysis.py \
  && python scripts/compare_plot.py
  echo "exit code $?"
} 2>&1 | tee results/eval_log.txt
