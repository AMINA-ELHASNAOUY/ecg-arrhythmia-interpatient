#!/bin/bash
# Extract beats, train the random-forest baseline and the CNN; everything is logged to results/run_log.txt
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results
rm -f results/RUN_DONE
{
  python -m src.data && python baseline.py && python train.py && python scripts/compare_plot.py
  echo "exit code $?"
} 2>&1 | tee results/run_log.txt
touch results/RUN_DONE
