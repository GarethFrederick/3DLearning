#!/bin/bash
# Scores a23_zerodose_r2 and a25_recency_r2 on the held-out years. With a19 they
# form a three-way read: a19 (all years + discount), a23 (all years, no
# discount), a25 (recent years only, no discount). All three use the narrow head
# and 30 rounds and score into the default results/scores/poc_eval_measure, so
# the read is like for like. run_poc_eval_measure.sh pins --constraint=milan.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }

for arm in a23_zerodose_r2 a25_recency_r2; do
  for s in ${SEEDS:-1 2 3}; do
    ls Utils/Models/ | grep -q "gt_${arm}_s${s}-ML-epoch" \
      || { echo "  SKIP $arm s$s: no checkpoint"; continue; }
    J=$(sbatch --parsable --job-name=sc_${arm}_s${s} \
      --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$arm,POC_SEEDS=$s \
      scripts/eval/run_poc_eval_measure.sh)
    echo "  $arm s$s -> $J"
  done
done
