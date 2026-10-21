#!/bin/bash
# Scores a44_long30 and a46_nogen30 on the held-out years, one job per (arm, seed).
# run_poc_eval_measure.sh pins --constraint=milan: CPU families disagree, so a
# scoring job without it is void.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }

for arm in a46_nogen30 a44_long30; do          # a46 first — it is the candidate headline
  for s in 1 2 3; do
    J=$(sbatch --parsable --job-name=sc_${arm}_s${s} \
      --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$arm,POC_SEEDS=$s \
      scripts/eval/run_poc_eval_measure.sh)
    echo "$arm s$s = $J"
  done
done
