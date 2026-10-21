#!/bin/bash
# Refuses a run name that is already in use: two submits under one name write the
# same model, restart and metadata files, and the second reads the first's restart.
# For each seed it checks for a queued or running job, a model file or a restart
# file with the name. Usage: bash scripts/arms/gates/check_run_name_free.sh <arm> <seed> [<seed> ...]
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
A="$1"; shift
for s in "$@"; do
  NAME="gt_${A}_s${s}"
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$NAME"; then
    echo "FATAL: a job named $NAME is already queued or running. Pick a new arm name."
    exit 1
  fi
  BASE="Utils/Models/3DLearning_mv_cond_${NAME}"
  if compgen -G "${BASE}-ML-epoch*.pth" > /dev/null; then
    echo "FATAL: model files already exist for $NAME. A second run would overwrite them."
    exit 1
  fi
  if [ -f "${BASE}_resume.pt" ]; then
    echo "FATAL: a restart file exists for $NAME. A second run would resume it."
    exit 1
  fi
done
echo "  | run names free: ${A} seeds $* |"
