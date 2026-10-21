#!/bin/bash
#SBATCH --job-name="trough_remeasure"
#SBATCH --time=02:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=milan
#SBATCH --output=slurm-%j.out
# Re-measures the trough location. Forward passes only, no solver.
# Run 1 is an A/A check: the default narrow pair with no env set must reproduce
# the published trough table, or the DIAG_ARMS patch changed behaviour and run 2
# is void. Run 2 is the wide pair, a72 against a71.
# Both runs print each model's own cheap-hour centre and the true centre: only
# the absolute hour separates an overnight-regime baseline from a rigid trough.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"

echo "################ RUN 1: A/A, default narrow pair (a34 vs a1_stall) ################"
$PY scripts/analysis/diag_trough_location.py

echo
echo "################ RUN 2: the WIDE pair (a72 vs a71) ################"
DIAG_ARMS="ours=a72_nogenwide,3DL=a71_a1wide" $PY scripts/analysis/diag_trough_location.py

echo "######## TROUGH_REMEASURE_DONE ########"
