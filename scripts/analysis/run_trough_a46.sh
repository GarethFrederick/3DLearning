#!/bin/bash
#SBATCH --job-name=trough_a46
#SBATCH --time=00:30:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=milan
#SBATCH --output=slurm-%j.out
# Re-measures the trough location for the narrow pair, a46 (trend-aim narrow,
# generator off, 30 rounds) against a1_stall, seed 1, and writes a new dump.
# The wide pair re-runs as an A/A check: it must print the wide-pair numbers
# already on record.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3; source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
echo "################ NARROW pair (a46 vs a1_stall) ################"
DIAG_ARMS="ours=a46_nogen30,3DL=a1_stall" DIAG_DUMP_JSON=results/analysis/trough_perwindow_narrow_a46.json $PY scripts/analysis/diag_trough_location.py
echo "################ WIDE pair (a72 vs a71), A/A ################"
DIAG_ARMS="ours=a72_nogenwide,3DL=a71_a1wide" $PY scripts/analysis/diag_trough_location.py
echo "######## TROUGH_A46_DONE ########"
