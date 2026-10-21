#!/bin/bash
#SBATCH --job-name=trough_figs
#SBATCH --time=00:30:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=milan
# 1. Re-runs the trough-location measurement on both pairs with the per-window
#    dump on. The printed tables must match the no-dump run exactly (the dump is
#    default-off and adds output only at the end); that is the A/A check.
# 2. Rebuilds every thesis figure.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY -m py_compile scripts/analysis/diag_trough_location.py scripts/figures/make_thesis_figures.py && echo "  | syntax OK |"
echo "################ NARROW pair (a34 vs a1_stall), dump on ################"
DIAG_DUMP_JSON=results/analysis/trough_perwindow_narrow.json $PY scripts/analysis/diag_trough_location.py
echo "################ WIDE pair (a72 vs a71), dump on ################"
DIAG_ARMS="ours=a72_nogenwide,3DL=a71_a1wide" DIAG_DUMP_JSON=results/analysis/trough_perwindow_wide.json $PY scripts/analysis/diag_trough_location.py
echo "################ figures ################"
$PY scripts/figures/make_thesis_figures.py
echo "######## TROUGH_FIGS_DONE ########"
