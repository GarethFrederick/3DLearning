#!/bin/bash
#SBATCH --job-name=fcshape
#SBATCH --time=00:20:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=milan
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3; source activate 3d-learning
export ICON_AGG=0 ICON_INSTANCE=$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY scripts/analysis/diag_forecast_shape.py
