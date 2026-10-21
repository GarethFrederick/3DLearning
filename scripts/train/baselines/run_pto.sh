#!/bin/bash
#SBATCH --job-name="energy_pto"
#SBATCH --time=12:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

# PTO trains on prices with SmoothL1Loss(beta=0.1) and solves no LP, so 16 cores
# suffice for the LSTM on CPU.

module load miniforge3
# Gurobi licence: needed only because Energy_Scheduling's __init__ creates a
# thread-local Gurobi env, even though PTO never solves an LP.
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/gurobi/13.0.0/gurobi.lic}"
unset GUROBI_HOME

# Cap PyTorch and BLAS thread pools at 1; the LSTM is small.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate 3d-learning
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

echo "  | GPUs allocated | ${CUDA_VISIBLE_DEVICES:-none}"
echo "  | Python         | $(which python)"
echo "  | STAGE          | pto"

export STAGE=pto
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY main_energy.py
