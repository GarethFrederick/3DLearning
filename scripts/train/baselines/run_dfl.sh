#!/bin/bash
#SBATCH --job-name="energy_dfl"
#SBATCH --time=48:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G

# DFL is LP-bound (Gurobi SPO+ on ICON sample01), so it runs on CPU nodes with
# 128 cores as a 64x2 worker/thread split. main_energy.py falls back to 'cpu'
# when no GPU is visible.

module load miniforge3
# Use the M3 site Gurobi licence, not a personal one (the WLS academic licence
# caps sessions). No `module load gurobi`: the conda env's bundled libgurobi must
# be the one used, with PATH and LD_LIBRARY_PATH left clean.
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/gurobi/13.0.0/gurobi.lic}"
unset GUROBI_HOME

# Cap PyTorch and BLAS thread pools at 1 so they do not contend with Gurobi's
# 64x2 LP worker pool on the same 128 cores; the LSTM work is tiny.
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
echo "  | STAGE          | dfl"

export STAGE=dfl
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY main_energy.py
