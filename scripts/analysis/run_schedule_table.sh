#!/bin/bash
#SBATCH --job-name=sched_table
#SBATCH --time=03:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --constraint=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
# Same measurement contract as run_poc_eval_measure.sh, so the MIP part of the
# table reproduces the scored regrets for its window.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
module load gurobi/13.0.0
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/gurobi/13.0.0/gurobi.lic}"
unset GUROBI_HOME
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY -m py_compile scripts/analysis/diag_schedule_table.py poc_eval_ood.py || { echo "SYNTAX ERROR"; exit 1; }
export ICON_AGG=0 ASSERT_NO_GREEDY=1 MIP_GAP=0.01 MIP_TL=150
export MIP_WORK_LIMIT=230 MIP_RESET=1 MIP_QUALITY=1 WORK_TL_SAFETY=20
export LP_WORKERS=8 LP_THREADS_PER_SOLVE=8 EVAL_ORIGIN_STRIDE=24
export ICON_INSTANCE=$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY scripts/analysis/diag_schedule_table.py
