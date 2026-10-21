#!/bin/bash
#SBATCH --job-name="poc_eval_measure"
#SBATCH --time=24:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
# Scoring must pin milan: Gurobi's work count repeats within a CPU family and not
# across families, so identical models scored on Intel and AMD disagree. Pinned
# here so a forgotten command line cannot void a run. #SBATCH lines are read at
# submit time, so editing them cannot affect a job already in the queue.
#SBATCH --constraint=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out

# Scores arms on the held-out years (poc_eval_ood.py) under the deterministic
# work-limit settings; writes results/scores/poc_eval_measure by default.
#  1. 64 cores and LP_WORKERS=8. Under a work limit the worker count changes run
#     time, never results: each solve gets its full budget, and MIP_RESET removes
#     carry-over between days. Runs at 8x8 and 16x8 are comparable.
#  2. LP_THREADS_PER_SOLVE stays 8 and is frozen: Gurobi's work count repeats
#     only at a fixed thread count. A run at any other value is void.
#  3. MIP_FOCUS is not set: feasibility-first search costs proof speed.
#  4. POC_EVAL_OUT_DIR is settable, so the same arm can be scored twice into
#     separate directories and the two runs compared exactly.
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

$PY -m py_compile poc_eval_ood.py Utils/Algos/ml.py \
    Utils/Downstream/energy_scheduling.py || { echo "SYNTAX ERROR"; exit 1; }

export ICON_AGG=0
export ASSERT_NO_GREEDY=1
export MIP_GAP="${MIP_GAP:-0.01}"
export MIP_TL="${MIP_TL:-150}"          # backstop base; safety net is 20x this

# The work-limit measurement settings:
export MIP_WORK_LIMIT="${MIP_WORK_LIMIT:-230}"
export MIP_RESET=1
export MIP_QUALITY=1
export WORK_TL_SAFETY="${WORK_TL_SAFETY:-20}"
export LP_WORKERS="${LP_WORKERS:-8}"
export LP_THREADS_PER_SOLVE=8           # FROZEN — see §8.47

export EVAL_ORIGIN_STRIDE="${EVAL_ORIGIN_STRIDE:-24}"
export ICON_INSTANCE="${ICON_INSTANCE:-$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "  | POC OOD EVAL (MEASURE) | instance=$ICON_INSTANCE stride=$EVAL_ORIGIN_STRIDE"
echo "  | work=$MIP_WORK_LIMIT reset=$MIP_RESET quality=$MIP_QUALITY safety=${WORK_TL_SAFETY}x"
echo "  | LP_WORKERS=$LP_WORKERS x $LP_THREADS_PER_SOLVE threads  GAP=$MIP_GAP"
# Default the output dir to the measure dir: poc_eval_ood.py defaults OUT_DIR to
# the old gate_out/poc_eval, and scores from the two rulers must not mix.
export POC_EVAL_OUT_DIR="${POC_EVAL_OUT_DIR:-$PWD/results/scores/poc_eval_measure}"
echo "  | out dir = $POC_EVAL_OUT_DIR"
$PY poc_eval_ood.py
echo "MEASURE_EVAL_DONE"
