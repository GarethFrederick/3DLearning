#!/bin/bash
#SBATCH --job-name=cal_worklimit
#SBATCH --account=fr57
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Measures Gurobi work units per evaluation MIP solve under the CURRENT
# wall-clock limit, so a WorkLimit can be chosen from data.
# Read-only: trains nothing, writes only gate_out/worklimit_calibration.json.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

module load miniforge3
conda activate 3d-learning

# Same solver settings the real evaluation uses. Serial outer loop, so one solve
# at a time and the work reading is unambiguous.
export LP_THREADS_PER_SOLVE=8
export MIP_FOCUS=1
export MIP_TL=150
export MIP_GAP=0.01
export ICON_AGG=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ICON_INSTANCE=$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt

# The measurement hook. MIP_WORK_LIMIT stays UNSET on purpose: this run must
# measure the OLD ruler. The script refuses to start if it is set.
export LOG_WORK=1
unset MIP_WORK_LIMIT
unset LP_WORK_LIMIT

export CAL_DAYS=${CAL_DAYS:-40}

echo "=== node: $(hostname)  date: $(date) ==="
python scripts/analysis/calibrate_worklimit.py
echo "EXIT=$?"
