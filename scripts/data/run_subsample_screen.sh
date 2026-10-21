#!/bin/bash
#SBATCH --job-name=sub07_screen
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=gate_out/slurm_sub07_screen_%j.out
#SBATCH --error=gate_out/slurm_sub07_screen_%j.err
set -eo pipefail   # NOT -u: conda activate references unbound vars

# Cheap screen for the sample07 subsample recipe: builds K in {50,100,150,200}
# x 3 seeds and checks format round-trip, single-task feasibility and LP
# response (SHIFT). LP only, no MIP.

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
mkdir -p gate_out

module load miniforge3
module load gurobi/13.0.0
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/gurobi/13.0.0/gurobi.lic}"
unset GUROBI_HOME
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"

export SUB_KS="${SUB_KS:-50,100,150,200}"
export SUB_SEEDS="${SUB_SEEDS:-0,1,2}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== sub07 cheap screen | KS=$SUB_KS SEEDS=$SUB_SEEDS | host=$(hostname) ==="
"$PY" -m py_compile scripts/data/subsample_icon.py
echo "    py_compile OK"
"$PY" scripts/data/subsample_icon.py
echo "=== sub07 screen DONE ==="
