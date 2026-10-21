#!/bin/bash
#SBATCH --job-name=fig1016
#SBATCH --time=00:15:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
# Builds thesis figures F0, F12, F21, F22 and F23. Order: syntax gate, then the
# checker's negative tests (its failure branches must fire), then the figures.
# Args pass through to fig_thesis_1016.py.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY -m py_compile scripts/figures/figcheck.py scripts/figures/fig_thesis_1016.py scripts/figures/fig_block_width.py && echo "  | syntax OK |"
$PY scripts/figures/figcheck.py
export FIG_OUT=$PWD/results/thesis/figures
echo "===== F12 ====="
$PY scripts/figures/fig_block_width.py
$PY scripts/figures/fig_thesis_1016.py "$@"
