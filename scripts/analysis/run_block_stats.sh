#!/bin/bash
#SBATCH --job-name="block_stats"
#SBATCH --time=00:20:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=milan
#SBATCH --output=slurm-%j.out
# Block-level statistics for the results section. Reads the scored JSONs only;
# no solver, no GPU. The window-to-block map is asserted, not assumed.
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
echo "===== selected checkpoints (the reported numbers) ====="
$PY scripts/analysis/block_stats.py
echo
echo "===== last-epoch checkpoints, no selection (the robustness check) ====="
STATS_KIND=fixed $PY scripts/analysis/block_stats.py
