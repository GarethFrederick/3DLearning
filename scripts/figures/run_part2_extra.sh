#!/bin/bash
#SBATCH --job-name="part2_figs"
#SBATCH --time=00:15:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
FIG_OUT=$PWD/results/thesis/figures PYTHONPATH="$PWD/scripts/figures${PYTHONPATH:+:$PYTHONPATH}" $PY -c "import make_thesis_figures as M; M.f7_output_range()"
$PY scripts/figures/fig_part2_extra.py
