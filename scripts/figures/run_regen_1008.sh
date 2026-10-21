#!/bin/bash
#SBATCH --job-name=regen1008
#SBATCH --time=00:20:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3; source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
$PY -m py_compile scripts/figures/make_thesis_figures.py scripts/figures/fig_supervisor.py scripts/figures/fig_results_1008.py scripts/figures/fig_worklimit.py && echo COMPILE_OK || exit 1
echo "===== F5 F9 F10 ====="
FIG_OUT=$PWD/results/thesis/figures PYTHONPATH="$PWD/scripts/figures${PYTHONPATH:+:$PYTHONPATH}" $PY -c "import make_thesis_figures as M; M.f5_random_aim(); M.f9_decomposition(); M.f10_narrow_ladder()"
echo "===== fig_results_1008 ====="; $PY scripts/figures/fig_results_1008.py
echo "===== fig_supervisor ====="; $PY scripts/figures/fig_supervisor.py
echo "===== fig_worklimit ====="; $PY scripts/figures/fig_worklimit.py
