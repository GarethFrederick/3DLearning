#!/bin/bash
#SBATCH --job-name="validate_ddpm_mv_cond"
#SBATCH --time=00:20:00
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Validates and calibrates the conditional F=8 mv DDPM. Reports uncalibrated vs
# calibrated price/demand marginals and spike rates, and writes the calibration
# and AEMO clip into the meta sidecar, where the 3DL z0 loader reads them.
# VAL_N defaults to 1024 in the Python script; a larger value tightens the
# spike-rate estimate.

module load miniforge3
source activate 3d-learning
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
echo "  | GPUs | ${CUDA_VISIBLE_DEVICES:-none}"
echo "  | Python | $(which python)"

export VAL_N=2048
$PY validate_ddpm_mv_cond.py
