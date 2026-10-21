#!/bin/bash
#SBATCH --job-name="train_ddpm_mv_cond"
#SBATCH --time=01:00:00
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Train + validate the CONDITIONAL multivariate DDPM (F=8, n_cond=6).
# Stoch channels: Price, Demand. Cond channels: sin/cos hour/dow/month.
# Writes:
#   Utils/Models/model_DDPM_mv_F8_L192_T500_best.pth
#   Utils/Models/model_DDPM_mv_F8_L192_T500_meta.json (with calibration+clip)
#   Utils/Models/model_DDPM_mv_F8_L192_T500_cond_pool.npy

module load miniforge3
source activate 3d-learning
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

echo "  | GPUs | ${CUDA_VISIBLE_DEVICES:-none}"
echo "  | Python | $(which python)"

PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
echo "  | syntax check |"
$PY -m py_compile train_ddpm_mv_cond.py validate_ddpm_mv_cond.py \
    Utils/Diffusion/DDPM.py Utils/Net/DDPM_net_mv.py Utils/diffusion_utils.py \
    || { echo "SYNTAX ERROR"; exit 1; }
echo "  | syntax OK |"

echo "  ===== TRAIN ====="
$PY train_ddpm_mv_cond.py || { echo "TRAIN FAILED"; exit 1; }

echo "  ===== VALIDATE / CALIBRATE ====="
$PY validate_ddpm_mv_cond.py || { echo "VALIDATE FAILED"; exit 1; }

echo "  | Done. |"
