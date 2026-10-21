#!/bin/bash
#SBATCH --job-name="gate_a72"
#SBATCH --time=00:05:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%j.out
# Gate the a72 fleet on the smoke log. Exits NON-ZERO on failure so the
# afterok dependency holds the seeds back.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
scripts/arms/gates/banner_check_wide.sh "$SMOKE_LOG" "$REF_LOG"
echo "A72_GATE_OK"
