#!/bin/bash
#SBATCH --job-name="regen_aemo"
#SBATCH --time=1:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Regenerates the AEMO base and OOD split CSVs with the resample labels set in
# preprocess_aemo.resample_30min (closed='right', label='right'). The train-era
# file must stay byte-unchanged; the md5sum before and after checks it.
#   1. preprocess_aemo.py     -> _train_2015_2020 / _test_2021_2022 / _test_2023_2025
#   2. make_ood_val_splits.py -> _oodval_* / _oodtest_*  (reads the _test_* files)
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"

set -e
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

echo "  | train-era checksum BEFORE (must be unchanged after) |"
md5sum Data/AEMO_VIC1_train_2015_2020.csv

echo "  | STAGE 1 | preprocess_aemo.py"
$PY Data/preprocess_aemo.py

echo "  | STAGE 2 | make_ood_val_splits.py"
$PY Data/make_ood_val_splits.py

echo "  | train-era checksum AFTER (must equal BEFORE) |"
md5sum Data/AEMO_VIC1_train_2015_2020.csv

echo "  | DONE |"
