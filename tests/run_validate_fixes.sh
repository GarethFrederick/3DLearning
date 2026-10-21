#!/bin/bash
#SBATCH --job-name="validate_fixes"
#SBATCH --time=00:30:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Syntax-gate every file touched by the silent-bug fix batch, then run the
# CPU-only regression tests (new + existing). No Gurobi, no training.
# sbatch spools this script, so its own path says nothing about the repo:
# take REPO_ROOT, else the submit directory, else $PWD, and refuse anything else.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root"; exit 1; }
cd "$REPO_ROOT"
module load miniforge3
source activate 3d-learning
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"

echo "===== py_compile edited files ====="
$PY -m py_compile \
    main_energy.py main_energy_mv.py \
    Utils/Algos/ddd_learning.py Utils/Algos/ml.py \
    Utils/Algos/ddd/*.py \
    Utils/Downstream/energy_scheduling.py \
    Utils/Downstream/opt_oracle.py \
    && echo "  py_compile OK" || { echo "  SYNTAX ERROR"; exit 1; }

echo "===== new silent-bug regression tests ====="
$PY tests/test_silent_bugs.py || { echo "  TEST FAIL"; exit 1; }

echo "===== existing tests (no regression) ====="
# Run directly (env has no pytest); both files self-run their checks under __main__.
$PY tests/test_spo_row_cache.py && $PY tests/test_ppo_pairing.py \
    || { echo "  EXISTING TEST FAIL"; exit 1; }

echo "######## VALIDATE DONE ########"

echo "===== root-level regression tests the suite never ran ====="
for t in tests/test_trend_knobs.py tests/test_eval_naming.py; do
    echo "--- $t"
    $PY "$t" || { echo "  ROOT TEST FAIL: $t"; exit 1; }
done

echo "===== head-range map and clock-distance characterization ====="
$PY tests/test_headrange_circdist.py || { echo "  HEADRANGE/CIRCDIST FAIL"; exit 1; }

echo "######## VALIDATE PLUS DONE ########"
