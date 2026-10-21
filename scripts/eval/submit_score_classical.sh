#!/bin/bash
# Scores the classical baselines (narrow output range, (0,1)) under the same
# measurement contract as every other row: poc_eval_ood.py via
# run_poc_eval_measure.sh (work 230, milan, stride 24, sub100_s0, selected = the
# only saved epoch). Each is a symlink pseudo-run to the checkpoint the
# classical trainer kept:
#   pto_base      -> PTO-epoch23
#   wdropto_base  -> WassersteinDRO-epoch100
#   wdrospo_base  -> WassersteinDRO_SPO-epoch92
#   dfl_warmstart -> DFL-epoch41  (already scored, not resubmitted)
# One training run each: seed count 1, labelled as such in every table.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
for arm in pto_base wdropto_base wdrospo_base; do
  ls Utils/Models/ | grep -q "gt_${arm}_s1-ML-epoch" || { echo "  SKIP $arm: no checkpoint"; continue; }
  J=$(sbatch --parsable --job-name=sc_${arm}_s1 \
    --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$arm,POC_SEEDS=1 \
    scripts/eval/run_poc_eval_measure.sh)
  echo "  $arm s1 -> $J"
done
