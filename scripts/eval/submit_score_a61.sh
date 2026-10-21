#!/bin/bash
# Score a61_nodecay30 on the two held-out shifted regimes.
#
# a61 is a34 with MV_NO_LR_DECAY=1 and 30 rounds. Its registered comparator is
# a44 (a34 at 30 rounds, decay ON), NOT a34 — a61 changes the decay and the round
# count relative to a34, so only a44 isolates the decay.
#
# CLAIM BOUNDARY. a1_stall and a36_oracle both ran with the decay ON. A win here
# does NOT restate the headline or the ceiling: that needs those two re-run with
# MV_NO_LR_DECAY=1 first, or it is a fixed arm against starved baselines.
#
# run_poc_eval_measure.sh carries --constraint=milan, the work limit and
# ICON_AGG=0 in the script itself.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
for s in 1 2 3; do
  ls Utils/Models/3DLearning_mv_cond_gt_a61_nodecay30_s${s}-ML-epoch*.pth >/dev/null 2>&1 \
    || { echo "FATAL: no checkpoints for seed $s — is it still training?"; exit 1; }
  J=$(sbatch --parsable --job-name=sc_a61_nodecay30_s${s} \
    --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=a61_nodecay30,POC_SEEDS=$s \
    scripts/eval/run_poc_eval_measure.sh)
  echo "a61_nodecay30 s$s = $J"
done
echo
echo "BEFORE QUOTING: ./banner_check_measure.sh slurm-<id>.out 230 -> MEASURE_GATE_OK"
echo "THEN: ARM_A=a44_long30 ARM_B=a61_nodecay30 python paired_test_seedlevel.py"
