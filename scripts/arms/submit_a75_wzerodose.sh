#!/bin/bash
# a75_wzerodose = the wide head with the augmentation at zero strength. It asks
# how much of the wide-head result is the head range and how much is the aim.
# The recipe is not "a72 plus one knob": a72 sets INSTR_HEAD_DEPTH_TREND=1, which
# overwrites the dose with the trend depth after INSTR_HEAD_ALPHA_MAX is read, so
# a72 + ALPHA_MAX=0 would train byte-identical to a72. The arm removes
# INSTR_HEAD_DEPTH_TREND and sets INSTR_HEAD_ALPHA_MAX=0.
# The aim stays on so the code path and random stream match a72; at zero
# amplitude the aim changes nothing in the data.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
# GO is checked below, AFTER the asserts, so a dry run can inspect the recipe.
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance missing"; exit 1; }
[ -x scripts/arms/gates/gate_a75_wzerodose.sh ] || { echo "FATAL: gate missing"; exit 1; }
REF=results/logs/train/slurm-59702669.out                      # a72 seed 1
[ -s "$REF" ] || { echo "FATAL: a72 reference log $REF missing"; exit 1; }
TAG=$(date +%m%d%H%M)
SEEDS="${SEEDS:-1 2 3 4 5}"

# Parse a72's environment verbatim, then make the two edits.
COMMON=$(grep -E '^COMMON=' scripts/arms/submit_a72_nogenwide.sh \
  | sed -E 's/^COMMON="\$COMMON,?//; s/^COMMON="//; s/"$//' | paste -sd, - | sed "s|\$INST|$INST|")
COMMON=$(echo "$COMMON" | tr ',' '\n' | grep -v '^INSTR_HEAD_DEPTH_TREND=' | paste -sd, -)
COMMON="$COMMON,INSTR_HEAD_ALPHA_MAX=0"

# Asserts. Each one is a way this arm could silently become something else.
echo "$COMMON" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: env not comma-safe"; exit 1; }
echo "$COMMON" | grep -q 'PRED_NORM_LO=-0.5'        || { echo "FATAL: lost the wide head"; exit 1; }
echo "$COMMON" | grep -q 'INSTR_HEAD_ALPHA_MAX=0'   || { echo "FATAL: zero dose not set"; exit 1; }
echo "$COMMON" | grep -q 'INSTR_HEAD_DEPTH_TREND'   && { echo "FATAL: depth trend still on; it would overwrite the zero"; exit 1; }
echo "$COMMON" | grep -q 'INSTR_HEAD_AIM_TREND=1'   || { echo "FATAL: lost the aim; the code path must match a72"; exit 1; }
echo "$COMMON" | grep -q 'MV_SKIP_GEN_TRAIN=1'      || { echo "FATAL: generator not skipped"; exit 1; }
echo "$COMMON" | grep -q 'MV_DRO_ITERS=30'          || { echo "FATAL: dose not 30 rounds"; exit 1; }

if [ "${GO:-0}" != "1" ]; then echo "DRY: $COMMON"; exit 0; fi
echo "env: $COMMON"
bash scripts/arms/gates/check_run_name_free.sh a75_wzerodose $SEEDS || exit 1

SM=$(sbatch --parsable --job-name="sm_a75_wzerodose_$TAG" --constraint=milan \
  --export=ALL,SMOKE=1,MV_DRO_ITERS=2,MV_SEED=1,$COMMON,MV_RUN_SUFFIX=_sm_a75_wzerodose_${TAG}_s1 \
  scripts/train/run_foldin_coevo.sh)
GT=$(sbatch --parsable --dependency=afterok:$SM --job-name="gate_a75" \
  --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G \
  --wrap="cd $REPO_ROOT; scripts/arms/gates/gate_a75_wzerodose.sh slurm-$SM.out $REF")
echo "smoke=$SM  gate=$GT"
for s in $SEEDS; do
  J=$(sbatch --parsable --dependency=afterok:$GT --job-name="gt_a75_wzerodose_s$s" \
    --time=48:00:00 --constraint=milan \
    --export=ALL,MV_SEED=$s,$COMMON,MV_RUN_SUFFIX=_gt_a75_wzerodose_s$s scripts/train/run_foldin_coevo.sh)
  SC=$(sbatch --parsable --dependency=afterok:$J --job-name="sc_a75_wzerodose_s$s" \
    --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=a75_wzerodose,POC_SEEDS=$s scripts/eval/run_poc_eval_measure.sh)
  echo "  seed$s=$J  score=$SC"
done
