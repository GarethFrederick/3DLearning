#!/bin/bash
# a76_block4: the block-4 probe. a74's environment with the random aim replaced
# by INSTR_HEAD_AIM_BLOCK=4.
# Chain: smoke -> gate -> one training run (seed 1) -> scoring.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance missing"; exit 1; }
A72="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1"
A72="$A72,GEN_TRANSPORT_ALPHA=120,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300"
A72="$A72,GEN_TRANSPORT_LEASH=1,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1"
A72="$A72,INSTR_HEAD_AUG=1,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_WIDTH=1"
A72="$A72,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1,INSTR_TREND_DEGREE_AUTO=1"
A72="$A72,MV_SKIP_GEN_TRAIN=1,MV_SKIP_GEN_SAMPLE=1"
E="$A72,INSTR_HEAD_AIM_BLOCK=4,PRED_NORM_LO=-0.5,PRED_NORM_HI=1.5"
N=a76_block4
echo "$E" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: env not comma-safe"; exit 1; }
bash scripts/arms/gates/check_run_name_free.sh "$N" 1 || exit 1
TAG=$(date +%m%d%H%M)
SMK=$(sbatch --parsable --job-name="sm_${N}_${TAG}" --constraint=milan \
  --export=ALL,SMOKE=1,MV_DRO_ITERS=2,KEEP_EPOCH_CKPTS=1,MV_SEED=1,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_sm_${N}_${TAG}_s1 \
  scripts/train/run_foldin_coevo.sh)
GJ=$(sbatch --parsable --dependency=afterok:$SMK --job-name="gate_$N" \
  --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G \
  --wrap="cd $REPO_ROOT; scripts/arms/gates/gate_a76_block4.sh slurm-$SMK.out")
TR=$(sbatch --parsable --dependency=afterok:$GJ --job-name="gt_${N}_s1" \
  --time=48:00:00 --constraint=milan \
  --export=ALL,MV_DRO_ITERS=30,KEEP_EPOCH_CKPTS=1,MV_SEED=1,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gt_${N}_s1 \
  scripts/train/run_foldin_coevo.sh)
SC=$(sbatch --parsable --dependency=afterok:$TR --job-name="sc_${N}_s1" \
  --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$N,POC_SEEDS=1 scripts/eval/run_poc_eval_measure.sh)
echo "$N: smoke=$SMK gate=$GJ train=$TR score=$SC"
