#!/bin/bash
# a78 block-width sensitivity: a72 recipe with INSTR_TREND_BLOCK_HOURS = 2, 6, 8.
# Per width: smoke -> gate -> seeds 1-3 -> score.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
TAG=$(date +%m%d%H%M)
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
COMMON=$(grep -E '^COMMON=' scripts/arms/submit_a72_nogenwide.sh | sed -E 's/^COMMON="\$COMMON,?//; s/^COMMON="//; s/"$//' | paste -sd, - | sed "s|\$INST|$INST|")
echo "$COMMON" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: env not comma-safe"; exit 1; }
echo "$COMMON" | grep -q "PRED_NORM_LO=-0.5" || { echo "FATAL: lost the wide head"; exit 1; }
for W in ${WIDTHS:-2 6 8}; do
  N=a78_bw$W
  E="$COMMON,INSTR_TREND_BLOCK_HOURS=$W"
  bash scripts/arms/gates/check_run_name_free.sh "$N" 1 2 3 || exit 1
  SM=$(sbatch --parsable --job-name="sm_${N}_$TAG" --constraint=milan \
    --export=ALL,SMOKE=1,MV_SEED=1,$E,MV_DRO_ITERS=2,MV_RUN_SUFFIX=_sm_${N}_${TAG}_s1 scripts/train/run_foldin_coevo.sh)
  GT=$(sbatch --parsable --dependency=afterok:$SM --job-name="gate_$N" \
    --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=4G \
    --wrap="cd $REPO_ROOT; scripts/arms/gates/gate_a78_blockwidth.sh slurm-$SM.out $W")
  echo "$N: smoke=$SM gate=$GT"
  for s in 1 2 3; do
    J=$(sbatch --parsable --dependency=afterok:$GT --job-name="gt_${N}_s$s" --time=48:00:00 \
      --constraint=milan --export=ALL,MV_SEED=$s,$E,MV_RUN_SUFFIX=_gt_${N}_s$s scripts/train/run_foldin_coevo.sh)
    SC=$(sbatch --parsable --dependency=afterok:$J --job-name="sc_${N}_s$s" \
      --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$N,POC_SEEDS=$s scripts/eval/run_poc_eval_measure.sh)
    echo "  seed$s train=$J score=$SC"
  done
done
