#!/bin/bash
# Seeds 6-20 of a72 (trend aim, wide head, no generator), one job each, with
# scoring chained per seed. The recipe is read verbatim from
# submit_a72_nogenwide.sh; smoke -> gate -> seeds as in submit_seeds45_a72_a74.sh.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
TAG=$(date +%m%d%H%M)
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
SEEDS="${SEEDS:-$(seq -s ' ' 6 20)}"
COMMON=$(grep -E '^COMMON=' scripts/arms/submit_a72_nogenwide.sh | sed -E 's/^COMMON="\$COMMON,?//; s/^COMMON="//; s/"$//' | paste -sd, - | sed "s|\$INST|$INST|")
echo "$COMMON" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: a72 env not comma-safe"; exit 1; }
echo "$COMMON" | grep -q "PRED_NORM_LO=-0.5" || { echo "FATAL: a72 env lost the wide head"; exit 1; }
bash scripts/arms/gates/check_run_name_free.sh a72_nogenwide $SEEDS || exit 1
SM=$(sbatch --parsable --job-name="sm_a72_nogenwide_$TAG" --constraint=milan \
  --export=ALL,SMOKE=1,MV_SEED=6,$COMMON,MV_DRO_ITERS=2,MV_RUN_SUFFIX=_sm_a72_nogenwide_${TAG}_s6 scripts/train/run_foldin_coevo.sh)
GT=$(sbatch --parsable --dependency=afterok:$SM --job-name="gate_a72_s6to20" \
  --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G \
  --wrap="cd $REPO_ROOT; scripts/arms/gates/banner_recon_check.sh slurm-$SM.out results/logs/train/slurm-59702669.out && grep -q 'Head range on warm-start | \[-0.5, 1.5\] (env override)' slurm-$SM.out && echo A72_WIDE_OK || { echo A72_GATE_FAIL; exit 1; }")
echo "a72: smoke=$SM gate=$GT"
for s in $SEEDS; do
  J=$(sbatch --parsable --dependency=afterok:$GT --job-name="gt_a72_nogenwide_s$s" --time=48:00:00 \
    --constraint=milan --export=ALL,MV_SEED=$s,$COMMON,MV_RUN_SUFFIX=_gt_a72_nogenwide_s$s scripts/train/run_foldin_coevo.sh)
  SC=$(sbatch --parsable --dependency=afterok:$J --job-name="sc_a72_nogenwide_s$s" \
    --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=a72_nogenwide,POC_SEEDS=$s scripts/eval/run_poc_eval_measure.sh)
  echo "  seed$s train=$J score=$SC"
done
