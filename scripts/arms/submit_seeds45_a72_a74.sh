#!/bin/bash
# Seeds 4 and 5 of a72 (trend aim) and a74 (random aim), scoring chained behind.
# Per arm: smoke (2 rounds) -> gate -> seeds 4 and 5 -> score each. Every link is
# afterok, so nothing downstream of a failure runs.
# a72's gate also greps the widened head range: banner_recon_check.sh never reads
# the "Head range on warm-start" line, so a forgotten PRED_NORM would pass it.
# a74's random block is seed-derived, RandomState(20260906 + MV_SEED) with the
# trend block excluded; gate_a74_randomaim.sh still refuses the trend block.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }
TAG=$(date +%m%d%H%M)
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
score() {  # $1 arm  $2 seed  $3 training job id
  sbatch --parsable --dependency=afterok:$3 --job-name="sc_$1_s$2" \
    --export=ALL,POC_RUN_PREFIX=gt,POC_ARMS=$1,POC_SEEDS=$2 scripts/eval/run_poc_eval_measure.sh
}

# a72: the recipe is submit_a72_nogenwide.sh, read verbatim.
COMMON=$(grep -E '^COMMON=' scripts/arms/submit_a72_nogenwide.sh | sed -E 's/^COMMON="\$COMMON,?//; s/^COMMON="//; s/"$//' | paste -sd, - | sed "s|\$INST|$INST|")
echo "$COMMON" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: a72 env not comma-safe"; exit 1; }
echo "$COMMON" | grep -q "PRED_NORM_LO=-0.5" || { echo "FATAL: a72 env lost the wide head"; exit 1; }
bash scripts/arms/gates/check_run_name_free.sh a72_nogenwide 4 5 || exit 1
SM=$(sbatch --parsable --job-name="sm_a72_nogenwide_$TAG" --constraint=milan \
  --export=ALL,SMOKE=1,MV_DRO_ITERS=2,MV_SEED=4,$COMMON,MV_RUN_SUFFIX=_sm_a72_nogenwide_${TAG}_s4 scripts/train/run_foldin_coevo.sh)
GT=$(sbatch --parsable --dependency=afterok:$SM --job-name="gate_a72_s45" \
  --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G \
  --wrap="cd $REPO_ROOT; scripts/arms/gates/banner_recon_check.sh slurm-$SM.out results/logs/train/slurm-59702669.out && grep -q 'Head range on warm-start | \[-0.5, 1.5\] (env override)' slurm-$SM.out && echo A72_WIDE_OK || { echo A72_GATE_FAIL; exit 1; }")
echo "a72: smoke=$SM gate=$GT"
for s in 4 5; do
  J=$(sbatch --parsable --dependency=afterok:$GT --job-name="gt_a72_nogenwide_s$s" --time=48:00:00 \
    --constraint=milan --export=ALL,MV_SEED=$s,$COMMON,MV_RUN_SUFFIX=_gt_a72_nogenwide_s$s scripts/train/run_foldin_coevo.sh)
  echo "    seed$s=$J  score=$(score a72_nogenwide $s $J)"
done

# a74: through its own launcher, then chain the scoring.
OUT=$(GO=1 ARMS="a74" SEEDS="4 5" scripts/arms/submit_controls_a73_a74.sh) || { echo "$OUT"; echo "FATAL: a74 launch failed"; exit 1; }
echo "$OUT"
for s in 4 5; do
  J=$(echo "$OUT" | grep -oE "seed$s=[0-9]+" | cut -d= -f2)
  [ -n "$J" ] || { echo "FATAL: no a74 seed$s job id parsed"; exit 1; }
  echo "    a74 seed$s=$J  score=$(score a74_randomaim $s $J)"
done
