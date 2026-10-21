#!/bin/bash
# Resubmits a24, a25, a26 and a22 on the registered instance (sub100_s0), with
# ICON_INSTANCE and KEEP_EPOCH_CKPTS=1 set. Without them the mv driver falls back
# to sample01, and the end-of-run cleanup deletes the per-epoch checkpoints that
# poc_eval_ood.py needs. CONSOL_EVERY is inert under FOLDIN_HEAD_ONLY=1 but kept
# so every knob matches the a19 reference. The _r2 suffix stops a same-name
# resubmit from resuming the old wrong-instance state.
# Each arm is A19ENV plus its own single knob. GO=1 submits; a bare run prints.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance file missing: $INST"; exit 1; }

# The a19 reference environment, unchanged. Every arm below is this plus one knob.
A19ENV="FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1,GEN_TRANSPORT_ALPHA=120,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300,GEN_TRANSPORT_LEASH=1,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1,INSTR_HEAD_AUG=1,HEAD_TRAIN_REAL=1,INSTR_HEAD_DISCOVER_WIDTH=1"

# arm -> the extra knobs that define it.
ARMS=(a24_trendaim_r2 a25_recency_r2 a26_aimbank_r2 a22_nogendepth_r2)
declare -A EXTRA=(
  [a24_trendaim_r2]="INSTR_HEAD_AIM_TREND=1"
  [a25_recency_r2]="HEAD_REAL_FROM_YEAR=2019,INSTR_HEAD_ALPHA_MAX=0"
  [a26_aimbank_r2]="INSTR_HEAD_ACCUM=1"
  [a22_nogendepth_r2]="INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_DEPTH=1"
)
# Each arm keeps its already-registered banner gate. The gates check banners, not
# the instance, so they are unchanged and still valid.
declare -A GATE=(
  [a24_trendaim_r2]=scripts/arms/gates/banner_check_a24.sh
  [a25_recency_r2]=scripts/arms/gates/banner_check_a25.sh
  [a26_aimbank_r2]=scripts/arms/gates/banner_check_a26.sh
  [a22_nogendepth_r2]=scripts/arms/gates/banner_check_a22.sh
)

# sbatch --export splits on commas, so no value may hold one.
for a in "${ARMS[@]}"; do
  E="$A19ENV,${EXTRA[$a]}"
  if echo "$E" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$'; then
    echo "FATAL: $a env is not comma-safe for --export"; exit 1
  fi
  [ -f "${GATE[$a]}" ] || { echo "FATAL: gate missing: ${GATE[$a]}"; exit 1; }
done
echo "  | all four arm envs are export-safe, all four gates present |"

if [ "${GO:-0}" != "1" ]; then
  echo
  echo "  DRY RUN. Set GO=1 to submit. Would launch, in this order:"
  for a in "${ARMS[@]}"; do
    echo "    $a  ->  ICON_INSTANCE=sub100_s0, KEEP_EPOCH_CKPTS=1, ${EXTRA[$a]}"
  done
  exit 0
fi

PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
PYC="module load miniforge3; source activate 3d-learning; cd $REPO_ROOT; $PY -m py_compile main_energy_mv.py Utils/Algos/ddd_learning.py && echo COMPILE_OK"

for a in "${ARMS[@]}"; do
  E="$A19ENV,${EXTRA[$a]}"
  CMP=$(sbatch --parsable --job-name="${a}_cmp" --account=fr57 --partition=fitc --qos=fitcq \
        --time=00:10:00 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --wrap="$PYC")
  SMK=$(sbatch --parsable --dependency=afterok:$CMP --job-name="gtsm_${a}" \
        --export=ALL,SMOKE=1,CONSOL_EVERY=1,KEEP_EPOCH_CKPTS=1,MV_SEED=1,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gtsm_${a}_s1 \
        scripts/train/run_foldin_coevo.sh)
  BAN=$(sbatch --parsable --dependency=afterok:$SMK --job-name="${a}_banner" \
        --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=4G \
        --wrap="cd $REPO_ROOT; SMOKE_LOG=slurm-$SMK.out bash ${GATE[$a]}")
  echo "$a: cmp=$CMP smoke=$SMK gate=$BAN"
  for s in 1 2 3; do
    J=$(sbatch --parsable --dependency=afterok:$BAN --job-name="gt_${a}_s$s" \
        --export=ALL,CONSOL_EVERY=5,KEEP_EPOCH_CKPTS=1,MV_SEED=$s,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gt_${a}_s$s \
        scripts/train/run_foldin_coevo.sh)
    echo "    seed$s=$J"
  done
done
