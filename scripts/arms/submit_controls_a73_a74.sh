#!/bin/bash
# Two controls at the wide head, each varying one knob against an existing arm.
# a73_vanilla3dl = a71_a1wide with FOLD_IN=0. The arm labelled "3D-Learning as
#   published" runs inside this project's co-evolution wrapper; a73 measures the
#   baseline without it. Same runner, same dose, same head: only FOLD_IN changes.
# a74_randomaim = a72_nogenwide with INSTR_HEAD_AIM_RANDOM=1: the same discount,
#   depth and width aimed at a random block instead of the derived one. If a74
#   ties a72, the trend rule does not earn its place.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance missing"; exit 1; }
WIDE="PRED_NORM_LO=-0.5,PRED_NORM_HI=1.5"

# a73: the bare co-evolution default with the wrapper switched OFF.
ENV_a73="FOLD_IN=0,$WIDE"
# a74: submit_a72_nogenwide.sh's COMMON, verbatim, plus one knob.
A72="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1"
A72="$A72,GEN_TRANSPORT_ALPHA=120,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300"
A72="$A72,GEN_TRANSPORT_LEASH=1,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1"
A72="$A72,INSTR_HEAD_AUG=1,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_WIDTH=1"
A72="$A72,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1,INSTR_TREND_DEGREE_AUTO=1"
A72="$A72,MV_SKIP_GEN_TRAIN=1,MV_SKIP_GEN_SAMPLE=1"
ENV_a74="$A72,INSTR_HEAD_AIM_RANDOM=1,$WIDE"
GATE_a73=scripts/arms/gates/gate_a73_vanilla.sh   ; NAME_a73=a73_vanilla3dl
GATE_a74=scripts/arms/gates/gate_a74_randomaim.sh ; NAME_a74=a74_randomaim

ARMS="${ARMS:-a73 a74}"
SEEDS="${SEEDS:-1 2 3}"
for a in $ARMS; do
  eval "E=\$ENV_$a; G=\$GATE_$a; N=\$NAME_$a"
  [ -x "$G" ] || { echo "FATAL: gate $G missing or not executable"; exit 1; }
  echo "$E" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: $a env not comma-safe"; exit 1; }
  if [ "${GO:-0}" != "1" ]; then echo "DRY RUN $N gate=$G seeds='$SEEDS'"; echo "   env: $E"; continue; fi
  bash scripts/arms/gates/check_run_name_free.sh "$N" $SEEDS || exit 1
# Unique smoke name per attempt: a fixed name lets a re-run skip on a stale
# checkpoint and "complete" in 10 seconds.
  SMOKE_TAG="${SMOKE_TAG:-$(date +%m%d%H%M)}"
  SMK=$(sbatch --parsable --job-name="sm_${N}_${SMOKE_TAG}" --constraint=milan \
    --export=ALL,SMOKE=1,MV_DRO_ITERS=2,KEEP_EPOCH_CKPTS=1,MV_SEED=1,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_sm_${N}_${SMOKE_TAG}_s1 \
    scripts/train/run_foldin_coevo.sh)
  GJ=$(sbatch --parsable --dependency=afterok:$SMK --job-name="gate_$N" \
    --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=4G \
    --wrap="cd $REPO_ROOT; ./$G slurm-$SMK.out")
  echo "$N: smoke=$SMK gate=$GJ"
  for s in $SEEDS; do
    J=$(sbatch --parsable --dependency=afterok:$GJ --job-name="gt_${N}_s$s" \
      --time=48:00:00 --constraint=milan \
      --export=ALL,MV_DRO_ITERS=30,KEEP_EPOCH_CKPTS=1,MV_SEED=$s,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gt_${N}_s$s \
      scripts/train/run_foldin_coevo.sh)
    echo "    seed$s=$J"
  done
done
