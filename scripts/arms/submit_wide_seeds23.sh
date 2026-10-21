#!/bin/bash
# Seeds 2 and 3 for the wide-head arms a70, a71 and a69. No submit script was
# saved for them, so their recipes were recovered from their own seed-1 logs:
#   a70 = a46's environment minus the two skip-generator knobs, at 8 rounds.
#   a71 = the bare co-evolution default at 30 rounds.
#   a69 = a70's environment with the trend aim off, the discount at zero, and
#         the oracle price file.
# Each arm smokes at 2 rounds, then banner_recon_check.sh compares the smoke's
# mechanism set against that arm's seed-1 log. banner_match.sh is not used: it
# reads only "[FOLD-IN]" lines, so a recipe that dropped INSTR_HEAD_AIM_TREND
# would pass it.
# The gate cannot see MV_DRO_ITERS (the smoke runs 2 rounds by design). Read the
# dose in each seed job's own log line "MV_DRO_ITERS=".
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance missing: $INST"; exit 1; }
[ -x scripts/arms/gates/banner_recon_check.sh ] || { echo "FATAL: gate missing or not executable"; exit 1; }

WIDE="PRED_NORM_LO=-0.5,PRED_NORM_HI=1.5"
A34="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1"
A34="$A34,GEN_TRANSPORT_ALPHA=120,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300"
A34="$A34,GEN_TRANSPORT_LEASH=1,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1"
A34="$A34,INSTR_HEAD_AUG=1,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_WIDTH=1"
A34="$A34,INSTR_TREND_DEGREE_AUTO=1"

ORACLE_CSV="$REPO_ROOT/Data/AEMO_VIC1_ORACLE_traintest.csv"
[ -f "$ORACLE_CSV" ] || { echo "FATAL: oracle price file missing: $ORACLE_CSV"; exit 1; }

ENV_a70="$A34,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1,$WIDE"
ENV_a71="$WIDE"
# HEAD_REAL_CSV as well as DATA_TRAIN: the head reads its real base rows from
# HEAD_REAL_CSV, not DATA_TRAIN. Setting only DATA_TRAIN trains the head on the
# 2015-2020 past while calling the arm an oracle.
ENV_a69="$A34,INSTR_HEAD_ALPHA_MAX=0,DATA_TRAIN=$ORACLE_CSV,HEAD_REAL_CSV=$ORACLE_CSV,$WIDE"
DOSE_a70=8 ; REF_a70=results/logs/train/slurm-59655382.out ; NAME_a70=a70_a34wide
DOSE_a71=30; REF_a71=results/logs/train/slurm-59655385.out ; NAME_a71=a71_a1wide
DOSE_a69=8 ; REF_a69=slurm-59655272.out ; NAME_a69=a69_oraclewide

ARMS="${ARMS:-a70 a71 a69}"
SEEDS="${SEEDS:-2 3}"
for a in $ARMS; do
  eval "E=\$ENV_$a; D=\$DOSE_$a; R=\$REF_$a; N=\$NAME_$a"
  [ -s "$R" ] || { echo "FATAL: reference log $R missing for $a"; exit 1; }
  echo "$E" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: $a env not comma-safe"; exit 1; }
  if [ "${GO:-0}" != "1" ]; then
    echo "DRY RUN $N  dose=$D  ref=$R  seeds='$SEEDS'"; echo "   env: $E"; continue
  fi
  bash scripts/arms/gates/check_run_name_free.sh "$N" $SEEDS || exit 1
# SMOKE_TAG gives every attempt a unique smoke name. A fixed name lets a re-run
# find a void attempt's checkpoint, print "[skip]" and exit having trained nothing.
  SMOKE_TAG="${SMOKE_TAG:-$(date +%m%d%H%M)}"
  SMK=$(sbatch --parsable --job-name="sm_${N}_${SMOKE_TAG}" --constraint=milan \
    --export=ALL,SMOKE=1,MV_DRO_ITERS=2,KEEP_EPOCH_CKPTS=1,MV_SEED=2,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_sm_${N}_${SMOKE_TAG}_s2 \
    scripts/train/run_foldin_coevo.sh)
  GATE=$(sbatch --parsable --dependency=afterok:$SMK --job-name="gate_$N" \
    --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=4G \
    --wrap="cd $REPO_ROOT; scripts/arms/gates/banner_recon_check.sh slurm-$SMK.out $R")
  echo "$N: smoke=$SMK gate=$GATE (ref $R, dose $D)"
  for s in $SEEDS; do
    J=$(sbatch --parsable --dependency=afterok:$GATE --job-name="gt_${N}_s$s" \
      --time=48:00:00 --constraint=milan \
      --export=ALL,MV_DRO_ITERS=$D,KEEP_EPOCH_CKPTS=1,MV_SEED=$s,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gt_${N}_s$s \
      scripts/train/run_foldin_coevo.sh)
    echo "    seed$s=$J"
  done
done
