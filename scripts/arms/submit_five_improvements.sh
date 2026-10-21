#!/bin/bash
# Five arms from the failure analysis of a50-a53. Each flips knobs that already
# exist; no code changed, so nothing here can disturb a34.
#  a55_twolobe  two opposed trend-derived lobes, narrow and summed. a34's 8h
#     half-width bell at 14.0h pushes prices down from 16h to 22h, where its own
#     training data says the block rises. AIM_TREND_MULTI keeps the top-2 blocks
#     with the sign each slope implies (14h down, 22h up), ADDITIVE gives both
#     full dose, and WIDTH_H=4 keeps each lobe inside its own 8h block, so the
#     lobes do not merge into a whole-day tilt.
#  a56/a57/a58  the width sweep with discovery off. a34 discovers its width, so
#     it is not the 8h point of this sweep; a57 is.
#  a59_drospo  warm-start from WassersteinDRO_SPO instead of DFL.
#     WassersteinDRO_SPO_final_epoch=92 is already in training_metadata.json.
# None of these may be promoted on an OODval score; read them as diagnostics
# until scored on the held-out years.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }

INST=$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt
# a34's exact knob set, with MV_DRO_ITERS and INSTR_HEAD_DISCOVER_WIDTH lifted
# out so each arm sets its own. Passing a key twice on one --export line is
# undefined, so neither may live in the base.
A34="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,KEEP_EPOCH_CKPTS=1,ICON_INSTANCE=$INST"
A34="$A34,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1,GEN_TRANSPORT_ALPHA=120"
A34="$A34,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300,GEN_TRANSPORT_LEASH=1"
A34="$A34,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1,INSTR_HEAD_AUG=1"
A34="$A34,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1"
A34="$A34,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1"

run () {  # $1 arm  $2 extra-knobs
  bash scripts/arms/gates/check_run_name_free.sh "$1" 1 2 3
  SM=$(sbatch --parsable --job-name=gtsm_$1 --time=03:00:00 \
      --export=ALL,SMOKE=1,MV_DRO_ITERS=2,MV_SEED=1,$A34,$2,MV_RUN_SUFFIX=_gtsm_$1 \
      scripts/train/run_foldin_coevo.sh)
  GA=$(sbatch --parsable --job-name=gate_$1 --dependency=afterok:$SM \
      --account=fr57 --partition=fitc --qos=fitcq --time=00:10:00 \
      --cpus-per-task=2 --mem=4G --wrap="bash scripts/arms/gates/banner_check_five.sh $1 slurm-$SM.out")
  echo "  $1: smoke=$SM gate=$GA"
  for s in 1 2 3; do
    J=$(sbatch --parsable --job-name=gt_$1_s$s --time=24:00:00 \
        --constraint=milan --dependency=afterok:$GA \
        --export=ALL,MV_DRO_ITERS=8,MV_SEED=$s,$A34,$2,MV_RUN_SUFFIX=_gt_$1_s$s \
        scripts/train/run_foldin_coevo.sh)
    echo "    seed $s = $J"
  done
}
W4="INSTR_HEAD_DISCOVER_WIDTH=0,INSTR_HEAD_WIDTH_H=4"
run a55_twolobe "$W4,INSTR_HEAD_AIM_TREND_MULTI=1,INSTR_HEAD_AIM_TREND_MULTI_K=2,INSTR_HEAD_MULTI_TIE=1,INSTR_HEAD_ADDITIVE=1"
run a56_width04 "$W4"
run a57_width08 "INSTR_HEAD_DISCOVER_WIDTH=0,INSTR_HEAD_WIDTH_H=8"
run a58_width12 "INSTR_HEAD_DISCOVER_WIDTH=0,INSTR_HEAD_WIDTH_H=12"
run a59_drospo  "INSTR_HEAD_DISCOVER_WIDTH=1,MV_WARM_SOURCE=WassersteinDRO_SPO"
