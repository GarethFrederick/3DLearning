#!/bin/bash
# a46_nogen30 — a34's recipe with the diffusion NEVER trained and NEVER sampled.
# Not byte-identical to a34: the diffusion consumes the global torch RNG, so
# removing it shifts every later draw including the head edit's amplitude and
# coin flip. Treat it as an independent draw of a34 that also proves the
# generator is unnecessary. If it lands OUTSIDE a34's seed spread, something
# reads the diffusion that the audit missed — that is the falsification test.
set -eu
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
[ "${GO:-0}" = "1" ] || { echo "Refusing to submit. Set GO=1."; exit 1; }

INST=$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt
COMMON="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,KEEP_EPOCH_CKPTS=1,ICON_INSTANCE=$INST"
COMMON="$COMMON,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1,GEN_TRANSPORT_ALPHA=120"
COMMON="$COMMON,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300,GEN_TRANSPORT_LEASH=1"
COMMON="$COMMON,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1,INSTR_HEAD_AUG=1"
COMMON="$COMMON,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_WIDTH=1"
COMMON="$COMMON,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1,MV_DRO_ITERS=30"
COMMON="$COMMON,MV_SKIP_GEN_TRAIN=1,MV_SKIP_GEN_SAMPLE=1,INSTR_TREND_DEGREE_AUTO=1"

for s in 1 2 3; do
  J=$(sbatch --parsable --job-name=gt_a46_nogen30_s$s --time=48:00:00 \
    --export=ALL,MV_SEED=$s,$COMMON,MV_RUN_SUFFIX=_gt_a46_nogen30_s$s \
    scripts/train/run_foldin_coevo.sh)
  echo "seed $s = $J"
done
