#!/bin/bash
#SBATCH --job-name="foldin_coevo"
#SBATCH --time=24:00:00
#SBATCH --partition=fitc
#SBATCH --qos=fitcq
#SBATCH --account=fr57
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

# Runs the 3D-Learning loop (ddd_learning.train via main_energy_mv.py) with the
# fold-in co-evolution wrapper. With FOLD_IN=1, each outer epoch folds the
# generator's selected worst-case windows into a replay buffer (the generator's
# s0 set and the head's training set) and re-derives the divergence reference
# from the live head. Per-round metrics land in gate_out/foldin_coevo_<RUN_NAME>.json.
# Requires ICON_INSTANCE; set KEEP_EPOCH_CKPTS=1 for any arm you will score.
# Overridable:
#   FOLDIN_OBJECTIVE  divergence|random|regret|off   (default divergence)
#   FOLDIN_REF        live-head|night-prior|flat      (default live-head)
#   FOLDIN_KEEP_FRAC  (default 0.2)   FOLDIN_REPLAY_FRAC (default 0.3)
#   SMOKE=1           fast 2-outer-epoch pipeline validation (see below)

module load miniforge3
module load gurobi/13.0.0
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/apps/gurobi/13.0.0/gurobi.lic}"
unset GUROBI_HOME
source activate 3d-learning
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"

PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
echo "  | syntax check |"
$PY -m py_compile main_energy_mv.py Utils/Algos/ddd_learning.py Utils/Algos/ddd/*.py Utils/utils.py \
    Utils/Diffusion/DDPM.py Utils/Net/DDPM_net_mv.py Utils/diffusion_utils.py \
    Utils/datasets.py Utils/Downstream/opt_oracle.py \
    || { echo "SYNTAX ERROR"; exit 1; }
echo "  | syntax OK |"

echo "  | Python | $(which python)  | STAGE_MV | cond_mv  | FOLD_IN | ${FOLD_IN:-1}"

# main_energy_mv.py falls back to the 25-task sample01 instance when ICON_INSTANCE
# is unset. Scores on another instance use a different regret denominator and
# cannot be compared, so a missing knob must crash here, not 24 hours later.
if [ -z "${ICON_INSTANCE:-}" ]; then
  echo "FATAL: ICON_INSTANCE is unset. The campaign instance is"
  echo "       \$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
  echo "       Pass it on the sbatch --export line. Refusing to run on a default."
  exit 1
fi
[ -f "$ICON_INSTANCE" ] || { echo "FATAL: ICON_INSTANCE not found: $ICON_INSTANCE"; exit 1; }

# The 2026-10-21 cleanup removed the code behind these knobs. A launch that sets
# one would run a different arm from the one it names, with no warning, so it
# must crash here. GEN_TRANSPORT_SANITY_FLOOR and INSTR_TREND_DEGREE_AUTO are
# not listed: the reported launchers still set them and they were inert before.
REMOVED_KNOBS="CONSOL_RECENT_ROUNDS CONSOL_RETAIN_THRESH CONSOL_RETAIN_TOL CONSOL_SESSION_EPOCHS
CONSOL_SESSION_LR CONSOL_SESSION_REPLAY CONSOL_VALLOSS_TOL FOLDIN_SEED GEN_ROWS_MULT
GEN_TRANSPORT_ALPHA_MIN GEN_TRANSPORT_CAP GEN_TRANSPORT_EDIT_DOWN GEN_TRANSPORT_EDIT_UP
GEN_TRANSPORT_FLOOR GEN_TRANSPORT_SUNRISE GEN_TRANSPORT_SUNSET INSTR_DEPTH_FROM_LOOCV
INSTR_HEAD_ACCUM INSTR_HEAD_ADDITIVE INSTR_HEAD_AIM_FILTER INSTR_HEAD_AIM_TREND_MULTI
INSTR_HEAD_AIM_TREND_MULTI_K INSTR_HEAD_CENTROID INSTR_HEAD_DEPTH_CANDS INSTR_HEAD_DISCOVER_DEPTH
INSTR_HEAD_MULTI_TIE INSTR_HEAD_MULTI_TIE_MAX INSTR_HEAD_MULTI_TIE_MEMORY INSTR_HEAD_MULTI_TIE_PCT
INSTR_HEAD_PROB INSTR_HEAD_REGIME INSTR_HEAD_REPLAY_ON INSTR_HEAD_WAVE INSTR_HEAD_WAVE_HARM
INSTR_HEAD_WIDTH_CANDS INSTR_HEAD_WIDTH_H INSTR_TREND_GRID_MULTI INSTR_TREND_GRID_TOPK
INSTR_TREND_QUANTILE INSTR_WAVE_BANK_K INSTR_WAVE_REFRESH_EPOCHS INSTR_WIDTH_SCORE_EXPECTED
MV_BUDGET MV_DRO_LOSS_CLAMP MV_DRO_LR MV_DRO_NO_EARLY_STOP MV_ML_LR MV_ML_NO_LR_DECAY
MV_NO_LR_DECAY PLOT_JSM SOLAR_ADV SOLAR_ADV_BETA SOLAR_ADV_GRID SOLAR_ADV_ORACLE SOLAR_ADV_SHUFFLE Z0_ONLY"
for k in $REMOVED_KNOBS; do
  if [ -n "${!k+x}" ]; then
    echo "FATAL: $k is set, but the code that read it was removed on 2026-10-21."
    echo "       Unset it; the arm you want may no longer exist. Refusing to run."
    exit 1
  fi
done
# KEEP_EPOCH_CKPTS=1 stops the end-of-run cleanup deleting the per-epoch head
# checkpoints that poc_eval_ood.py loads; without it the eval silently snaps to
# the one kept checkpoint. Warn rather than fail: some probes want cleanup.
if [ "${KEEP_EPOCH_CKPTS:-0}" != "1" ]; then
  echo "  | WARNING: KEEP_EPOCH_CKPTS is not 1 — per-epoch checkpoints WILL be"
  echo "  |          deleted at the end. poc_eval_ood.py will then have only one"
  echo "  |          checkpoint to score. Set it for any arm you intend to eval. |"
fi
echo "  | ICON_INSTANCE=$ICON_INSTANCE  KEEP_EPOCH_CKPTS=${KEEP_EPOCH_CKPTS:-0} |"

# 3DL environment.
export STAGE_MV=cond_mv
# Gurobi parallelism: 16 workers x 8 threads = 128 cores. Threads per solve is
# the lever: 64 workers x 2 threads starved each solve, which hit TIME_LIMIT with
# no incumbent and fell back to greedy. MIP_FOCUS=1 forces an early incumbent.
# OMP/MKL/OPENBLAS=1 so the worker processes do not oversubscribe.
export TORCH_THREADS=64
export LP_WORKERS=16
export LP_THREADS_PER_SOLVE=8
export MIP_FOCUS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MV_GEN_ITERS=8
export MV_DRO_INNER=10
export MV_ML_INNER=5
export MV_ADJUST=10
export MV_ML_NO_EARLY_STOP=1
# ICON_AGG=1 trains at aggregated T=48: at native T=288 each epoch of in-loop
# SPO+ head training takes about an hour. Rare greedy fallbacks are tolerated on
# this LP training path. Reported regret always comes from the separate native
# T=288 MIP eval (ICON_AGG=0, hard-fails on greedy), never from T=48.
export ICON_AGG="${ICON_AGG:-1}"

# Seed and epoch-selection rule.
# MV_SEED: launchers set it per seed. Unset, every run uses 42.
export MV_SEED="${MV_SEED:-42}"
# VAL_OOD_REGRET / OOD_VAL_EPSILON: pick the epoch by constrained worst-case
# regret over both validation regimes (2021-22 gas, 2023-25 renewables), with the
# test windows held out. Every arm run through this file uses the same rule; the
# classical baselines (main_energy.py) select on 2015-2020 validation loss instead.
# Do not scope the selector to one regime, or the arms stop being comparable.
export VAL_OOD_REGRET="${VAL_OOD_REGRET:-1}"
export OOD_VAL_EPSILON="${OOD_VAL_EPSILON:-0.2}"
# VAL_OOD_REGRET_MIP=1 selects epochs by native MIP; default off. A MIP solve on
# every validation window each epoch is intractable, and on the aggregated grid
# the MIP path falls back to greedy and crashes under ASSERT_NO_GREEDY=1. LP-
# relaxation selection applies identically to every arm, so the comparison stays
# fair. EVAL_ORIGIN_STRIDE only affects test_task evals.
export VAL_OOD_REGRET_MIP="${VAL_OOD_REGRET_MIP:-0}"
export EVAL_ORIGIN_STRIDE="${EVAL_ORIGIN_STRIDE:-24}"

# Fold-in co-evolution env, all overridable from the submit line.
export FOLD_IN="${FOLD_IN:-1}"
export FOLDIN_OBJECTIVE="${FOLDIN_OBJECTIVE:-divergence}"
export FOLDIN_REF="${FOLDIN_REF:-live-head}"
export FOLDIN_KEEP_FRAC="${FOLDIN_KEEP_FRAC:-0.2}"
export FOLDIN_REPLAY_FRAC="${FOLDIN_REPLAY_FRAC:-0.3}"
export MV_RUN_SUFFIX="${MV_RUN_SUFFIX:-_coevo_div}"
# ASSERT_NO_GREEDY=0 (default): a greedy LP solve during fold-in oracle work is
# detected, its c_ref does not steer the next round, the JSON row is flagged
# greedy_contaminated and a warning prints, but the run continues. 1 hard-fails
# on the first greedy fallback. The reported MIP path hard-fails on greedy
# whatever this flag says.
export ASSERT_NO_GREEDY="${ASSERT_NO_GREEDY:-0}"

# SMOKE=1 shrinks the outer and inner loops so the whole fold-in pipeline runs in
# minutes. An explicit MV_DRO_ITERS on --export overrides the smoke default of 2.
if [ "${SMOKE:-0}" = "1" ]; then
  echo "  | SMOKE=1 — fast pipeline validation (2 outer epochs) |"
  export MV_DRO_ITERS="${MV_DRO_ITERS:-2}"
  export MV_DRO_INNER=2
  export MV_ML_INNER=2
  export MV_GEN_ITERS=1
else
  export MV_DRO_ITERS="${MV_DRO_ITERS:-30}"
fi

echo "  | FOLD_IN=$FOLD_IN  OBJECTIVE=$FOLDIN_OBJECTIVE  REF=$FOLDIN_REF"
echo "  | KEEP_FRAC=$FOLDIN_KEEP_FRAC  REPLAY_FRAC=$FOLDIN_REPLAY_FRAC  SUFFIX=$MV_RUN_SUFFIX"
echo "  | MV_DRO_ITERS=$MV_DRO_ITERS  ICON_AGG=$ICON_AGG"

$PY main_energy_mv.py
