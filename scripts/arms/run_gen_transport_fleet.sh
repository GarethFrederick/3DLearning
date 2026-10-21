#!/bin/bash
# Launches the GEN_TRANSPORT co-evolution fleet (FOLDIN_OBJECTIVE=divergence,
# MV_DRO_ITERS=30) on the sub100 instance through run_foldin_coevo.sh, one job
# per arm and seed. GO=1 submits; ARMS and SEEDS pick a subset.
#   a1_stall       = plain generator fold-in, no transport
#   gt_headonly    = head-side solar transport only, no GEN_TRANSPORT; the
#                    generator's own share is a3_leashon minus gt_headonly
#   a3_leashon     = GEN_TRANSPORT alpha=120, leash on (the treatment)
#   a3_dis_valid   = fixed alpha=2000, leash on: deep but valid (clipped at -1000)
#   a3_dis_inval   = fixed alpha=2000, leash off: below the market floor
#   a4_retrainreal = DDPM trained on real midday prices (the ceiling)
# The a3_dis pair fixes alpha with ALPHA_MIN=ALPHA=2000, so a shared seed matches
# the pair window by window and only the leash differs.
# Score with poc_eval_ood.py (POC_RUN_PREFIX=gt). The in-training midday_p5 is a
# mechanism check, never the headline.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
INST="$PWD/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
SEEDS="${SEEDS:-1 2 3 4 5}"

# Only the thesis arm survives in this launcher. The other cells it once held
# (gt_headonly, a3_*, a4_retrainreal) used knobs that the 2026-10-21 cleanup
# removed, so they would have run as a1_stall under a different name.
declare -A ARM_ENV=(
  [a1_stall]="FOLDIN_OBJECTIVE=divergence"
)
ARMS="${ARMS:-a1_stall}"
for arm in $ARMS; do
  [ -n "${ARM_ENV[$arm]:-}" ] || { echo "FATAL: unknown arm '$arm' (known: ${!ARM_ENV[*]})"; exit 1; }
done

# Safety: require an explicit GO=1 so a stray `bash run_gen_transport_fleet.sh`
# cannot fire 30 full-node 24h jobs before the arm set is confirmed.
if [ "${GO:-0}" != "1" ]; then
  n_arms=$(echo $ARMS | wc -w); n_seeds=$(echo $SEEDS | wc -w)
  echo "GUARD: this launches $((n_arms*n_seeds)) full-node 24h jobs (arms: $ARMS | seeds: $SEEDS)."
  echo "  GO=1 bash scripts/arms/run_gen_transport_fleet.sh"
  echo "  GO=1 ARMS=\"a1_stall\" SEEDS=\"1 2 3\" bash scripts/arms/run_gen_transport_fleet.sh  # subset"
  exit 0
fi

echo "GEN_TRANSPORT fleet on sub100 | arms: $ARMS | seeds: $SEEDS"
# KEEP_EPOCH_CKPTS=1: the default cleanup keeps only the best epoch, which
# collapses the generator-alone trajectory and poc_eval_ood's 'fixed' comparison
# to one point.
for arm in $ARMS; do
  for s in $SEEDS; do
    jid=$(sbatch --parsable --job-name="gt_${arm}_s${s}" \
      --export=ALL,ICON_INSTANCE="$INST",MV_SEED=$s,KEEP_EPOCH_CKPTS=1,${ARM_ENV[$arm]},MV_RUN_SUFFIX=_gt_${arm}_s${s} \
      scripts/train/run_foldin_coevo.sh)
    echo "  submitted gt_${arm}_s${s} -> $jid"
  done
done
