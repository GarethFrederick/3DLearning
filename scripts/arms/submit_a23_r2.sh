#!/bin/bash
# a23_zerodose_r2 = the a19 environment plus one knob, INSTR_HEAD_ALPHA_MAX=0:
# all years, no discount. Same submit shape as
# resubmit_a22_a26_correct_instance.sh, so ICON_INSTANCE and KEEP_EPOCH_CKPTS
# cannot go missing. GO=1 to submit.
set -u
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$REPO_ROOT/main_energy_mv.py" ] || { echo "FATAL: run from the repo root (or set REPO_ROOT)"; exit 1; }
cd "$REPO_ROOT"
INST="$REPO_ROOT/Data/sampleinstances/_sub07/sub100_s0/instance.txt"
[ -f "$INST" ] || { echo "FATAL: instance missing: $INST"; exit 1; }
A19ENV="FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1,GEN_TRANSPORT_ALPHA=120,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300,GEN_TRANSPORT_LEASH=1,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1,INSTR_HEAD_AUG=1,HEAD_TRAIN_REAL=1,INSTR_HEAD_DISCOVER_WIDTH=1"
E="$A19ENV,INSTR_HEAD_ALPHA_MAX=0"
A=a23_zerodose_r2
echo "$E" | tr ',' '\n' | grep -qvE '^[A-Z_]+=[^,]*$' && { echo "FATAL: env not comma-safe"; exit 1; }
[ -f scripts/arms/gates/banner_check_a23.sh ] || { echo "FATAL: gate missing"; exit 1; }
echo "  | env export-safe, gate present |"
# Refuse a run name already in use (check_run_name_free.sh).
bash scripts/arms/gates/check_run_name_free.sh "$A" 1 2 3 || exit 1
[ "${GO:-0}" = "1" ] || { echo "  DRY RUN. GO=1 to submit $A with $E"; exit 0; }
PY="${PY:-$HOME/scratch/conda/envs/3d-learning/bin/python}"
PYC="module load miniforge3; source activate 3d-learning; cd $REPO_ROOT; $PY -m py_compile main_energy_mv.py Utils/Algos/ddd_learning.py && echo COMPILE_OK"
CMP=$(sbatch --parsable --job-name="${A}_cmp" --account=fr57 --partition=fitc --qos=fitcq \
      --time=00:10:00 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --wrap="$PYC")
SMK=$(sbatch --parsable --dependency=afterok:$CMP --job-name="gtsm_${A}" \
      --export=ALL,SMOKE=1,CONSOL_EVERY=1,KEEP_EPOCH_CKPTS=1,MV_SEED=1,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gtsm_${A}_s1 scripts/train/run_foldin_coevo.sh)
BAN=$(sbatch --parsable --dependency=afterok:$SMK --job-name="${A}_banner" \
      --account=fr57 --partition=fitc --qos=fitcq --time=00:05:00 --nodes=1 --ntasks=1 \
      --cpus-per-task=1 --mem=4G \
      --wrap="cd $REPO_ROOT; SMOKE_LOG=slurm-$SMK.out bash scripts/arms/gates/banner_check_a23.sh")
echo "$A: cmp=$CMP smoke=$SMK gate=$BAN"
for s in 1 2 3; do
  J=$(sbatch --parsable --dependency=afterok:$BAN --job-name="gt_${A}_s$s" \
      --export=ALL,CONSOL_EVERY=5,KEEP_EPOCH_CKPTS=1,MV_SEED=$s,ICON_INSTANCE=$INST,$E,MV_RUN_SUFFIX=_gt_${A}_s$s scripts/train/run_foldin_coevo.sh)
  echo "    seed$s=$J"
done
