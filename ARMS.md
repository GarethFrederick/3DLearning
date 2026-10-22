# Arms that the thesis reports

An **arm** is one named set of switches under test. Each training run of an arm
uses one seed. The run suffix is `_gt_<arm>_s<seed>`. The arm names match
`scripts/analysis/build_tseedstats.py` and the main table, `t2-main.tex`.

Every launcher sits in `scripts/arms/`, except the classical baselines. Each
launcher refuses to submit unless you set `GO=1`. Each launcher submits
`scripts/train/run_foldin_coevo.sh` once per seed.

## Names used in the table

- **COMMON72**: the `COMMON=` lines of `submit_a72_nogenwide.sh`. The next
  section lists them.
- **A19ENV**: the switches in `submit_a23_r2.sh`.
- **A34**: the switches of arm `a34_honestdose`: the a44 `COMMON` without
  `INSTR_TREND_DEGREE_AUTO`, with `MV_DRO_ITERS=8`.
- **WIDE**: `PRED_NORM_LO=-0.5,PRED_NORM_HI=1.5`. This widens the forecaster's
  output range.
- **Rounds**: `MV_DRO_ITERS`, the number of passes of the outer training loop.

## The table

| Thesis row | Arm | Launcher | Env |
|---|---|---|---|
| Wide, reported, 20 seeds | a72_nogenwide | submit_a72_nogenwide.sh (s1-3), submit_seeds45_a72_a74.sh, submit_a72_seeds6to20.sh | COMMON72 |
| Wide, generator on, 8 rounds | a70_a34wide | submit_wide_seeds23.sh (s1 ad hoc, chain 59655380-82) | A34 + trend aim/depth + WIDE, MV_DRO_ITERS=8 |
| Wide, block width 2/6/8 h | a78_bw2/6/8 | submit_a78_blockwidth.sh | COMMON72 + INSTR_TREND_BLOCK_HOURS |
| Wide, real reference pool | a79_realref | submit_a79_realref.sh | COMMON72 + FOLDIN_REF_POOL=real |
| Wide, 3D-Learning | a71_a1wide | submit_wide_seeds23.sh (+ ad hoc s1, s4-5) | WIDE only |
| Wide, wrapper off | a73_vanilla3dl | submit_controls_a73_a74.sh | FOLD_IN=0 + WIDE |
| Wide, zero discount | a75_wzerodose | submit_a75_wzerodose.sh | COMMON72 − DEPTH_TREND + INSTR_HEAD_ALPHA_MAX=0 |
| Wide, random aim, 5 draws | a74_randomaim | submit_controls_a73_a74.sh, submit_seeds45_a72_a74.sh | COMMON72 + INSTR_HEAD_AIM_RANDOM=1 |
| Wide, block 16-20 h | a76_block4 | submit_a76_block4.sh | COMMON72 + INSTR_HEAD_AIM_BLOCK=4 |
| Wide, fixed $120 cap, no trend depth | a80_fixed120 | submit_a80_fixed120.sh (s1-3) | COMMON72 − DEPTH_TREND + INSTR_HEAD_ALPHA_MAX=120 |
| Narrow, no generator, 30 | a46_nogen30 | submit_a46.sh | COMMON72 − WIDE |
| Narrow, generator on, 30 | a44_long30 | submit_a44.sh | a46 − MV_SKIP_GEN_* |
| Narrow, generator on, 8 | a34_honestdose | ad hoc (logs 58782818-20, chain 59613309-12) | a44 COMMON − DEGREE_AUTO, MV_DRO_ITERS=8 |
| Narrow, WDRO warm start | a59_drospo | submit_five_improvements.sh | A34 + MV_WARM_SOURCE=WassersteinDRO_SPO, 8 rounds (A34 already runs the width search) |
| Narrow, observed depth | a77_obsdepth | submit_a77_obsdepth.sh | a46 + INSTR_HEAD_DEPTH_OBSERVED=1 |
| Narrow, 3D-Learning | a1_stall | run_gen_transport_fleet.sh:51 (+ chain 59613305-08) | FOLDIN_OBJECTIVE=divergence only |
| Divergence-aimed $120 | a19_discwidth | ad hoc (§8.24, log 58587032) | A19ENV + CONSOL_EVERY=5 |
| No discount, all years | a23_zerodose_r2 | submit_a23_r2.sh | A19ENV + ALPHA_MAX=0 + CONSOL_EVERY=5 |
| No discount, 2019-20 | a25_recency_r2 | resubmit_a22_a26_correct_instance.sh | A19ENV + HEAD_REAL_FROM_YEAR=2019 + ALPHA_MAX=0 |
| Classical baselines | pto/wdropto/wdrospo/dfl | scripts/train/baselines/run_pto.sh, run_dfl.sh, run_wdro.sh | main_energy.py STAGE=pto/dfl/wdro |
| Persistence, and persistence with a $120 dip at 14:00 | persist, persistdip120 | scripts/eval/make_persistence_models.py, then scoring | no training |

"Ad hoc" means a person typed the `sbatch` line. No launcher file holds it. The
numbers after "chain" and "logs" are Slurm job ids. "§8.24" names a section of
the preregistration record, which is not in this repository.

The classical baselines and the warm start `DFL-epoch41.pth` were trained on
the ICON instance `sample01` (25 tasks), with `main_energy.py`'s default
`ICON_INSTANCE`. Every other arm trains on `sub100_s0`. All rows are scored on
`sub100_s0`.

## Selection without 2021-2025 data

Set `POC_INDIST=1` on `scripts/eval/run_poc_eval_measure.sh` to score, for each
seed, the saved round with the best in-distribution validation regret (the
validation split inside the 2015-2020 file) in place of the selected and last
rounds. Set `POC_EVAL_OUT_DIR` to a separate folder. The rule reads the per-round
`Val Norm Regret` lines of the training log, so the log must be at the
repository root or in `results/logs/train/`.

## How COMMON72 is built

`submit_a72_nogenwide.sh` builds one string, `COMMON`, from these lines:

```bash
INST=<repo>/Data/sampleinstances/_sub07/sub100_s0/instance.txt
COMMON="HEAD_ROWS_MULT=5,CONSOL_EVERY=5,KEEP_EPOCH_CKPTS=1,ICON_INSTANCE=$INST"
COMMON="$COMMON,FOLDIN_OBJECTIVE=divergence,GEN_TRANSPORT=1,GEN_TRANSPORT_ALPHA=120"
COMMON="$COMMON,GEN_TRANSPORT_RAMP=800,GEN_TRANSPORT_SANITY_FLOOR=-300,GEN_TRANSPORT_LEASH=1"
COMMON="$COMMON,GEN_TRANSPORT_MODE=instructor_typical,FOLDIN_HEAD_ONLY=1,INSTR_HEAD_AUG=1"
COMMON="$COMMON,HEAD_TRAIN_REAL=1,INSTR_REAL_WINDOWS=1,INSTR_HEAD_DISCOVER_WIDTH=1"
COMMON="$COMMON,INSTR_HEAD_AIM_TREND=1,INSTR_HEAD_DEPTH_TREND=1,MV_DRO_ITERS=30"
COMMON="$COMMON,MV_SKIP_GEN_TRAIN=1,MV_SKIP_GEN_SAMPLE=1,INSTR_TREND_DEGREE_AUTO=1"
COMMON="$COMMON,PRED_NORM_LO=-0.5,PRED_NORM_HI=1.5"
```

The last line is WIDE. It is the only line that differs from `submit_a46.sh`.

Two switches in this string are inert. `GEN_TRANSPORT_SANITY_FLOOR` and
`INSTR_TREND_DEGREE_AUTO` were read only by code paths that the reported
method never used, and the 2026-10-21 cleanup removed those paths. The
launchers keep the two switches so the recorded recipe stays exact; no code
reads them, and setting them to any value changes nothing.
The launcher then passes `--export=ALL,MV_SEED=$s,$COMMON,MV_RUN_SUFFIX=_gt_a72_nogenwide_s$s`
to `sbatch`, once for each seed in `SEEDS` (default `1 2 3`). Do not remove a
switch from this list. A missing switch makes a different arm and prints no
warning.

## The rule for a smoke run

A smoke run is a short test run of 2 rounds. `SMOKE=1` alone does not shorten a
run, because `COMMON` already sets `MV_DRO_ITERS=30`. Put `MV_DRO_ITERS=2` after
`$COMMON` on the `--export` line. The later value wins. Two old launchers put
`SMOKE=1,MV_DRO_ITERS=2` before `$COMMON`, and their smoke runs ran 30 rounds.
Build `$COMMON` from the lines above first. Then use this order:

```bash
sbatch --constraint=milan --time=03:00:00 \
  --export=ALL,MV_SEED=1,$COMMON,SMOKE=1,MV_DRO_ITERS=2,MV_RUN_SUFFIX=_smoke_a72_s1 \
  scripts/train/run_foldin_coevo.sh
```

Check the log after the run. It must show `MV_DRO_ITERS=2` and at most 2
`| Outer epoch` lines.
