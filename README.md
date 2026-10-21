# Robust price forecasts for electricity-aware task scheduling

This repository trains a price forecaster for a task scheduler. The scheduler
plans 100 tasks against the next day's electricity prices. Prices in later years
differ from the training years. The forecaster must still lead the scheduler to
cheap schedules in those later years.

The code extends the public release of 3D-Learning (paper: arXiv 2602.02943,
accepted at INFOCOM 2026). The original release carries the MIT licence,
"Copyright (c) 2026 JiaqiWen-25". The original paper applies the method to
resource provisioning for language-model servers. This repository applies the
method to electricity prices, changes it, and compares the changed method with
the original.

## Words used here

- **The scheduler**: a mixed-integer program that places tasks on machines in
  half-hour slots. Gurobi solves it.
- **The forecaster**: a neural network that predicts 48 half-hour prices for the
  next day. The code calls it *the head*.
- **Decision regret**: the extra cost that the scheduler pays because it planned
  against the forecast and not against the true prices.
- **Normalised decision regret**: the total decision regret over all scored days,
  divided by the total size of the best possible cost on those days
  (`Utils/Algos/ml.py`, `avg_norm_task_loss`). Lower is better.
- **An arm**: one named set of switches under test, for example `a72_nogenwide`.
  `ARMS.md` lists every arm that the thesis reports.
- **3D-Learning**: the published method. It trains a diffusion model to invent
  severe price days, and then trains the forecaster on them.

## The data and the task

- Prices: AEMO half-hourly prices and demand for Victoria. The training years are
  2015-2020. The held-out years are 2021-2022 (the "gas era") and 2023-2025 (the
  "renewables era"). Files: `Data/AEMO_VIC1_*.csv`.
- Task: the ICON energy-cost aware scheduling problem (CSPlib problem 059).
  The thesis uses instance `sub100_s0`: 100 tasks, 2 machines, 3 resources
  (`Data/sampleinstances/_sub07/sub100_s0/instance.txt`).

## The reported method

The reported method is arm `a72_nogenwide`. It has three parts.

1. **Trend-aimed augmentation.** The method copies real 2015-2020 price days. It
   makes one part of each copy cheaper. It reads two things from the 2015-2020
   data only: which hours fell fastest in price, and how far to push them.
2. **No generative model.** No output of the diffusion model reaches the
   forecaster. The code still contains the inherited diffusion parts.
3. **A wide head.** The forecaster's output range is wider than the training
   prices. The launcher comment gives about -$107 to $358, against $9.20 to
   $242.07 for the narrow head (`scripts/arms/submit_a72_nogenwide.sh`).

`ALGORITHM.md` gives each step with its source line. It documents arm
`a34_honestdose`, which uses the same augmentation with the narrow head, the
generator on and 8 rounds. Arm `a72_nogenwide` differs in those three settings.
`ARMS.md` gives the exact switches.

## The headline result

Both rows use the wide head. The score is normalised decision regret; lower is
better. Each mean is over 5 seeds. Source: the scored files that
`scripts/eval/run_poc_eval_measure.sh` writes, as read on 2026-09-24. The
thesis holds those files; this repository holds the code and data to
regenerate them.

| held-out era | 3D-Learning (5 seeds) | this method, `a72_nogenwide` (5 seeds) |
|---|---|---|
| renewables, 2023-2025 | 0.8884 | 0.4496 |
| gas, 2021-2022 | 0.6112 | 0.3718 |

The thesis holds every other number, with its spread, its seed count and its
caveats. Do not quote these two rows without the thesis caveats.

## What you need

- A Slurm cluster. Every step in this repository runs as an `sbatch` job. The
  `#SBATCH` lines name the partition and account of the Monash M3 cluster.
  Change them for your cluster.
- A GPU for the diffusion-model training step only.
- Gurobi. Gurobi is a commercial solver and a required dependency. This project
  used `gurobipy` 13.0.2 under an academic licence. Get an academic licence at
  https://www.gurobi.com/academia/academic-program-and-licenses/. This
  repository does not contain Gurobi.
- Python 3.12.0 and the packages in `requirements.txt`.

Create the environment:

```bash
module load miniforge3
conda create -n 3d-learning python=3.12.0
conda activate 3d-learning
pip install -r requirements.txt
```

The launchers call the environment's Python at
`~/scratch/conda/envs/3d-learning/bin/python`. Edit the `PY` line if your path
differs.

## Layout

```
main_energy_mv.py        training entry point for every arm
main_energy.py           training entry point for the classical baselines
poc_eval_ood.py          scoring entry point
train_ddpm_mv_cond.py    diffusion-model training
validate_ddpm_mv_cond.py diffusion-model check and calibration
Utils/                   the method: Algos/, Diffusion/, Net/, the scheduler code
scripts/data/            build the AEMO files
scripts/train/           arm runner, diffusion training, baselines/
scripts/arms/            one launcher per reported arm, and gates/
scripts/eval/            scoring and solver-budget launchers
scripts/analysis/        statistics and diagnostics
scripts/figures/         thesis figures and tables
tests/                   unit tests
Data/                    AEMO prices and the ICON instances
results/                 created by the scripts: scored files, training rows, figures, tables
```

## How to reproduce the thesis

Submit every job from the repository root.

1. **Build the data.** Run `sbatch scripts/data/run_regen_aemo.sh`. It runs
   `Data/preprocess_aemo.py`, then `Data/make_ood_val_splits.py`. It prints the
   training file's checksum before and after. The two must match.
2. **Train the diffusion model.** Run
   `sbatch scripts/train/run_train_ddpm_mv_cond.sh`. It writes the model to
   `Utils/Models/`. The reported arm does not use the model's output, but the
   arm runner passes the model's path to the training loop in every run.
3. **Train the classical baselines.** Run the three launchers in
   `scripts/train/baselines/`: `run_pto.sh`, `run_dfl.sh` and `run_wdro.sh`.
   They call `main_energy.py`.
4. **Train an arm.** Run `GO=1 bash scripts/arms/submit_<arm>.sh`. Without
   `GO=1` the launcher refuses to submit. Each launcher submits
   `scripts/train/run_foldin_coevo.sh` once per seed. `ARMS.md` names the
   launcher for each thesis row.
5. **Score the arms.** Run `scripts/eval/run_poc_eval_measure.sh` through
   `sbatch`. Set `POC_RUN_PREFIX`, `POC_ARMS`, `POC_SEEDS` and
   `POC_EVAL_REGIMES` on `--export`. The scored files go to
   `POC_EVAL_OUT_DIR`. Scoring must run on one CPU family: the launcher pins
   `--constraint=milan`. Gurobi's results differ between CPU families. Two
   scorings of the same models disagreed on 274 of 320 windows across families
   (launcher comment). Do not remove the constraint.
6. **Build the figures and tables.** The scripts in `scripts/figures/` write the
   figures. The scripts in `scripts/analysis/` write the statistics.
   `scripts/analysis/build_tseedstats.py` builds the per-seed table and checks
   each seed count against the main table.

| thesis output | script |
|---|---|
| F2, F4-F10 | `scripts/figures/make_thesis_figures.py` |
| F0, F21-F23 | `scripts/figures/fig_thesis_1016.py` |
| F12 | `scripts/figures/fig_block_width.py` |
| F13-F16, `t2-main.tex`, `t-seed-stats.tex` | `scripts/figures/fig_supervisor.py` |
| F17, F19 | `scripts/figures/fig_part2_extra.py` |
| F18 | `scripts/figures/fig_worklimit.py` |
| F20 | `scripts/figures/fig_f20.py` |
| `t5-random.tex` | `scripts/figures/fig_results_1008.py` |
| `t-seed-stats.tex` (check against `t2-main.tex`) | `scripts/analysis/build_tseedstats.py` |

No script writes `comparisons.tex`, `schedule.tex`, `t6-mechanism.tex`,
`t7-recency.tex` or `table_configs.tex`. Each of them except `t7-recency.tex`
describes its values in a header comment.
`scripts/figures/fig_thesis_1016.py` checks figure F23 against `schedule.tex`.

## What is not in the repository

- Model checkpoints. They take 282 GB on disk (`du -sh Utils/Models`,
  2026-10-21). Git ignores `Utils/Models/`, `Utils/Gen_Data/` and
  `Utils/Results/`.
- Scheduler logs (`slurm-*.out`).
- Run outputs. Git ignores `results/`; the scripts create it. The scorer
  writes to `results/scores/`, the analysis scripts to `results/analysis/`,
  the figure scripts to `results/thesis/`, and the training runner writes a
  per-round JSON row per run to `results/training_rows/`. The scorer reads
  the training log of a run (`slurm-<job>.out`, at the repository root or
  under `results/logs/train/`) to find the saved round that the run kept.
- The British carbon-intensity work (a second domain that the thesis does not
  report). It stays outside this repository.

## Credits for data

- AEMO prices and demand: "Contains AEMO data."
- ICON scheduling instances: "ICON energy-cost aware scheduling instances by
  Helmut Simonis, Barry O'Sullivan, Deepak Mehta, Barry Hurley and Milan De
  Cauwer, CSPlib problem 059, used under CC-BY 4.0."
- The ICON instance `sub100_s0` is a 100-task subsample that
  `scripts/data/subsample_icon.py` draws from the CSPlib instances.

`DATA_SOURCES.md` gives each source, its licence and how the licence was checked.

## Licence

MIT. See `LICENSE`. The original copyright of the 3D-Learning authors stays.
The extensions carry a second copyright line.
