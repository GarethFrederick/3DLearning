# The method, written out

This document gives the algorithm of the arm `a34_honestdose`. That arm has the
best score of all arms. This text follows ASD-STE100.

I read every step in the code. I then confirmed every step against the log of
a34 seed 1 (`slurm-58782818.out`, job 58782818). Each step names its source. No
step comes from the name of a switch.

One step is still open. Step 1 selects the hours to change. The arm a43 tests a
different rule for that step. Section 8 gives both rules.

**A limit on this document.** ASD-STE100 has an approved word list. I do not hold
that list. I obey the grammar rules and the sentence rules. I select the most
common word that I know. Some names in this project have no common-word
equivalent. I keep those names. I explain each one the first time it occurs.

---

## 1. Words used in this document

Use these words with one meaning each.

- **The scheduler**: the system that buys electric power in advance. It solves a
  schedule task for 100 jobs on 2 machines.
- **The forecaster**: the network that predicts 48 half-hour prices for the next
  day. The code calls it *the head*.
- **Decision regret**: the extra cost that the scheduler pays because it used a
  prediction and not the true prices. A low value is good.
- **5th-percentile price**: the price that 5% of the half-hours fall below. It
  shows the cheap tail of the day. Solar power pushes this tail down.
- **Decision divergence**: the distance between two schedules. The first schedule
  is the best one for a changed price day. The second is the best one for a
  reference price day. This number shows if a change moves a decision.
- **The pick score**: the worst decision regret on the shifted-year validation
  months. The loop uses it to select which saved model to keep.
- **A round**: one pass of the outer loop. The run does 8 rounds.
- **An arm**: one set of switches under test.
- **A run**: one training job for one arm and one seed.
- **The edit**: the change that the method applies to a copy of a price day.

## 2. What the method does

The scheduler buys electric power in advance. It first predicts prices. It then
solves the schedule task against that prediction. Later the true prices move away
from the 2015-2020 data. The prediction then misleads the scheduler. The schedule
costs more than necessary.

The published method (3D-Learning) trains a diffusion model. That model invents
severe price days. The method then trains the forecaster on those days. On real
Victorian prices this fails. The generator cannot reach the true shift.

This method removes the generator. It trains the forecaster on real past price
days. It also makes a copy of each day and makes one part of that day cheaper.
It reads two things from the 2015-2020 data: which hours to change, and how much
cheaper to make them. No future data touches any choice.

## 3. Inputs

| input | value | source |
|---|---|---|
| price data | AEMO Victoria, half-hourly, 2015-2020, 105,215 rows | `Data/AEMO_VIC1_train_2015_2020.csv` |
| forecaster inputs | 8 features, 48 half-hours in, 48 half-hours out | `Utils/datasets.py:94-101`, `main_energy_mv.py:98` |
| stored window length | 192 half-hours (4 days) | `main_energy_mv.py:109` |
| schedule task | ICON `sub100_s0`, 100 jobs, 2 machines, 3 resources, T=288 | the submit line |
| horizon year | 2024 | `INSTR_HEAD_DEPTH_HORIZON` |

The forecaster does not see the year. It does not see the date. All six calendar
inputs are cyclic. Therefore the forecaster cannot learn the multi-year fall by
itself. This is why the method must put that fall into the training data.

## 4. The algorithm

### Step 1 — Select the hours to change

Use only the 2015-2020 data.

1. Divide the day into six blocks of 4 hours.
2. For each block, find the 5th-percentile price in each year from 2015 to 2020.
3. Fit a straight line of those six values against the year. Keep the slope.
4. Select the block with the most negative slope. That block has the fastest fall.

Source: `_trend_aim`,
[ddd_learning.py:2290](Utils/Algos/ddd_learning.py#L2290).

Result on a34 seed 1:

| block | 00-04h | 04-08h | 08-12h | 12-16h | 16-20h | 20-24h |
|---|---|---|---|---|---|---|
| slope, $/yr | −0.10 | −0.40 | −1.56 | **−5.09** | +3.09 | +4.93 |

The winner is block 3. Its hours are 12 to 16. Its centre is **14.0h**. The
direction is DOWN. A tail that falls means that the method must make those hours
cheaper.

Record two facts. First, the divergence objective in the loop selects block 4.
The trend rule overrides it. Second, the code computes the centre as `4·b + 2`.
The number 4 is fixed in that line. The block grid is a switch. No arm changes
that grid. Therefore no result is wrong. The two parts would disagree if a later
arm set a different grid.

### Step 2 — Select how much cheaper

Use only the 2015-2020 data. Use the block from step 1.

1. Find that block's 5th-percentile price in each year.
2. Fit a **quadratic** curve of those values against the year.
3. Extend the curve to the horizon year.
4. Subtract the extended value from the last year's value. This difference is the
   depth.

Source: `_trend_depth`,
[ddd_learning.py:2339](Utils/Algos/ddd_learning.py#L2339).

Result on a34 seed 1:

| year | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|
| block-3 5th-percentile, $/MWh | +16.20 | +13.36 | +44.91 | +27.34 | +10.06 | −13.91 |

The acceleration is −10.75 $/yr². The 2020 value is −$13.91. The extended 2024
value is −$228.13. Therefore the **depth is $214.21**.

Use a curve and not a straight line. The fall gets faster each year. A line
explains 24% of the variation. A curve explains 81%. Remove any single year and
the curve still gives $162 to $292. Remove 2020 and the line fails completely.

The code stops the run if the depth is zero or negative. This guard prevents an
edit with no strength that still prints as active.

### Step 3 — Select the width of the edit

The edit is a raised-cosine bell. Its centre is the hour from step 1. The loop
searches for its half-width.

1. For each half-width in {2, 4, 6, 8, 10, 12} hours, build a bell of that width.
2. Apply that bell at full depth to the real windows.
3. Score the result by decision divergence.
4. Keep the width with the highest score.

Source: [ddd_learning.py:3464-3485](Utils/Algos/ddd_learning.py#L3464). The score
function is `_score` at
[ddd_learning.py:3173](Utils/Algos/ddd_learning.py#L3173).

Result on a34 seed 1:

| half-width | 2h | 4h | 6h | 8h | 10h | 12h |
|---|---|---|---|---|---|---|
| divergence score | +28,449 | +30,092 | +33,103 | +36,356 | **+37,293** | +37,177 |

The winner is **10h**. The score increases and then decreases. Therefore the
search does not simply select the widest bell. A very wide bell becomes a flat
shift of the whole day. A flat shift moves no decision. At a half-width of 10h
the bell is above zero from 04h to 24h. Its peak is at 14h. It covers most of the
day.

### Step 4 — Build the changed day

Do these steps for each sequence in a batch.

1. Compute the hour of the day from the two cyclic calendar channels. Use `atan2`.
2. Compute the circular distance `d` from that hour to the centre hour.
3. Compute the bell. It is `0.5 + 0.5·cos(π·d/w)` when `d < w`. It is 0 elsewhere.
   Its peak is 1 at the centre. It is 0 at the distance `w`.
4. Draw one amplitude for the sequence. Draw it uniform between 0 and the depth.
5. Set that amplitude to zero with a probability of 0.5. Half of the sequences
   stay unchanged.
6. Subtract `amplitude × bell` from the price channel. Do this in normalised
   space.

Source: `_instr_head_augment_price`, [ml.py:178](Utils/Algos/ml.py#L178).

The mean push is `depth/4`, which is about $53. Two causes make it smaller than
the depth. The amplitude is a uniform draw below the depth. Half of the sequences
get no edit at all.

**The depth is a ceiling. It is not the push that the forecaster sees.** Do not
quote "$214" alone. That number overstates the treatment.

### Step 5 — Train the forecaster

Do these steps in each round.

1. Take a continuous slice of real 2015-2020 days as the base data. The switch
   `HEAD_ROWS_MULT=5` makes this 5 batches for each inner epoch. Source:
   `_real_base_sample`,
   [ddd_learning.py:2549](Utils/Algos/ddd_learning.py#L2549).
2. Score the real price windows by decision divergence. Keep the best 20%. On a34
   this is 13 windows in each round. The buffer grows: 13, then 26, then 39.
3. Build the loader from the base days plus a 30% replay from that buffer.
4. Run 5 inner epochs. Apply the edit of step 4 to every batch.

The dose is 5 batches × 5 inner epochs × 8 rounds = **200 optimiser steps**. The
arm a35 ran 400 steps. It gained 0.0103. That gain is too small to read.
Therefore 200 steps are sufficient.

### Step 6 — Select which saved model to keep

Score the forecaster two ways after each round.

1. Find the decision regret on the held-out 2015-2020 validation split. This is
   the in-distribution score.
2. Find the decision regret on both shifted validation sets, 2021-22 and 2023-25.
   Keep the **worse** of the two.

A round is eligible only if its in-distribution regret stays at or below 1.2
times the best in-distribution regret up to that round. Of the eligible rounds,
keep the round with the lowest worst shifted-year regret.

Source: [ddd_learning.py:1530-1540](Utils/Algos/ddd_learning.py#L1530).

Stop the run early if four rounds give no improvement. The patience is half the
round count, with a maximum of 25
([ddd_learning.py:72](Utils/Algos/ddd_learning.py#L72)).

## 5. Pseudocode

```
Input: price data P (2015-2020), schedule task, horizon year H

# --- computed once, from 2015-2020 data only ---
for each 4h block b:
    s[b] = slope of quantile_05(P, b, year) against year
b*      = argmin_b s[b]                       # block with the fastest fall
centre  = 4*b* + 2                            # hours
q[y]    = quantile_05(P, b*, y) for y in 2015..2020
depth   = q[2020] - quadratic_fit(q)(H)       # stop if this is <= 0
for each half-width w in {2,4,6,8,10,12}:
    score[w] = mean decision_divergence( bell(centre, w, depth) on real windows )
width   = argmax_w score[w]

# --- the training loop ---
buffer = empty
for round = 1 .. 8:
    train the diffusion parts            # the output is unused, see section 6
    kept    = best 20% of real windows by decision divergence
    buffer += kept
    data    = real_days_sample() + 30% replay from buffer
    for inner_epoch = 1 .. 5:
        for each batch in data:              # 5 batches
            for each sequence:
                d    = circular_distance(hour_of_day, centre)
                bell = 0.5 + 0.5*cos(pi*d/width)  if d < width else 0
                a    = Uniform(0, depth)
                a    = 0 with probability 0.5
                price -= a * bell
            take one optimiser step on the forecaster
    r_in  = decision regret, in-distribution validation
    r_ood = max( regret on 2021-22 val, regret on 2023-25 val )
    if r_in <= 1.2 * best_r_in and r_ood < best_r_ood:
        save this round as the candidate model
    if four rounds give no improvement: stop
return the saved model
```

## 6. What the method does not contain

The code still runs the diffusion parts that it inherited. It runs the DRO inner
loop. It runs the guided sample steps. It saves 5 diffusion checkpoints in each
run. **No output of those parts reaches the forecaster.** The log of a34 states
this:

```
| [INSTR-REAL-WIN] instructor judging on REAL windows — NO generator output used anywhere |
| [HEAD-REAL] loaded 105216 real train rows as head base data (diffusion s_theta REPLACED for the head) |
```

The method has no generative model. The code still holds the old parts. State
this fact plainly. Do not describe the runs as if those parts were absent.

The measured cost to remove the generator is 0.0157 on the renewable years. It is
0.0165 on the gas years. Three seeds gave these numbers. You cannot read either
number. The repeat error on a gap is 0.0106. Three seeds resolve nothing smaller
than 0.0711.

**Write this: "the measured cost is 0.0157, and this study cannot resolve it".
Never write "the generator costs nothing to remove".** (§8.45)

## 7. Settings, and who selected each one

| setting | value | selected by |
|---|---|---|
| hours to change | 12-16h, centre 14.0h | a fit to 2015-2020; it holds in 7 of 7 folds |
| direction | down | the sign of the fitted slope |
| depth | $214.21 | a curve on 2015-2020; it holds in 6 of 6 folds |
| bell half-width | 10h | the loop, from the real windows |
| edit probability | 0.5 | an old default, `INSTR_HEAD_PROB` |
| amplitude draw | Uniform(0, depth) | inherited from the solar edit |
| rounds | 8 | us, and we state it as ours |
| optimiser steps | 200 | us, and we state it as ours |
| selection tolerance | ε = 0.2 | us, registered before the runs |

A person sets nothing else.

## 8. The one open step

The arm a43 replaces step 1. It also adds a cap to step 3. It trains now. It has
no test score.

**The rule of a34 (settled).** Use one grid of 4 hours. Fix the curve degree at
2. Use one aim: 14.0h, down.

**The rule of a43 (under test).** Select the curve degree by leave-one-year-out
prediction skill. Degree 2 scores 0.919. It is the only degree that beats
persistence. Keep every block grid that the same test cannot separate. All seven
grids score inside a band of 3%. Therefore all seven stay. Each grid gives its
strongest block. Merge the aims that share a direction and sit closer together
than the narrowest bell. Three aims remain:

| centre | direction | depth |
|---|---|---|
| 20.2h | up | $168.35 |
| 14.0h | down | $214.21 |
| 6.0h | down | $96.48 |

Cap each aim's half-width at the circular distance to its nearest other aim. Each
bell then falls to zero where the next aim starts. Without this cap the bells
overlap and cancel each other. At a half-width of 12h the up-aim peak falls from
+$167 to +$82.

If a43 wins, step 1 gets the self-selected degree and the multi-grid aim, and
step 3 gets the cap. If a43 loses, both steps stay as a34 has them.

## 9. Sources for the numbers

| claim | source |
|---|---|
| block slopes, depth curve, width scores | `slurm-58782818.out` (a34 seed 1) |
| a34 beats the published method, 6/6 cells, p = 0.0312 | `slurm-59035764.out` |
| a34 beats the same recipe at equal dose, 6/6, p = 0.0312 | `slurm-59075629.out` |
| 400 steps add nothing, 3/6, p = 1.0 | `slurm-59075123.out` |
| ~~gap recovery: 81.7-86.4% gas, 94.9-99.5% renewables~~ WITHDRAWN | `slurm-59082228.out` |
| the measurement floor is exactly zero | job 58862389 |
| a43's three aims and the merge | `slurm-59098337.out` |
| a43's cap holds in the live pipeline | `slurm-59103982.out` |

Use `paired_test_seedlevel.py` for the significance test. The pooled per-window
p-value is withdrawn. The 2,880 windows are not independent. The stride is 24
against a window of 48 slots. Only 6 trained models exist. Quote p = 0.0312.
Never quote 1e-275.

The gap-recovery line above is WITHDRAWN (2026-09-02). PREREGISTRATION
8.56 forbids it. The oracle arm a36 and the method a34 differ by 0.0103 on
the renewables test set, and a34's own seed spread is 0.0307. A gap smaller
than the noise cannot carry a percentage. Do not quote any share of the gap
to perfect knowledge until the wide-head arms settle whether the oracle's
ceiling is the head clamp or the task. Registered reads sit in 8.58 and 8.61.
