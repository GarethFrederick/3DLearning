"""Thesis figures F0, F21, F22, F23. Built at print size, 15 cm wide.

Data
  Prices: Data/AEMO_VIC1_train_2015_2020.csv + Data/AEMO_VIC1_ORACLE_traintest.csv
  (2021-2025), nominal $/MWh, half-hourly. AEMO stamps are INTERVAL-ENDING
  (preprocess_aemo.py), so a row stamped 10:30 is the half-hour 10:00-10:30.
  F0 and F21 place every row by the START of its half-hour, for both the hour
  of day and the calendar year. "10:00-14:00" = half-hours starting 10:00..13:30.
  F23: results/analysis/schedule_day48_renewables.json (diag_schedule_table.py with
  SCHED_DUMP48=1, run_schedule_day48.sh) and results/thesis/tables/schedule.tex, which
  the script parses and checks against.

Colour: slots 1-3 of the project palette. Years: 2015-2020 blue ramp,
2021-2022 aqua (slot 3), 2023-2025 orange ramp. Every figure goes through
figcheck.finalize (size, >= 9 pt).

Writes <FIG_OUT>/F0_price_shift, F21_negative_share, F22_pipelines,
F23_one_day (.pdf, .png); default FIG_OUT is results/thesis/figures.
Usage: python fig_thesis_1016.py [F0 F21 F22 F23]   (no args = all four)
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import figcheck  # noqa: E402

FIG = os.environ.get('FIG_OUT', f'{ROOT}/results/thesis/figures')
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#1a1a19', '#4a4a46', '#8a8a82'
CM = figcheck.CM
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'legend.frameon': False, 'lines.linewidth': 1.4,
})
YEARS = list(range(2015, 2026))
TRAIN = list(range(2015, 2021))
GAS = [2021, 2022]
REN = [2023, 2024, 2025]


def year_colours():
    blues = plt.cm.Blues(np.linspace(0.35, 1.0, len(TRAIN)))
    oranges = plt.cm.Oranges(np.linspace(0.45, 0.9, len(REN)))
    col, ls = {}, {}
    for y, c in zip(TRAIN, blues):
        col[y], ls[y] = c, '-'
    for y, l in zip(GAS, ((0, (3, 1.5)), '-')):
        col[y], ls[y] = C3, l
    for y, c in zip(REN, oranges):
        col[y], ls[y] = c, '-'
    return col, ls


def load_prices():
    """Both files, one row per half-hour, with start-of-interval year and slot.
    Fails loudly on overlap, gaps, or a missing half-hour in any year."""
    a = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv', parse_dates=['TIMESTAMP'])
    b = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_ORACLE_traintest.csv', parse_dates=['TIMESTAMP'])
    assert a['TIMESTAMP'].max() < b['TIMESTAMP'].min(), 'files overlap'
    d = pd.concat([a, b], ignore_index=True)
    assert d['TIMESTAMP'].is_unique
    step = d['TIMESTAMP'].diff().dropna()
    assert (step == pd.Timedelta(minutes=30)).all(), f'gaps: {step.value_counts().head()}'
    # ORACLE must be the two test files joined, else "existing files" disagree
    t = pd.concat([pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_test_2021_2022.csv', parse_dates=['TIMESTAMP']),
                   pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_test_2023_2025.csv', parse_dates=['TIMESTAMP'])],
                  ignore_index=True)
    assert (t['TIMESTAMP'].values == b['TIMESTAMP'].values).all()
    print(f'  ORACLE vs test files: max |price diff| {np.abs(t["Price"] - b["Price"]).max():.3g}')
    start = d['TIMESTAMP'] - pd.Timedelta(minutes=30)
    d['year'] = start.dt.year
    d['slot'] = start.dt.hour * 2 + start.dt.minute // 30
    cnt = d.groupby('year').size()
    print('  half-hours per year: ' + ', '.join(f'{y} {n}' for y, n in cnt.items()))
    for y, n in cnt.items():
        assert n == (17568 if y % 4 == 0 else 17520), (y, n)
    assert sorted(cnt.index) == YEARS, list(cnt.index)
    return d


def hhmm(slot):
    return f'{slot // 2:02d}:{30 * (slot % 2):02d}'


# F0
def f0_price_shift(d):
    W, H = 15.0, 6.5
    med = d.groupby(['year', 'slot'])['Price'].median().unstack()      # (11, 48)
    assert med.shape == (11, 48)
    vals = med.to_numpy().ravel()
    p2, p98 = np.percentile(vals, [2, 98])
    print(f'  F0 y-axis = 2nd-98th percentile of the {vals.size} plotted medians: '
          f'${p2:.2f} to ${p98:.2f}')
    col, ls = year_colours()
    x = np.arange(48) / 2 + 0.25                                       # half-hour midpoints
    fig, ax = plt.subplots(figsize=(W * CM, H * CM), constrained_layout=True)
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    first_mid = None
    outside = []
    print('  F0 minimum median half-hour by year:')
    for y in YEARS:
        v = med.loc[y].to_numpy()
        k = int(np.argmin(v))
        ties = int((np.abs(v - v[k]) < 0.005).sum())
        lw = 1.6 if y in REN else 1.2
        ax.plot(x, v, color=col[y], ls=ls[y], lw=lw, label=str(y), zorder=3)
        ax.plot(x[k], v[k], 'o', ms=4, color=col[y], mec='white', mew=0.6, zorder=4)
        if not p2 <= v[k] <= p98:
            outside.append(y)
        mid = 20 <= k <= 31                                            # starts 10:00..15:30
        if mid and first_mid is None:
            first_mid = y
        print(f'    {y}: {hhmm(k)}-{hhmm(k + 1) if k < 47 else "24:00"}  median ${v[k]:7.2f}'
              f'{"  (" + str(ties) + " half-hours tie)" if ties > 1 else ""}'
              f'{"  <- 10:00-16:00" if mid else ""}')
    print(f'  F0 first year with the minimum inside 10:00-16:00: {first_mid}')
    print(f'  F0 minimum dots outside the clipped y-axis: {outside or "none"}')
    # No clipping: the axis spans every plotted median. p2/p98 are printed only.
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = 0.04 * (vmax - vmin)
    ax.set_ylim(vmin - pad, vmax + pad)
    print(f'  F0 y-axis drawn: ${vmin - pad:.2f} to ${vmax + pad:.2f} (all medians '
          f'${vmin:.2f} to ${vmax:.2f}, no clip)')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel('hour of day (start of half-hour)')
    ax.set_ylabel('median price ($/MWh)')
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.6,
              labelspacing=0.15, borderaxespad=0)
    figcheck.finalize(fig, f'{FIG}/F0_price_shift', W, H)
    return {'p2': p2, 'p98': p98, 'first_mid': first_mid, 'outside': outside}


# F21
def f21_negative_share(d):
    W, H = 15.0, 7.0
    neg = d.assign(n=d['Price'] < 0).groupby(['year', 'slot'])['n'].mean().unstack() * 100
    assert neg.shape == (11, 48)
    vmax = float(neg.to_numpy().max())
    fig, ax = plt.subplots(figsize=(W * CM, H * CM), constrained_layout=True)
    im = ax.imshow(neg.to_numpy(), aspect='auto', cmap='Blues', origin='lower',
                   vmin=0, vmax=vmax, interpolation='nearest',
                   extent=[0, 24, YEARS[0] - 0.5, YEARS[-1] + 0.5])
    ax.axhline(2020.5, color=INK, lw=1.2)
    ax.text(0.25, 2020.62, 'training ends', color=INK, va='bottom', ha='left', fontsize=9)
    ax.set_yticks(YEARS)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel('hour of day (start of half-hour)')
    ax.set_ylabel('year')
    ax.grid(False)
    ax.spines[['left', 'bottom']].set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label('half-hours below $0 (%)')
    cb.outline.set_edgecolor(MUTED)
    m = d[(d['slot'] >= 20) & (d['slot'] <= 27)]                       # 10:00-14:00
    share = (m['Price'] < 0).groupby(m['year']).mean() * 100
    for y in (2019, 2020, 2023, 2024):
        print(f'  F21 share below $0, 10:00-14:00, {y}: {share[y]:.1f}%  '
              f'({int((m[m["year"] == y]["Price"] < 0).sum())} of {int((m["year"] == y).sum())})')
    over = [y for y in YEARS if (neg.loc[y] > 10).any()]
    first = over[0] if over else None
    if first:
        cols = [hhmm(s) for s in neg.columns if neg.loc[first, s] > 10]
        print(f'  F21 first year any half-hour column > 10%: {first} '
              f'(columns {", ".join(cols)}; max {neg.loc[first].max():.1f}%)')
    else:
        print('  F21 no column ever exceeds 10%')
    print(f'  F21 colour scale 0 to {vmax:.1f}% (max at {neg.stack().idxmax()})')
    figcheck.finalize(fig, f'{FIG}/F21_negative_share', W, H)


# F22
def f22_pipelines():
    """Schematic. Loss text follows the code: DFL (ml.py) and 3D-Learning
    (ddd_learning.py) also add 0.5 x smooth-L1 to SPO+,
    so smooth-L1 does not set trend-aim apart and is not highlighted."""
    W, H = 15.0, 8.0
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    # columns: x left edge, width (cm)
    gap = 0.28
    widths = [1.75, 2.4, 1.9, 1.85, 1.75, 1.75]
    x0 = W - 0.05 - sum(widths) - gap * (len(widths) - 1)
    xs = []
    for w in widths:
        xs.append(x0)
        x0 += w + gap
    rows = [
        (('(a)', 'PTO'), 1.2, ['historical\ndays', None, None, 'forecaster', 'scheduler\n(not in\nloop)', 'MSE'], ()),
        (('(b)', 'DFL'), 1.2, ['historical\ndays', None, None, 'forecaster', 'scheduler', 'SPO+ and\nsmooth-L1'], ()),
        (('(c)', '3D-Learning'), 1.7, ['historical\ndays', 'diffusion\ngenerator', 'worst-case\ngenerated\ndays',
                                    'forecaster', 'scheduler', 'SPO+ and\nsmooth-L1'], (1, 2)),
        (('(d)', 'Trend-aim'), 1.7, ['historical\ndays', 'trend rule:\nfalling block\nand projected\ndepth',
                                   'edited\nreal days', 'forecaster', 'scheduler', 'SPO+ and\nsmooth-L1'], (1, 2)),
    ]
    pitch = [1.75, 1.6, 2.25, 2.25]
    inside = []
    ytop = H - 0.1
    label_box = FancyBboxPatch((0.02, 0), xs[0] - 0.08, H, boxstyle='square,pad=0',
                               fc='none', ec='none')
    ax.add_patch(label_box)
    for (label, bh, boxes, hl), p in zip(rows, pitch):
        yc = ytop - p / 2
        ytop -= p
        for txt, dy, fw in ((label[0], 0.2, 'bold'), (label[1], -0.2, 'normal')):
            inside.append((ax.text(0.05, yc + dy, txt, ha='left', va='center', fontsize=9,
                                   color=INK, fontweight=fw), label_box))
        drawn = []
        for i, txt in enumerate(boxes):
            if txt is None:
                continue
            dashed = 'not in' in txt
            fc = '#fbe3d8' if i in hl else 'white'
            ec = C2 if i in hl else (MUTED if dashed else INK2)
            patch = FancyBboxPatch((xs[i], yc - bh / 2), widths[i], bh,
                                   boxstyle='round,pad=0,rounding_size=0.12', fc=fc, ec=ec,
                                   lw=1.4 if i in hl else 0.9, ls=(0, (3, 2)) if dashed else '-',
                                   zorder=2)
            ax.add_patch(patch)
            t = ax.text(xs[i] + widths[i] / 2, yc, txt, ha='center', va='center', fontsize=9,
                        color=MUTED if dashed else INK, zorder=3, linespacing=1.15)
            inside.append((t, patch))
            drawn.append(i)
        # arrows between consecutive drawn boxes; PTO's forecaster goes to the
        # loss over the top of the scheduler, which is not in its training loop
        pairs = list(zip(drawn[:-1], drawn[1:]))
        if 'not in' in boxes[4]:
            pairs = [(a, b) for a, b in pairs if b != 4 and a != 4]
            ytop_box = yc + bh / 2
            xa, xb = xs[3] + widths[3] / 2, xs[5] + widths[5] / 2
            ax.add_patch(FancyArrowPatch(path=matplotlib.path.Path(
                [(xa, ytop_box), (xa, ytop_box + 0.18), (xb, ytop_box + 0.18), (xb, ytop_box)],
                [1, 2, 2, 2]), arrowstyle='-|>', mutation_scale=8, color=INK2, lw=0.9, zorder=1))
        for a, b in pairs:
            ax.add_patch(FancyArrowPatch((xs[a] + widths[a], yc), (xs[b], yc), arrowstyle='-|>',
                                         mutation_scale=8, color=INK2, lw=0.9, shrinkA=0,
                                         shrinkB=0, zorder=1))
    figcheck.finalize(fig, f'{FIG}/F22_pipelines', W, H, inside=inside)


# F23
def parse_schedule_tex(path):
    """-> {block: [true, trend, 3dl, opt_e, trend_e, 3dl_e]}, plus cost/regret rows."""
    def num(s):
        s = s.replace('$-$', '-').replace('{,}', '').replace('\\\\', '').strip()
        return float(s)
    rows, extra = {}, {}
    for line in open(path):
        m = re.match(r'^(\d\d)--(\d\d) h &(.*)\\\\', line.strip())
        if m:
            cells = [c for c in m.group(3).split('&') if c.strip()]
            rows[(int(m.group(1)), int(m.group(2)))] = [num(c) for c in cells]
        for key in ('Cost under true prices', 'Regret'):
            if line.strip().startswith(key):
                cells = [c.strip() for c in line.split('&')[1:]]
                extra[key] = [None if c.replace('\\\\', '').strip() in ('---', '') else num(c)
                              for c in cells if c.replace('\\\\', '').strip()]
    assert len(rows) == 6 and all(len(v) == 6 for v in rows.values()), rows
    return rows, extra


def f23_one_day():
    W, H = 15.0, 8.0
    src = f'{ROOT}/results/analysis/schedule_day48_renewables.json'
    B = json.load(open(src))['oodtest_2023_2025_renewables']['B']
    assert B['start'] == '2023-10-05 00:00:00', B['start']
    true = np.asarray(B['true_price48'])
    fc = {k: np.asarray(v) for k, v in B['forecast48'].items()}
    en = {k: np.asarray(v['energy48']) for k, v in B['mip'].items()}
    for v in [true, *fc.values(), *en.values()]:
        assert v.shape == (48,), v.shape

    # Check against schedule.tex (parsed, not typed).
    rows, extra = parse_schedule_tex(f'{ROOT}/results/thesis/tables/schedule.tex')
    bad = 0
    print('  F23 CHECK against results/thesis/tables/schedule.tex (per four-hour block):')
    print('    block    energy MWh: optimal  trend-aim  3D-Learning   |  price $/MWh: true  trend-aim  3D-L')
    for bi, (h0, h1) in enumerate([(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]):
        sl = slice(8 * bi, 8 * bi + 8)
        e = [en['optimal'][sl].sum(), en['trend-aim'][sl].sum(), en['3D-Learning'][sl].sum()]
        p = [true[sl].mean(), fc['trend-aim'][sl].mean(), fc['3D-Learning'][sl].mean()]
        t = rows[(h0, h1)]
        ok_e = all(round(a) == b for a, b in zip(e, t[3:]))
        ok_p = all(abs(round(a, 2) - b) < 1e-9 for a, b in zip(p, t[:3]))
        bad += (not ok_e) + (not ok_p)
        print(f'    {h0:02d}-{h1:02d} h  '
              + '  '.join(f'{a:8.1f}/{b:5.0f}' for a, b in zip(e, t[3:]))
              + ('  OK ' if ok_e else '  <-- MISMATCH ') + ' | '
              + '  '.join(f'{a:7.2f}/{b:7.2f}' for a, b in zip(p, t[:3]))
              + ('  OK' if ok_p else '  <-- MISMATCH'))
    cost = [B['mip'][k]['cost_true'] for k in ('optimal', 'trend-aim', '3D-Learning')]
    reg = [B['mip'][k]['regret'] for k in ('trend-aim', '3D-Learning')]
    ok_c = all(round(a) == b for a, b in zip(cost, extra['Cost under true prices']))
    ok_r = all(round(a) == b for a, b in zip(reg, [v for v in extra['Regret'] if v is not None]))
    bad += (not ok_c) + (not ok_r)
    print(f'    cost  {[round(c) for c in cost]} vs table {extra["Cost under true prices"]}  '
          + ('OK' if ok_c else '<-- MISMATCH'))
    print(f'    regret {[round(r) for r in reg]} vs table {extra["Regret"]}  '
          + ('OK' if ok_r else '<-- MISMATCH'))
    print(f'    scored regrets (eval files): {B["scored_regret"]}')
    print(f'  F23 CHECK result: {"ALL OK" if bad == 0 else f"{bad} MISMATCH(ES)"}')

    edges = np.arange(49) / 2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W * CM, H * CM), sharex=True,
                                 constrained_layout=True)
    for a in (a1, a2):
        for x in range(4, 24, 4):
            a.axvline(x, color=MUTED, lw=0.6, ls=':', zorder=0)
        a.yaxis.grid(True, zorder=0)
    a1.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    a1.stairs(true, edges, baseline=None, color=INK, lw=1.6, label='true price', zorder=4)
    a1.stairs(fc['trend-aim'], edges, baseline=None, color=C1, lw=1.4, label='trend-aim', zorder=3)
    a1.stairs(fc['3D-Learning'], edges, baseline=None, color=C2, lw=1.4, ls=(0, (4, 2)),
              label='3D-Learning', zorder=3)
    a1.set_ylabel('price ($/MWh)')
    a1.legend(loc='lower left', bbox_to_anchor=(0.06, 1.0), ncol=3, borderaxespad=0.1,
              handlelength=1.8)
    a1.text(0.0, 1.02, '(a)', transform=a1.transAxes, ha='left', va='bottom', fontsize=9,
            fontweight='bold')
    a2.stairs(en['optimal'], edges, fill=True, color='#d6d6d0', label='optimal', zorder=2)
    a2.stairs(en['trend-aim'], edges, baseline=None, color=C1, lw=1.4, label='trend-aim', zorder=3)
    a2.stairs(en['3D-Learning'], edges, baseline=None, color=C2, lw=1.4, ls=(0, (4, 2)),
              label='3D-Learning', zorder=3)
    a2.set_ylabel('energy (MWh)')
    a2.legend(loc='lower left', bbox_to_anchor=(0.06, 1.0), ncol=3, borderaxespad=0.1,
              handlelength=1.8)
    a2.text(0.0, 1.02, '(b)', transform=a2.transAxes, ha='left', va='bottom', fontsize=9,
            fontweight='bold')
    a2.set_xlim(0, 24)
    a2.set_ylim(bottom=0)
    a2.set_xticks(range(0, 25, 4))
    a2.set_xlabel('hours into the horizon (window of 5 October 2023)')
    figcheck.finalize(fig, f'{FIG}/F23_one_day', W, H)
    return bad


if __name__ == '__main__':
    want = sys.argv[1:] or ['F0', 'F21', 'F22', 'F23']
    failed = []
    d = load_prices() if {'F0', 'F21'} & set(want) else None
    for name, fn in (('F0', lambda: f0_price_shift(d)), ('F21', lambda: f21_negative_share(d)),
                     ('F22', f22_pipelines), ('F23', f23_one_day)):
        if name not in want:
            continue
        print(f'\n===== {name} =====', flush=True)
        try:
            r = fn()
            if name == 'F23' and r:
                failed.append('F23 table check')
        except Exception as e:                      # keep going; report loudly
            print(f'  FAILED {name}: {type(e).__name__}: {e}', flush=True)
            failed.append(name)
    print(f'\nFAILED: {failed}' if failed else '\nALL FIGURES OK')
    print('FIG1016_DONE')
    sys.exit(1 if failed else 0)
