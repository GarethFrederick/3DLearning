"""F12 block width: why the trend-aim rule splits the day into 4 h blocks.

Training years 2015-2020 only (Data/AEMO_VIC1_train_2015_2020.csv). The hour
convention matches _trend_aim and diag_grid_invariance.sh: hour taken straight
from TIMESTAMP, so the figure and the rule see the same blocks.

(a) 5th-percentile price at each hour of the day, one line per training year.
(b) Linear slope of that 5th percentile across the years, per hour. The
    "falling zone" is the run of hours next to the steepest hour whose slope
    is at least half the steepest slope.
(c) For each block width, the slope of the block the rule picks. Self-check:
    4 h must give block 3, slope -5.09, depth $214.21.

Writes <FIG_OUT>/F12_block_width.{pdf,png} (default results/thesis/figures).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
OUT = os.environ.get('FIG_OUT', f'{ROOT}/results/thesis/figures')
C1, C2 = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#1a1a19', '#4a4a46', '#8a8a82'
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'legend.frameon': False, 'lines.linewidth': 2.0,
})

d = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv', parse_dates=['TIMESTAMP'])
d['y'] = d['TIMESTAMP'].dt.year
d = d[(d['y'] >= 2015) & (d['y'] <= 2020)].copy()
d['h'] = d['TIMESTAMP'].dt.hour + d['TIMESTAMP'].dt.minute / 60.
yrs = np.array(sorted(d['y'].unique()), float)
xc = yrs - yrs.mean()


def slope(v):
    return float((xc * (v - v.mean())).sum() / (xc ** 2).sum())


def p5_by_year(mask):
    s = d[mask]
    return np.array([s[s['y'] == int(y)]['Price'].quantile(0.05) for y in yrs])


# (a)+(b): hourly
hr = np.clip(d['h'] // 1, 0, 23).astype(int)
P = np.array([p5_by_year(hr == h) for h in range(24)])          # (24, n_years)
S = np.array([slope(P[h]) for h in range(24)])
hmin = int(np.argmin(S))
lo_h = hi_h = hmin
while lo_h - 1 >= 0 and S[lo_h - 1] <= 0.5 * S[hmin]:
    lo_h -= 1
while hi_h + 1 <= 23 and S[hi_h + 1] <= 0.5 * S[hmin]:
    hi_h += 1
zone_w = hi_h - lo_h + 1
print(f'  hourly p5 slope: steepest hour {hmin}:00 ({S[hmin]:+.2f} $/yr); '
      f'falling zone {lo_h:02d}:00-{hi_h + 1:02d}:00 = {zone_w} h')
print('  hourly slopes: ' + ' '.join(f'{h}:{S[h]:+.1f}' for h in range(24)))

# (c): grid widths
rows = []
for g in [1, 2, 3, 4, 6, 8, 12]:
    nb = int(round(24 / g))
    b = np.clip((d['h'] // g).astype(int), 0, nb - 1)
    v = [p5_by_year(b == i) for i in range(nb)]
    sl = [slope(x) for x in v]
    i = int(np.argmin(sl))
    c = np.polyfit(yrs, v[i], 2)
    dep = float(v[i][-1] - np.polyval(c, 2024.0))
    lin = np.polyval(np.polyfit(yrs, v[i], 1), yrs)
    resid = float(np.std(v[i] - lin, ddof=2))
    rows.append((g, i, i * g, (i + 1) * g, sl[i], dep, resid))
    print(f'  {g:2d} h: block {i:2d} ({i * g:02d}-{(i + 1) * g:02d} h)  slope {sl[i]:+6.2f}  '
          f'depth ${dep:7.2f}  resid SD ${resid:5.2f}')
r4 = [r for r in rows if r[0] == 4][0]
ok = r4[1] == 3 and abs(r4[4] + 5.09) < 0.01 and abs(r4[5] - 214.21) < 0.01
print('  CHECK vs job 60266418 (4 h: block 3, -5.09, $214.21): ' + ('OK' if ok else '<-- MISMATCH'))

# Exactly 15 x 16 cm, no tight bbox (it trims the page), text >= 9 pt, panel
# letters only (the caption carries titles). figcheck.finalize enforces size and text.
import sys
sys.path.insert(0, ROOT)
import figcheck
plt.rcParams.update({'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'axes.titlesize': 9, 'legend.fontsize': 9, 'savefig.bbox': None})
fig = plt.figure(figsize=(15 * figcheck.CM, 16 * figcheck.CM), constrained_layout=True)
_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.15, 1.0])
ax = [fig.add_subplot(_gs[0, :]), fig.add_subplot(_gs[1, 0]), fig.add_subplot(_gs[1, 1])]
for a in ax[:2]:
    a.axvspan(12, 16, color=C2, alpha=0.10, lw=0)
    for x in range(4, 24, 4):
        a.axvline(x, color=MUTED, lw=0.6, ls=':')
    a.set_xlim(0, 24)
    a.set_xticks(range(0, 25, 4))
    a.set_xlabel('Hour of day')

ramp = plt.cm.Blues(np.linspace(0.35, 1.0, len(yrs)))
hx = np.arange(24) + 0.5
for k, y in enumerate(yrs):
    ax[0].plot(hx, P[:, k], color=ramp[k], lw=1.6, label=f'{int(y)}')
ax[0].legend(ncol=6, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
ax[0].set_ylim(-22, 90)   # headroom so the year legend clears the lines
ax[0].set_ylabel('5th-percentile price ($/MWh)')
ax[0].set_title('(a)', loc='left')
ax[0].grid(axis='y')

cols = [C2 if lo_h <= h <= hi_h else '#c8c8c2' for h in range(24)]
ax[1].bar(hx, S, width=0.8, color=cols, edgecolor='white', linewidth=0.6)
ax[1].axhline(0, color=MUTED, lw=0.8)
ax[1].axhline(0.5 * S[hmin], color=INK2, lw=0.8, ls='--')
ax[1].text(0.3, 0.5 * S[hmin] - 0.25, 'half the\nsteepest\nfall', color=INK2, fontsize=9, va='top')
ax[1].set_ylabel('Change in floor\n($/MWh per year)')
ax[1].set_title('(b)', loc='left')
print(f'  caption text: (a) daily price floor by year; (b) fastest fall {lo_h:02d}-{hi_h + 1:02d} h; '
      f'(c) block chosen at each width')
ax[1].grid(axis='y')

gw = [r[0] for r in rows]
gs = [r[4] for r in rows]
xs = np.arange(len(gw))
ax[2].bar(xs, gs, width=0.6, color=[C2 if g == 4 else C1 for g in gw], edgecolor='white', linewidth=0.6)
for x, r in zip(xs, rows):
    ax[2].text(x, r[4] - 0.15, f'{r[2]:02d}-{r[3]:02d}', ha='center', va='top', fontsize=9, color=INK2,
               rotation=90)
ax[2].axhline(0, color=MUTED, lw=0.8)
ax[2].set_xticks(xs)
ax[2].set_xticklabels([f'{g}' for g in gw])
ax[2].set_xlabel('Block width (h)')
ax[2].set_ylabel('Slope of chosen block\n($/MWh per year)')
ax[2].set_ylim(min(gs) - 2.6, 0.5)
ax[2].set_title('(c)', loc='left')
ax[2].grid(axis='y')

figcheck.finalize(fig, f'{OUT}/F12_block_width', 15, 16)
print('F12_DONE')
