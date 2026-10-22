"""part2 figures F17 (clamp share) and F19 (one edited training window).

F17 parses the two diag_forecast_shape.py logs in results/logs/analysis/
(renewables slurm-60483431.out, gas slurm-60597260.out), not typed numbers.
F19 rebuilds the training edit as ml._instr_head_augment_price does: hour from
the INTERVAL-ENDING stamp, circular distance d to 14:00, bell
0.5+0.5cos(pi d/w) for d<w, price minus delta*bell in dollars (the code does
the same in normalised units). delta=$214.21; w=8 h, the half-width a72 chose
most often.
Writes results/thesis/figures/F17_clamp_share and F19_edit_example (.pdf, .png).
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
FIG = f'{ROOT}/results/thesis/figures'
os.makedirs(FIG, exist_ok=True)
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
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


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG}/{name}.{ext}')
    plt.close(fig)
    print(f'  wrote {FIG}/{name}.pdf')


# F17
def read_shape(path):
    out = {}
    for line in open(path):
        m = re.match(r'^(.+?)\s+[\d.]+%\s+[\d.]+\s+([\d.]+)%\s+\d+', line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


ren = read_shape(f'{ROOT}/results/logs/analysis/slurm-60483431.out')
gas = read_shape(f'{ROOT}/results/logs/analysis/slurm-60597260.out')
ORDER = [('PTO', 'PTO'), ('DFL', 'DFL'), ('WDRO-SPO', 'WDRO-SPO'),
         ('3D-Learning (narrow)', '3D-Learning, narrow'), ('3D-Learning (wide)', '3D-Learning, wide'),
         ('trend-aim (narrow)', 'Trend-aim, narrow'), ('trend-aim (wide)', 'Trend-aim, wide')]
WANT = {'PTO': (32.4, 31.0), 'DFL': (28.7, 29.0), 'WDRO-SPO': (18.3, 19.3),
        '3D-Learning (narrow)': (27.8, 28.3), '3D-Learning (wide)': (0.6, 0.9),
        'trend-aim (narrow)': (12.8, 13.8), 'trend-aim (wide)': (0.2, 0.9)}
for k, (r, g) in WANT.items():
    ok = abs(ren[k] - r) < 1e-9 and abs(gas[k] - g) < 1e-9
    print(f'  F17 check {k:22s} renewables {ren[k]:5.1f}% gas {gas[k]:5.1f}% '
          + ('OK' if ok else '<-- MISMATCH'))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
y = np.arange(len(ORDER))[::-1]
h = 0.36
ax.barh(y + h / 2, [ren[k] for k, _ in ORDER], h, color=C1, label='renewables 2023-2025', zorder=3)
ax.barh(y - h / 2, [gas[k] for k, _ in ORDER], h, color=C2, hatch='///', edgecolor='white',
        lw=0, label='gas 2021-2022', zorder=3)
for yy, (k, _) in zip(y, ORDER):
    ax.text(ren[k] + 0.4, yy + h / 2, f'{ren[k]:.1f}%', va='center', fontsize=7.5, color=INK2)
    ax.text(gas[k] + 0.4, yy - h / 2, f'{gas[k]:.1f}%', va='center', fontsize=7.5, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab in ORDER])
ax.set_xlabel('share of forecast half-hours clipped to a bound of the output range (%)')
ax.set_xlim(0, 38)
ax.xaxis.grid(True, zorder=0)
ax.legend(loc='lower right', fontsize=8)
save(fig, 'F17_clamp_share')

# F19
DELTA, HW, CENTRE = 214.21, 8.0, 14.0
d = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv', parse_dates=['TIMESTAMP'])
d = d.reset_index(drop=True)
d['h'] = d['TIMESTAMP'].dt.hour + d['TIMESTAMP'].dt.minute / 60.0
# A "day" D = the 48 half-hours labelled D 00:30 .. D+1 00:00 (interval-ending),
# i.e. clock 00:00-24:00 on D. Its 12-16 h price = labels 12:30 .. 16:00.
d['day'] = (d['TIMESTAMP'] - pd.Timedelta(minutes=30)).dt.normalize()
mid = d[(d['h'] > 12.0) & (d['h'] <= 16.0)]
md = mid.groupby('day')['Price'].mean()
md = md[(md.index.year == 2020)]
full = d.groupby('day').size()
md = md[[full.get(k, 0) == 48 for k in md.index]]
target = md.median()
day = (md - target).abs().idxmin()
i0 = int(d.index[d['day'] == day][0])            # first horizon row
assert i0 >= 48 and d.loc[i0, 'TIMESTAMP'].strftime('%H:%M') == '00:30'
w = d.iloc[i0 - 48:i0 + 48].copy()
dist = np.abs(w['h'].to_numpy() - CENTRE)
dist = np.minimum(dist, 24.0 - dist)
bell = np.where(dist < HW, 0.5 + 0.5 * np.cos(np.pi * dist / HW), 0.0)
w['edited'] = w['Price'] - DELTA * bell
print(f'  F19 day {day.date()} (2020 median 12-16 h mean ${target:.2f}; this day ${md[day]:.2f}); '
      f'window rows {w["TIMESTAMP"].iloc[0]} .. {w["TIMESTAMP"].iloc[-1]}; '
      f'horizon starts {d.loc[i0, "TIMESTAMP"]} (clock 00:00); half-width {HW:g} h; '
      f'delta ${DELTA}; min edited ${w["edited"].min():.2f} at {w.loc[w["edited"].idxmin(), "TIMESTAMP"]}')

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                             gridspec_kw={'height_ratios': [2.3, 1]})
x = w['TIMESTAMP'] - pd.Timedelta(minutes=15)    # plot at the middle of each half-hour
for a in (a1, a2):
    a.axvspan(x.iloc[0], x.iloc[47] + pd.Timedelta(minutes=15), color=MUTED, alpha=0.10, lw=0)
    a.yaxis.grid(True, zorder=0)
a1.plot(x, w['Price'], color=INK2, lw=1.6, label='real price', zorder=3)
a1.plot(x, w['edited'], color=C2, lw=2.0, label=f'after the edit (full depth \\${DELTA:.2f})', zorder=4)
a1.axhline(0, color=MUTED, lw=0.8)
a1.text(x.iloc[2], a1.get_ylim()[1], 'history (48 half-hours)', color=INK2, fontsize=8, va='top')
a1.text(x.iloc[50], a1.get_ylim()[1], 'forecast horizon (48 half-hours)', color=INK2, fontsize=8, va='top')
a1.set_ylabel('price ($/MWh)')
a1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8)
a2.fill_between(x, 0, bell, color=C2, alpha=0.25, lw=0)
a2.plot(x, bell, color=C2, lw=1.6)
a2.set_ylim(0, 1.1)
a2.set_ylabel('bell weight')
a2.text(x.iloc[49], 0.6, f'centre 14:00\nhalf-width {HW:g} h', color=INK2, fontsize=8)
import matplotlib.dates as mdates
a2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
a2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
save(fig, 'F19_edit_example')
print('PART2_EXTRA_DONE')
