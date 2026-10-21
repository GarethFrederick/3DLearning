"""F20 block-width sensitivity, built at print size.

15 x 13 cm, every text >= 9 pt, saved through figcheck.finalize (no tight bbox).
Layout as F12: (a) renewables and (b) gas by block width on the top row, (c) both
eras ordered by the aimed block's CENTRE hour across the full bottom row, so the
claim "regret rises with the centre hour" is read from one panel.
Data: score JSONs results/scores/poc_eval_measure/<arm>_s<seed>_<era>_selected.json,
selected checkpoint. 4 h = a72 (all seeds); 2/6/8 h = a78_bw{2,6,8} (3 seeds).
Writes <FIG_OUT>/F20_block_sensitivity.{pdf,png}; supersedes F20 in fig_results_1008.py.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import figcheck  # noqa: E402

FIG = os.environ.get('FIG_OUT', f'{ROOT}/results/thesis/figures')
C1, C2 = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#1a1a19', '#4a4a46', '#8a8a82'
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': None,
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'legend.frameon': False, 'lines.linewidth': 1.6,
})
ERAS = [('oodtest_2023_2025_renewables', 'Renewables 2023-2025'),
        ('oodtest_2021_2022_gas', 'Gas 2021-2022')]
WIDTHS = [(2, 'a78_bw2'), (4, 'a72_nogenwide'), (6, 'a78_bw6'), (8, 'a78_bw8')]
HOURS = {2: '12-14', 4: '12-16', 6: '12-18', 8: '08-16'}
CENTRE = {2: 13, 4: 14, 6: 15, 8: 12}
# Published seed means: the figure refuses to build if any differs.
REF = {(2, 0): 0.4041, (4, 0): 0.4438, (6, 0): 0.4893, (8, 0): 0.3848,
       (2, 1): 0.3542, (4, 1): 0.3760, (6, 1): 0.4016, (8, 1): 0.3509}


def load(arm, era):
    v = []
    for f in sorted(glob.glob(f'{ROOT}/results/scores/poc_eval_measure/{arm}_s*_{era}_selected.json')):
        d = json.load(open(f))
        assert d.get('greedy_solves', 0) == 0, f
        v.append(float(d['norm_regret']))
    return np.array(v)


res = {}
for k, (era, _) in enumerate(ERAS):
    for w, arm in WIDTHS:
        v = load(arm, era)
        assert abs(v.mean() - REF[(w, k)]) < 5e-5, f'{arm} {era} mean {v.mean():.4f} != {REF[(w, k)]}'
        res[(w, k)] = v
        print(f'  {ERAS[k][1]:22s} {w} h ({HOURS[w]} h, centre {CENTRE[w]}:00) n={len(v):2d} '
              f'mean {v.mean():.4f} min {v.min():.4f} max {v.max():.4f}')

W_CM, H_CM = 15, 13
fig = plt.figure(figsize=(W_CM * figcheck.CM, H_CM * figcheck.CM), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

for k, (ax, (era, name)) in enumerate(zip(axes[:2], ERAS)):
    for w, _ in WIDTHS:
        v = res[(w, k)]
        col = C2 if w == 4 else C1
        jit = np.random.RandomState(w).uniform(-0.15, 0.15, len(v))
        ax.scatter(np.full(len(v), w) + jit, v, s=12, color=col, alpha=0.55,
                   edgecolor='none', zorder=3)
        ax.plot([w - 0.45, w + 0.45], [v.mean()] * 2, color=col, lw=2.4, zorder=4)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xticklabels([f'{w} h\n{HOURS[w]}' for w in (2, 4, 6, 8)])
    ax.set_xlim(1.0, 9.0)
    ax.set_title(f'({"ab"[k]}) {name}', loc='left')
    ax.set_xlabel('block width (hours covered)')
    ax.yaxis.grid(True, zorder=0)
axes[0].set_ylabel('normalised regret')

ax = axes[2]
order = sorted(CENTRE, key=lambda w: CENTRE[w])
xs = [CENTRE[w] for w in order]
for k, (col, mk, ls, name) in enumerate(((C1, 'o', '-', 'renewables'), (INK2, 's', '--', 'gas'))):
    ys = [res[(w, k)].mean() for w in order]
    ax.plot(xs, ys, color=col, marker=mk, ls=ls, ms=6, zorder=3, label=name)
    for x, y, w in zip(xs, ys, order):
        ax.annotate(f'{w} h  {y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 7 if k == 0 else -15), ha='center', fontsize=9,
                    color=C2 if w == 4 else INK2)
ax.set_xticks(xs)
ax.set_xticklabels([f'{x}:00' for x in xs])
ax.set_xlim(11.5, 15.5)
lo = min(res[(w, 1)].mean() for w in order)
hi = max(res[(w, 0)].mean() for w in order)
ax.set_ylim(lo - 0.03, hi + 0.03)
ax.set_xlabel('centre hour of the aimed block')
ax.set_ylabel('mean normalised regret')
ax.set_title('(c) Means ordered by centre hour; orange = 4 h, the width used', loc='left')
ax.legend(loc='upper left')
ax.yaxis.grid(True, zorder=0)
for k in (0, 1):
    print(f'  (c) {ERAS[k][1]}: ' + ', '.join(f'{CENTRE[w]}:00 {res[(w, k)].mean():.4f}' for w in order))

figcheck.finalize(fig, f'{FIG}/F20_block_sensitivity', W_CM, H_CM)
print('F20_DONE')
