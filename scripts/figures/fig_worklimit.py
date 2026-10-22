"""F18: regret against the Gurobi work limit (diag_worklimit.py outputs).
Seed 1, wide range, first 128 renewables windows in scorer order. The optimal
cost is re-solved at each limit. Reads results/scores/worklimit/wl_*.json only.
Writes results/thesis/figures/F18_worklimit.{pdf,png}."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
C1, C2 = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#1a1a19', '#4a4a46', '#8a8a82'
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.edgecolor': MUTED,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.labelcolor': INK,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 2.0})
WL = [230, 1000, 3000, 10000]
R = {w: json.load(open(f'{ROOT}/results/scores/worklimit/wl_{w}.json')) for w in WL}
for w in WL:
    for lab in ('trend-aim', '3D-Learning'):
        assert R[w][lab]['safety_hits'] == 0, f'safety net fired at {w} for {lab}: VOID'
print('  limit   trend-aim (stops)     3D-Learning (stops)   gap    optimal stops')
for w in WL:
    a, b = R[w]['trend-aim'], R[w]['3D-Learning']
    os_ = 100 * np.mean([s == 16 for s in R[w]['opt']['status']])
    print(f'  {w:6d}  {a["norm_regret"]:.4f} ({100*a["timeout_share"]:4.1f}%)   '
          f'{b["norm_regret"]:.4f} ({100*b["timeout_share"]:4.1f}%)   '
          f'{b["norm_regret"]-a["norm_regret"]:.4f}   {os_:.1f}%')
fig, ax = plt.subplots(figsize=(5.8, 3.5))
for lab, col, mk in (('3D-Learning', C1, 'o'), ('trend-aim', C2, 's')):
    y = [R[w][lab]['norm_regret'] for w in WL]
    ax.plot(WL, y, color=col, marker=mk, markersize=7, zorder=3)
    for w, v in zip(WL, y):
        ax.annotate(f'{v:.3f}\n{100*R[w][lab]["timeout_share"]:.0f}% stopped', (w, v),
                    textcoords='offset points', xytext=(0, 9 if lab == '3D-Learning' else -24),
                    ha='center', fontsize=7, color=INK2)
    ax.text(WL[-1] * 1.25, y[-1], 'Trend-aim' if lab == 'trend-aim' else lab, color=col,
            fontsize=9, va='center')
for w in (230, 10000):
    ta, dl = R[w]['trend-aim']['norm_regret'], R[w]['3D-Learning']['norm_regret']
    adv = 100 * (1 - ta / dl)
    ax.annotate('', xy=(w, ta + 0.012), xytext=(w, dl - 0.012),
                arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.9))
    ax.text(w * 1.12, (ta + dl) / 2 + (0.05 if w == 230 else 0), f'trend-aim\n{adv:.1f}% lower',
            fontsize=7.5, color=INK, va='center')
    print(f'  advantage at {w}: {adv:.1f}%')
ax.set_xscale('log')
ax.set_xticks(WL)
ax.set_xticklabels([str(w) for w in WL])
ax.set_xlim(170, 30000)
ax.set_ylim(0.85, 1.48)
ax.axvline(230, color=MUTED, lw=0.8, ls=':')
ax.text(245, 1.06, 'limit used for\nthe reported tables', fontsize=7.5, color=INK2, va='center')
ax.set_xlabel('Gurobi work limit (work units, log scale)')
ax.set_ylabel('normalised decision regret\n(lower is better)')
ax.yaxis.grid(True, zorder=0)
for ext in ('pdf', 'png'):
    fig.savefig(f'{ROOT}/results/thesis/figures/F18_worklimit.{ext}')
print('F18_DONE')
