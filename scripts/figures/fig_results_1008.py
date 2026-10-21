"""Prints the trend-aim seed spread and the a77/a78/a79 reads against their
pre-registered comparators; writes results/thesis/tables/t5-random.tex.
Reads results/scores/poc_eval_measure/*_selected.json only. F20 is drawn but
not saved (fig_f20.py saves it); fig_supervisor.py writes t2-main.tex."""
import glob, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
EVAL = f'{ROOT}/results/scores/poc_eval_measure'
C1, C2 = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#1a1a19', '#4a4a46', '#8a8a82'
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.edgecolor': MUTED,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.labelcolor': INK,
    'grid.color': '#e6e6e2', 'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 2.0})
ERAS = [('oodtest_2023_2025_renewables', 'renewables'), ('oodtest_2021_2022_gas', 'gas')]


def load(arm, era):
    out = {}
    for f in glob.glob(f'{EVAL}/{arm}_s*_{era}_selected.json'):
        d = json.load(open(f))
        assert d.get('greedy_solves', 0) == 0, f'greedy in {f}'
        out[int(d['seed'])] = float(d['norm_regret'])
    return dict(sorted(out.items()))


def ms(v):
    v = np.array(list(v.values()) if isinstance(v, dict) else v)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan')), len(v)


print('== ITEM 5  trend-aim wide, 20 seeds ==')
for era, en in ERAS:
    t = load('a72_nogenwide', era); d = load('a71_a1wide', era)
    m, s, n = ms(t); m5, s5, _ = ms([t[k] for k in range(1, 6)])
    print(f'  {en:10s} trend-aim n={n} mean {m:.4f} sd {s:.4f} (seeds 1-5: {m5:.4f} sd {s5:.4f}) '
          f'min {min(t.values()):.4f} max {max(t.values()):.4f} | 3DL wide n={len(d)} mean {ms(d)[0]:.4f} '
          f'| trend-aim lower by {100*(1-m/ms(d)[0]):.1f}%')
    w, b = max(t.values()), min(d.values())
    print(f'             worst trend-aim seed {w:.4f} (s{max(t, key=t.get)}) vs best 3DL seed {b:.4f} '
          f'(s{min(d, key=d.get)}): trend-aim lower by {100*(1-w/b):.1f}%')

print('\n== ITEM 8  observed depth (narrow) vs trend-aim narrow a46 (§8.77) ==')
for era, en in ERAS:
    a, r = load('a77_obsdepth', era), load('a46_nogen30', era)
    print(f'  {en:10s} a77 {list(a.values())} mean {ms(a)[0]:.4f} | a46 {list(r.values())} mean {ms(r)[0]:.4f} '
          f'range {min(r.values()):.4f}-{max(r.values()):.4f} | change {100*(ms(a)[0]/ms(r)[0]-1):+.1f}%')

print('\n== ITEM 7  block width (wide) vs 4 h a72 (§8.78: registered range = seeds 1-5) ==')
WIDTHS = [(2, 'a78_bw2'), (4, 'a72_nogenwide'), (6, 'a78_bw6'), (8, 'a78_bw8')]
res = {}
for era, en in ERAS:
    ref = load('a72_nogenwide', era); r5 = [ref[k] for k in range(1, 6)]; r3 = [ref[k] for k in range(1, 4)]
    for w, arm in WIDTHS:
        v = load(arm, era); res[(w, en)] = v
        if w == 4:
            print(f'  {en:10s} 4 h  20 seeds mean {ms(v)[0]:.4f} | seeds 1-3 mean {np.mean(r3):.4f} | '
                  f'seeds 1-5 range {min(r5):.4f}-{max(r5):.4f}')
            continue
        m = ms(v)[0]
        read = 'i inside' if min(r5) <= m <= max(r5) else ('ii worse' if m > max(r5) else 'iii better')
        print(f'  {en:10s} {w} h  {[round(x,4) for x in v.values()]} mean {m:.4f} | vs 4 h seeds 1-3 '
              f'{100*(m/np.mean(r3)-1):+.1f}% | vs 4 h 20-seed {100*(m/ms(ref)[0]-1):+.1f}% | read ({read})')

print('\n== a79  real reference pool (wide) vs a72 (§8.79: registered range = seeds 1-5) ==')
for era, en in ERAS:
    a, ref = load('a79_realref', era), load('a72_nogenwide', era)
    r5 = [ref[k] for k in range(1, 6)]; r3 = [ref[k] for k in range(1, 4)]
    m = ms(a)[0]
    read = 'i inside' if min(r5) <= m <= max(r5) else ('ii worse' if m > max(r5) else 'iii better')
    print(f'  {en:10s} a79 {list(a.values())} mean {m:.4f} | a72 seeds 1-3 {[round(x,4) for x in r3]} '
          f'mean {np.mean(r3):.4f} | paired change {100*(m/np.mean(r3)-1):+.1f}% | a72 20-seed range '
          f'{min(ref.values()):.4f}-{max(ref.values()):.4f} | read ({read})')

# F20 draft: drawn but not saved; fig_f20.py saves F20 at print size.
CENTRE = {2: 13, 4: 14, 6: 15, 8: 12}
HOURS = {2: '12-14 h', 4: '12-16 h', 6: '12-18 h', 8: '08-16 h'}
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), gridspec_kw={'width_ratios': [1, 1, 0.9]})
for ax, (era, en) in zip(axes[:2], ERAS):
    for w, arm in WIDTHS:
        v = np.array(list(res[(w, en)].values()))
        col = C2 if w == 4 else C1
        ax.scatter(np.full(len(v), w) + np.random.RandomState(w).uniform(-0.12, 0.12, len(v)), v,
                   s=14, color=col, alpha=0.55, edgecolor='none', zorder=3)
        ax.plot([w - 0.35, w + 0.35], [v.mean()] * 2, color=col, lw=2.4, zorder=4)
        ax.text(w + 0.42, v.mean(), f'{v.mean():.3f}', va='center', fontsize=7.5, color=INK2)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xticklabels([f'{w} h{" (used)" if w == 4 else ""}\n{HOURS[w]}\ncentre {CENTRE[w]}:00'
                        for w in (2, 4, 6, 8)], fontsize=8)
    ax.set_xlim(1.2, 9.2)
    ax.set_title(f'{en} 2023-2025' if en == 'renewables' else f'{en} 2021-2022')
    ax.set_xlabel('block width (block the rule chose)')
    ax.yaxis.grid(True, zorder=0)
axes[0].set_ylabel('normalised decision regret\n(lower is better)')
ax = axes[2]
for (era, en), col, mk, ls in ((ERAS[0], C1, 'o', '-'), (ERAS[1], INK2, 's', '--')):
    order = sorted(CENTRE, key=lambda w: CENTRE[w])
    xs = [CENTRE[w] for w in order]
    ys = [np.mean(list(res[(w, en)].values())) for w in order]
    ax.plot(xs, ys, color=col, marker=mk, ls=ls, ms=6, zorder=3)
    ax.text(15.12, ys[-1], en, color=col, fontsize=8, va='center')
    for x, y, w in zip(xs, ys, order):
        ax.annotate(f'{w} h', (x, y), textcoords='offset points', xytext=(0, 7), ha='center',
                    fontsize=7, color=INK2)
    print(f'  F20 by centre {en}: ' + ', '.join(f'{x}:00 {y:.4f}' for x, y in zip(xs, ys)))
ax.set_xticks([12, 13, 14, 15]); ax.set_xticklabels(['12:00', '13:00', '14:00', '15:00'])
ax.set_xlim(11.6, 15.9)
ax.set_xlabel('centre hour of the aimed block')
ax.set_title('Ordered by centre hour (means)')
ax.yaxis.grid(True, zorder=0)
fig.text(0.5, -0.06, 'Dots are seeds, bars are means. 4 h (orange) is the width used, 20 seeds; '
         'other widths 3 seeds.', ha='center', fontsize=8, color=INK2)
fig.suptitle('Sensitivity to the block width', y=1.02)
fig.tight_layout()
for ext in ('pdf', 'png'):
    pass  # not saved; see fig_f20.py
print('  F20 skipped here (see fig_f20.py)'); plt.close(fig)

# t5-random.tex: one row per block 0-5, from the score files.
# Block drawn by each random-aim seed, read from the [RANDOM-AIM] log banners.
DRAWN = {1: 5, 2: 0, 3: 1, 4: 2, 5: 1}
rows = {}
for era, en in ERAS:
    ra = load('a74_randomaim', era)
    for s, v in ra.items():
        rows.setdefault(DRAWN[s], {}).setdefault(en, []).append(v)
    rows.setdefault(4, {})[en] = list(load('a76_block4', era).values())
    rows.setdefault(3, {})[en] = list(load('a72_nogenwide', era).values())
T = [r'\begin{table}[H]', r'\centering\footnotesize\setstretch{1}',
     r"\caption{Discount aimed at each block. Bold is the trend rule's block, a "
     r'twenty-seed mean; other rows are single runs, and block 1 is the mean of two.}',
     r'\label{tab:random}', r'\begin{tabular}{@{}c c r r@{}}', r'\toprule',
     r'Block & Hours & Renewables & Gas \\', r'\midrule']
print('  t5-random:')
for b in range(6):
    r, g = np.mean(rows[b]['renewables']), np.mean(rows[b]['gas'])
    hrs = f'{4 * b:02d}--{4 * b + 4:02d}'
    print(f'    block {b} {hrs} n={len(rows[b]["renewables"])} renewables {r:.4f} gas {g:.4f}')
    if b == 3:
        T.append(rf'\textbf{{{b}}} & \textbf{{{hrs}}} & \textbf{{{r:.4f}}} & \textbf{{{g:.4f}}} \\')
    else:
        T.append(f'{b} & {hrs} & {r:.4f} & {g:.4f} \\\\')
T += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{ROOT}/results/thesis/tables/t5-random.tex', 'w').write('\n'.join(T) + '\n')
print('  wrote results/thesis/tables/t5-random.tex')
print('RESULTS_1008_DONE')
