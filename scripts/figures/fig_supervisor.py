"""Supervisor-review figures and tables (printed as ITEM 1, 3a-3d, 6).

Reads the scored files (selected checkpoint) and the training CSV only. No
solver, no model. The style is copied from make_thesis_figures.py, not
imported, because importing it runs every figure. Outputs (.pdf and .png):
  results/thesis/figures/F13_regret_box        wide configurations, both eras
  results/thesis/figures/F14_regret_box_narrow narrow configurations, both eras
  results/thesis/figures/F15_block_advantage_box
  results/thesis/figures/F16_depth_fit
  results/thesis/tables/t-seed-stats.tex   (build_tseedstats.py writes it too;
                                            the last one run wins)
  results/thesis/tables/t2-main.tex
Self-checks against known numbers print OK / MISMATCH.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
EVAL = f'{ROOT}/results/scores/poc_eval_measure'
FIG = f'{ROOT}/results/thesis/figures'
TAB = f'{ROOT}/results/thesis/tables'
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

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
ERAS = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
        ('oodtest_2021_2022_gas', 'gas 2021-2022')]


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG}/{name}.{ext}')
    plt.close(fig)
    print(f'  wrote {FIG}/{name}.pdf')


def load(arm, era):
    """-> {seed: (norm_regret, sum_regret, sum_abs_opt)} from selected files."""
    out = {}
    for f in sorted(glob.glob(f'{EVAL}/{arm}_s*_{era}_selected.json')):
        d = json.load(open(f))
        if d.get('norm_regret') is None:
            continue
        r = np.asarray(d['regret_per_window'], float)
        nr = float(d['norm_regret'])
        out[int(d['seed'])] = (nr, float(r.sum()), float(r.sum() / nr),
                               int(d.get('mip_timeouts', 0)), int(d.get('mip_solves', 0)))
    return out


def check(label, got, want, tol=5e-5):
    ok = abs(got - want) <= tol
    print(f'  CHECK {label}: {got:.4f} vs {want:.4f} ' + ('OK' if ok else '<-- MISMATCH'))


# Item 1d: seed-statistics table.
CONFIGS = [  # (group, label, arm, range)
    ('Reported method and variants', 'Trend-aim', 'a72_nogenwide', 'wide'),
    ('Reported method and variants', 'Trend-aim, generator on (8 rounds)', 'a70_a34wide', 'wide'),
    ('Reported method and variants', 'Trend-aim', 'a46_nogen30', 'narrow'),
    ('Reported method and variants', 'Trend-aim, generator on', 'a44_long30', 'narrow'),
    ('Reported method and variants', 'Trend-aim, generator on (8 rounds)', 'a34_honestdose', 'narrow'),
    ('Reported method and variants', 'As above, WDRO warm start', 'a59_drospo', 'narrow'),
    ('Published method', '3D-Learning', 'a71_a1wide', 'wide'),
    ('Published method', '3D-Learning, wrapper off', 'a73_vanilla3dl', 'wide'),
    ('Published method', '3D-Learning', 'a1_stall', 'narrow'),
    ('Controls', 'Random aim block', 'a74_randomaim', 'wide'),
    ('Controls', 'Aim fixed at 16--20\\,h', 'a76_block4', 'wide'),
    ('Controls', 'Zero discount', 'a75_wzerodose', 'wide'),
    ('Controls', 'Trend-aim, observed \\$24 depth', 'a77_obsdepth', 'narrow'),
    ('Controls', 'Block width 2\\,h (12--14\\,h, centre 13:00)', 'a78_bw2', 'wide'),
    ('Controls', 'Block width 6\\,h (12--18\\,h, centre 15:00)', 'a78_bw6', 'wide'),
    ('Controls', 'Block width 8\\,h (08--16\\,h, centre 12:00)', 'a78_bw8', 'wide'),
    ('Controls', 'Real-window reference', 'a79_realref', 'wide'),
    ('Controls', 'Divergence-aimed discount', 'a19_discwidth', 'narrow'),
    ('Controls', 'No discount, all training years', 'a23_zerodose_r2', 'narrow'),
    ('Controls', 'No discount, 2019--2020 only', 'a25_recency_r2', 'narrow'),
    ('Classical baselines', 'PTO', 'pto_base', 'narrow'),
    ('Classical baselines', 'WDRO-PTO', 'wdropto_base', 'narrow'),
    ('Classical baselines', 'DFL', 'dfl_warmstart', 'narrow'),
    ('Classical baselines', 'WDRO-SPO', 'wdrospo_base', 'narrow'),
]
data = {(arm, era): load(arm, era) for _, _, arm, _ in CONFIGS for era, _ in ERAS}
check('a72 renewables mean, seeds 1-5', np.mean([v[0] for s, v in data[('a72_nogenwide', ERAS[0][0])].items() if s <= 5]), 0.4496)
check('a46 renewables mean', np.mean([v[0] for v in data[('a46_nogen30', ERAS[0][0])].values()]), 0.5495)
check('a46 gas mean', np.mean([v[0] for v in data[('a46_nogen30', ERAS[1][0])].values()]), 0.4296)
check('a1_stall renewables mean', np.mean([v[0] for v in data[('a1_stall', ERAS[0][0])].values()]), 0.9383)


def stats(v):
    v = np.asarray(v, float)
    sd = f'{v.std(ddof=1):.4f}' if len(v) > 1 else '--'
    return len(v), v.min(), v.max(), np.median(v), v.mean(), sd


lines = [r'\begin{table}[H]', r'\centering\footnotesize\setstretch{1}',
         r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Regret over seeds for every configuration.}',
         r'\label{tab:seedstats}',
         r'\begin{tabular}{@{}l l l r r r r r r@{}}', r'\toprule',
         r'Configuration & Range & Era & $n$ & Min & Max & Median & Mean & SD \\',
         r'\midrule']
print('\n  ITEM 1d  seed statistics (n, min, max, median, mean, SD)')
grp_prev = None
for grp, lab, arm, rng in CONFIGS:
    if grp != grp_prev:
        if grp_prev is not None:
            lines.append(r'\addlinespace')
        lines.append(r'\multicolumn{9}{@{}l}{\textit{' + grp + r'}}\\')
        grp_prev = grp
    for k, (era, et) in enumerate(ERAS):
        v = [x[0] for x in data[(arm, era)].values()]
        if not v:
            print(f'  MISSING {arm} {era}')
            continue
        n, mn, mx, md, me, sd = stats(v)
        name = lab if k == 0 else ''
        rr = rng if k == 0 else ''
        lines.append(f'{name} & {rr} & {et.split()[0]} & {n} & {mn:.4f} & {mx:.4f} & '
                     f'{md:.4f} & {me:.4f} & {sd} \\\\')
        print(f'  {lab:38s} {rng:6s} {et:22s} n={n} min={mn:.4f} max={mx:.4f} '
              f'median={md:.4f} mean={me:.4f} sd={sd}')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{TAB}/t-seed-stats.tex', 'w').write('\n'.join(lines) + '\n')
print(f'  wrote {TAB}/t-seed-stats.tex')

# t2-main: same configurations, mean and timeout share per era.
# Timeout share = sum(mip_timeouts) / sum(mip_solves) over the seeds; every
# solve counts (optimal and forecast), as in table_baselines.tex.
t2 = [r'\begin{table}[H]', r'\centering\footnotesize\setstretch{1}',
      r'\setlength{\tabcolsep}{3pt}',
      r'\caption{Regret and timeout rate for every configuration.}', r'\label{tab:main}',
      r'\begin{tabular}{@{}l l r r r r r@{}}', r'\toprule',
      r' & & & \multicolumn{2}{c}{Renewables 2023--25} & \multicolumn{2}{c}{Gas 2021--22} \\',
      r'Configuration & Range & $n$ & Mean & t/o\% & Mean & t/o\% \\', r'\midrule']
print('\n  T2-MAIN  mean and timeout share per era')
grp_prev = None
for grp, lab, arm, rng in CONFIGS:
    if grp != grp_prev:
        if grp_prev is not None:
            t2.append(r'\addlinespace')
        t2.append(r'\multicolumn{7}{@{}l}{\textit{' + grp + r'}}\\')
        grp_prev = grp
    cells, ns = [], []
    for era, et in ERAS:
        dd = data[(arm, era)]
        m = np.mean([x[0] for x in dd.values()])
        to = 100.0 * sum(x[3] for x in dd.values()) / max(1, sum(x[4] for x in dd.values()))
        cells += [f'{m:.4f}', f'{to:.1f}']
        ns.append(len(dd))
    assert ns[0] == ns[1], (arm, ns)
    t2.append(f'{lab} & {rng} & {ns[0]} & ' + ' & '.join(cells) + r' \\')
    print(f'  {lab:48s} {rng:6s} n={ns[0]:2d}  ' + '  '.join(cells))
t2 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{TAB}/t2-main.tex', 'w').write('\n'.join(t2) + '\n')
print(f'  wrote {TAB}/t2-main.tex')


# Items 1a/1b: box plots.
def box_fig(cfgs, name, title):
    # Horizontal boxes: long configuration names stay readable on one axis.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.45 * len(cfgs) + 1.4), sharey=True)
    for ax, (era, et) in zip(axes, ERAS):
        vals = [[x[0] for x in data[(arm, era)].values()] for _, arm in cfgs]
        pos = np.arange(len(cfgs))
        ax.boxplot(vals, positions=pos, widths=0.5, showfliers=False, vert=False,
                   medianprops=dict(color=INK, lw=1.4),
                   boxprops=dict(color=INK2, lw=0.9), whiskerprops=dict(color=INK2, lw=0.9),
                   capprops=dict(color=INK2, lw=0.9))
        rs = np.random.RandomState(0)
        for p, (lab, arm), v in zip(pos, cfgs, vals):
            col = C2 if lab.startswith('Trend-aim') else C1
            mk = 's' if lab.startswith('Trend-aim') else 'o'
            ax.scatter(v, p + rs.uniform(-0.12, 0.12, len(v)), s=22, color=col, marker=mk,
                       edgecolor='white', linewidth=0.6, zorder=3)
        for (lab, _), v in zip(cfgs, vals):
            print(f'  {name} {et:22s} {lab.replace(chr(10), " "):30s} n={len(v):2d} '
                  f'min={min(v):.4f} median={np.median(v):.4f} max={max(v):.4f}')
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{lab.replace(chr(10), ' ')} ({len(v)} seeds)" for (lab, _), v in zip(cfgs, vals)], fontsize=8)
        ax.set_title(et)
        ax.xaxis.grid(True, zorder=0)
        ax.set_xlabel('normalised decision regret (lower is better)')
    fig.tight_layout()
    save(fig, name)


box_fig([('3D-Learning', 'a71_a1wide'), ('3D-Learning,\nwrapper off', 'a73_vanilla3dl'),
         ('Zero discount', 'a75_wzerodose'), ('Random block', 'a74_randomaim'),
         ('Real-window reference', 'a79_realref'), ('Trend-aim', 'a72_nogenwide')],
        'F13_regret_box', 'Regret across seeds, wide output range')
box_fig([('3D-Learning', 'a1_stall'), ('No discount,\nall years', 'a23_zerodose_r2'),
         ('No discount,\n2019-2020', 'a25_recency_r2'), ('Divergence block', 'a19_discwidth'),
         ('Trend-aim, gen on,\n8 rounds', 'a34_honestdose'), ('Trend-aim, gen on,\nWDRO start', 'a59_drospo'),
         ('Trend-aim,\ngenerator on', 'a44_long30'),
         ('Trend-aim, observed \\$24 depth', 'a77_obsdepth'), ('Trend-aim', 'a46_nogen30')],
        'F14_regret_box_narrow', 'Regret across seeds, narrow output range')

# Item 1c: per-block advantage.
import block_stats as B
print('\n  ITEM 1c  per-block advantage, 3D-Learning regret minus trend-aim regret, $ per block')
fig, ax = plt.subplots(figsize=(6.4, 3.4))
groups, labels = [], []
for era, et in ERAS:
    for rng, a, b in [('wide', 'a72_nogenwide', 'a71_a1wide'), ('narrow', 'a46_nogen30', 'a1_stall')]:
        ra, _ = B.load(a, era)
        rb, _ = B.load(b, era)
        # Each arm is averaged over ALL its own seeds (trend-aim wide has 20,
        # 3D-Learning 5); matched-seed averaging would drop 15.
        seeds = (sorted(ra), sorted(rb))
        bma = np.mean([B.block_means(ra[s]) for s in seeds[0]], axis=0)
        bmb = np.mean([B.block_means(rb[s]) for s in seeds[1]], axis=0)
        d = bmb - bma
        groups.append(d / 1000.0)
        nneg = int((d < 0).sum())
        labels.append(f'{et.split()[0]}\n{rng}\n{nneg} of {len(d)} below 0')
        if era.endswith('renewables') and rng == 'wide':
            # Block b is rows b*192 .. b*192+191 of the test file (block_stats map:
            # block = batch*64 + sequence, 192 rows per sequence, no shuffle).
            _bi = int(np.argmin(d))
            _ts = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_oodtest_2023_2025.csv',
                              parse_dates=['TIMESTAMP'])['TIMESTAMP']
            _t0, _t1 = _ts.iloc[_bi * 192], _ts.iloc[_bi * 192 + 191]
            outlier = (len(groups) - 1, d[_bi] / 1000.0,
                       f'{_t0:%d %b}–{_t1:%d %b %Y}')
            print(f'  most negative renewables-wide block: #{_bi} ${d[_bi]:,.0f} '
                  f'rows {_bi * 192}-{_bi * 192 + 191} = {_t0} .. {_t1} (interval-ending stamps)')
        print(f'  {et:22s} {rng:6s} blocks below zero: {nneg} of {len(d)}')
        q1, md, q3 = np.percentile(d, [25, 50, 75])
        blo, bhi = B.boot_ci(d, np.random.default_rng(B.SEED))
        print(f'  {et:22s} {rng:6s} seeds {seeds} blocks {len(d)}  median ${md:,.0f}  '
              f'IQR ${q1:,.0f}..${q3:,.0f}  mean ${d.mean():,.0f}  min ${d.min():,.0f}  '
              f'max ${d.max():,.0f}  trend-aim cheaper {100 * (d > 0).mean():.1f}%  '
              f'95% block-bootstrap CI of the mean ${blo:,.0f}..${bhi:,.0f}')
pos = np.arange(len(groups))
ax.boxplot(groups, positions=pos, widths=0.5, showfliers=True,
           flierprops=dict(marker='o', markersize=3, markerfacecolor=MUTED, markeredgecolor='none'),
           medianprops=dict(color=INK, lw=1.4), boxprops=dict(color=INK2, lw=0.9),
           whiskerprops=dict(color=INK2, lw=0.9), capprops=dict(color=INK2, lw=0.9))
ax.axhline(0, color=C2, lw=1.0, ls='--')
ax.annotate(f'−\\${abs(outlier[1]):,.0f}k\n{outlier[2]}', (outlier[0], outlier[1]),
            textcoords='offset points', xytext=(10, 0), fontsize=7.5, color=INK2, va='center')
ax.text(2.5, -25, 'zero difference', color=C2, fontsize=8, va='top', ha='center')
ax.set_xticks(pos)
ax.set_xticklabels(labels)
ax.set_ylabel('3D-Learning regret minus trend-aim\nregret (thousand \\$ per four-day test block)')
ax.yaxis.grid(True, zorder=0)
save(fig, 'F15_block_advantage_box')

# Item 6: pooled eras.
print('\n  ITEM 6  pooled eras (gas + renewables windows together), sum(regret)/sum(|opt|)')
for lab, arm in [('trend-aim wide', 'a72_nogenwide'), ('3D-Learning wide', 'a71_a1wide')]:
    g, r = data[(arm, ERAS[1][0])], data[(arm, ERAS[0][0])]
    seeds = sorted(set(g) & set(r))
    pooled = [(g[s][1] + r[s][1]) / (g[s][2] + r[s][2]) for s in seeds]
    print(f'  {lab:18s} seeds {seeds} pooled per seed ' + ' '.join(f'{p:.4f}' for p in pooled)
          + f'  mean {np.mean(pooled):.4f}  sd {np.std(pooled, ddof=1):.4f}')

# Items 3a-3d: block floor, depth, statistic, degree.
tr = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv', parse_dates=['TIMESTAMP'])
tr['y'] = tr['TIMESTAMP'].dt.year
tr = tr[(tr['y'] >= 2015) & (tr['y'] <= 2020)].copy()
tr['b'] = np.clip(((tr['TIMESTAMP'].dt.hour + tr['TIMESTAMP'].dt.minute / 60.) // 4).astype(int), 0, 5)
yrs = np.array(sorted(tr['y'].unique()), float)
xc = yrs - yrs.mean()
BL = [f'{4 * i:02d}-{4 * i + 4:02d}h' for i in range(6)]


def slope(v):
    return float((xc * (v - v.mean())).sum() / (xc ** 2).sum())


def table(col, fn):
    return np.array([[fn(tr[(tr['b'] == i) & (tr['y'] == int(y))][col]) for y in yrs] for i in range(6)])


p5 = table('Price', lambda s: s.quantile(0.05))
print('\n  ITEM 3a  block floor (5th-percentile price), $/MWh')
for i in range(6):
    print(f'  {BL[i]}  by year ' + ' '.join(f'{v:+7.2f}' for v in p5[i])
          + f'  | mean {p5[i].mean():+7.2f}  slope {slope(p5[i]):+6.2f}')
print(f'  lowest MEAN floor 2015-2020: {BL[int(np.argmin(p5.mean(1)))]} (${p5.mean(1).min():.2f})')
print(f'  lowest floor in 2020 alone : {BL[int(np.argmin(p5[:, -1]))]} (${p5[:, -1].min():.2f})')
print(f'  fastest-falling floor      : {BL[int(np.argmin([slope(v) for v in p5]))]}')
check('block 3 slope', slope(p5[3]), -5.09, tol=0.005)

print('\n  ITEM 3b  depth inputs, block 12-16h')
print(f'  observed 2019->2020 fall: ${p5[3][-2]:.2f} -> ${p5[3][-1]:.2f} = ${p5[3][-2] - p5[3][-1]:.2f}')

STATS = [('1st pct', lambda s: s.quantile(0.01)), ('5th pct', lambda s: s.quantile(0.05)),
         ('10th pct', lambda s: s.quantile(0.10)), ('25th pct', lambda s: s.quantile(0.25)),
         ('median', lambda s: s.median()), ('mean', lambda s: s.mean()), ('minimum', lambda s: s.min())]
for col, unit in [('Price', '$/MWh'), ('TOTALDEMAND_MW', 'MW')]:
    print(f'\n  ITEM 3c  statistic choice on {col} ({unit}); picks = block with the most negative slope')
    print(f'  {"statistic":10s} {"12-16h slope/yr":>16s} {"picks":>8s} {"SD of 6 years":>14s} '
          f'{"SD about trend":>15s}')
    for nm, fn in STATS:
        t = table(col, fn)
        sl = [slope(v) for v in t]
        pick = int(np.argmin(sl))
        res = t[3] - np.polyval(np.polyfit(yrs, t[3], 1), yrs)
        print(f'  {nm:10s} {sl[3]:+16.2f} {BL[pick]:>8s} {t[3].std(ddof=1):14.2f} '
              f'{np.sqrt((res ** 2).sum() / (len(yrs) - 2)):15.2f}')

print('\n  ITEM 3d  polynomial degree, block 12-16h floor extended to 2024')
fig, ax = plt.subplots(figsize=(5.6, 3.4))
xx = np.linspace(2015, 2024, 200)
ax.scatter(yrs, p5[3], s=34, color=INK, zorder=4, label='observed floor, 2015-2020')
for deg, col, ls in [(1, C1, '-'), (2, C2, '-'), (3, C3, '--')]:
    c = np.polyfit(yrs, p5[3], deg)
    at = float(np.polyval(c, 2024.0))
    dep = float(p5[3][-1]) - at
    print(f'  degree {deg}: floor at 2024 ${at:9.2f}  depth ${dep:8.2f}')
    ax.plot(xx, np.polyval(c, xx), color=col, ls=ls, lw=1.8,
            label=f'degree {deg}: depth \\${dep:,.0f}')
    if deg == 2:
        check('degree-2 depth', dep, 214.21, tol=0.005)
ax.axvline(2020.5, color=MUTED, lw=0.8, ls=':')
ax.text(2020.6, ax.get_ylim()[1], 'training ends', color=INK2, fontsize=8, va='top')
ax.set_xlabel('year')
ax.set_ylabel('12-16 h 5th-percentile price ($/MWh)')
ax.legend(loc='lower left', fontsize=8)
ax.yaxis.grid(True, zorder=0)
save(fig, 'F16_depth_fit')
print('SUPERVISOR_DONE')
