"""Thesis figures F1-F6. Static, print-first, greyscale-safe.

COLOR. Slots 1-3 of the reference categorical palette, unmodified and in fixed
order: blue #2a78d6, orange #eb6834, aqua #1baf7a. Their colour-blind contrast
rests on palette.md's recorded check, not a fresh run (the validator needs
node, absent here). At most three hues, no dual axis, and every figure has
direct labels plus a non-colour channel (hatch, marker, position).

Each figure names its own data source; numbers come from the scored JSONs or
the CSVs. Writes <FIG_OUT>/F<n>_*.{png,pdf} (default results/thesis/figures).
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
EVAL = f'{ROOT}/results/scores/poc_eval_measure'
# FIG_OUT overrides the output folder; unset = results/thesis/figures.
OUT = os.environ.get('FIG_OUT', f'{ROOT}/results/thesis/figures')
os.makedirs(OUT, exist_ok=True)

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
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}')
    plt.close(fig)
    print(f'  wrote {OUT}/{name}.png and .pdf')


def load_arm(arm, era, kind='selected'):
    """-> list of (seed, norm_regret) from the scored files."""
    out = []
    for f in sorted(glob.glob(f'{EVAL}/{arm}_s*_{era}_{kind}.json')):
        d = json.load(open(f))
        if d.get('norm_regret') is not None:
            out.append((int(d['seed']), float(d['norm_regret'])))
    return out


def f1_the_shift():
    """The motivation: where the cheap half-hours sit, by year.

    Heatmap of hour x year, one hue; a ring marks each year's cheapest hour so
    it reads without colour. Source: the training file (2015-2020) and the
    oracle file (2021-2026), which cover every year without overlap.
    """
    parts = []
    for p in (f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv',
              f'{ROOT}/Data/AEMO_VIC1_ORACLE_traintest.csv'):
        if os.path.exists(p):
            parts.append(pd.read_csv(p, parse_dates=['TIMESTAMP']))
    df = pd.concat(parts, ignore_index=True).drop_duplicates('TIMESTAMP')
    df['year'] = df['TIMESTAMP'].dt.year
    df['hour'] = df['TIMESTAMP'].dt.hour
    piv = df.groupby(['year', 'hour'])['Price'].quantile(0.05).unstack()
    piv = piv.loc[[y for y in piv.index if 2015 <= y <= 2025]]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(piv.values, aspect='auto', cmap='Blues_r', origin='lower',
                   extent=[-0.5, 23.5, -0.5, len(piv) - 0.5])
    for r, y in enumerate(piv.index):
        h = int(piv.loc[y].idxmin())
        ax.plot(h, r, marker='o', ms=6, mfc='none', mec=C2, mew=1.8, zorder=3)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel('hour of day')
    ax.set_ylabel('year')
    ax.set_title('5th-percentile price by hour and year; ring marks the cheapest hour')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('5th-percentile price ($/MWh)')
    cb.outline.set_edgecolor(MUTED)
    save(fig, 'F1_the_shift')


def f2_mechanism():
    """Where each model puts its cheapest hours: the per-window DISTRIBUTION.

    One panel per era, wide range: histogram of the true cheap-hour centre
    (bars) and each model's predicted centre (lines), with circular means.
    Source: results/analysis/trough_perwindow_{narrow,wide}.json, written by
    diag_trough_location.py under DIAG_DUMP_JSON. Falls back to the
    bar-of-means version if either dump is absent.
    """
    src = {'narrow': f'{ROOT}/results/analysis/trough_perwindow_narrow.json',
           'wide': f'{ROOT}/results/analysis/trough_perwindow_wide.json'}
    data = {k: json.load(open(p)) for k, p in src.items() if os.path.exists(p)}
    if len(data) < 2:
        print('  F2: per-window dumps missing, drawing the bar version instead')
        return f2_mechanism_bars('F2_mechanism')
    eras = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
            ('oodtest_2021_2022_gas', 'gas 2021-2022')]
    bins = np.arange(0, 25, 1)
    centres = bins[:-1] + 0.5

    def circ_mean(h):
        a = 2 * np.pi * np.asarray(h) / 24.0
        return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi)) / (2 * np.pi) * 24

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    for ax, (era, title) in zip(axes, eras):
        d = data['wide'][era]
        th = np.asarray(d['true_hr']) % 24
        n = len(th)
        ax.bar(centres, 100 * np.histogram(th, bins)[0] / n, width=0.9, color='#d6d6d0',
               zorder=2, label='true cheapest hours')
        for lab, col, name, ls in (('ours', C1, 'trend-aim', '-'),
                                   ('3DL', C2, '3D-Learning', (0, (4, 2)))):
            hr = np.asarray(d[lab]['hr']) % 24
            ax.plot(centres, 100 * np.histogram(hr, bins)[0] / n, color=col, ls=ls,
                    lw=2.0, marker='o', ms=3, zorder=4, label=f'{name} forecast')
            ax.axvline(circ_mean(hr), color=col, ls=ls, lw=1.0, zorder=3)
        ax.axvline(circ_mean(th), color=INK2, lw=1.0, zorder=3)
        # One label in the empty morning band; next to the lines they overprint.
        ax.text(0.21, 0.74,
                f'centre\n'
                f'true        {circ_mean(th):4.1f} h\n'
                f'trend-aim   {circ_mean(np.asarray(d["ours"]["hr"]) % 24):4.1f} h\n'
                f'3D-Learning {circ_mean(np.asarray(d["3DL"]["hr"]) % 24):4.1f} h',
                transform=ax.transAxes, va='top', ha='left', fontsize=7,
                family='monospace', color=INK,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=MUTED, lw=0.6))
        print(f'  F2 check {title}: centres true {circ_mean(th):.2f}h, trend-aim '
              f'{circ_mean(np.asarray(d["ours"]["hr"]) % 24):.2f}h, 3D-Learning '
              f'{circ_mean(np.asarray(d["3DL"]["hr"]) % 24):.2f}h, windows {n}')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel('hour of day')
        ax.set_title(title)
        ax.yaxis.grid(True, zorder=0)
    axes[0].set_ylabel('share of test windows (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    save(fig, 'F2_mechanism')


def f2_mechanism_bars(name='F2_mechanism_means'):
    """The four cell means as grouped bars, with the 6.0 h random rule.
    Fallback for f2_mechanism. The values are typed in: diag_trough_location.py
    means over 960 windows per era."""
    cells = ['renewables\nnarrow', 'renewables\nwide', 'gas\nnarrow', 'gas\nwide']
    ours = [3.14, 2.67, 3.57, 3.51]
    base = [7.57, 6.43, 6.81, 5.66]
    x = np.arange(len(cells))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    b1 = ax.bar(x - w / 2, ours, w, color=C1, label='trend-aim', zorder=3)
    b2 = ax.bar(x + w / 2, base, w, color=C2, label='3D-Learning', zorder=3,
                hatch='///', edgecolor='white', linewidth=0)
    ax.axhline(6.0, color=INK2, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(len(cells) - 0.42, 6.12, 'a random guess sits here', fontsize=8,
            color=INK2, ha='right')
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12,
                    f'{r.get_height():.2f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_ylabel('hours from the true cheap-hour centre')
    ax.set_title('Distance to the true cheapest hours (lower is better)')
    ax.set_ylim(0, 8.6)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc='upper left', ncol=2)
    save(fig, name)


def f3_edited_day():
    """What the method actually does to one training day.

    Illustrative: the edit from a72's banners (block 3, centre 14:00, depth
    $214.21) applied to one real training day, 12 June 2019.
    """
    # Raised-cosine bell as in ml._instr_head_augment_price; half-width 8 h is
    # the width seed 1 chose most often. Strength is drawn per window from
    # U(0, depth), so the band is every possible edit and the dashed line the
    # deepest. AEMO stamps the END of each half-hour, so select on slot START.
    df = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv',
                     parse_dates=['TIMESTAMP'])
    start = df['TIMESTAMP'] - pd.Timedelta(minutes=30)
    day = df[start.dt.date == pd.Timestamp('2019-06-12').date()]
    assert len(day) == 48, f'2019-06-12 has {len(day)} half-hours, expected 48'
    p = day['Price'].to_numpy(float)
    h = np.arange(48) / 2.0
    centre, half_width, depth = 14.0, 8.0, 214.21
    d = np.abs(h - centre)
    d = np.minimum(d, 24.0 - d)
    beta = np.where(d < half_width, 0.5 + 0.5 * np.cos(np.pi * d / half_width), 0.0)
    deepest = p - depth * beta

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.fill_between(h, deepest, p, color=C2, alpha=0.14, lw=0, zorder=2,
                    label='possible edits (strength drawn from \\$0 to \\$214)')
    ax.plot(h, p, color=C1, marker='o', ms=3, zorder=4,
            label='a real training day (12 June 2019)')
    ax.plot(h, deepest, color=C2, ls=(0, (5, 2)), marker='s', ms=3, zorder=3,
            label='the same day at full depth')
    ax.axvline(centre, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.annotate('aimed at 14:00\nhalf-width 8 h', xy=(centre, deepest.min()),
                xytext=(centre + 3.2, deepest.min() + 30), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlabel('hour of day')
    ax.set_ylabel('price (\\$/MWh)')
    ax.set_title('The edit: one training day, made cheaper where the trend points')
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 23.5)
    ax.yaxis.grid(True)
    ax.legend(loc='upper left', fontsize=8)
    print(f'  F3 check: bell support {h[beta > 0].min():.1f}-{h[beta > 0].max():.1f} h, '
          f'max cut ${(depth * beta).max():.2f} at {h[beta.argmax()]:.1f} h')
    save(fig, 'F3_edited_day')


def f4_main_results():
    """The headline table as a figure.

    Horizontal bars ordered by score, one panel per era, seeds as dots.
    """
    arms = [('a71_a1wide', '3D-Learning'),
            ('a73_vanilla3dl', '3D-Learning, wrapper off'),
            ('a74_randomaim', 'random aim'),
            ('a75_wzerodose', 'trend-aim, zero discount'),
            ('a70_a34wide', 'trend-aim, generator on (8 rounds)'),
            ('a72_nogenwide', 'trend-aim')]
    eras = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
            ('oodtest_2021_2022_gas', 'gas 2021-2022')]
    # ONE row order for BOTH panels, from the renewables era. With sharey=True
    # the second panel's set_yticklabels overwrites the first's, so per-panel
    # sorting mislabels the renewables panel.
    base = []
    for arm, label in arms:
        v = [x for _, x in load_arm(arm, eras[0][0])]
        if v:
            base.append((arm, label, float(np.mean(v))))
    base.sort(key=lambda r: -r[2])
    order = [(a, l) for a, l, _ in base]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharey=True)
    for ax, (era, title) in zip(axes, eras):
        y, means, allv = [], [], []
        for i, (arm, label) in enumerate(order):
            vals = [v for _, v in load_arm(arm, era)]
            y.append(i)
            means.append(np.mean(vals) if vals else np.nan)
            allv.append(vals)
        xmax = np.nanmax(means) * 1.55
        ax.barh(y, means, 0.6, color=C1, zorder=3)
        for i, (m, vals) in enumerate(zip(means, allv)):
            if not vals:
                continue
            ax.plot(vals, [i] * len(vals), 'o', ms=4, mfc='white', mec=INK,
                    mew=1.0, zorder=4)
            # text clear of BOTH the bar end and the right-most seed dot
            xt = max(m, max(vals)) + xmax * 0.035
            ax.text(xt, i, f'{m:.4f}', va='center',
                    fontsize=8, color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([l for _, l in order])
        ax.set_xlabel('normalised decision regret (lower is better)')
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.xaxis.grid(True, zorder=0)
    axes[0].invert_yaxis()      # best arm at the top, where the eye starts
    fig.suptitle('Held-out decision regret; dots are individual runs', y=1.02)
    save(fig, 'F4_main_results')


def f5_random_aim():
    """Aiming the discount at each block of the day against the trend rule.

    One mark per block; runs that drew the same block are averaged. The random
    control never drew block 4 (16-20 h); a76_block4 fills it, else the column
    shows 'pending'. The trend rule (12-16 h) is the mean of ALL its runs, with
    its min-max range. `drawn` holds each seed's block from the [RANDOM-AIM]
    log banners.
    """
    drawn = {1: 5, 2: 0, 3: 1, 4: 2, 5: 1}
    labels = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-24']
    eras = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
            ('oodtest_2021_2022_gas', 'gas 2021-2022')]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, (era, title) in zip(axes, eras):
        per_block = {}
        for s, v in load_arm('a74_randomaim', era):
            per_block.setdefault(drawn[s], []).append(v)
        for _, v in load_arm('a76_block4', era):
            per_block.setdefault(4, []).append(v)
        _rv = [v for _, v in load_arm('a72_nogenwide', era)]
        rule, rmin, rmax = float(np.mean(_rv)), float(min(_rv)), float(max(_rv))
        xs = sorted(per_block)
        ys = [float(np.mean(per_block[b])) for b in xs]
        ax.plot(xs, ys, 'o', ms=8, mfc='white', mec=C1, mew=1.8, zorder=4,
                label='discount centred on this block')
        # The min-max bar lets a single-run block be read against the seed spread.
        ax.plot([3, 3], [rmin, rmax], color=C2, lw=1.4, zorder=3)
        ax.plot([2.9, 3.1], [rmin, rmin], color=C2, lw=1.4, zorder=3)
        ax.plot([2.9, 3.1], [rmax, rmax], color=C2, lw=1.4, zorder=3)
        ax.plot([3], [rule], 's', ms=8, color=C2, zorder=4,
                label=f'trend rule: mean and range over {len(_rv)} seeds')
        ax.annotate(f'{rule:.3f}', (3, rule), textcoords='offset points', xytext=(9, 0),
                    ha='left', va='center', fontsize=7.5, color=INK)
        for x, y in list(zip(xs, ys)):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=7.5, color=INK)
        allv = ys + [rule, rmin, rmax]
        lo, hi = min(allv), max(allv)
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - 2 * pad, hi + 2.2 * pad)
        if 4 not in per_block:
            ax.axvspan(3.6, 4.4, color=MUTED, alpha=0.12, lw=0, zorder=1)
            ax.text(4, lo - 1.5 * pad, 'pending', ha='center', va='bottom',
                    fontsize=7.5, color=INK2)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, 5.5)
        ax.set_xlabel('block on which the discount is centred')
        ax.set_title(title)
        ax.yaxis.grid(True, zorder=0)
        print(f'  F5 check {title}: ' + ', '.join(f'b{b} {np.mean(per_block[b]):.4f} '
              f'(runs {len(per_block[b])})' for b in xs) + f'; rule {rule:.4f} '
              f'(n={len(_rv)}, min {rmin:.4f}, max {rmax:.4f})')
    axes[0].set_ylabel('normalised decision regret\n(lower is better)')
    axes[0].legend(loc='center left', fontsize=8)
    save(fig, 'F5_random_aim')


def f6_head_range():
    """Narrow against wide output range, same arms.

    Slope chart, renewables era; every line is directly labelled.
    """
    pairs = [('3D-Learning', 'a1_stall', 'a71_a1wide'),
             ('trend-aim, generator on (8 rounds)', 'a34_honestdose', 'a70_a34wide'),
             ('trend-aim', 'a46_nogen30', 'a72_nogenwide')]
    era = 'oodtest_2023_2025_renewables'
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    rows = []
    for label, narrow, wide in pairs:
        vn = [v for _, v in load_arm(narrow, era)]
        vw = [v for _, v in load_arm(wide, era)]
        if not vn or not vw:
            continue
        a, b = float(np.mean(vn)), float(np.mean(vw))
        rows.append((label, a, b, len(vn), len(vw)))
        col = C2 if 'Learning' in label else C1
        ls = (0, (4, 2)) if 'generator on' in label else '-'
        ax.plot([0, 1], [a, b], color=col, ls=ls, marker='o', ms=6, zorder=3)

    # Two lines sit within 0.01 of each other. Push labels apart to a minimum
    # gap in data units; the markers stay where the data are.
    def stagger(vals, gap=0.032):
        out = list(vals)
        order = list(np.argsort(vals))
        for i in range(1, len(order)):
            lo_i, hi_i = order[i - 1], order[i]
            if out[hi_i] - out[lo_i] < gap:
                out[hi_i] = out[lo_i] + gap
        return out

    yl = stagger([r[1] for r in rows])
    yr = stagger([r[2] for r in rows])
    for (label, a, b, nn, nw), ya, yb in zip(rows, yl, yr):
        ax.text(-0.04, ya, f'{a:.4f}', ha='right', va='center',
                fontsize=8, color=INK)
        ax.text(1.04, yb, f'{b:.4f}  {label}', ha='left', va='center',
                fontsize=8, color=INK)
    ax.set_xticks([0, 1])
    # \$ keeps the dollar signs out of mathtext. Bounds are computed, not
    # typed: the wide top is lo + 1.5*span = $358.4975, printed $358 as in F7.
    _lo, _hi = _norm_bounds()
    _sp = _hi - _lo
    ax.set_xticklabels([f'narrow output range\n\\${_lo:.2f} to \\${_half_up(_hi)}',
                        f'wide output range\n−\\${abs(_lo - 0.5 * _sp):.0f} to '
                        f'\\${_lo + 1.5 * _sp:.0f}'])
    ax.set_xlim(-0.6, 2.35)
    ax.set_ylabel('normalised decision regret')
    ax.set_title('Widening what the forecaster may say')
    ax.yaxis.grid(True)
    save(fig, 'F6_head_range')


# Figures F7-F11. Each prints a self-check against a number quoted in the
# thesis, so a data or code slip shows up in the job log.
TRAIN_CSV = f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv'
TEST_CSV = {'renewables 2023-2025': f'{ROOT}/Data/AEMO_VIC1_oodtest_2023_2025.csv',
            'gas 2021-2022': f'{ROOT}/Data/AEMO_VIC1_oodtest_2021_2022.csv'}


def _norm_bounds():
    """The forecaster's normalisation bounds: 1st/99th percentile of the first
    70% of the training file, as main_energy_mv.py computes them."""
    tr = pd.read_csv(TRAIN_CSV)
    fold = tr['Price'].iloc[:int(0.70 * len(tr))]
    return float(fold.quantile(0.01)), float(fold.quantile(0.99))


def _half_up(v):
    """Two decimals, half rounded UP. The upper bound is 242.065, which '.2f'
    prints as 242.06 (binary float sits just below); the thesis uses 242.07."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(f'{v:.3f}').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def f7_output_range():
    """What share of deployment-era prices each output range can express.

    Histogram per era with the two reachable bands shaded (colour AND hatch).
    Source: the held-out test files and the training fold's bounds.
    Self-check: narrow / wide 65.5% / 98.9% renewables, 68.3% / 97.4% gas.
    """
    lo, hi = _norm_bounds()
    span = hi - lo
    wl, wh = lo - 0.5 * span, lo + 1.5 * span
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=False)
    for ax, (title, path) in zip(axes, TEST_CSV.items()):
        p = pd.read_csv(path)['Price'].to_numpy(float)
        nar = 100 * ((p >= lo) & (p <= hi)).mean()
        wid = 100 * ((p >= wl) & (p <= wh)).mean()
        xlo, xhi = -250, 450
        out = 100 * ((p < xlo) | (p > xhi)).mean()
        ax.axvspan(wl, wh, color=C1, alpha=0.10, lw=0, zorder=1)
        ax.axvspan(lo, hi, color=C2, alpha=0.18, lw=0, hatch='///', zorder=1)
        ax.hist(np.clip(p, xlo, xhi), bins=np.arange(xlo, xhi + 5, 5),
                color=INK2, alpha=0.85, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
        ymax = ax.get_ylim()[1]
        ax.text(lo + span * 0.62, ymax * 0.92, f'narrow range\n{nar:.1f}% of half-hours',
                ha='center', fontsize=8, color=INK)
        ax.text(wh - 6, ymax * 0.62, f'wide range\n{wid:.1f}%', ha='right',
                fontsize=8, color=INK)
        ax.set_xlim(xlo, xhi)
        ax.set_title(title)
        ax.set_xlabel('half-hourly price (\\$/MWh)'
                      + (f'; {out:.1f}% lie beyond the axis' if out >= 0.05 else ''))
        print(f'  F7 check {title}: narrow {nar:.2f}%  wide {wid:.2f}%')
    axes[0].set_ylabel('half-hours')
    save(fig, 'F7_output_range')


def _midday_p5(path):
    """5th-percentile price over half-hours stamped 10:00-16:00 inclusive,
    nearest-rank. Reference values: $13.16 / -55.17 / -67.76."""
    d = pd.read_csv(path, parse_dates=['TIMESTAMP'])
    hr = d['TIMESTAMP'].dt.hour + d['TIMESTAMP'].dt.minute / 60.0
    v = np.sort(d.loc[(hr >= 10) & (hr <= 16), 'Price'].to_numpy(float))
    return float(v[int(0.05 * len(v))])


def f8_generator_stall():
    """The midday 5th-percentile price of the 3D-Learning generator, per round.

    One line per training run: the generator's midday 5th-percentile price
    per round, against the real values for the training years and both eras.
    Source: results/training_rows/foldin_coevo_*_{a1_stall,a71_a1wide}_s{1,2,3}.json (midday_p5).
    Self-check: pooled mean $20.83 over 120 rounds.
    """
    refs = [('training years 2015-2020', _midday_p5(TRAIN_CSV)),
            ('gas era 2021-2022', _midday_p5(TEST_CSV['gas 2021-2022'])),
            ('renewables era 2023-2025', _midday_p5(TEST_CSV['renewables 2023-2025']))]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    allv = []
    for arm, ls, rng in (('a1_stall', '-', 'narrow'), ('a71_a1wide', (0, (4, 2)), 'wide')):
        for s in (1, 2, 3):
            f = f'{ROOT}/results/training_rows/foldin_coevo_3DLearning_mv_cond_gt_{arm}_s{s}.json'
            if not os.path.exists(f):
                continue
            rows = json.load(open(f))['rows']
            v = [r['midday_p5'] for r in rows if r.get('midday_p5') is not None]
            allv += v
            ax.plot(np.arange(1, len(v) + 1), v, color=C2, ls=ls, lw=1.2, alpha=0.85,
                    label=f'3D-Learning generator, {rng} range' if s == 1 else None,
                    zorder=3)
    xr = ax.get_xlim()[1]
    for (name, val), ls in zip(refs, ((0, (1, 2)), (0, (3, 2)), '-')):
        ax.axhline(val, color=INK2, lw=1.0, ls=ls, zorder=2)
        ax.text(xr, val + 2.5, f'{name}: ' + (f'\u2212\\${abs(val):.2f}' if val < 0 else f'\\${val:.2f}'), ha='right', fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlabel('training round')
    ax.set_ylabel('midday 5th-percentile price (\\$/MWh)')
    ax.set_ylim(-80, 38)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc='center right', fontsize=8)
    print(f'  F8 check: {len(allv)} rounds, mean ${np.mean(allv):.2f}, '
          f'range ${min(allv):.2f} to ${max(allv):.2f}; refs '
          + ', '.join(f'{n} ${v:.2f}' for n, v in refs))
    save(fig, 'F8_generator_stall')


def f9_decomposition():
    """Where the improvement over 3D-Learning comes from, at the wide range.

    One bar per era from trend-aim to 3D-Learning, split at the zero-discount
    run: real days instead of generated ones (3D-Learning -> zero discount),
    then the trend-aimed discount (zero discount -> trend-aim). NOT the output
    range: all three runs are wide. Self-check (real days / aim): renewables
    0.1759 / 0.2629, gas 0.1219 / 0.1175.
    """
    eras = [('oodtest_2023_2025_renewables', 'renewables\n2023-2025'),
            ('oodtest_2021_2022_gas', 'gas\n2021-2022')]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for i, (era, label) in enumerate(eras):
        m71 = np.mean([v for _, v in load_arm('a71_a1wide', era)])
        m75 = np.mean([v for _, v in load_arm('a75_wzerodose', era)])
        m72 = np.mean([v for _, v in load_arm('a72_nogenwide', era)])
        tot = m71 - m72
        ax.barh(i, m75 - m72, left=m72, height=0.5, color=C1, zorder=3,
                label='the trend-rule discount' if i == 0 else None)
        ax.barh(i, m71 - m75, left=m75, height=0.5, color=C3, hatch='///',
                edgecolor='white', lw=0, zorder=3,
                label='real training days instead of generated ones' if i == 0 else None)
        ax.text((m72 + m75) / 2, i, f'{m75 - m72:.4f}\n({100 * (m75 - m72) / tot:.1f}%)',
                ha='center', va='center', fontsize=8, color='white', zorder=4)
        ax.text((m75 + m71) / 2, i, f'{m71 - m75:.4f}\n({100 * (m71 - m75) / tot:.1f}%)',
                ha='center', va='center', fontsize=8, color=INK, zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
        for x, t in ((m72, 'trend-aim'), (m75, 'zero discount'), (m71, '3D-Learning')):
            ax.plot([x, x], [i - 0.3, i + 0.3], color=INK, lw=1.2, zorder=5)
            ax.text(x, i + 0.33, f'{t}\n{x:.4f}', ha='center', va='top', fontsize=7.5,
                    color=INK)
        print(f'  F9 check {label.split()[0]}: real days {m71 - m75:.4f}, '
              f'aim {m75 - m72:.4f}, total {tot:.4f}')
    ax.set_yticks([0, 1])
    ax.set_yticklabels([l for _, l in eras])
    ax.set_ylim(-0.4, 1.75)
    ax.invert_yaxis()
    ax.set_xlabel('normalised decision regret (lower is better), wide output range')
    ax.xaxis.grid(True, zorder=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8)
    save(fig, 'F9_decomposition')


def f10_narrow_ladder():
    """The narrow-range controls in one ordered bar chart.

    Bars with seeds as dots, one order for both panels. Every rung has a
    30-round cap, but early stopping keeps rounds 1-14, so the rungs are not
    dose-matched, and most steps change several switches at once (see
    ARMS.md). Trend-aim here is the 30-round, generator-off run (a46), not the
    8-round one. Source: scored files.
    """
    rungs = [('a1_stall', '3D-Learning'),
             ('a23_zerodose_r2', 'real days, no discount'),
             ('a25_recency_r2', 'real days 2019-2020 only, no discount'),
             ('a77_obsdepth', 'real days, observed \\$24 discount'),
             ('a19_discwidth', 'real days, \\$120 discount at the divergence block'),
             ('a46_nogen30', 'trend-aim')]
    eras = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
            ('oodtest_2021_2022_gas', 'gas 2021-2022')]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.1), sharey=True)
    for ax, (era, title) in zip(axes, eras):
        means = []
        for i, (arm, _) in enumerate(rungs):
            vals = [v for _, v in load_arm(arm, era)]
            m = float(np.mean(vals))
            means.append(m)
            ax.barh(i, m, 0.6, color=C2 if arm == 'a1_stall' else C1, zorder=3)
            ax.plot(vals, [i] * len(vals), 'o', ms=4, mfc='white', mec=INK, mew=1.0,
                    zorder=4)
            ax.text(max(m, max(vals)) + 0.02, i, f'{m:.4f}', va='center', fontsize=8,
                    color=INK)
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels([l for _, l in rungs])
        ax.set_xlim(0, max(means) * 1.25)
        ax.set_title(title)
        ax.set_xlabel('normalised decision regret (lower is better)')
        ax.xaxis.grid(True, zorder=0)
        print(f'  F10 check {title}: ' + ', '.join(f'{a} {m:.4f}' for (a, _), m in
                                                   zip(rungs, means)))
    axes[0].invert_yaxis()
    save(fig, 'F10_ladder')


def f11_per_block():
    """Per-block advantage of trend-aim over 3D-Learning, wide range.

    Every 192-slot block as one sorted bar; windows overlap, so the block is
    the unit of inference. Uses block_stats.py's block map and matched-seed
    averaging, so it must reproduce block_stats.py: win share / mean advantage
    86.7% / $122,422 renewables, 78.1% / $60,419 gas.
    """
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
    import block_stats as B
    eras = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
            ('oodtest_2021_2022_gas', 'gas 2021-2022')]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for ax, (era, title) in zip(axes, eras):
        ra, _ = B.load('a72_nogenwide', era)
        rb, _ = B.load('a71_a1wide', era)
        seeds = sorted(set(ra) & set(rb))
        bma = np.mean([B.block_means(ra[s]) for s in seeds], axis=0)
        bmb = np.mean([B.block_means(rb[s]) for s in seeds], axis=0)
        d = np.sort(bmb - bma) / 1000.0
        win = 100 * (d > 0).mean()
        x = np.arange(len(d))
        pos = d > 0
        ax.bar(x[pos], d[pos], 0.9, color=C1, zorder=3,
               label='trend-aim cheaper')
        ax.bar(x[~pos], d[~pos], 0.9, color=C2, hatch='///', edgecolor='white', lw=0,
               zorder=3, label='3D-Learning cheaper')
        ax.axhline(0, color=INK2, lw=0.8, zorder=4)
        ax.text(0.03, 0.95, f'trend-aim cheaper in {win:.1f}% of {len(d)} blocks\n'
                            f'mean advantage \\${1000 * d.mean():,.0f} per block',
                transform=ax.transAxes, va='top', fontsize=8, color=INK)
        ax.set_title(title)
        ax.set_xlabel('four-day blocks, sorted by advantage')
        ax.set_xticks([])
        ax.yaxis.grid(True, zorder=0)
        print(f'  F11 check {title}: seeds {seeds}, win {win:.1f}%, '
              f'mean ${1000 * d.mean():,.0f}, blocks {len(d)}')
    axes[0].set_ylabel('3D-Learning regret minus trend-aim regret\n(thousand \\$ per block)')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Block by block, wide output range', y=1.02)
    save(fig, 'F11_per_block')


if __name__ == '__main__':
    for fn in (f1_the_shift, f2_mechanism, f2_mechanism_bars, f3_edited_day,
               f4_main_results, f5_random_aim, f6_head_range, f7_output_range,
               f8_generator_stall, f9_decomposition, f10_narrow_ladder, f11_per_block):
        try:
            fn()
        except Exception as e:                      # keep going; report loudly
            print(f'  FAILED {fn.__name__}: {type(e).__name__}: {e}')
    print('######## FIGURES_DONE ########')
