"""Block-level statistics for the thesis results section.

The scorer cuts 5 overlapping 48-half-hour windows from each 192-half-hour
block, so windows are not independent. This resamples BLOCKS, not windows.

Window-to-block map (ml.test_task order: batch-major, then origin, then
block-within-batch; batch_size 64, 5 origins per block):

    block_id(k) = (k // 320) * 64 + (k % 64)

That gives 128 blocks for a 640-window era and 64 for a 320-window era. The
script ASSERTS this, so a change in batch size or stride fails loudly.

Reads results/scores/poc_eval_measure/<arm>_s*_<era>_<STATS_KIND>.json.
Prints, per era: each arm's block-mean regret per seed; paired per-block
differences with a block-bootstrap CI, sign test and win share; and a map
check. The map check tests autocorrelation PERIODICITY (peaks at multiples of
64). Do not test within-block > across-block correlation: the five origins sit
12 h apart, so within-block correlation is lower even when the map is right.

Regret is in dollars per window. The oracle schedule does not depend on the
method, so dollar regrets compare across arms on the same window.

Log layout: run_block_stats.sh runs this twice into one log, selected
checkpoints THEN STATS_KIND=fixed (no selection). Each section ends with
BLOCK_STATS_DONE. Slice the log by section header, never by tail.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
EVAL = os.environ.get('STATS_EVAL_DIR', f'{ROOT}/results/scores/poc_eval_measure')
KIND = os.environ.get('STATS_KIND', 'selected')
NBOOT = int(os.environ.get('STATS_NBOOT', '10000'))
SEED = int(os.environ.get('STATS_SEED', '20260918'))
ERAS = [('oodtest_2023_2025_renewables', 'renewables 2023-2025'),
        ('oodtest_2021_2022_gas', 'gas 2021-2022')]
ORIGINS = 5
BATCH = 64


def block_ids(n):
    """Flat window index -> block id. See the module docstring."""
    per_batch = ORIGINS * BATCH
    k = np.arange(n)
    ids = (k // per_batch) * BATCH + (k % BATCH)
    n_blocks, counts = np.unique(ids, return_counts=True)
    assert n % per_batch == 0, f'{n} windows is not a whole number of batches'
    assert len(n_blocks) == n // ORIGINS, (
        f'{n} windows mapped to {len(n_blocks)} blocks, expected {n // ORIGINS}')
    assert set(counts.tolist()) == {ORIGINS}, (
        f'blocks do not all hold {ORIGINS} windows: {sorted(set(counts.tolist()))}')
    return ids


def load(arm, era):
    """-> {seed: per-window regret array}, and the normalised headline per seed."""
    out, norm = {}, {}
    for f in sorted(glob.glob(f'{EVAL}/{arm}_s*_{era}_{KIND}.json')):
        d = json.load(open(f))
        r = np.asarray(d.get('regret_per_window', []), float)
        if r.size == 0:
            continue
        out[int(d['seed'])] = r
        norm[int(d['seed'])] = d.get('norm_regret')
    return out, norm


def block_means(r):
    ids = block_ids(len(r))
    nb = ids.max() + 1
    return np.array([r[ids == b].mean() for b in range(nb)])


def validate_map(r):
    """Periodicity test. Same-block windows sit 64 apart under this map, so the
    autocorrelation must peak at multiples of 64. See the module docstring."""
    def ac(x, l):
        return float(np.corrcoef(x[:-l], x[l:])[0, 1])
    n = len(r)
    mult = [ac(r, l) for l in (ORIGINS * 0 + BATCH, 2 * BATCH, 3 * BATCH) if l < n]
    other = [ac(r, l) for l in (1, 2, 3, 5, 7, 11, BATCH - 1, BATCH + 1) if l < n]
    return float(np.mean(mult)), float(np.mean(other))


def boot_ci(diff_blocks, rng, n=NBOOT, ci=95):
    nb = len(diff_blocks)
    stat = np.empty(n)
    for i in range(n):
        stat[i] = diff_blocks[rng.integers(0, nb, nb)].mean()
    lo, hi = np.percentile(stat, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def sign_p(diff_blocks):
    """Two-sided sign test on block differences. No distribution assumed."""
    from math import comb
    pos = int((diff_blocks > 0).sum())
    n = int((diff_blocks != 0).sum())
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2 * tail))


def main():
    arms = os.environ.get(
        'STATS_ARMS',
        'a72_nogenwide,a71_a1wide,a70_a34wide,a73_vanilla3dl,a74_randomaim,'
        'a75_wzerodose,a34_honestdose,a1_stall,a23_zerodose_r2,a25_recency_r2'
    ).split(',')
    pairs = [p.split(':') for p in os.environ.get(
        'STATS_PAIRS', 'a72_nogenwide:a71_a1wide,a72_nogenwide:a74_randomaim,'
                       'a72_nogenwide:a70_a34wide,a34_honestdose:a1_stall'
    ).split(',')]
    rng = np.random.default_rng(SEED)
    print(f'eval dir : {EVAL}\ncheckpoint kind : {KIND}\nbootstrap draws : {NBOOT}\n')

    for era, label in ERAS:
        print('=' * 78)
        print(f'{label}')
        print('=' * 78)
        store = {}
        for arm in arms:
            r, norm = load(arm, era)
            if not r:
                print(f'  {arm:22s} no scores yet')
                continue
            store[arm] = r
            per_seed = {s: block_means(v).mean() for s, v in r.items()}
            vals = np.array(list(per_seed.values()))
            nv = [norm[s] for s in sorted(norm) if norm[s] is not None]
            print(f'  {arm:22s} n={len(vals)}  block-mean regret $'
                  f'{vals.mean():,.0f}  spread ${vals.max()-vals.min():,.0f}'
                  + (f'  |  normalised {np.mean(nv):.4f}' if nv else ''))
        if store:
            a0 = sorted(store)[0]
            w, a = validate_map(sorted(store[a0].items())[0][1])
            print(f'\n  map check on {a0}: corr at multiples of {BATCH} '
                  f'{w:+.3f}, at other lags {a:+.3f}'
                  + ('  OK, same-block windows are correlated as the map predicts'
                     if w > 2 * max(a, 1e-6)
                     else '  <-- WARNING: no block periodicity; map may be wrong'))
        print()
        for A, B in pairs:
            if A not in store or B not in store:
                continue
            seeds = sorted(set(store[A]) & set(store[B]))
            use = seeds if seeds else None
            bmA = np.mean([block_means(store[A][s]) for s in (use or sorted(store[A]))], axis=0)
            bmB = np.mean([block_means(store[B][s]) for s in (use or sorted(store[B]))], axis=0)
            d = bmB - bmA                       # positive => A is better
            lo, hi = boot_ci(d, rng)
            p = sign_p(d)
            win = 100.0 * (d > 0).mean()
            tag = (f'matched seeds {use}' if use else
                   f'UNMATCHED: {A} n={len(store[A])} vs {B} n={len(store[B])}')
            print(f'  {A} vs {B}   ({tag}, {len(d)} blocks)')
            print(f'     mean per-block advantage to {A}: ${d.mean():,.0f}'
                  f'   95% block-bootstrap CI [${lo:,.0f}, ${hi:,.0f}]'
                  f'   {"EXCLUDES 0" if lo > 0 or hi < 0 else "INCLUDES 0"}')
            print(f'     {A} wins {win:.1f}% of blocks   sign test p={p:.2}')
        print()
    print('######## BLOCK_STATS_DONE ########')


if __name__ == '__main__':
    sys.exit(main())
