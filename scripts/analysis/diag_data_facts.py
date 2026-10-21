"""Data facts for the data chapter. Read-only, no model, no solver; prints only.
Reads Data/AEMO/<year>/ raw files and the Data/AEMO_VIC1_*.csv 30-min files.

Per era: raw rows, raw duplicate stamps, raw prices outside the clip band, the
30-min rows written, missing 30-min slots, NaNs. Per scored test file: the start
hour of every scored window, and how many 4-day blocks / windows straddle a
removed month (the test file keeps alternate months only, and the loader cuts it
into 192-row blocks with no regard to month ends).
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.chdir(ROOT)
ERAS = {'train 2015-2020': range(2015, 2021), 'gas 2021-2022': range(2021, 2023),
        'renewables 2023-2025': range(2023, 2026)}
CSV = {'train 2015-2020': 'Data/AEMO_VIC1_train_2015_2020.csv',
       'gas 2021-2022': 'Data/AEMO_VIC1_test_2021_2022.csv',
       'renewables 2023-2025': 'Data/AEMO_VIC1_test_2023_2025.csv'}

print('== RAW FILES ==')
for era, yrs in ERAS.items():
    fs = sorted(f for y in yrs for f in glob.glob(f'Data/AEMO/{y}/PRICE_AND_DEMAND_{y}*_VIC1.csv'))
    raw = pd.concat([pd.read_csv(f, usecols=['SETTLEMENTDATE', 'RRP', 'TOTALDEMAND', 'PERIODTYPE'])
                     for f in fs], ignore_index=True)
    trade = raw[raw['PERIODTYPE'] == 'TRADE']
    ts = pd.to_datetime(trade['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    step = ts.sort_values().diff().dt.total_seconds().div(60).value_counts().head(3).to_dict()
    print(f'  {era:22s} files {len(fs):3d} ({fs[0].split("_")[-2]}..{fs[-1].split("_")[-2]})  '
          f'rows {len(raw):,}  non-TRADE {len(raw) - len(trade):,}  '
          f'duplicate stamps {int(ts.duplicated().sum()):,}  '
          f'RRP<-1000 {int((trade["RRP"] < -1000).sum())}  RRP>17500 {int((trade["RRP"] > 17500).sum())}  '
          f'min RRP {trade["RRP"].min():.2f} max RRP {trade["RRP"].max():.2f}  '
          f'stamp gaps (min:count) {step}')
    if era != 'train 2015-2020':
        cut = pd.Timestamp('2021-10-01')
        pre, post = ts[ts <= cut], ts[ts > cut]
        print(f'      stamps <= 2021-10-01: {len(pre):,}   after: {len(post):,}   '
              f'first 5-min stamp: {ts[ts.dt.minute % 30 != 0].min()}')

print('\n== 30-MIN FILES ==')
for era, path in CSV.items():
    d = pd.read_csv(path, parse_dates=['TIMESTAMP'])
    full = pd.date_range(d['TIMESTAMP'].min(), d['TIMESTAMP'].max(), freq='30min')
    miss = full.difference(d['TIMESTAMP'])
    print(f'  {era:22s} {path}: rows {len(d):,}  span {d["TIMESTAMP"].min()} .. {d["TIMESTAMP"].max()}  '
          f'missing slots {len(miss)}  NaN price {int(d["Price"].isna().sum())}  '
          f'NaN demand {int(d["TOTALDEMAND_MW"].isna().sum())}  duplicate stamps {int(d["TIMESTAMP"].duplicated().sum())}')
    if len(miss):
        print(f'      first missing: {list(miss[:5])}')
    print(f'      demand MW: min {d["TOTALDEMAND_MW"].min():.0f} mean {d["TOTALDEMAND_MW"].mean():.0f} '
          f'max {d["TOTALDEMAND_MW"].max():.0f}')

print('\n== SCORED WINDOWS (loader: 192-row blocks, origins 48..144 step 24, drop_last) ==')
SEQ, B, W = 192, 64, 48
for era, path in [('gas 2021-2022', 'Data/AEMO_VIC1_oodtest_2021_2022.csv'),
                  ('renewables 2023-2025', 'Data/AEMO_VIC1_oodtest_2023_2025.csv')]:
    d = pd.read_csv(path, parse_dates=['TIMESTAMP'])
    nblk = (len(d) // SEQ) // B * B
    ts = d['TIMESTAMP'].to_numpy()[:nblk * SEQ].reshape(nblk, SEQ)
    starts, seam_win, seam_blk = [], 0, 0
    for k in range(nblk):
        gaps = np.diff(ts[k]).astype('timedelta64[m]').astype(int)
        bad = np.where(gaps != 30)[0]
        if len(bad):
            seam_blk += 1
        for t in range(W, SEQ - W + 1, 24):
            starts.append(pd.Timestamp(ts[k][t]))
            if np.any((bad >= t - W) & (bad < t + W - 1)):
                seam_win += 1
    hrs = pd.Series([s.strftime('%H:%M') for s in starts]).value_counts().to_dict()
    months = d['TIMESTAMP'].dt.to_period('M').unique()
    print(f'  {era:22s} rows {len(d):,}  months kept {len(months)} ({months[0]}..{months[-1]})  '
          f'blocks scored {nblk} of {len(d) // SEQ}  windows {len(starts)}  '
          f'rows dropped at tail {len(d) - nblk * SEQ:,} (from {pd.Timestamp(d["TIMESTAMP"].iloc[nblk * SEQ])})')
    print(f'      first forecast slot of each window (interval-ending label): {hrs}')
    print(f'      blocks containing a removed-month seam: {seam_blk}   windows whose 96 rows cross a seam: {seam_win}')
    print(f'      last scored row: {pd.Timestamp(ts[-1][-1])}')
print('DATA_FACTS_DONE')
