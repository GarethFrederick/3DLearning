"""
Preprocess AEMO Victoria electricity price data for 3D-Learning experiments.

Input:  Data/AEMO/<year>/PRICE_AND_DEMAND_<YYYYMM>_VIC1.csv
        Columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE
        Resolution: 30-min (2015-Oct 2021) then 5-min (Oct 2021 onwards, AEMO 5MS reform)

Output: One CSV per split with columns TIMESTAMP, Price, TOTALDEMAND_MW — all at 30-min resolution
        - AEMO_VIC1_train_2015_2020.csv   (training / in-distribution)
        - AEMO_VIC1_test_2021_2022.csv    (mild OOD)
        - AEMO_VIC1_test_2023_2025.csv    (strong OOD — high renewable penetration)

Preprocessing steps:
  1. Filter to PERIODTYPE == 'TRADE' (drops predispatch/other records)
  2. Clip RRP to [PRICE_FLOOR, PRICE_CAP] — removes extreme spikes while
     preserving negative prices (common during renewable surplus)
  3. Resample to 30-min by averaging, labelled by interval END (see resample_30min)
  4. Rename SETTLEMENTDATE → TIMESTAMP, RRP → Price
  5. Sort by timestamp, drop duplicates
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_ROOT  = os.path.join(os.path.dirname(__file__), 'AEMO')
OUT_DIR    = os.path.dirname(__file__)   # write outputs to Data/

PRICE_FLOOR =  -1000.0  # AUD/MWh  (AEMO market floor price)
PRICE_CAP   =  17500.0  # AUD/MWh  (AEMO market cap; was $14,500 pre-2022).
                        # The true cap keeps spike events in the data.

SPLITS = {
    'AEMO_VIC1_train_2015_2020': list(range(2015, 2021)),
    'AEMO_VIC1_test_2021_2022':  list(range(2021, 2023)),
    'AEMO_VIC1_test_2023_2025':  list(range(2023, 2026)),
}


def load_year(year: int) -> pd.DataFrame:
    pattern = os.path.join(DATA_ROOT, str(year), f'PRICE_AND_DEMAND_{year}*_VIC1.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files found for year {year} at {pattern}')

    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=['SETTLEMENTDATE', 'RRP', 'TOTALDEMAND', 'PERIODTYPE'])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, parse, and clip raw AEMO trade records. NO resampling here —
    the resample is deferred to after concatenation so 30-min bins straddling
    year boundaries get all 6 contributing 5-min observations rather than 1+5
    split across files."""
    df = df[df['PERIODTYPE'] == 'TRADE'].copy()
    df['TIMESTAMP'] = pd.to_datetime(df['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    df['Price'] = df['RRP'].clip(lower=PRICE_FLOOR, upper=PRICE_CAP)
    return (df[['TIMESTAMP', 'Price', 'TOTALDEMAND']]
            .sort_values('TIMESTAMP')
            .drop_duplicates('TIMESTAMP')
            .reset_index(drop=True))


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 30-min resolution, averaging 5-min observations (post-Oct 2021).

    AEMO SETTLEMENTDATE is INTERVAL-ENDING. pandas defaults (closed='left',
    label='left') label 30-min-era bins by END but 5-min-era bins by START, a
    one-slot shift between the training and OOD eras. closed='right',
    label='right' makes every bin (t-30min, t] labelled by its END t, in both
    eras; the 30-min era is unchanged by it."""
    df = (df.set_index('TIMESTAMP')
            .resample('30min', closed='right', label='right').mean()
            .dropna()
            .reset_index())
    return df.rename(columns={'TOTALDEMAND': 'TOTALDEMAND_MW'})


if __name__ == '__main__':
    for split_name, years in SPLITS.items():
        print(f'Processing {split_name} (years: {years[0]}–{years[-1]}) ...')

        all_years = []
        for year in years:
            print(f'  Loading {year} ...', end=' ', flush=True)
            raw = load_year(year)
            clean = preprocess(raw)
            all_years.append(clean)
            print(f'{len(clean):,} rows')

        combined = pd.concat(all_years, ignore_index=True)
        combined = combined.sort_values('TIMESTAMP').drop_duplicates('TIMESTAMP').reset_index(drop=True)
        # Resample once on the full multi-year frame so year-boundary 30-min bins
        # are averaged from all underlying observations, not split across files.
        combined = resample_30min(combined)

        out_path = os.path.join(OUT_DIR, f'{split_name}.csv')
        combined.to_csv(out_path, index=False)

        print(f'  → {out_path}')
        print(f'     rows: {len(combined):,}')
        print(f'     price range: [{combined["Price"].min():.2f}, {combined["Price"].max():.2f}]')
        print(f'     negative prices: {(combined["Price"] < 0).sum():,} '
              f'({100*(combined["Price"] < 0).mean():.1f}%)')
        print(f'     demand range (MW): [{combined["TOTALDEMAND_MW"].min():.0f}, '
              f'{combined["TOTALDEMAND_MW"].max():.0f}]  '
              f'mean={combined["TOTALDEMAND_MW"].mean():.0f}')
        print()

    print('Done.')
