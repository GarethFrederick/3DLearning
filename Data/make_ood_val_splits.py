"""Split each natural-shift OOD CSV into an OOD-VAL half and a locked OOD-TEST
half, for leakage-safe model/epoch/eps selection.

The shift grows over time in each era, so a chronological 50/50 would put each
era's extreme only in the test half. Instead, alternate whole calendar months
to val/test: both halves span the full period. The only leakage is at month
boundaries, acceptable because val is used for SELECTION, not training.

Run from the repo root (paths are relative). Reads Data/AEMO_VIC1_test_XXXX.csv;
writes, per year-range XXXX (overwrites):
  Data/AEMO_VIC1_oodval_XXXX.csv   (even-index months)
  Data/AEMO_VIC1_oodtest_XXXX.csv  (odd-index months)
"""
import pandas as pd

SOURCES = {
    '2021_2022': './Data/AEMO_VIC1_test_2021_2022.csv',
    '2023_2025': './Data/AEMO_VIC1_test_2023_2025.csv',
}


def _skew(df):
    p = df['Price']
    return (f"n={len(p):6d}  mean=${p.mean():7.1f}  %neg={100*(p<0).mean():4.1f}%  "
            f"p1=${p.quantile(0.01):8.1f}  p99=${p.quantile(0.99):8.1f}  max=${p.max():8.1f}")


for tag, path in SOURCES.items():
    df = pd.read_csv(path, parse_dates=['TIMESTAMP'])
    # Ordered unique calendar months -> alternate whole months to val/test.
    ym = df['TIMESTAMP'].dt.to_period('M')
    months = list(dict.fromkeys(ym))                 # preserves chronological order
    month_idx = {m: i for i, m in enumerate(months)}
    is_val = ym.map(month_idx).mod(2).eq(0)          # even-index months -> val

    val_df = df[is_val].reset_index(drop=True)
    test_df = df[~is_val].reset_index(drop=True)

    val_path = f'./Data/AEMO_VIC1_oodval_{tag}.csv'
    test_path = f'./Data/AEMO_VIC1_oodtest_{tag}.csv'
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n[{tag}]  {len(months)} months -> "
          f"val={sum(month_idx[m] % 2 == 0 for m in months)} / "
          f"test={sum(month_idx[m] % 2 == 1 for m in months)} months")
    print(f"  VAL  {val_path}")
    print(f"       {_skew(val_df)}")
    print(f"  TEST {test_path}")
    print(f"       {_skew(test_df)}")
    print(f"  FULL {_skew(df)}")
