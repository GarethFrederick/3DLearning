"""Join every future price file into one training file for the upper bound.

The upper bound trains on real 2021-2025 prices, including the days it is later
scored on. That is deliberate: it is a reference, never a method, and no number
from it may be quoted as one.

All four files, because the two test files alone give 43488 rows (3 batches),
and dose must match the method arm's 61440 rows (5 batches). The validation
months add rows without changing what the bound means.

Writes Data/AEMO_VIC1_ORACLE_traintest.csv. The name is loud on purpose.
"""
import os
from pathlib import Path
import pandas as pd

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
SRC = ['Data/AEMO_VIC1_oodtest_2021_2022.csv',
       'Data/AEMO_VIC1_oodtest_2023_2025.csv',
       'Data/AEMO_VIC1_oodval_2021_2022.csv',
       'Data/AEMO_VIC1_oodval_2023_2025.csv']
OUT = 'Data/AEMO_VIC1_ORACLE_traintest.csv'
NEEDED = 61440          # 5 batches x 64 sequences x 192 rows, matching the method arm

parts = []
for s in SRC:
    d = pd.read_csv(os.path.join(ROOT, s), parse_dates=['TIMESTAMP'])
    print(f'  | {s}: {len(d)} rows, '
          f'{d["TIMESTAMP"].min().date()} to {d["TIMESTAMP"].max().date()}', flush=True)
    parts.append(d)

df = pd.concat(parts, ignore_index=True).sort_values('TIMESTAMP')
df = df.drop_duplicates('TIMESTAMP').reset_index(drop=True)

# No training year may enter, or the comparison stops being "what if you knew
# the future" and becomes "what if you had more data".
yrs = sorted(int(y) for y in df['TIMESTAMP'].dt.year.unique())
assert min(yrs) >= 2021, f'training years leaked in: {yrs}'
assert len(df) >= NEEDED, (
    f'only {len(df)} rows, need {NEEDED} to match the method arm dose')
print(f'  | joined: {len(df)} rows, years {yrs}', flush=True)
print(f'  | enough for {len(df) // 192 // 64} batches per pass '
      f'(the method arm gets 5)', flush=True)

df.to_csv(os.path.join(ROOT, OUT), index=False)
print(f'  | wrote {OUT}', flush=True)
