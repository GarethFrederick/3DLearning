"""Prove the new trend knobs reproduce the old hardcoded behaviour EXACTLY.

Calls the real _trend_blocks method on the real training CSV and compares it to
the expression it replaced. Also checks the knobs actually bite when changed, so
this test can fail in both directions.
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Utils'))
from Utils.Algos.ddd_learning import DDD_Learning

tr = pd.read_csv(f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv',
                 parse_dates=['TIMESTAMP'], date_format='mixed')
yr = tr['TIMESTAMP'].dt.year
tr = tr[(yr >= 2015) & (yr <= 2020)]

# bare object carrying only what _trend_blocks reads
o = DDD_Learning.__new__(DDD_Learning)

# (1) default must equal the OLD hardcoded expression, exactly
o._trend_block_hours = 4.0
new_b, new_nb = DDD_Learning._trend_blocks(o, tr)
old_b = np.clip(((tr['TIMESTAMP'].dt.hour
                  + tr['TIMESTAMP'].dt.minute / 60.0) // 4).astype(int), 0, 5)
same = bool((new_b.to_numpy() == old_b.to_numpy()).all())
print(f'default 4h: n_blocks={new_nb} (old fixed 6)   rows identical: {same}')
assert new_nb == 6, 'default must give 6 blocks'
assert same, 'FAIL: default no longer reproduces the old hardcoded grid'

# (2) the knob must actually change something, or the test is inert
o._trend_block_hours = 1.0
b1, nb1 = DDD_Learning._trend_blocks(o, tr)
print(f'knob 1h  : n_blocks={nb1}   differs from default: '
      f'{not bool((b1.to_numpy() == new_b.to_numpy()).all())}')
assert nb1 == 24, 'INSTR_TREND_BLOCK_HOURS=1 must give 24 blocks'
assert not (b1.to_numpy() == new_b.to_numpy()).all(), 'FAIL: knob is inert'

# (3) quantile + degree defaults reproduce the shipped numbers
years = np.array(sorted(tr['TIMESTAMP'].dt.year.unique()), dtype=float)
p5 = np.array([tr['Price'][(new_b == 3) & (tr['TIMESTAMP'].dt.year == int(y))]
               .quantile(0.05) for y in years])
d = float(p5[-1]) - float(np.polyval(np.polyfit(years, p5, 2), 2024.0))
print(f'block 3 depth at degree 2, horizon 2024 = ${d:.2f}   (shipped $214.21)')
assert abs(d - 214.21) < 0.05, f'FAIL: depth drifted to {d:.2f}'

print('\nALL CHECKS PASSED — defaults byte-identical, knobs live.')
