"""Why do some methods' forecasts time out the MIP more often?
Hypothesis: flat or tied forecasts make many schedules near-equal in cost, so
branch-and-bound cannot prove optimality within the work limit. Forward passes
only, no solves; prints only. For every scored window of one era (same
loader/order as the scorer) and each seed-1 head named in the scored JSONs:
forecast range ($), share of slots on the head's clamp bounds, number of
distinct values, and the gap between the cheapest and 6th-cheapest slot."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import poc_eval_ood as P
from Utils.datasets import Data_Loader, build_ml_features
EVAL = f'{P.ROOT}/results/scores/poc_eval_measure'
ARMS = [('trend-aim (narrow)', 'a46_nogen30'), ('3D-Learning (narrow)', 'a1_stall'),
        ('trend-aim (wide)', 'a72_nogenwide'), ('3D-Learning (wide)', 'a71_a1wide'),
        ('DFL', 'dfl_warmstart'), ('WDRO-SPO', 'wdrospo_base'), ('PTO', 'pto_base')]
# SHAPE_REGIME: substring of an OOD_REGIMES name; unset = renewables.
regime, csv = [r for r in P.OOD_REGIMES if os.environ.get('SHAPE_REGIME', 'renewables') in r[0]][0]
print(f'  | era {regime}')
price_b, demand_b = P.train_fold_bounds(); lo, hi = price_b; span = hi - lo
feat = build_ml_features(pd.read_csv(csv, parse_dates=['TIMESTAMP']))[P.feature_cols_for(8)]
loader = Data_Loader(df=feat).ml_set2loader_mv(split_test=False, normlized=True, device='cpu',
    batch_size=P.BATCH_SIZE, pic_size=P.ML_SEQ_LEN,
    norm_bounds={'Price': price_b, 'TOTALDEMAND_MW': demand_b}, drop_last=True)
batches = [b[0] for b in loader]
print(f'{"method":22s} {"t/o":>6s} {"range$":>8s} {"clamped%":>9s} {"distinct":>9s} {"trough$":>8s} {"true range$":>11s}')
for name, arm in ARMS:
    j = json.load(open(f'{EVAL}/{arm}_s1_{regime}_selected.json'))
    m = torch.load(j['head_path'], weights_only=False, map_location='cpu'); m.eval()
    plo, phi = getattr(m, 'pred_lo', 0.0), getattr(m, 'pred_hi', 1.0)
    rng, clamp, dist, trough, trng = [], [], [], [], []
    with torch.no_grad():
        for sb in batches:
            for t in range(48, sb.shape[1] - 48 + 1, 24):
                yp = m(sb[:, t - 48:t, :]).numpy()
                yt = sb[:, t:t + 48, 0].numpy() * span + lo
                clamp += list(((np.abs(yp - plo) < 1e-6) | (np.abs(yp - phi) < 1e-6)).mean(1))
                yr = yp * span + lo
                rng += list(yr.max(1) - yr.min(1))
                dist += [len(np.unique(np.round(r, 2))) for r in yr]
                s = np.sort(yr, 1); trough += list(s[:, 5] - s[:, 0])
                trng += list(yt.max(1) - yt.min(1))
    to = 100 * j['mip_timeouts'] / j['mip_solves']
    print(f'{name:22s} {to:5.1f}% {np.median(rng):8.1f} {100*np.mean(clamp):8.1f}% '
          f'{np.median(dist):9.0f} {np.median(trough):8.1f} {np.median(trng):11.1f}   n={len(rng)}')
print('SHAPE_DONE')
