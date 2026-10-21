"""Regret against the Gurobi work limit.

Seed-1 heads of trend-aim (a72_nogenwide) and 3D-Learning (a71_a1wide), wide
range, renewables 2023-2025, the FIRST N windows in scorer order (k = 0..N-1,
batch-major, then origin, then sequence -- the order of regret_per_window).
MIP_WORK_LIMIT (required) sets one limit per job. Writes
results/scores/worklimit/wl_<MIP_WORK_LIMIT>.json.

The optimal cost is re-solved at the SAME limit as the forecasts, as the scorer
does. Every solve's Gurobi status is recorded: 2 = proven optimal, 16 = stopped
on the work limit ("timeout" in the tables), 9 = stopped on the wall-clock
safety net (non-deterministic; any 9 makes that limit's row VOID).

Self-check: at MIP_WORK_LIMIT=230 the recomputed per-window regrets must match
the scored files' first N entries (the scored contract is work 230).
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import poc_eval_ood as P
from Utils.datasets import Data_Loader, build_ml_features

ROOT = P.ROOT
EVAL = f'{ROOT}/results/scores/poc_eval_measure'
ARMS = {'trend-aim': 'a72_nogenwide', '3D-Learning': 'a71_a1wide'}
SEED = 1
N = int(os.environ.get('WL_N_WINDOWS', '128'))
REGIME = 'oodtest_2023_2025_renewables'
WL = os.environ['MIP_WORK_LIMIT']
W = PL = 48
STRIDE = int(os.environ.get('EVAL_ORIGIN_STRIDE', '24'))
OUT = f'{ROOT}/results/scores/worklimit/wl_{WL}.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

task = P.build_task()
statuses = []
_orig_solve = task._solve


def _rec_solve(price_vec, relax=True):
    r = _orig_solve(price_vec, relax=relax)
    if not relax:
        statuses.append(int(r.get('status', -99)))
    return r


task._solve = _rec_solve
price_b, demand_b = P.train_fold_bounds()
lo, hi = price_b
norm_full = {'Price': price_b, 'TOTALDEMAND_MW': demand_b}
csv = dict(P.OOD_REGIMES)[REGIME]

heads, jsons = {}, {}
for lab, arm in ARMS.items():
    j = json.load(open(f'{EVAL}/{arm}_s{SEED}_{REGIME}_selected.json'))
    jsons[lab] = j
    m = torch.load(j['head_path'], weights_only=False, map_location='cpu'); m.eval()
    heads[lab] = m
F = P.detect_F(jsons['trend-aim']['head_path'])
cols = P.feature_cols_for(F or 8)
raw = pd.read_csv(csv, parse_dates=['TIMESTAMP'])
feat = build_ml_features(raw)[cols]
nb = {k: v for k, v in norm_full.items() if k in cols}
loader = Data_Loader(df=feat).ml_set2loader_mv(
    split_test=False, normlized=True, device='cpu',
    batch_size=P.BATCH_SIZE, pic_size=P.ML_SEQ_LEN, norm_bounds=nb, drop_last=True)

y_true, y_pred = [], {lab: [] for lab in ARMS}
with torch.no_grad():
    for batch in loader:
        sb = batch[0]
        if sb.dim() == 2:
            sb = sb.unsqueeze(-1)
        for t in range(W, sb.shape[1] - PL + 1, STRIDE):
            inp, tgt = sb[:, t - W:t, :], sb[:, t:t + PL, 0]
            y_true.append(tgt * (hi - lo) + lo)
            for lab, m in heads.items():
                y_pred[lab].append(m(inp) * (hi - lo) + lo)
y_true = torch.cat(y_true).float()
y_pred = {k: torch.cat(v).float() for k, v in y_pred.items()}
assert y_true.shape[0] == len(jsons['trend-aim']['regret_per_window']), 'window count drift'
yt = y_true[:N]
print(f'  | work limit {WL} | {N} of {y_true.shape[0]} windows | bounds {lo:.2f}..{hi:.2f}', flush=True)

res = {'work_limit': float(WL), 'n_windows': N}
statuses.clear()
x_opt, _ = task.find_best_scheduling_mip(yt)
c_opt = task.loss(x_opt, yt).detach().numpy().astype(float)
res['opt'] = {'status': list(statuses), 'cost': c_opt.tolist()}
for lab in ARMS:
    statuses.clear()
    x, _ = task.find_best_scheduling_mip(y_pred[lab][:N])
    c = task.loss(x, yt).detach().numpy().astype(float)
    reg = c - c_opt
    st = list(statuses)
    res[lab] = {'status': st, 'regret': reg.tolist(),
                'norm_regret': float(reg.sum() / np.abs(c_opt).sum()),
                'timeout_share': float(np.mean([s == 16 for s in st])),
                'safety_hits': int(sum(s == 9 for s in st))}
    print(f'  | {lab:12s} norm regret {res[lab]["norm_regret"]:.4f}  '
          f'work-limit stops {100 * res[lab]["timeout_share"]:.1f}%  '
          f'safety-net stops {res[lab]["safety_hits"]}', flush=True)
st = res['opt']['status']
print(f'  | optimal      work-limit stops {100 * np.mean([s == 16 for s in st]):.1f}%  '
      f'safety-net stops {sum(s == 9 for s in st)}', flush=True)
if float(WL) == 230.0:
    for lab in ARMS:
        ref = np.asarray(jsons[lab]['regret_per_window'][:N], float)
        d = np.abs(np.asarray(res[lab]['regret']) - ref).max()
        print(f'  | CHECK {lab}: max |recomputed - scored| over {N} windows = ${d:,.2f} '
              + ('OK' if d <= 50 else '<-- MISMATCH'), flush=True)
json.dump(res, open(OUT + '.tmp', 'w'))
os.replace(OUT + '.tmp', OUT)
print(f'  wrote {OUT}\nWORKLIMIT_DONE')
