"""Schedule comparison for the thesis: where each model puts the load.

Seed-1 heads of trend-aim (a72_nogenwide) and 3D-Learning (a71_a1wide), on
the SAME windows the eval scored. Part A (LP over all windows) is removed; the
JSON keeps 'A': null.

Part B: one window, full MIP under the measurement contract (work 230, reset,
quality, 8 threads). The window is the one whose (3D-Learning regret -
trend-aim regret) is the era MEDIAN, so it is neither best nor worst case. It
reports per-block energy, true-price cost and regret, and checks the regrets
against the scored file's entry for that window; a mismatch means the window
was not reproduced.

Task, loader, window order and denormalisation mirror poc_eval_ood.eval_one and
ml.test_task (window k = batch-major, then origin, then sequence).
Writes results/analysis/schedule_table.json (or SCHED_OUT).
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
W = PL = 48
STRIDE = int(os.environ.get('EVAL_ORIGIN_STRIDE', '24'))
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
OUT = f'{ROOT}/results/analysis/schedule_table.json'
# All three unset = both eras, default OUT, no extra keys. SCHED_REGIME limits
# the loop to one era; SCHED_DUMP48=1 adds per-half-hour arrays. Either one
# needs SCHED_OUT, so a partial run never overwrites the stored table JSON.
_ONLY = os.environ.get('SCHED_REGIME', '')
REGIMES = [r for r in P.OOD_REGIMES if not _ONLY or r[0] == _ONLY]
assert REGIMES, f'SCHED_REGIME={_ONLY!r} not in {[r[0] for r in P.OOD_REGIMES]}'
DUMP48 = os.environ.get('SCHED_DUMP48', '0')
assert DUMP48 in ('0', '1'), f'SCHED_DUMP48={DUMP48!r}'
DUMP48 = DUMP48 == '1'
if (_ONLY or DUMP48) and not os.environ.get('SCHED_OUT'):
    raise SystemExit('SCHED_REGIME/SCHED_DUMP48 set without SCHED_OUT: refusing to overwrite '
                     + OUT)
OUT = os.environ.get('SCHED_OUT', OUT)

task = P.build_task()
T, UP = task.T, task.upsample_factor
assert task.T_external == 48 and T == 48 * UP, (task.T_external, T, UP)
price_b, demand_b = P.train_fold_bounds()
lo, hi = price_b
norm_full = {'Price': price_b, 'TOTALDEMAND_MW': demand_b}
denorm = np.array([lo, hi])
print(f'  | task T={T} upsample={UP} work={os.environ.get("MIP_WORK_LIMIT")} '
      f'reset={os.environ.get("MIP_RESET")} stride={STRIDE} | bounds {lo:.2f}..{hi:.2f}', flush=True)


def block_of_slots48(v48):
    """(..., 48) external slots -> (..., 6) four-hour blocks by summation."""
    return v48.reshape(*v48.shape[:-1], 6, 8).sum(-1)


def energy48(x):
    """x (B, T+1) from the task -> energy per external slot (B, 48)."""
    w = x[:, :T].reshape(x.shape[0], 48, UP).sum(-1)
    return w


results = {}
for regime, csv in REGIMES:
    print(f'\n===== {regime} =====', flush=True)
    heads, jsons = {}, {}
    for lab, arm in ARMS.items():
        j = json.load(open(f'{EVAL}/{arm}_s{SEED}_{regime}_selected.json'))
        jsons[lab] = j
        m = torch.load(j['head_path'], weights_only=False, map_location='cpu'); m.eval()
        heads[lab] = m
        print(f'  | {lab:12s} head={os.path.basename(j["head_path"])} '
              f'norm_regret={j["norm_regret"]:.4f} n_win={j["n_windows"]}', flush=True)
    F = P.detect_F(jsons['trend-aim']['head_path'])
    cols = P.feature_cols_for(F or 8)
    raw = pd.read_csv(csv, parse_dates=['TIMESTAMP'])
    feat = build_ml_features(raw)[cols]
    assert len(feat) == len(raw), (len(feat), len(raw))
    nb = {k: v for k, v in norm_full.items() if k in cols}
    loader = Data_Loader(df=feat).ml_set2loader_mv(
        split_test=False, normlized=True, device='cpu',
        batch_size=P.BATCH_SIZE, pic_size=P.ML_SEQ_LEN, norm_bounds=nb, drop_last=True)

    # Windows in scorer order.
    y_true, y_pred = [], {lab: [] for lab in ARMS}
    meta = []                                   # (batch i, origin t, seq b) per window
    with torch.no_grad():
        for i, batch in enumerate(loader):
            sb = batch[0]
            if sb.dim() == 2:
                sb = sb.unsqueeze(-1)
            for t in range(W, sb.shape[1] - PL + 1, STRIDE):
                inp, tgt = sb[:, t - W:t, :], sb[:, t:t + PL, 0]
                y_true.append(tgt * (hi - lo) + lo)
                for lab, m in heads.items():
                    y_pred[lab].append(m(inp) * (hi - lo) + lo)
                meta += [(i, t, b) for b in range(sb.shape[0])]
    y_true = torch.cat(y_true).float()
    y_pred = {k: torch.cat(v).float() for k, v in y_pred.items()}
    n = y_true.shape[0]
    reg72 = np.asarray(jsons['trend-aim']['regret_per_window'], float)
    reg71 = np.asarray(jsons['3D-Learning']['regret_per_window'], float)
    assert n == len(reg72) == len(reg71), (n, len(reg72), len(reg71))
    print(f'  | {n} windows rebuilt, matches the scored files', flush=True)

    # Part A removed: at the native T=288 grid the batched LP fell back to greedy
    # on most gas windows. The table uses Part B (full MIP).
    A = None

    # Part B: the median-gap window, full MIP.
    gap = reg71 - reg72
    k = int(np.argsort(np.abs(gap - np.median(gap)))[0])
    i, t, b = meta[k]
    row0 = (i * P.BATCH_SIZE + b) * P.ML_SEQ_LEN + t
    start_ts = str(raw['TIMESTAMP'].iloc[row0])
    print(f'  | B: median-gap window k={k} (batch {i}, origin {t}, seq {b}) '
          f'starts {start_ts}; scored regrets trend-aim ${reg72[k]:,.0f} 3DL ${reg71[k]:,.0f} '
          f'(median gap ${np.median(gap):,.0f})', flush=True)
    Bres = {'window_index': k, 'start': start_ts, 'scored_regret': {'trend-aim': float(reg72[k]),
            '3D-Learning': float(reg71[k])}, 'median_gap': float(np.median(gap))}
    yt = y_true[k:k + 1]
    mip = {}
    for lab, y in [('optimal', yt)] + [(lab, y_pred[lab][k:k + 1]) for lab in ARMS]:
        x, _ = task.find_best_scheduling_mip(y)
        cost = float(task.loss(x, yt).item())
        e48 = energy48(x.detach().cpu().numpy())[0]
        mip[lab] = {'cost_true': cost, 'energy_block': block_of_slots48(e48).tolist(),
                    'energy_total': float(e48.sum())}
        if DUMP48:
            mip[lab]['energy48'] = e48.tolist()
    for lab in ARMS:
        mip[lab]['regret'] = mip[lab]['cost_true'] - mip['optimal']['cost_true']
    Bres['true_price_block'] = (block_of_slots48(yt.numpy())[0] / 8.0).tolist()
    Bres['forecast_block'] = {lab: (block_of_slots48(y_pred[lab][k:k + 1].numpy())[0] / 8.0).tolist()
                              for lab in ARMS}
    Bres['mip'] = mip
    if DUMP48:
        Bres['true_price48'] = yt.numpy()[0].tolist()
        Bres['forecast48'] = {lab: y_pred[lab][k].numpy().tolist() for lab in ARMS}
        # The start stamp comes from row arithmetic, not from the loader. If the
        # arithmetic is off, the CSV rows from row0 will not be this window's prices.
        csv48 = raw['Price'].iloc[row0:row0 + PL].to_numpy(float)
        Bres['csv_stamps48'] = [str(s) for s in raw['TIMESTAMP'].iloc[row0:row0 + PL]]
        dmax = float(np.abs(csv48 - yt.numpy()[0]).max())
        print(f'  | DUMP48 CHECK: window true prices vs CSV rows from {start_ts}: '
              f'max |diff| ${dmax:.4f}  ' + ('OK' if dmax < 0.01 else '<-- MISMATCH'), flush=True)
    print(f'  | B (MIP): cost under true prices  optimal ${mip["optimal"]["cost_true"]:,.0f}  '
          + '  '.join(f'{lab} ${mip[lab]["cost_true"]:,.0f} (regret ${mip[lab]["regret"]:,.0f}; '
                      f'scored ${Bres["scored_regret"][lab]:,.0f})' for lab in ARMS), flush=True)
    print('    block   true$  ' + '  '.join(f'{k2+" fc$":>14s}' for k2 in ARMS)
          + '   ' + '  '.join(f'{k2+" MWh":>14s}' for k2 in ['optimal'] + list(ARMS)))
    for bi, (h0, h1) in enumerate(BLOCKS):
        print(f'    {h0:02d}-{h1:02d}h {Bres["true_price_block"][bi]:7.2f}  '
              + '  '.join(f'{Bres["forecast_block"][k2][bi]:14.2f}' for k2 in ARMS)
              + '   ' + '  '.join(f'{mip[k2]["energy_block"][bi]:14.2f}' for k2 in ['optimal'] + list(ARMS)))
    for lab in ARMS:
        d = abs(mip[lab]['regret'] - Bres['scored_regret'][lab])
        print(f'  | CHECK {lab}: recomputed regret ${mip[lab]["regret"]:,.0f} vs scored '
              f'${Bres["scored_regret"][lab]:,.0f}  diff ${d:,.0f}  '
              + ('OK' if d <= 0.01 * max(abs(Bres['scored_regret'][lab]), 1000) + 50 else '<-- MISMATCH'))
    results[regime] = {'A': A, 'B': Bres}

json.dump(results, open(OUT, 'w'), indent=1)
print(f'\n  wrote {OUT}')
print('SCHEDULE_TABLE_DONE')
