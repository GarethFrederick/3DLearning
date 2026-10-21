"""Measure Gurobi work units per evaluation MIP solve, so a WorkLimit can be
chosen rather than guessed.

A wall-clock TimeLimit makes the incumbent depend on thread timing; WorkLimit
is deterministic. Work units are NOT seconds, so measure them on this instance
and data, through the eval's own code path, under the CURRENT time limit.

Needs LOG_WORK on and MIP_WORK_LIMIT unset. Trains nothing. Writes only the
summary JSON at CAL_OUT (default results/scores/worklimit/
worklimit_calibration.json).
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

# Use the eval driver's own task builder: a differently built task would
# calibrate the wrong solver.
from poc_eval_ood import build_task, PREDICT_LEN          # noqa: E402
from Utils.datasets import build_ml_features              # noqa: E402

N_DAYS = int(os.environ.get('CAL_DAYS', '40'))
CSV = os.environ.get('CAL_CSV', f'{ROOT}/Data/AEMO_VIC1_oodtest_2023_2025.csv')
OUT = os.environ.get('CAL_OUT', f'{ROOT}/results/scores/worklimit/worklimit_calibration.json')


def main():
    print('=== WORKLIMIT CALIBRATION ===', flush=True)
    print(f'  | instance     : {os.environ.get("ICON_INSTANCE")}', flush=True)
    print(f'  | price csv    : {CSV}', flush=True)
    print(f'  | days         : {N_DAYS}', flush=True)
    print(f'  | MIP_TL       : {os.environ.get("MIP_TL")}', flush=True)
    print(f'  | MIP_FOCUS    : {os.environ.get("MIP_FOCUS")}', flush=True)
    print(f'  | LOG_WORK     : {os.environ.get("LOG_WORK")}', flush=True)
    if os.environ.get('LOG_WORK') in (None, '', '0'):
        print('CAL_GATE_FAIL: LOG_WORK is off, so no work would be recorded.')
        return 4
    if os.environ.get('MIP_WORK_LIMIT') not in (None, ''):
        print('CAL_GATE_FAIL: MIP_WORK_LIMIT is set. Calibration must run under '
              'the CURRENT time limit, or it measures the new ruler, not the old.')
        return 4

    task = build_task()
    print(f'  | MIP_TIMELIMIT= {task.MIP_TIMELIMIT}s  T={task.T} '
          f'T_external={task.T_external}', flush=True)

    df = pd.read_csv(CSV, parse_dates=['TIMESTAMP'])
    price = build_ml_features(df)['Price'].to_numpy(dtype=float)
    n_full = len(price) // PREDICT_LEN
    if n_full < N_DAYS:
        print(f'CAL_GATE_FAIL: only {n_full} whole days in {CSV}, need {N_DAYS}')
        return 4
    # Even stride across the file so the sample spans the whole period rather
    # than one season. Deterministic: no RNG, so a rerun measures the same days.
    stride = max(1, n_full // N_DAYS)
    day_idx = [i * stride for i in range(N_DAYS)]

    rows = []
    t0 = time.time()
    for n, di in enumerate(day_idx, 1):
        y48 = price[di * PREDICT_LEN:(di + 1) * PREDICT_LEN]
        y288 = task._upsample(torch.tensor(y48, dtype=torch.float32).unsqueeze(0))
        y288 = y288.squeeze(0).detach().cpu().numpy()
        t_solve = time.time()
        task._solve(y288, relax=False)
        rows.append({
            'day': int(di),
            'work': task._last_work,
            'status': task._last_status,
            'runtime_s': task._last_runtime,
            'solcount': task._last_solcount,
            'mipgap': task._last_gap,
            'wall_s': round(time.time() - t_solve, 2),
        })
        r = rows[-1]
        print(f'  [{n:3d}/{N_DAYS}] day={di:5d} work={r["work"]} '
              f'status={r["status"]} runtime={r["runtime_s"]} '
              f'sols={r["solcount"]} gap={r["mipgap"]}', flush=True)

    works = [r['work'] for r in rows if r['work'] is not None]
    # Gurobi status codes: 2 OPTIMAL, 9 TIME_LIMIT, 16 WORK_LIMIT.
    n_tl = sum(1 for r in rows if r['status'] == 9)
    n_opt = sum(1 for r in rows if r['status'] == 2)
    n_nosol = sum(1 for r in rows if (r['solcount'] or 0) == 0)

    summary = {
        'csv': CSV, 'days': N_DAYS,
        'mip_timelimit_s': task.MIP_TIMELIMIT,
        'n_optimal': n_opt, 'n_timelimit': n_tl,
        'frac_timelimit': round(n_tl / len(rows), 4),
        'n_no_incumbent': n_nosol,
        'work_min': min(works) if works else None,
        'work_p50': float(np.percentile(works, 50)) if works else None,
        'work_p90': float(np.percentile(works, 90)) if works else None,
        'work_p99': float(np.percentile(works, 99)) if works else None,
        'work_max': max(works) if works else None,
        'work_at_timelimit_min': min([r['work'] for r in rows if r['status'] == 9],
                                     default=None),
        'total_wall_s': round(time.time() - t0, 1),
        'rows': rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    os.replace(OUT + '.tmp', OUT)

    print('\n=== SUMMARY ===', flush=True)
    for k in ('n_optimal', 'n_timelimit', 'frac_timelimit', 'n_no_incumbent',
              'work_min', 'work_p50', 'work_p90', 'work_p99', 'work_max',
              'work_at_timelimit_min', 'total_wall_s'):
        print(f'  {k:24s} {summary[k]}', flush=True)
    print(f'\nwrote {OUT}', flush=True)
    print('CALIBRATION_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
