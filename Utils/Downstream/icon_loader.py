"""
Loader for ICON Challenge 2014 instance files (Simonis et al., 2014).

Format of `instance.txt` (whitespace-separated):
    L1                : period_minutes (int — granularity, typically 5 → T=288/day)
    L2                : n_resources
    L3                : n_machines
    [per machine, 2 lines]:
        line A        : machine_id  idle_power  startup_cost  shutdown_cost
        line B        : cap_r1 cap_r2 ... cap_rR        (R = n_resources)
    Lline_after       : n_tasks
    [per task, 2 lines]:
        line A        : task_id  duration  earliest_start  latest_end  power
        line B        : req_r1 req_r2 ... req_rR

Returns a dict that `Energy_Scheduling.from_icon_instance` consumes directly.

The accompanying `forecast.txt` (288 step prices) is intentionally ignored —
we use AEMO spot prices instead.
"""

import os
import numpy as np


def _read_tokens(path: str):
    """Yield whitespace-separated tokens lazily, skipping blanks."""
    with open(path, 'r') as f:
        for line in f:
            for tok in line.strip().split():
                yield tok


def _next_ints(it, n):
    return [int(next(it)) for _ in range(n)]


def _next_floats(it, n):
    return [float(next(it)) for _ in range(n)]


def load_icon_instance(path: str) -> dict:
    """
    Parse an ICON Challenge instance.txt file.

    Args:
        path : path to instance.txt
    Returns:
        dict with keys
          period_minutes, T, n_resources, n_machines, n_tasks,
          machine_idle, machine_startup_cost, machine_shutdown_cost,
          machine_capacity        (n_machines, n_resources),
          task_duration, task_earliest_start, task_latest_end, task_power,
          task_resources          (n_tasks, n_resources),
          source_path
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'ICON instance not found: {path}')

    it = _read_tokens(path)

    period_minutes = int(next(it))
    n_resources    = int(next(it))
    n_machines     = int(next(it))

    machine_idle           = np.zeros(n_machines, dtype=np.float64)
    machine_startup_cost   = np.zeros(n_machines, dtype=np.float64)
    machine_shutdown_cost  = np.zeros(n_machines, dtype=np.float64)
    machine_capacity       = np.zeros((n_machines, n_resources), dtype=np.float64)

    for _ in range(n_machines):
        mid    = int(next(it))
        idle   = float(next(it))
        c_up   = float(next(it))
        c_dn   = float(next(it))
        if not (0 <= mid < n_machines):
            raise ValueError(f'machine id {mid} out of range [0,{n_machines})')
        machine_idle[mid]          = idle
        machine_startup_cost[mid]  = c_up
        machine_shutdown_cost[mid] = c_dn
        machine_capacity[mid]      = _next_floats(it, n_resources)

    n_tasks = int(next(it))

    task_duration        = np.zeros(n_tasks, dtype=np.int64)
    task_earliest_start  = np.zeros(n_tasks, dtype=np.int64)
    task_latest_end      = np.zeros(n_tasks, dtype=np.int64)
    task_power           = np.zeros(n_tasks, dtype=np.float64)
    task_resources       = np.zeros((n_tasks, n_resources), dtype=np.float64)

    for _ in range(n_tasks):
        tid  = int(next(it))
        dur  = int(next(it))
        e    = int(next(it))
        l    = int(next(it))
        pwr  = float(next(it))
        if not (0 <= tid < n_tasks):
            raise ValueError(f'task id {tid} out of range [0,{n_tasks})')
        task_duration[tid]       = dur
        task_earliest_start[tid] = e
        task_latest_end[tid]     = l
        task_power[tid]          = pwr
        task_resources[tid]      = _next_floats(it, n_resources)

    # Time horizon for the ICON grid (one scheduling day).
    minutes_per_day = 24 * 60
    if minutes_per_day % period_minutes != 0:
        raise ValueError(
            f'period_minutes={period_minutes} does not divide a 24h day evenly')
    T = minutes_per_day // period_minutes

    # Validate task feasibility on the ICON grid: e + d <= l and l <= T.
    for j in range(n_tasks):
        e, d, l = task_earliest_start[j], task_duration[j], task_latest_end[j]
        if e + d > l:
            raise ValueError(
                f'task {j}: earliest_start({e}) + duration({d}) > latest_end({l})')
        if l > T:
            # Some ICON instances mark `latest_end` up to T inclusive; clamp.
            task_latest_end[j] = T

    return {
        'period_minutes':         period_minutes,
        'T':                      T,
        'n_resources':            n_resources,
        'n_machines':             n_machines,
        'n_tasks':                n_tasks,
        'machine_idle':           machine_idle,
        'machine_startup_cost':   machine_startup_cost,
        'machine_shutdown_cost':  machine_shutdown_cost,
        'machine_capacity':       machine_capacity,
        'task_duration':          task_duration,
        'task_earliest_start':    task_earliest_start,
        'task_latest_end':        task_latest_end,
        'task_power':             task_power,
        'task_resources':         task_resources,
        'source_path':            os.path.abspath(path),
    }


def aggregate_icon_to_period(inst: dict, target_period_minutes: int) -> dict:
    """
    Coarsen an ICON instance's time grid to ``target_period_minutes``.

    Aggregation rule (factor = target / source, must be integer ≥ 1):
        T_new           = T  // factor
        duration_new    = ceil(duration / factor)        — round up (preserve runtime)
        earliest_new    = floor(earliest / factor)       — widen window start
        latest_new      = ceil(latest / factor)          — widen window end (clamp ≤ T_new)
        power           = unchanged (MW is rate-based)
        resources / cap = unchanged (per-period, instantaneous)
        idle / startup  = unchanged (per-event / per-period)

    The objective sum_t pj·q·price_t still gives the same energy cost since
    pj·q·factor (the new contribution of one aggregated period) cancels with
    duration shrinking by the same factor.
    """
    src_period = int(inst['period_minutes'])
    if target_period_minutes < src_period:
        raise ValueError(
            f"can't aggregate {src_period}min → {target_period_minutes}min "
            f"(target finer than source)")
    if target_period_minutes % src_period != 0:
        raise ValueError(
            f"target_period_minutes ({target_period_minutes}) must be a multiple "
            f"of source period ({src_period})")
    factor = target_period_minutes // src_period
    if factor == 1:
        return inst

    src_T = int(inst['T'])
    if src_T % factor != 0:
        raise ValueError(
            f'aggregation factor {factor} does not divide T={src_T} evenly')
    new_T = src_T // factor

    d = np.asarray(inst['task_duration'], dtype=np.int64)
    e = np.asarray(inst['task_earliest_start'], dtype=np.int64)
    l = np.asarray(inst['task_latest_end'], dtype=np.int64)

    new_duration = np.ceil(d / factor).astype(np.int64)
    new_earliest = (e // factor).astype(np.int64)
    new_latest   = np.ceil(l / factor).astype(np.int64)
    new_latest   = np.minimum(new_latest, new_T)
    # Ensure window can fit duration after aggregation; if not, clip duration.
    for j in range(len(new_duration)):
        if new_earliest[j] + new_duration[j] > new_latest[j]:
            new_duration[j] = max(1, new_latest[j] - new_earliest[j])

    out = dict(inst)
    out.update({
        'period_minutes':      target_period_minutes,
        'T':                   new_T,
        'task_duration':       new_duration,
        'task_earliest_start': new_earliest,
        'task_latest_end':     new_latest,
    })
    return out


def summarise_instance(inst: dict) -> str:
    """One-paragraph summary suitable for log printing."""
    n_j, n_m, n_r, T = inst['n_tasks'], inst['n_machines'], inst['n_resources'], inst['T']
    d = inst['task_duration']
    e = inst['task_earliest_start']
    l = inst['task_latest_end']
    win = (l - e).astype(float)

    # Rough capacity utilisation if all tasks ran simultaneously somewhere
    total_req = inst['task_resources'].sum(axis=0)        # (R,)
    total_cap = inst['machine_capacity'].sum(axis=0)      # (R,)
    util = total_req / np.maximum(total_cap, 1e-9)

    return (
        f'ICON instance: {os.path.basename(os.path.dirname(inst["source_path"]))}  '
        f'tasks={n_j}  machines={n_m}  resources={n_r}  T={T}  '
        f'period={inst["period_minutes"]}min\n'
        f'  duration: min={int(d.min())} mean={d.mean():.1f} max={int(d.max())}\n'
        f'  window:   min={int(win.min())} mean={win.mean():.1f} max={int(win.max())}\n'
        f'  resource pressure (sum_req / sum_cap): '
        + '  '.join(f'r{i}={u:.2f}' for i, u in enumerate(util))
    )
