import os
import threading
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

_GUROBI_AVAILABLE = False
try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    pass

# The pip gurobipy "Restricted license" caps models at 2,000 vars x 2,000
# constraints. The first solve that hits the cap flips the instance to the
# scipy (LP) / greedy (MIP) fallback for the rest of the process.

# Lock for one-time scipy LP-static build per Energy_Scheduling instance.
_LP_STATIC_BUILD_LOCK = threading.Lock()

# Core budget = LP_WORKERS (thread-pool workers) x LP_THREADS_PER_SOLVE
# (Gurobi Threads). Without Threads, Gurobi takes ALL cores per solve and the
# pool oversubscribes the CPU; a starved MIP hits TIME_LIMIT with no incumbent
# and falls to greedy. sched_getaffinity respects SLURM's CPU mask.
_LP_TOTAL_CORES = (
    len(os.sched_getaffinity(0))
    if hasattr(os, 'sched_getaffinity')
    else (os.cpu_count() or 4)
)
# Default 2 threads per solve, rest as workers (16 cores -> 8 x 2), tuned for
# the aggregated T=48 LP. The native T=288 MIP wants more threads per solve;
# set both env vars explicitly.
_LP_THREADS_PER_SOLVE = int(os.environ.get(
    'LP_THREADS_PER_SOLVE',
    max(1, min(2, _LP_TOTAL_CORES)),
))
_LP_WORKERS = int(os.environ.get(
    'LP_WORKERS',
    max(1, _LP_TOTAL_CORES // _LP_THREADS_PER_SOLVE),
))


def cache_solve_per_row(price_np, cache, solve_fn, greedy_count_fn=None):
    """Per-row schedule cache for FIXED-row price arrays solved repeatedly
    (e.g. spo_plus_loss's w*(y_true)). Keyed per row, so it hits however the
    batches are composed, and bounded by the number of distinct rows.

    Greedy guard: if the greedy counter rises during the miss-solve, the
    results serve THIS call but are NOT cached. Cache hits bypass _solve_batch,
    so a cached greedy schedule would be served, uncounted, forever.

    Args:
        price_np       : (B, T) fixed-row prices to solve.
        cache          : dict {row.tobytes(): w_row} persisted on the task.
        solve_fn       : callable(arr)->w_array, e.g. lambda a: _solve_batch(a)[0].
        greedy_count_fn: callable()->int returning the task's cumulative greedy
                         solve count, or None to skip the guard.
    Returns:
        (B, T) w array in the original row order.
    """
    row_keys = [price_np[r].tobytes() for r in range(price_np.shape[0])]
    miss_idx = [r for r, k in enumerate(row_keys) if k not in cache]
    out = [cache.get(k) for k in row_keys]
    if miss_idx:
        _g0 = greedy_count_fn() if greedy_count_fn is not None else None
        miss_w = solve_fn(price_np[miss_idx])
        _contaminated = (greedy_count_fn is not None and greedy_count_fn() != _g0)
        for _j, _r in enumerate(miss_idx):
            out[_r] = miss_w[_j]
            if not _contaminated:           # never cache a greedy result
                cache[row_keys[_r]] = miss_w[_j]
    return np.stack(out)


class Energy_Scheduling:
    """
    ICON Challenge 2014 energy-cost-aware scheduling (Simonis et al., 2014).

    Full multi-period formulation over one day of T periods (48 synthetic,
    288 native ICON). Decision variables per Simonis et al.:
        x_jmt ∈ {0,1}  — task j starts on machine m at time t
        v_mt  ∈ {0,1}  — machine m is on at time t
        y_mt  ∈ {0,1}  — machine m starts up at time t
        z_mt  ∈ {0,1}  — machine m shuts down at time t

    Objective (minimise): C = C_J + C_M + C_up + C_down
        C_J    = task energy cost      (linear in actual prices r_t)
        C_M    = machine idle cost     (linear in actual prices r_t)
        C_up   = machine startup cost  (price-independent)
        C_down = machine shutdown cost (price-independent)

    The schedule is encoded as x of shape (batch, T+1):
        x[:, :T]  — energy profile w_t (MWh consumed at each period, price-sensitive)
        x[:, T]   — fixed cost K = C_up + C_down (AUD, price-independent)

    loss(x, y) = sum_t(r_t * w_t) + K   [total actual cost in AUD; negative
    when prices are negative, because load is paid]

    Training uses the LP relaxation; scoring uses find_best_scheduling_mip.
    """

    def __init__(self, price_range, demand_stats=None, n_tasks=10,
                 n_machines=3, T=48, seed=None,
                 lp_timelimit=10.0, mip_timelimit=30.0,
                 period_minutes=30, T_external=None, n_resources=3,
                 _icon_params=None, **kwargs):
        """
        Args:
            price_range    : [min_price, max_price] AUD/MWh (after clipping)
            demand_stats   : dict with keys 'mean_mwh', 'price_p25', 'price_p75'
                             (required for synthetic mode; ignored if _icon_params)
            n_tasks        : number of optional tasks  |J|
            n_machines     : number of machines        |M|
            T              : number of LP time periods (synthetic: 48 for 30-min;
                             ICON: 288 for 5-min). This is the INTERNAL grid.
            seed           : int for reproducibility; None = random (printed to log)
            lp_timelimit   : Gurobi time limit (s) for LP relaxation (SPO+ training)
            mip_timelimit  : Gurobi time limit (s) for full MIP (evaluation)
            period_minutes : LP period length in minutes (drives q_h = period/60)
            T_external     : external grid (LSTM/AEMO predictions). Defaults to T.
                             When T_external < T, prices fed to loss/find_best/SPO+
                             are upsampled by repeat_interleave(T // T_external).
                             Must divide T evenly.
            n_resources    : number of resource types (3 for synthetic; from instance for ICON)
            _icon_params   : pre-loaded ICON instance dict (from
                             icon_loader.load_icon_instance). Bypasses synthesis.
                             Use Energy_Scheduling.from_icon_instance(...) instead
                             of passing this directly.
        """
        self.PRICE_MIN     = float(price_range[0])
        self.PRICE_MAX     = float(price_range[1])
        self.T             = int(T)
        self.T_external    = int(T_external) if T_external is not None else int(T)
        if self.T % self.T_external != 0:
            raise ValueError(
                f'T ({self.T}) must be a multiple of T_external ({self.T_external})')
        self.upsample_factor = self.T // self.T_external
        self.q_h           = float(period_minutes) / 60.0
        self.LP_TIMELIMIT  = lp_timelimit
        self.MIP_TIMELIMIT = mip_timelimit
        # Relative MIP gap at which a solve stops; the native T=288 MIP cannot
        # close to Gurobi's 1e-4 default in reasonable time.
        self.MIP_GAP       = float(os.environ.get('MIP_GAP', '0.01'))

        # Deterministic stopping. A wall-clock TimeLimit makes the incumbent
        # depend on thread timing; WorkLimit counts deterministic work units.
        # It repeats only on ONE CPU family: score on AMD Milan
        # (sbatch --constraint=milan). Both unset = TimeLimit only.
        _mwl = os.environ.get('MIP_WORK_LIMIT')
        _lwl = os.environ.get('LP_WORK_LIMIT')
        self.MIP_WORKLIMIT = float(_mwl) if _mwl not in (None, '') else None
        self.LP_WORKLIMIT  = float(_lwl) if _lwl not in (None, '') else None
        # Wall-clock safety net under a work limit (this multiple of the time
        # limit). If it binds, the solve is NOT deterministic: void its row. A
        # TIME_LIMIT status while a work limit is set means it fired.
        self.WORK_TL_SAFETY = float(os.environ.get('WORK_TL_SAFETY', '20.0'))

        # A cached Gurobi model keeps solution state across optimize() calls, so
        # a day's answer can depend on the day that thread solved before.
        # MIP_RESET=1 clears it; REQUIRED for a work limit to repeat.
        self.MIP_RESET = os.environ.get('MIP_RESET') not in (None, '', '0')
        # Cuts and presolve; deterministic given fixed threads.
        self.MIP_QUALITY = os.environ.get('MIP_QUALITY') not in (None, '', '0')
        # See the cache lookup in _solve.
        self.MIP_FRESH_MODEL = os.environ.get('MIP_FRESH_MODEL') not in (None, '', '0')
        # The [MEASURE] banner prints at the FIRST solve, not here: callers
        # change MIP_TIMELIMIT after construction (poc_eval_ood applies MIP_TL).
        self._measure_banner_done = False
        # Set after each optimize() when LOG_WORK is on.
        self._last_work = None
        self._last_status = None
        self._last_runtime = None
        self._last_solcount = None
        self._last_gap = None

        # Thread-local Gurobi models, slots 'lp' and 'mip'. Only the objective
        # depends on price, so each model is built once per worker thread.
        self._lp_thread_cache = threading.local()

        self.seed = int(np.random.randint(0, 100_000)) if seed is None else int(seed)

        if _icon_params is not None:
            self._load_icon_params(_icon_params)
            mode = f'ICON({os.path.basename(os.path.dirname(_icon_params["source_path"]))})'
        else:
            self.n_tasks     = int(n_tasks)
            self.n_machines  = int(n_machines)
            self.n_resources = int(n_resources)
            self._synthesise_params(demand_stats)
            mode = 'synthetic'

        print(f'  [Energy_Scheduling/{mode}] n_tasks={self.n_tasks}'
              f'  n_machines={self.n_machines}  n_res={self.n_resources}'
              f'  T={self.T} (period={period_minutes}min)'
              f'  T_external={self.T_external} (upsample={self.upsample_factor})'
              f'  seed={self.seed}')

        # valid[j, m, t]: can task j start on machine m at time t?
        self._valid = np.zeros(
            (self.n_tasks, self.n_machines, self.T), dtype=bool)
        for j in range(self.n_tasks):
            e_j = self.task_earliest_start[j]
            l_j = self.task_latest_end[j]
            d_j = self.task_duration[j]
            for t in range(e_j, min(self.T, l_j - d_j + 1)):
                for mi in range(self.n_machines):
                    self._valid[j, mi, t] = True

        print(f'  [ICON] Valid start slots: {int(self._valid.sum())}'
              f'  ({self._valid.sum() / max(1, self.n_tasks):.1f} avg per task)')
        print(f'  [ICON] Mean task duration: {self.task_duration.mean():.1f} '
              f'({self.task_duration.mean() * self.q_h:.2f}h)')
        print(f'  [ICON] Mean task power: {self.task_power.mean():.2f} '
              f'(min={self.task_power.min():.2f}, max={self.task_power.max():.2f})')
        print(f'  [ICON] Machine idle: {np.round(self.machine_idle, 3).tolist()}')
        print(f'  [ICON] Startup cost: {np.round(self.machine_startup_cost, 3).tolist()}')

    def _apply_solve_limits(self, mdl, relax: bool):
        """Set the stopping limit for one solve.

        Called from BOTH the fresh-build and the cached-model paths. They must
        never diverge, or one run measures with two different rulers.
        """
        work = self.LP_WORKLIMIT if relax else self.MIP_WORKLIMIT
        tl = self.LP_TIMELIMIT if relax else self.MIP_TIMELIMIT
        # Print RESOLVED values once, only when a measurement knob is set.
        if not self._measure_banner_done and (
                self.MIP_WORKLIMIT is not None or self.LP_WORKLIMIT is not None
                or self.MIP_RESET or self.MIP_QUALITY):
            self._measure_banner_done = True
            print(f'  | [MEASURE] MIP_WORKLIMIT={self.MIP_WORKLIMIT} '
                  f'LP_WORKLIMIT={self.LP_WORKLIMIT} RESET={self.MIP_RESET} '
                  f'QUALITY={self.MIP_QUALITY} FRESH={self.MIP_FRESH_MODEL} '
                  f'MIP_TL={self.MIP_TIMELIMIT} LP_TL={self.LP_TIMELIMIT} '
                  f'safety={self.WORK_TL_SAFETY}x GAP={self.MIP_GAP} '
                  f'threads={os.environ.get("LP_THREADS_PER_SOLVE")}', flush=True)
        if work is None:
            mdl.setParam('TimeLimit', tl)
        else:
            mdl.setParam('WorkLimit', work)
            mdl.setParam('TimeLimit', tl * self.WORK_TL_SAFETY)

        # Cuts target the fixed-charge idle dodge and the near-symmetric
        # machine plateau. Deterministic given a fixed thread count.
        if self.MIP_QUALITY and not relax:
            mdl.setParam('Cuts', 2)
            mdl.setParam('FlowCoverCuts', 2)
            mdl.setParam('MIRCuts', 2)
            mdl.setParam('Presolve', 2)

    def _synthesise_params(self, demand_stats):
        """Procedural calibration from AEMO demand stats (synthetic mode)."""
        if demand_stats is None:
            raise ValueError(
                'demand_stats is required for synthetic Energy_Scheduling '
                '(use from_icon_instance() to load a real ICON file instead)')

        rng = np.random.RandomState(self.seed)
        T, n_tasks, n_machines = self.T, self.n_tasks, self.n_machines

        # Task energy: 15% of mean interval demand split across n_tasks
        schedulable_mwh = demand_stats['mean_mwh'] * 0.15
        per_task_mean   = schedulable_mwh / n_tasks

        self.task_duration = rng.randint(1, 5, n_tasks).astype(int)
        task_energy_mwh    = rng.uniform(
            0.5 * per_task_mean, 1.5 * per_task_mean, n_tasks)
        self.task_power    = task_energy_mwh / (self.task_duration * self.q_h)

        self.task_earliest_start = rng.randint(0, T // 2, n_tasks).astype(int)
        self.task_latest_end = np.array([
            rng.randint(
                self.task_earliest_start[j] + self.task_duration[j],
                T + 1
            )
            for j in range(n_tasks)
        ], dtype=int)

        self.task_resources = np.column_stack([
            rng.randint(1, 5, n_tasks),   # CPU
            rng.randint(1, 4, n_tasks),   # Memory
            rng.randint(1, 3, n_tasks),   # I/O
        ]).astype(float)

        max_concurrent = max(1, n_tasks // n_machines)
        self.machine_capacity = np.column_stack([
            np.full(n_machines, max_concurrent * 3.0),
            np.full(n_machines, max_concurrent * 2.5),
            np.full(n_machines, max_concurrent * 2.0),
        ])

        self.machine_idle = rng.uniform(0.01, 0.05, n_machines) * float(np.mean(self.task_power))

        price_mid = float(demand_stats['price_p25'] + demand_stats['price_p75']) / 2.0
        typical_task_cost = float(np.mean(self.task_power)) * self.q_h * price_mid
        self.machine_startup_cost  = rng.uniform(0.001, 0.01, n_machines) * typical_task_cost
        self.machine_shutdown_cost = rng.uniform(0.001, 0.01, n_machines) * typical_task_cost

    def _load_icon_params(self, p):
        """Adopt task/machine parameters from a parsed ICON instance dict."""
        self.n_tasks     = int(p['n_tasks'])
        self.n_machines  = int(p['n_machines'])
        self.n_resources = int(p['n_resources'])
        self.task_duration         = np.asarray(p['task_duration'],       dtype=int)
        self.task_earliest_start   = np.asarray(p['task_earliest_start'], dtype=int)
        self.task_latest_end       = np.asarray(p['task_latest_end'],     dtype=int)
        self.task_power            = np.asarray(p['task_power'],          dtype=float)
        self.task_resources        = np.asarray(p['task_resources'],      dtype=float)
        self.machine_idle          = np.asarray(p['machine_idle'],        dtype=float)
        self.machine_startup_cost  = np.asarray(p['machine_startup_cost'],dtype=float)
        self.machine_shutdown_cost = np.asarray(p['machine_shutdown_cost'],dtype=float)
        self.machine_capacity      = np.asarray(p['machine_capacity'],    dtype=float)
        # ICON instance T must match the LP grid we asked for.
        if int(p['T']) != self.T:
            raise ValueError(
                f'ICON instance T={p["T"]} does not match Energy_Scheduling.T={self.T}'
                f' (period_minutes mismatch?)')

    @classmethod
    def from_icon_instance(cls, instance_path, price_range,
                           T_external=48, lp_timelimit=30.0, mip_timelimit=120.0,
                           seed=None, aggregate_to_external=False, **kwargs):
        """
        Load an ICON Challenge 2014 instance file and build a faithful
        Energy_Scheduling task on its native time grid.

        AEMO 30-min prices fed to loss/find_best/spo_plus_loss are upsampled
        internally by repeat_interleave(T // T_external).

        Args:
            instance_path  : path to ICON instance.txt
            price_range    : [min_price, max_price] AUD/MWh
            T_external     : LSTM/AEMO grid length (default 48 for 30-min/day)
            lp_timelimit   : Gurobi LP time limit (s)
            mip_timelimit  : Gurobi MIP time limit (s)
            seed           : reproducibility (only affects None-fields, the
                             instance itself is deterministic)
            aggregate_to_external : if True, coarsen the ICON time grid from
                             native (typically 5min) up to AEMO grid
                             (1440 / T_external minutes per period). Task
                             durations round UP to the new grid. The aggregated
                             MIP can be infeasible: score with False.
        Returns:
            Energy_Scheduling configured for the loaded instance.
        """
        # Lazy import: icon_loader loads only when an ICON task is built.
        from Utils.Downstream.icon_loader import (
            load_icon_instance, summarise_instance, aggregate_icon_to_period,
        )
        inst = load_icon_instance(instance_path)
        if aggregate_to_external:
            target_period = (24 * 60) // int(T_external)
            inst = aggregate_icon_to_period(inst, target_period)
            print(f'  [ICON] Aggregated to {target_period}min grid '
                  f'(T={inst["T"]}, factor={target_period // 5 if target_period >= 5 else 1})')
        print(summarise_instance(inst))
        return cls(
            price_range=price_range,
            demand_stats=None,
            T=inst['T'],
            period_minutes=inst['period_minutes'],
            T_external=T_external,
            n_resources=inst['n_resources'],
            n_tasks=inst['n_tasks'],
            n_machines=inst['n_machines'],
            seed=seed,
            lp_timelimit=lp_timelimit,
            mip_timelimit=mip_timelimit,
            _icon_params=inst,
            **kwargs,
        )

    def _solve(self, price_vec: np.ndarray, relax: bool = True) -> dict:
        """
        Solve the ICON scheduling problem for a T-period price vector.

        Equations 1–12 from Simonis et al. / ICON Challenge 2014 specification:
            Eq 1  : each task assigned to exactly one (machine, start-time)
            Eq 2-3: task windows (earliest start, latest end - duration)
            Eq 4  : resource capacity per machine per period
            Eq 5-7: machine state continuity (startup/shutdown linkage)
            Eq 8  : minimise C_J + C_M + C_up + C_down
            Eq 9  : C_J  = task energy cost
            Eq 10 : C_M  = machine idle energy cost
            Eq 11 : C_up = startup cost
            Eq 12 : C_down = shutdown cost

        Args:
            price_vec : shape (T,) in AUD/MWh
            relax     : LP relaxation if True, full MIP if False
        Returns:
            dict with 'w' (energy profile, T), 'k' (fixed cost), 'obj' (total cost)
        """
        J, M, T, R = self.n_tasks, self.n_machines, self.T, self.n_resources
        q = self.q_h
        d  = self.task_duration
        pj = self.task_power
        ur = self.task_resources    # [J, R]
        cap = self.machine_capacity  # [M, R]

        if not _GUROBI_AVAILABLE:
            return self._greedy_solve(price_vec)

        # A prior license-size error on this instance: skip Gurobi.
        if getattr(self, '_gurobi_unavailable', False):
            if relax:
                return self._scipy_solve(price_vec)
            return self._greedy_solve(price_vec)

        _verbose = int(os.environ.get('GUROBI_VERBOSE', '0'))

        # Get-or-build the thread-local model; a cache hit skips a ~7s build.
        _cache_key = 'lp' if relax else 'mip'
        cached = getattr(self._lp_thread_cache, _cache_key, None)
        # MIP_FRESH_MODEL=1 builds every model from scratch (~7s at T=288).
        # Whether a day lands on a fresh or a reused model depends on thread
        # order, and reset(1) does not make a reused model identical to fresh.
        if self.MIP_FRESH_MODEL:
            cached = None
        if cached is not None:
            mdl, x, v, yu, zd, x_var_list, v_var_list = cached
            if _verbose:
                _t_build_start = None  # signal "skip build print"
        else:
            if _verbose:
                _t_build_start = time.time()
            # Per-thread quiet Gurobi env. OutputFlag must be set BEFORE
            # env.start() to suppress the license preamble.
            env = getattr(self._lp_thread_cache, 'env', None)
            if env is None:
                env = gp.Env(empty=True)
                env.setParam('OutputFlag', _verbose)
                env.start()
                self._lp_thread_cache.env = env
            mdl = gp.Model(env=env)
            mdl.setParam('OutputFlag', _verbose)
            self._apply_solve_limits(mdl, relax)
            mdl.setParam('Threads', _LP_THREADS_PER_SOLVE)
            if relax:
                # Barrier without crossover: an interior optimum is enough for
                # the SPO+ subgradient, which needs w*(y), not a vertex.
                mdl.setParam('Method', 2)
                mdl.setParam('Crossover', 0)

            # Decision variables
            # LP: continuous [0,1]; MIP: binary
            if relax:
                x  = mdl.addVars(J, M, T, lb=0.0, ub=1.0, name='x')
                v  = mdl.addVars(M, T, lb=0.0, ub=1.0, name='v')
                yu = mdl.addVars(M, T, lb=0.0, ub=1.0, name='yu')
                zd = mdl.addVars(M, T, lb=0.0, ub=1.0, name='zd')
            else:
                x  = mdl.addVars(J, M, T, vtype=GRB.BINARY, name='x')
                v  = mdl.addVars(M, T, vtype=GRB.BINARY, name='v')
                yu = mdl.addVars(M, T, vtype=GRB.BINARY, name='yu')
                zd = mdl.addVars(M, T, vtype=GRB.BINARY, name='zd')

            # Constraint 1: each task assigned exactly once (Eq 1)
            for j in range(J):
                mdl.addConstr(
                    gp.quicksum(x[j, mi, t]
                                for mi in range(M) for t in range(T)
                                if self._valid[j, mi, t]) == 1,
                    name=f'assign_{j}'
                )
                # Force out-of-window slots to zero (Eqs 2–3)
                for mi in range(M):
                    for t in range(T):
                        if not self._valid[j, mi, t]:
                            mdl.addConstr(x[j, mi, t] == 0)

            # Constraint 4: resource capacity per machine per period
            for mi in range(M):
                for t in range(T):
                    for r in range(R):
                        # Running tasks: all tasks j that started at t_s ≤ t < t_s + d_j
                        mdl.addConstr(
                            gp.quicksum(
                                x[j, mi, t_s] * ur[j, r]
                                for j in range(J)
                                for t_s in range(max(0, t - d[j] + 1), t + 1)
                                if self._valid[j, mi, t_s]
                            ) <= cap[mi, r],
                            name=f'cap_{mi}_{t}_{r}'
                        )

            # Constraints 5–7: machine state linkage
            for mi in range(M):
                for t in range(T):
                    # Machine must be on when any task is running (Eq 7)
                    for j in range(J):
                        for t_s in range(max(0, t - d[j] + 1), t + 1):
                            if self._valid[j, mi, t_s]:
                                mdl.addConstr(x[j, mi, t_s] <= v[mi, t],
                                              name=f'on_{j}_{mi}_{t}')

                    # Startup/shutdown indicators (Eqs 5–6)
                    if t == 0:
                        mdl.addConstr(yu[mi, t] >= v[mi, t],
                                      name=f'startup0_{mi}')
                        mdl.addConstr(zd[mi, t] == 0,
                                      name=f'shutdown0_{mi}')
                    else:
                        mdl.addConstr(yu[mi, t] >= v[mi, t] - v[mi, t - 1],
                                      name=f'startup_{mi}_{t}')
                        mdl.addConstr(zd[mi, t] >= v[mi, t - 1] - v[mi, t],
                                      name=f'shutdown_{mi}_{t}')

            # Price-independent objective layout, built once per instance:
            # x[j,mi,t_s] coeff = pj[j]*q*sum(price[t_s : t_s+d[j]]); v[mi,t]
            # coeff = machine_idle[mi]*q*price[t]. Each solve then sets all
            # coeffs in one numpy pass + one setAttr (a LinExpr build holds the GIL).
            if not getattr(self, '_obj_struct_built', False):
                _xk, _xs, _xe1, _xm = [], [], [], []
                for j in range(J):
                    for mi in range(M):
                        for t_s in range(T):
                            if self._valid[j, mi, t_s]:
                                _xk.append((j, mi, t_s))
                                _xs.append(t_s)
                                _xe1.append(min(t_s + d[j], T))
                                _xm.append(pj[j] * q)
                self._obj_x_keys   = _xk
                self._obj_x_tstart = np.asarray(_xs,  dtype=np.int64)
                self._obj_x_tend1  = np.asarray(_xe1, dtype=np.int64)
                self._obj_x_mult   = np.asarray(_xm,  dtype=np.float64)
                _vk, _vt, _vm = [], [], []
                for mi in range(M):
                    for t in range(T):
                        _vk.append((mi, t))
                        _vt.append(t)
                        _vm.append(self.machine_idle[mi] * q)
                self._obj_v_keys = _vk
                self._obj_v_t    = np.asarray(_vt, dtype=np.int64)
                self._obj_v_mult = np.asarray(_vm, dtype=np.float64)
                self._obj_struct_built = True

            # Per-model var handles in the structure's canonical order.
            x_var_list = [x[k] for k in self._obj_x_keys]
            v_var_list = [v[k] for k in self._obj_v_keys]
            # Fixed (price-independent) startup/shutdown costs — set once.
            _yu_vars = [yu[mi, t] for mi in range(M) for t in range(T)]
            _zd_vars = [zd[mi, t] for mi in range(M) for t in range(T)]
            mdl.setAttr('Obj', _yu_vars,
                        [self.machine_startup_cost[mi] for mi in range(M) for t in range(T)])
            mdl.setAttr('Obj', _zd_vars,
                        [self.machine_shutdown_cost[mi] for mi in range(M) for t in range(T)])
            mdl.ModelSense = GRB.MINIMIZE

            setattr(self._lp_thread_cache, _cache_key,
                    (mdl, x, v, yu, zd, x_var_list, v_var_list))

        # Apply per-solve params on EVERY call so cached models honour the
        # current limits and MIPGap.
        self._apply_solve_limits(mdl, relax)
        if not relax:
            mdl.setParam('MIPGap', self.MIP_GAP)
            # Optional feasibility-first levers (default unset). The greedy
            # fallback fires when SolCount==0; these push Gurobi to find SOME
            # incumbent fast.
            _mip_focus = os.environ.get('MIP_FOCUS')
            if _mip_focus not in (None, '', '0'):
                mdl.setParam('MIPFocus', int(_mip_focus))
            _norel = os.environ.get('NOREL_HEUR_TIME')
            if _norel not in (None, ''):
                mdl.setParam('NoRelHeurTime', float(_norel))
            _heur = os.environ.get('GRB_HEURISTICS')
            if _heur not in (None, ''):
                mdl.setParam('Heuristics', float(_heur))

        # Objective (Eqs 8-12): refresh only the price-dependent coeffs (C_J via
        # a price cumsum, C_M); C_up/C_down were set at build.
        _pv = np.asarray(price_vec, dtype=np.float64)
        _cs = np.concatenate(([0.0], np.cumsum(_pv)))   # _cs[k] = sum(price[:k]), len T+1
        _x_coeffs = self._obj_x_mult * (_cs[self._obj_x_tend1] - _cs[self._obj_x_tstart])
        _v_coeffs = self._obj_v_mult * _pv[self._obj_v_t]
        mdl.setAttr('Obj', x_var_list, _x_coeffs.tolist())
        mdl.setAttr('Obj', v_var_list, _v_coeffs.tolist())
        if _verbose:
            if _t_build_start is not None:
                print(f'  [Gurobi] FRESH build (constraints + objective): '
                      f'{time.time() - _t_build_start:.2f}s '
                      f'(NumVars={mdl.NumVars}, NumConstrs={mdl.NumConstrs})', flush=True)
            else:
                print(f'  [Gurobi] CACHED model — only objective updated, '
                      f'NumVars={mdl.NumVars} NumConstrs={mdl.NumConstrs}', flush=True)
            _t_solve_start = time.time()
        try:
            if self.MIP_RESET:
                # AFTER the objective update, BEFORE optimize. reset(1) clears
                # the retained solution as well as the basis.
                mdl.reset(1)
            mdl.optimize()
            # LOG_WORK=1 records the work units each solve used, to calibrate
            # a work limit.
            if os.environ.get('LOG_WORK') not in (None, '', '0'):
                try:
                    # A solve stopped on its limit gives a LOWER bound on work.
                    self._last_work = float(mdl.Work)
                    self._last_status = int(mdl.Status)
                    self._last_runtime = float(mdl.Runtime)
                    self._last_solcount = int(mdl.SolCount)
                    self._last_gap = float(mdl.MIPGap) if not relax and mdl.SolCount > 0 else None
                except Exception:
                    self._last_work = None
                    self._last_status = None
        except gp.GurobiError as e:
            msg = str(e).lower()
            if 'size-limited' in msg or 'too large' in msg:
                # Restricted license cap hit: later calls skip Gurobi.
                if not getattr(self, '_gurobi_unavailable', False):
                    self._gurobi_unavailable = True
                    print(f'  [Gurobi] License size-limit hit; switching to '
                          f'{"scipy" if relax else "greedy"} fallback for this Energy_Scheduling instance.')
                if relax:
                    return self._scipy_solve(price_vec)
                return self._greedy_solve(price_vec)
            raise

        # Statuses with a usable incumbent. WORK_LIMIT MUST be here, or every
        # work-limited solve falls to greedy. getattr: gurobipy < 9.5 lacks it.
        _usable = (GRB.OPTIMAL, GRB.TIME_LIMIT, getattr(GRB, 'WORK_LIMIT', 16))
        if mdl.status not in _usable or mdl.SolCount == 0:
            return self._greedy_solve(price_vec)

        # Extract energy profile w[t]
        w_arr = np.zeros(T, dtype=np.float64)
        for t in range(T):
            et = 0.0
            for j in range(J):
                for t_s in range(max(0, t - d[j] + 1), t + 1):
                    for mi in range(M):
                        if self._valid[j, mi, t_s]:
                            try:
                                et += x[j, mi, t_s].X * pj[j] * q
                            except AttributeError:
                                pass
            for mi in range(M):
                try:
                    et += v[mi, t].X * self.machine_idle[mi] * q
                except AttributeError:
                    pass
            w_arr[t] = et

        # Fixed cost K = C_up + C_down
        k_val = 0.0
        for mi in range(M):
            for t in range(T):
                try:
                    k_val += yu[mi, t].X * self.machine_startup_cost[mi]
                    k_val += zd[mi, t].X * self.machine_shutdown_cost[mi]
                except AttributeError:
                    pass

        # MIPGap applies only to MIP solves; the LP gap is 0.
        try:
            gap = float(mdl.MIPGap) if not relax else 0.0
        except (AttributeError, gp.GurobiError):
            gap = 0.0

        return {
            'w':     w_arr,
            'k':     float(k_val),
            'obj':   float(mdl.ObjVal),
            'gap':   gap,
            'status': int(mdl.status),
        }

    # scipy.linprog (HiGHS) LP fallback for the restricted license. Constraint
    # matrices are built once per instance.

    def _build_lp_static_for_scipy(self):
        """Build cached LP constraint matrices and objective coefficient maps.
        Idempotent + thread-safe: called from the first _scipy_solve() in any
        worker thread of the ThreadPoolExecutor."""
        if getattr(self, '_scipy_built', False):
            return
        with _LP_STATIC_BUILD_LOCK:
            if getattr(self, '_scipy_built', False):
                return

            J, M, T, R = self.n_tasks, self.n_machines, self.T, self.n_resources
            q   = self.q_h
            d   = self.task_duration
            pj  = self.task_power
            ur  = self.task_resources
            cap = self.machine_capacity
            valid = self._valid

            # Variable layout (linear order):
            #   x[j,m,t]  → idx = j*M*T + m*T + t        (range [0, J*M*T))
            #   v[m,t]    → idx = J*M*T + m*T + t        (length M*T)
            #   yu[m,t]   → idx = J*M*T + M*T + m*T + t  (length M*T)
            #   zd[m,t]   → idx = J*M*T + 2*M*T + m*T + t (length M*T)
            n_x  = J * M * T
            n_mt = M * T
            x_off, v_off, yu_off, zd_off = 0, n_x, n_x + n_mt, n_x + 2 * n_mt
            n_vars = n_x + 3 * n_mt

            def x_idx(j, m, t):  return x_off  + j * M * T + m * T + t
            def v_idx(m, t):     return v_off  + m * T + t
            def yu_idx(m, t):    return yu_off + m * T + t
            def zd_idx(m, t):    return zd_off + m * T + t

            # Bounds: [0,1] for all; invalid x[j,m,t] forced to [0,0].
            # The boolean mask applies only to the x section (first n_x slots).
            bounds_lo = np.zeros(n_vars)
            bounds_hi = np.ones(n_vars)
            invalid_x_mask = ~valid.flatten()  # length n_x
            x_section = bounds_hi[:n_x]
            x_section[invalid_x_mask] = 0.0
            bounds_hi[:n_x] = x_section
            self._scipy_bounds = list(zip(bounds_lo, bounds_hi))

            # Equality constraints
            # 1. Each task assigned exactly once: sum_{m,t valid} x[j,m,t] = 1
            # 2. zd[m, 0] = 0
            eq_rows, eq_cols, eq_data = [], [], []
            eq_b_list = []
            row = 0
            for j in range(J):
                for mi in range(M):
                    for t in range(T):
                        if valid[j, mi, t]:
                            eq_rows.append(row)
                            eq_cols.append(x_idx(j, mi, t))
                            eq_data.append(1.0)
                eq_b_list.append(1.0)
                row += 1
            for mi in range(M):
                eq_rows.append(row); eq_cols.append(zd_idx(mi, 0)); eq_data.append(1.0)
                eq_b_list.append(0.0); row += 1
            self._scipy_A_eq = csr_matrix(
                (eq_data, (eq_rows, eq_cols)), shape=(row, n_vars))
            self._scipy_b_eq = np.asarray(eq_b_list)

            # Inequality constraints
            ub_rows, ub_cols, ub_data = [], [], []
            ub_b_list = []
            row = 0

            # Capacity (Eq 4): for each (m, t, r), sum running tasks * ur[j,r] <= cap[m,r]
            for mi in range(M):
                for t in range(T):
                    for r in range(R):
                        for j in range(J):
                            for t_s in range(max(0, t - d[j] + 1), t + 1):
                                if valid[j, mi, t_s]:
                                    ub_rows.append(row)
                                    ub_cols.append(x_idx(j, mi, t_s))
                                    ub_data.append(float(ur[j, r]))
                        ub_b_list.append(float(cap[mi, r]))
                        row += 1

            # Machine on (Eq 7): x[j,m,t_s] - v[m,t] <= 0 for t in [t_s, t_s+d_j-1]
            for mi in range(M):
                for t in range(T):
                    for j in range(J):
                        for t_s in range(max(0, t - d[j] + 1), t + 1):
                            if valid[j, mi, t_s]:
                                ub_rows.append(row); ub_cols.append(x_idx(j, mi, t_s)); ub_data.append(1.0)
                                ub_rows.append(row); ub_cols.append(v_idx(mi, t));     ub_data.append(-1.0)
                                ub_b_list.append(0.0)
                                row += 1

            # Startup (Eq 5): yu >= v - v_prev (with v_prev=0 at t=0)
            # Shutdown (Eq 6): zd >= v_prev - v (only for t > 0)
            for mi in range(M):
                for t in range(T):
                    if t == 0:
                        # yu[m,0] >= v[m,0]  →  v - yu <= 0
                        ub_rows.append(row); ub_cols.append(v_idx(mi, t));  ub_data.append(1.0)
                        ub_rows.append(row); ub_cols.append(yu_idx(mi, t)); ub_data.append(-1.0)
                        ub_b_list.append(0.0); row += 1
                    else:
                        # yu[m,t] >= v[m,t] - v[m,t-1]  →  v - v_prev - yu <= 0
                        ub_rows.append(row); ub_cols.append(v_idx(mi, t));    ub_data.append(1.0)
                        ub_rows.append(row); ub_cols.append(v_idx(mi, t-1));  ub_data.append(-1.0)
                        ub_rows.append(row); ub_cols.append(yu_idx(mi, t));   ub_data.append(-1.0)
                        ub_b_list.append(0.0); row += 1
                        # zd[m,t] >= v[m,t-1] - v[m,t]  →  v_prev - v - zd <= 0
                        ub_rows.append(row); ub_cols.append(v_idx(mi, t-1)); ub_data.append(1.0)
                        ub_rows.append(row); ub_cols.append(v_idx(mi, t));   ub_data.append(-1.0)
                        ub_rows.append(row); ub_cols.append(zd_idx(mi, t));  ub_data.append(-1.0)
                        ub_b_list.append(0.0); row += 1

            self._scipy_A_ub = csr_matrix(
                (ub_data, (ub_rows, ub_cols)), shape=(row, n_vars))
            self._scipy_b_ub = np.asarray(ub_b_list)

            # Objective coefficient maps
            # The objective is C_J + C_M + C_fixed, where C_J and C_M are linear
            # in price_vec but C_fixed is price-independent. Cache:
            #   _scipy_fixed_c    : np.array(n_vars)  — yu/zd contributions
            #   _scipy_obj_x_idx[t]   : indices of x vars active at time t
            #   _scipy_obj_x_coeff[t] : coeff per x var (= pj[j]*q)
            #   _scipy_obj_v_idx      : flat indices of v vars (len n_mt)
            #   _scipy_obj_v_coeff    : coeff per v var (= machine_idle[m]*q)
            #     such that c[v_idx[m,t]] = obj_v_coeff[m*T+t] * price_vec[t]
            obj_x_idx_per_t   = [[] for _ in range(T)]
            obj_x_coeff_per_t = [[] for _ in range(T)]
            for j in range(J):
                coeff_j = float(pj[j]) * q
                for mi in range(M):
                    for t_s in range(T):
                        if valid[j, mi, t_s]:
                            var_idx = x_idx(j, mi, t_s)
                            for t in range(t_s, min(T, t_s + d[j])):
                                obj_x_idx_per_t[t].append(var_idx)
                                obj_x_coeff_per_t[t].append(coeff_j)
            self._scipy_obj_x_idx   = [np.asarray(l, dtype=np.int64) for l in obj_x_idx_per_t]
            self._scipy_obj_x_coeff = [np.asarray(l, dtype=np.float64) for l in obj_x_coeff_per_t]

            obj_v_idx   = np.empty(n_mt, dtype=np.int64)
            obj_v_coeff = np.empty(n_mt, dtype=np.float64)
            for mi in range(M):
                for t in range(T):
                    obj_v_idx[mi * T + t]   = v_idx(mi, t)
                    obj_v_coeff[mi * T + t] = float(self.machine_idle[mi]) * q
            self._scipy_obj_v_idx   = obj_v_idx
            self._scipy_obj_v_coeff = obj_v_coeff

            fixed_c = np.zeros(n_vars)
            for mi in range(M):
                for t in range(T):
                    fixed_c[yu_idx(mi, t)] = float(self.machine_startup_cost[mi])
                    fixed_c[zd_idx(mi, t)] = float(self.machine_shutdown_cost[mi])
            self._scipy_fixed_c = fixed_c

            # Stash offsets/sizes for solution extraction
            self._scipy_n_vars = n_vars
            self._scipy_v_off = v_off
            self._scipy_T_local = T
            self._scipy_M_local = M

            self._scipy_built = True

    def _scipy_solve(self, price_vec: np.ndarray) -> dict:
        """Solve LP relaxation via scipy.linprog (HiGHS).
        Used as fallback when the restricted gurobipy license can't handle the
        problem size. Returns the same dict contract as _solve()/_greedy_solve()."""
        self._build_lp_static_for_scipy()

        T = self._scipy_T_local
        M = self._scipy_M_local
        q = self.q_h

        # Build cost vector c using cached coefficient maps.
        c = self._scipy_fixed_c.copy()
        for t in range(T):
            idx = self._scipy_obj_x_idx[t]
            if idx.size:
                np.add.at(c, idx, float(price_vec[t]) * self._scipy_obj_x_coeff[t])
        # v[m,t] contribution: c[v_idx[m,t]] += machine_idle[m]*q * price[t]
        # _scipy_obj_v_idx is laid out [m=0,t=0..T-1, m=1,t=0..T-1, ...]
        price_repeated = np.tile(np.asarray(price_vec, dtype=np.float64), M)
        np.add.at(c, self._scipy_obj_v_idx, self._scipy_obj_v_coeff * price_repeated)

        # Capped at LP_TIMELIMIT; on failure fall through to greedy.
        res = linprog(
            c,
            A_ub=self._scipy_A_ub, b_ub=self._scipy_b_ub,
            A_eq=self._scipy_A_eq, b_eq=self._scipy_b_eq,
            bounds=self._scipy_bounds,
            method='highs',
            options={'time_limit': float(self.LP_TIMELIMIT)},
        )

        if not res.success or res.x is None:
            return self._greedy_solve(price_vec)

        sol = res.x

        # w[t] reuses the cached objective coefficient maps.
        w_arr = np.zeros(T, dtype=np.float64)
        for t in range(T):
            idx = self._scipy_obj_x_idx[t]
            if idx.size:
                w_arr[t] += float(np.dot(sol[idx], self._scipy_obj_x_coeff[t]))
        # v contribution to w[t]: machine_idle[m]*q * v[m,t]
        for mi in range(M):
            for t in range(T):
                w_arr[t] += sol[self._scipy_v_off + mi * T + t] * float(self.machine_idle[mi]) * q

        k_val = float(np.dot(sol, self._scipy_fixed_c))

        return {
            'w':      w_arr,
            'k':      k_val,
            'obj':    float(res.fun),
            'gap':    0.0,
            'status': 0,
        }

    def _greedy_solve(self, price_vec: np.ndarray) -> dict:
        """
        Greedy fallback: each task at its cheapest feasible window on machine 0,
        ignoring capacity and idle cost. Returns status=-1.
        """
        J, T = self.n_tasks, self.T
        d  = self.task_duration
        pj = self.task_power
        w  = np.zeros(T, dtype=np.float64)

        for j in range(J):
            best_t, best_cost = None, np.inf
            for t_s in range(T):
                if self._valid[j, 0, t_s]:
                    cost = sum(
                        price_vec[min(T - 1, t_s + dt)] * pj[j] * self.q_h
                        for dt in range(d[j])
                    )
                    if cost < best_cost:
                        best_t, best_cost = t_s, cost
            if best_t is not None:
                for dt in range(d[j]):
                    w[min(T - 1, best_t + dt)] += pj[j] * self.q_h

        k_val = 0.0
        obj   = float(np.dot(w, price_vec) + k_val)
        # Sentinel gap 1.0 so high-gap counters include greedy solves.
        return {'w': w, 'k': k_val, 'obj': obj, 'gap': 1.0, 'status': -1}

    def _solve_batch(self, price_batch: np.ndarray, relax: bool = True):
        """
        Solve for each price vector in a batch, in parallel across CPU threads.

        Each worker thread holds its own gp.Model; Gurobi releases the GIL.
        Workers are capped at _LP_WORKERS.

        Args:
            price_batch : shape (batch, T)
            relax       : LP relaxation if True, full MIP if False
        Returns:
            w_batch : shape (batch, T)
            k_batch : shape (batch,)
        """
        B = price_batch.shape[0]
        n_workers = min(_LP_WORKERS, B)

        _debug = int(os.environ.get('SOLVE_BATCH_DEBUG', '0'))
        # SOLVE_BATCH_SUMMARY=1: one per-batch line of per-LP wall-time stats.
        _summary = int(os.environ.get('SOLVE_BATCH_SUMMARY', '0'))
        _t0_batch = time.time() if (_debug or _summary) else None
        _solve_dts = [] if _summary else None

        def solve_one(b):
            if _debug:
                tid = threading.get_ident()
                t_in = time.time() - _t0_batch
                print(f'  [batch] tid={tid} b={b:>3} START   @ +{t_in:6.2f}s', flush=True)
            t_solve_start = time.time() if (_debug or _summary) else None
            res = self._solve(price_batch[b], relax=relax)
            if _debug or _summary:
                dt = time.time() - t_solve_start
                if _summary:
                    _solve_dts.append((dt, res.get('status', -1)))
                if _debug:
                    t_out = time.time() - _t0_batch
                    print(f'  [batch] tid={threading.get_ident()} b={b:>3} DONE    @ +{t_out:6.2f}s '
                          f'(solve_dt={dt:5.2f}s, status={res.get("status")})', flush=True)
            return res

        if _debug:
            print(f'  [batch] === B={B}, n_workers={n_workers}, '
                  f'_LP_THREADS_PER_SOLVE={_LP_THREADS_PER_SOLVE} ===', flush=True)

        # Persistent pool for the instance's lifetime. A per-call pool would
        # drop the per-thread model cache and re-fetch the license each call.
        pool = getattr(self, '_solve_pool', None)
        if pool is None or self._solve_pool_workers < n_workers:
            if pool is not None:
                pool.shutdown(wait=True)
            pool = ThreadPoolExecutor(max_workers=n_workers,
                                      thread_name_prefix='lp_solve')
            self._solve_pool = pool
            self._solve_pool_workers = n_workers
        results = list(pool.map(solve_one, range(B)))
        if _debug:
            print(f'  [batch] === all {B} LPs done in {time.time()-_t0_batch:.2f}s ===', flush=True)

        if _summary and _solve_dts:
            wall = time.time() - _t0_batch
            dts = np.array([d for d, _ in _solve_dts])
            statuses = [s for _, s in _solve_dts]
            from collections import Counter
            status_counts = Counter(statuses)
            status_str = ' '.join(f'{s}:{n}' for s, n in sorted(status_counts.items()))
            # First-call diagnostic: thread environment that can throttle Gurobi.
            if not getattr(self, '_first_batch_logged', False):
                import torch as _torch_for_diag
                try:
                    nthreads = _torch_for_diag.get_num_threads()
                except Exception:
                    nthreads = '?'
                print(f'  [batch] === thread env: torch.get_num_threads={nthreads}  '
                      f'OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS", "unset")}  '
                      f'MKL_NUM_THREADS={os.environ.get("MKL_NUM_THREADS", "unset")}  '
                      f'OPENBLAS_NUM_THREADS={os.environ.get("OPENBLAS_NUM_THREADS", "unset")}  '
                      f'cpus={len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "?"}  ===',
                      flush=True)
                self._first_batch_logged = True
            print(f'  [batch] B={B}  workers={n_workers}×{_LP_THREADS_PER_SOLVE}t  '
                  f'wall={wall:.1f}s  per-LP solve_dt mean={dts.mean():.2f}s '
                  f'med={np.median(dts):.2f}s max={dts.max():.2f}s min={dts.min():.2f}s  '
                  f'status: {status_str}', flush=True)

        w_batch = np.stack([r['w'] for r in results])
        k_batch = np.array([r['k'] for r in results], dtype=np.float64)

        # Greedy monitor (status=-1): count greedy solves on the task so
        # test_task can report them.
        _n_greedy = sum(1 for r in results if r.get('status', 0) == -1)
        if _n_greedy > 0:
            self._greedy_solves = getattr(self, '_greedy_solves', 0) + _n_greedy
            self._total_solves  = getattr(self, '_total_solves', 0) + B
            if not getattr(self, '_greedy_warned', False):
                print(f'  [GREEDY FALLBACK] {_n_greedy}/{B} solves fell back to '
                      f'greedy (status=-1) — "MIP" regret is CONTAMINATED. '
                      f'Check ICON_AGGREGATE_TO_EXTERNAL / Gurobi license.', flush=True)
                self._greedy_warned = True
            # A greedy MIP result is NEVER a valid number: the MIP path always
            # crashes here. The LP path tolerates a transient fallback in
            # training unless ASSERT_NO_GREEDY=1.
            if (not relax) or os.environ.get('ASSERT_NO_GREEDY') == '1':
                raise RuntimeError(
                    f'GREEDY on {"MIP" if not relax else "LP"} path: {_n_greedy}/{B} '
                    f'solves used the greedy fallback (status=-1). Refusing to '
                    f'produce contaminated regret. For native-MIP eval set '
                    f'ICON_AGG=0 (T=288 is feasible; aggregated T=48 is not).')
        else:
            self._total_solves = getattr(self, '_total_solves', 0) + B

        # MIP quality monitor: a truncated solve returns an UPPER bound on the
        # optimum, so regret is UNDERESTIMATED. Prints the first and every 50th
        # batch with an issue.
        if not relax and _GUROBI_AVAILABLE:
            gaps = np.array([r.get('gap', 0.0) for r in results])
            # WORK_LIMIT is a truncation too.
            _trunc = (GRB.TIME_LIMIT, getattr(GRB, 'WORK_LIMIT', 16))
            timeouts = sum(1 for r in results if r.get('status', 0) in _trunc)
            high_gap = int((gaps > 0.01).sum())
            if timeouts > 0 or high_gap > 0:
                self._mip_batches_with_issue = getattr(self, '_mip_batches_with_issue', 0) + 1
                self._mip_total_timeouts    = getattr(self, '_mip_total_timeouts', 0) + timeouts
                self._mip_total_high_gap    = getattr(self, '_mip_total_high_gap', 0) + high_gap
                self._mip_total_instances   = getattr(self, '_mip_total_instances', 0) + B
                _n = self._mip_batches_with_issue
                if _n == 1 or _n % 50 == 0:
                    print(f'  [MIP cumulative @ batch #{_n}]  '
                          f'timeouts={self._mip_total_timeouts}/{self._mip_total_instances}  '
                          f'>1%_gap={self._mip_total_high_gap}/{self._mip_total_instances}  '
                          f'latest max_gap={gaps.max():.4f}'
                          + ('  (regret UNDERESTIMATED on time-out instances)' if timeouts > 0 else ''),
                          flush=True)

        return w_batch, k_batch

    def _upsample(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map external (LSTM/AEMO) price grid → internal LP grid by step-replicate.

        AEMO is 30-min (T_external=48); ICON LP is 5-min (T=288). Each AEMO price
        applies to upsample_factor=6 LP periods. Identity when upsample_factor=1.
        """
        if self.upsample_factor == 1:
            return y
        return y.repeat_interleave(self.upsample_factor, dim=-1)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute actual total scheduling cost C = C_J + C_M + C_up + C_down.

        Eq 8, Simonis et al. Negative when spot prices are negative: load is
        paid to consume.

        Args:
            x : schedule encoding, shape (batch, T+1) on LP grid
                  x[:, :T]  — energy profile w_t (MWh/period, price-sensitive)
                  x[:, T]   — fixed cost K = C_up + C_down (AUD)
            y : actual prices (AUD/MWh), shape (batch, T_external) on AEMO grid
        Returns:
            total cost per batch element, shape (batch,)
        """
        w = x[:, :self.T]   # (batch, T)
        k = x[:,  self.T]   # (batch,)
        y_full = self._upsample(y)
        return (y_full * w).sum(dim=-1) + k

    def find_best_scheduling(self, y: torch.Tensor):
        """
        Solve ICON LP relaxation for predicted prices (fast; used in training).

        Args:
            y : predicted prices (AUD/MWh), shape (batch, T_external)
        Returns:
            best_x       : schedule encoding (batch, T+1) for loss()
            lowest_loss  : optimal cost under y (batch,)
        """
        y_full = self._upsample(y).detach().cpu().numpy()
        w_b, k_b = self._solve_batch(y_full, relax=True)

        best_x = torch.zeros(y.shape[0], self.T + 1,
                              dtype=torch.float32, device=y.device)
        best_x[:, :self.T] = torch.tensor(w_b, dtype=torch.float32, device=y.device)
        best_x[:,  self.T] = torch.tensor(k_b, dtype=torch.float32, device=y.device)

        lowest_loss = self.loss(best_x, y)
        return best_x, lowest_loss

    def find_best_scheduling_mip(self, y: torch.Tensor):
        """
        Full MIP solve for scoring; not called during training.
        """
        y_full = self._upsample(y).detach().cpu().numpy()
        w_b, k_b = self._solve_batch(y_full, relax=False)

        best_x = torch.zeros(y.shape[0], self.T + 1,
                              dtype=torch.float32, device=y.device)
        best_x[:, :self.T] = torch.tensor(w_b, dtype=torch.float32, device=y.device)
        best_x[:,  self.T] = torch.tensor(k_b, dtype=torch.float32, device=y.device)

        lowest_loss = self.loss(best_x, y)
        return best_x, lowest_loss

    def spo_plus_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        SPO+ surrogate loss for training through the ICON LP relaxation.

        From Elmachtoub & Grigas (2022), Theorem 2, the SPO+ subgradient is:
            ∂ℓ_SPO+/∂ŷ = 2·(w*(y) − w*(2ŷ − y))

        Implemented as the loss expression that autograd differentiates to give
        this gradient (treating the schedule tensors w as detached constants):
            ℓ = (2·ŷ·w_true − (2·ŷ − y)·w_pert).sum(dim=−1).mean()
            ∂ℓ/∂ŷ = 2·w_true − 2·w_pert = 2·(w_true − w_pert)  ✓

        where  w_true = w*(y)            [LP optimal schedule for TRUE price]
               w_pert = w*(2ŷ − y)       [LP optimal for SPO+ perturbed price]

        The C_up + C_down fixed terms cancel in the gradient (price-independent)
        so only the energy profiles w enter the gradient computation.

        With upsample_factor>1, autograd through repeat_interleave sums the
        LP-grid gradient back over each block of replicates.

        Args:
            y_pred : predicted prices (AUD/MWh), shape (batch, T_external),
                     requires_grad=True
            y_true : actual prices   (AUD/MWh), shape (batch, T_external)
        Returns:
            scalar SPO+ loss
        """
        # Upsample external→internal grid (autograd-compatible)
        y_pred_full = self._upsample(y_pred)
        y_true_full = self._upsample(y_true)

        y_pred_np    = y_pred_full.detach().cpu().numpy()
        y_true_np    = y_true_full.detach().cpu().numpy()
        perturbed_np = 2.0 * y_pred_np - y_true_np

        # w*(y_true) cache keyed PER ROW: DFL subsamples batches with randperm,
        # so a batch-level key would always miss and grow without bound.
        _spo_cache = getattr(self, '_spo_true_cache', None)
        if _spo_cache is None:
            _spo_cache = self._spo_true_cache = {}
        w_true_np = cache_solve_per_row(
            y_true_np, _spo_cache,
            solve_fn=lambda a: self._solve_batch(a, relax=True)[0],
            greedy_count_fn=lambda: getattr(self, '_greedy_solves', 0),
        )
        w_pert_np, _ = self._solve_batch(perturbed_np, relax=True)

        w_true = torch.tensor(w_true_np, dtype=torch.float32, device=y_pred.device)
        w_pert = torch.tensor(w_pert_np, dtype=torch.float32, device=y_pred.device)

        loss = (2.0 * y_pred_full * w_true
                - (2.0 * y_pred_full - y_true_full) * w_pert).sum(dim=-1).mean()

        # LOSS SCALE from price_range (PRICE_RANGE_LO/HI in main_energy_mv.py).
        price_scale = max(abs(self.PRICE_MAX - self.PRICE_MIN), 1.0)
        return loss / price_scale
