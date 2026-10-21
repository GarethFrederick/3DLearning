"""
Build and screen K-task subsamples of the REAL ICON sample07, without any MIP.

Recipe:
  * keep a seeded random subset of K of sample07's 1000 task rows VERBATIM,
    re-indexed 0..K-1;
  * leave the MACHINE block UNCHANGED. Scaling capacity by K/1000 makes some
    tasks exceed a machine's cap -> infeasible MIP -> silent greedy fallback.

Checks, for each K (SUB_KS) x seed (SUB_SEEDS):
  1. the file round-trips through the real loader;
  2. every kept task fits the smallest machine on every resource;
  3. LP-oracle SHIFT (midday share, OOD minus in-dist prices) >= +0.60.

Writes Data/sampleinstances/_sub07/sub<K>_s<seed>/instance.txt. The two-level
nesting keeps behavior_instances.py's one-level glob from picking them up.
"""
import os, sys
import numpy as np
import torch

from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Utils'))

from Utils.Downstream.icon_loader import load_icon_instance
from Utils.Downstream.energy_scheduling import Energy_Scheduling
from Utils.Downstream.opt_oracle import build_oracle

SRC      = f'{ROOT}/Data/sampleinstances/sample07/instance.txt'
OUT_BASE = f'{ROOT}/Data/sampleinstances/_sub07'

# SHIFT probe prices must stay IDENTICAL to behavior_instances.py, so the
# subsample SHIFT compares with the full instance ladder.
T   = 48
H   = np.arange(T) / T * 24.0
MID = (H >= 11) & (H <= 15)
p_in  = np.full(T, 50.0); p_in[(H >= 0) & (H <= 5)] = 20.0; p_in[MID] = 90.0       # midday dear
p_ood = np.full(T, 55.0); p_ood[MID] = -30.0; p_ood[(H >= 17) & (H <= 20)] = 150.0  # midday cheap


def _fmt(x):
    """Write whole numbers as ints, else shortest round-trip float (loader reads
    caps/reqs/power via float(), so int-looking tokens parse fine)."""
    xf = float(x)
    return str(int(round(xf))) if abs(xf - round(xf)) < 1e-9 else repr(xf)


def build_subsample(K, seed, out_dir):
    """Write a K-task verbatim subsample of sample07 (machines UNCHANGED)."""
    inst = load_icon_instance(SRC)
    n, R, M = inst['n_tasks'], inst['n_resources'], inst['n_machines']
    idx = np.sort(np.random.RandomState(seed).choice(n, K, replace=False))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'instance.txt')

    L = [str(int(inst['period_minutes'])), str(int(R)), str(int(M))]
    for m in range(M):                                   # machine block — verbatim
        L.append(f"{m} {_fmt(inst['machine_idle'][m])} "
                 f"{_fmt(inst['machine_startup_cost'][m])} "
                 f"{_fmt(inst['machine_shutdown_cost'][m])}")
        L.append(' '.join(_fmt(c) for c in inst['machine_capacity'][m]))
    L.append(str(int(K)))
    for new_id, j in enumerate(idx):                     # K real tasks — re-indexed
        L.append(f"{new_id} {int(inst['task_duration'][j])} "
                 f"{int(inst['task_earliest_start'][j])} "
                 f"{int(inst['task_latest_end'][j])} {_fmt(inst['task_power'][j])}")
        L.append(' '.join(_fmt(v) for v in inst['task_resources'][j]))
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return path


def single_task_feasible(inst):
    """THE key check: every task fits the smallest machine on every resource."""
    min_cap = inst['machine_capacity'].min(axis=0)               # (R,)
    bad_rows = (inst['task_resources'] > min_cap[None, :] + 1e-9).any(axis=1)
    return (not bad_rows.any()), int(bad_rows.sum())


def shift_of(path):
    """LP-oracle responsiveness (SHIFT) at T=48 aggregated — same as the ladder."""
    task = Energy_Scheduling.from_icon_instance(
        path, price_range=np.array([-1000.0, 17500.0]),
        T_external=T, aggregate_to_external=True)
    orc = build_oracle(task, device='cpu')

    def midday_frac(price):
        w = np.asarray(orc.sensitivity(
            torch.tensor(price[None, :], dtype=torch.float32))).reshape(-1)
        w = np.clip(w, 0, None); tot = w.sum()
        return float(w[MID].sum() / tot) if tot > 1e-9 else 0.0

    fi, fo = midday_frac(p_in), midday_frac(p_ood)
    return fi, fo, fo - fi


def main():
    Ks    = [int(x) for x in os.environ.get('SUB_KS', '50,100,150,200').split(',')]
    seeds = [int(x) for x in os.environ.get('SUB_SEEDS', '0,1,2').split(',')]
    print("=== sample07 subsample CHEAP SCREEN (build + feasibility + LP SHIFT) ===",
          flush=True)
    print(f"  src={SRC}", flush=True)
    print(f"\n  {'K':>5} {'seed':>4} {'parse':>6} {'feasible':>9} {'bad':>4} "
          f"{'SHIFT':>7} {'mid_in':>7} {'mid_ood':>8}", flush=True)
    rows = []
    for K in Ks:
        for s in seeds:
            out_dir = f"{OUT_BASE}/sub{K}_s{s}"
            try:
                path = build_subsample(K, s, out_dir)
                inst2 = load_icon_instance(path)            # round-trip / format check
                assert inst2['n_tasks'] == K, f"n_tasks {inst2['n_tasks']} != {K}"
                feas, nbad = single_task_feasible(inst2)
                fi, fo, sh = shift_of(path)
            except Exception as e:
                print(f"  {K:>5} {s:>4}  ERROR {type(e).__name__}: {e}", flush=True)
                rows.append((K, s, False, -1, float('nan')))
                continue
            print(f"  {K:>5} {s:>4} {'ok':>6} {('OK' if feas else 'INFEAS'):>9} "
                  f"{nbad:>4} {sh:>+7.2f} {fi:>7.2f} {fo:>8.2f}", flush=True)
            rows.append((K, s, feas, nbad, sh))

    print("\n  === SUMMARY (target: feasible AND SHIFT >= +0.60) ===", flush=True)
    ok_overall = True
    for K in Ks:
        good = [r for r in rows if r[0] == K and r[2] and not np.isnan(r[4])]
        shs = [r[4] for r in good]
        n_feas = sum(1 for r in rows if r[0] == K and r[2])
        if shs:
            hi = min(shs) >= 0.60
            print(f"  K={K:>4}: feasible {n_feas}/{len(seeds)}  "
                  f"SHIFT mean={np.mean(shs):+.2f} range[{min(shs):+.2f},{max(shs):+.2f}]"
                  f"  {'PASS' if (n_feas == len(seeds) and hi) else 'CHECK'}", flush=True)
            ok_overall &= (n_feas == len(seeds) and hi)
        else:
            print(f"  K={K:>4}: NO feasible+valid subsample", flush=True)
            ok_overall = False
    print(f"\n  VERDICT: {'ALL PASS — subsample recipe valid' if ok_overall else 'SOME FAILED — inspect above'}",
          flush=True)
    print("######## DONE ########", flush=True)


if __name__ == '__main__':
    main()
