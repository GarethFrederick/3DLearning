"""
Per-checkpoint native-MIP decision regret on the two held-out OOD-TEST regimes.

For every existing checkpoint
Utils/Models/3DLearning_mv_cond_<RUN_PREFIX>_<arm>_s<seed>-ML-epoch<N>.pth
it scores BOTH:
  (a) 'selected': the epoch in the slurm log line "Restored best 3D-Learning
      checkpoint from outer epoch <N>", else the LOWEST saved ML epoch, and
  (b) 'fixed': the HIGHEST saved ML epoch.

Scoring is native T=288 MIP (ICON_AGG=0) via ML.test_task on the sub100 ICON
instance, at EVAL_ORIGIN_STRIDE, on the OOD-TEST regimes (NOT oodval). Only
the ML head is loaded; the DDPM path is recorded for provenance only.

Per (arm, seed, regime, checkpoint-kind) it dumps:
  - the full PER-WINDOW regret vector (real $, ordered by origin),
  - the per-call native-MIP timeout count and total solves,
  - whether any greedy fallback occurred,
to gate_out/poc_eval/<arm>_s<seed>_<regime>_<kind>.json.

Re-runnable: runs without checkpoints are skipped, and so is any existing
result JSON, error JSONs included (POC_EVAL_FORCE=1 recomputes). One failed
checkpoint never kills the sweep.
"""
import os, sys, re, glob, json, traceback
import numpy as np
import pandas as pd
import torch

# REPO_ROOT points the scorer at another checkout; unset = this file's folder.
# Analysis scripts read P.ROOT, so the name must stay ROOT and a str.
ROOT = os.environ.get('REPO_ROOT') or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'Utils'))
from Utils.Algos.ml import ML
from Utils.Downstream.energy_scheduling import Energy_Scheduling
from Utils.datasets import Data_Loader, build_ml_features
from Utils.utils import set_global_seed

DEVICE = 'cpu'; SEED = 68268
set_global_seed(SEED)

# The skip-if-present rule reads this directory; set a new one to score twice.
OUT_DIR = os.environ.get('POC_EVAL_OUT_DIR', f'{ROOT}/gate_out/poc_eval')
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = f'{ROOT}/Utils/Models'

WINDOW_LEN = PREDICT_LEN = 48
ML_SEQ_LEN = 192
BATCH_SIZE = 64

# EVAL_FULL_COVERAGE=1 keeps every four-day block. Unset, the loader drops the
# last incomplete batch: floor(N/BATCH_SIZE) x BATCH_SIZE blocks, tail lost.
# Unset is the path behind every reported AEMO number. NEVER MIX THE TWO
# SETTINGS IN ONE COMPARISON; each result JSON carries full_coverage.
FULL_COVERAGE = os.environ.get('EVAL_FULL_COVERAGE', '0') == '1'
if FULL_COVERAGE:
    # Anti-mix guard: the filename carries no coverage field. Old scores have no
    # full_coverage key, so a missing key counts as the old setting.
    import glob as _glob
    import json as _json
    _mixed = []
    for _p in _glob.glob(f'{OUT_DIR}/*.json'):
        try:
            with open(_p) as _fh:
                if _json.load(_fh).get('full_coverage', False) is not True:
                    _mixed.append(os.path.basename(_p))
        except Exception:
            continue
    if _mixed:
        raise SystemExit(
            f'REFUSING TO RUN: EVAL_FULL_COVERAGE=1 but {OUT_DIR} already holds '
            f'{len(_mixed)} score(s) from the old 64-block setting, e.g. '
            f'{_mixed[:3]}. Point POC_EVAL_OUT_DIR at a new directory. '
            f'PREREGISTRATION.md §8.73 amendment 1 bars mixing the two.')
CAL_COLS = ['sin_h', 'cos_h', 'sin_dow', 'cos_dow', 'sin_m', 'cos_m']

RUN_PREFIX = os.environ.get('POC_RUN_PREFIX', 'poc')
# An arm missing from this list is a SILENT omission (exit 0), so main()
# prints the resolved ARMS. Arms without checkpoints are skipped.
_DEFAULT_ARMS = ('div,regret,random,off,flat,noaug,transport,retrainreal'
                 if RUN_PREFIX == 'poc'
                 else 'a1_stall,gt_headonly,a3_leashon,a4_retrainreal,'
                      'hd_dis_valid,hd_dis_inval,a3_full,a5_consolidate,'
                      'a6_instructor,c1_legalnoise,c2_noreteach,a8_retain,'
                      'a9_retain,a10_typical,a11_instrhead,a12_instrbroad,'
                      'a13_instrwide,a14_instrcentroid,a15_wavesuper,a16_wavebank,a17_pfnregime,'
                      'a18_realbase,a19_discwidth,a20_nogen,a21_discdepth,'
                      # a22/a24/a25/a26 without _r2 ran on the wrong instance:
                      # NEVER quote them; quote the _r2 runs.
                      'a22_nogendepth,a24_trendaim,a25_recency,a26_aimbank,'
                      'a22_nogendepth_r2,a23_zerodose_r2,a24_trendaim_r2,'
                      'a25_recency_r2,a26_aimbank_r2,a27_trenddepth_r2,a28_aimfilter,'
                      'a29_wassfilter,a29b_wassfilter,a30_recentdisc,a31_dose,'
                      'a32_ceilingdose,a33_roundsonly,a34_honestdose,a35_dose400,a36_oracle,'
                      # a1_stall / a36_oracle are superseded: quote the _r2 runs.
                      'a1_stall_r2,a36_oracle_r2,'
                      'a37_multitie,a38_widthtie,a40_tiememory,'
                      'a41_aimmulti,a42_a1dose,'
                      # a43 only as _r2: the bare name has no files and would
                      # skip in silence (run names match exactly, not by prefix).
                      'a43_gridfree_r2,'
                      'a44_long30,a46_nogen30,'
                      # dfl_warmstart is NOT an arm: its checkpoint is a SYMLINK
                      # to Utils/Models/DFL-epoch41.pth, the shared warm start.
                      # Score it with POC_SEEDS=1 only; it has nothing to pair.
                      'dfl_warmstart,'
                      'a47_prob025,a48_prob075,a49_prob100,'
                      'a50_additive,a51_loocvdepth,a52_widthexp,a53_replay,'
                      'a54_coldstart,'
                      'a55_twolobe,a56_width04,a57_width08,a58_width12,'
                      'a59_drospo,'
                      'a60_hilr30,a61_nodecay30,'
                      'a66_horizon2022,a67_horizon2026,a68_derivedsign')
ARMS  = [a.strip() for a in os.environ.get('POC_ARMS', _DEFAULT_ARMS).split(',') if a.strip()]
SEEDS = [int(s) for s in os.environ.get('POC_SEEDS', '1,2,3,4,5').split(',') if s.strip()]

ICON_INSTANCE = os.environ.get(
    'ICON_INSTANCE', f'{ROOT}/Data/sampleinstances/_sub07/sub100_s0/instance.txt')

# Held-out OOD-TEST regimes (NOT oodval). DATA_OODTEST_A/B and DATA_TRAIN let
# a second domain be scored; defaults are the AEMO paths.
OOD_REGIMES = [
    (os.environ.get('DATA_OODTEST_A_LBL', 'oodtest_2021_2022_gas'),
     os.environ.get('DATA_OODTEST_A',
                    f'{ROOT}/Data/AEMO_VIC1_oodtest_2021_2022.csv')),
    (os.environ.get('DATA_OODTEST_B_LBL', 'oodtest_2023_2025_renewables'),
     os.environ.get('DATA_OODTEST_B',
                    f'{ROOT}/Data/AEMO_VIC1_oodtest_2023_2025.csv')),
]
# POC_EVAL_REGIMES filters the list; unset keeps both. An unknown name must
# crash: an empty regime list would report a clean pass.
_want = [r.strip() for r in os.environ.get('POC_EVAL_REGIMES', '').split(',') if r.strip()]
if _want:
    _known = {name for name, _ in OOD_REGIMES}
    _bad = [r for r in _want if r not in _known]
    if _bad:
        raise SystemExit(f'POC_EVAL_REGIMES names unknown regimes {_bad}; known: {sorted(_known)}')
    OOD_REGIMES = [(n, c) for n, c in OOD_REGIMES if n in _want]

FORCE = os.environ.get('POC_EVAL_FORCE', '0') == '1'


# Feature subset per detected input dim F.
def feature_cols_for(F):
    if F >= 8:  return ['Price', 'TOTALDEMAND_MW'] + CAL_COLS
    if F == 2:  return ['Price', 'TOTALDEMAND_MW']
    return ['Price']


def train_fold_bounds():
    """70%-train-FOLD bounds, the de-normalisation the heads were trained with.
    Full-file quantiles silently mis-scale every prediction and the regret."""
    train = pd.read_csv(
        os.environ.get('DATA_TRAIN', f'{ROOT}/Data/AEMO_VIC1_train_2015_2020.csv'),
        parse_dates=['TIMESTAMP'])
    feat = build_ml_features(train)
    _n = int(0.70 * len(feat))
    pf = feat['Price'].iloc[:_n]; dfd = feat['TOTALDEMAND_MW'].iloc[:_n]
    price_b = (float(pf.quantile(0.01)), float(pf.quantile(0.99)))
    demand_b = (float(dfd.min()), float(dfd.max()))
    return price_b, demand_b


def detect_F(head_path):
    m = torch.load(head_path, weights_only=False, map_location=DEVICE)
    return int(getattr(getattr(m, 'lstm1', None), 'input_size', 0)) or None


def run_basename(arm, seed):
    return f'3DLearning_mv_cond_{RUN_PREFIX}_{arm}_s{seed}'


def saved_ml_epochs(arm, seed):
    """Sorted list of saved ML-head epochs for a run, [] if none exist."""
    base = run_basename(arm, seed)
    eps = []
    for p in glob.glob(f'{MODELS_DIR}/{base}-ML-epoch*.pth'):
        m = re.search(r'-ML-epoch(\d+)\.pth$', os.path.basename(p))
        if m:
            eps.append(int(m.group(1)))
    return sorted(set(eps))


def find_selected_epoch(arm, seed, available_epochs):
    """Resolve the SELECTED (best-OODval) epoch from the matching slurm log.

    Each fold-in slurm log prints 'SUFFIX=_poc_<arm>_s<seed>' and later
    'Restored best 3D-Learning checkpoint from outer epoch <N>'. We match the
    log by its SUFFIX line so we don't depend on the slurm job-id mapping.

    Fallback (no log / no line): the LOWEST saved epoch. Returns
    (epoch, source_str), source 'slurm_log:<basename>' or
    'fallback:lowest_saved_epoch'. The epoch snaps to the nearest SAVED
    checkpoint epoch."""
    suffix = f'_{RUN_PREFIX}_{arm}_s{seed}'
    best_ep = None; src = 'fallback:lowest_saved_epoch'
    # Search the repo root and results/logs/train/, newest first.
    _logs = (glob.glob(f'{ROOT}/slurm-*.out')
             + glob.glob(f'{ROOT}/results/logs/train/slurm-*.out'))
    for log in sorted(_logs, key=os.path.getmtime, reverse=True):
        try:
            with open(log, 'r', errors='ignore') as fh:
                txt = fh.read()
        except OSError:
            continue
        # Boundary-aware match, so the s1 query cannot match an s10 log.
        if not re.search(rf'(?:MV_RUN_)?SUFFIX={re.escape(suffix)}(?![A-Za-z0-9_])', txt):
            continue
        matches = re.findall(
            r'Restored best 3D-Learning checkpoint from outer epoch (\d+)', txt)
        if matches:
            best_ep = int(matches[-1])
            src = f'slurm_log:{os.path.basename(log)}'
            break
    if best_ep is None:
        best_ep = available_epochs[0]
    # Snap to the nearest saved checkpoint epoch.
    snapped = min(available_epochs, key=lambda e: (abs(e - best_ep), e))
    if snapped != best_ep:
        src += f'(snapped {best_ep}->{snapped})'
    return snapped, src


def build_task():
    """Native-T=288 MIP task on ICON_INSTANCE (ICON_AGG=0), with MIP_TL override."""
    task = Energy_Scheduling.from_icon_instance(
        instance_path=ICON_INSTANCE,
        price_range=np.array([-1000.0, 17500.0]),
        T_external=PREDICT_LEN, seed=SEED,
        aggregate_to_external=(os.environ.get('ICON_AGG', '0') == '1'))
    if os.environ.get('MIP_TL'):
        task.MIP_TIMELIMIT = float(os.environ['MIP_TL'])
        print(f"  | MIP_TIMELIMIT override = {task.MIP_TIMELIMIT}s (MIP_TL env)",
              flush=True)
    return task


def eval_one(ml, task, head_path, ddpm_path, norm_full, denorm_range,
             arm, seed, regime_label, csv, kind, epoch):
    """Run native-MIP test_task for one (checkpoint, regime); return a result
    dict. Greedy/timeout counts are per-call deltas. The caller catches
    exceptions."""
    F = detect_F(head_path)
    cols = feature_cols_for(F or 8)
    feat = build_ml_features(pd.read_csv(csv, parse_dates=['TIMESTAMP']))[cols]
    nb = {k: v for k, v in norm_full.items() if k in cols}
    loader = Data_Loader(df=feat).ml_set2loader_mv(
        split_test=False, normlized=True, device=DEVICE,
        batch_size=BATCH_SIZE, pic_size=ML_SEQ_LEN, norm_bounds=nb,
        drop_last=not FULL_COVERAGE)

    # Print the blocks the loader yields, not the knob.
    _total = len(loader.dataset)
    _kept = len(loader) * BATCH_SIZE if (not FULL_COVERAGE) else _total
    print(f'  | [COVERAGE] blocks total={_total}  scored={_kept}  '
          f'dropped={_total - _kept}  '
          f'share_lost={100.0 * (_total - _kept) / max(_total, 1):.1f}%  '
          f'full_coverage={int(FULL_COVERAGE)}', flush=True)

    # Fresh oracle cache per regime, or a schedule from the other CSV is served.
    if hasattr(task, '_oracle_cache'):
        del task._oracle_cache

    g0 = int(getattr(task, '_greedy_solves', 0))
    to0 = int(getattr(task, '_mip_total_timeouts', 0))
    ti0 = int(getattr(task, '_mip_total_instances', 0))

    out = ml.test_task(loader, head_path, task, denorm_range)

    g1 = int(getattr(task, '_greedy_solves', 0))
    to1 = int(getattr(task, '_mip_total_timeouts', 0))
    ti1 = int(getattr(task, '_mip_total_instances', 0))

    reg = [float(x) for x in out.get('regret_per_instance', [])]
    res = {
        'arm': arm, 'seed': seed, 'regime': regime_label,
        'checkpoint_kind': kind, 'epoch': epoch,
        'head_path': head_path, 'ddpm_path': ddpm_path,
        'input_F': F, 'features': cols,
        'instance': ICON_INSTANCE,
        'eval_origin_stride': int(os.environ.get('EVAL_ORIGIN_STRIDE', '1')),
        'icon_agg': os.environ.get('ICON_AGG', '0'),
        'mip_tl': os.environ.get('MIP_TL'),
        'n_windows': len(reg),
        'full_coverage': FULL_COVERAGE,
        'regret_per_window': reg,                 # real-$, ordered by origin
        'norm_regret': out.get('Norm Regret'),
        'task_regret_mean': out.get('Task Regret'),
        'mse_loss': out.get('MSE Loss'),
        'actual_cost': out.get('Actual Cost'),
        'mip_timeouts': to1 - to0,
        'mip_solves': ti1 - ti0,
        'greedy_fallback': bool(g1 - g0 > 0),
        'greedy_solves': g1 - g0,
    }
    return res


def main():
    print("=== POC OOD EVAL — native-MIP per-window regret, 2 OOD-test regimes ===",
          flush=True)
    # Print the resolved scope so an omitted arm shows in the log.
    print(f"  | RUN_PREFIX = {RUN_PREFIX}  ARMS = {','.join(ARMS)}  "
          f"SEEDS = {SEEDS}", flush=True)
    print(f"  | instance = {ICON_INSTANCE}", flush=True)
    print(f"  | EVAL_ORIGIN_STRIDE = {os.environ.get('EVAL_ORIGIN_STRIDE','1')}  "
          f"ICON_AGG = {os.environ.get('ICON_AGG','0')}  "
          f"MIP_TL = {os.environ.get('MIP_TL')}", flush=True)
    if not os.path.exists(ICON_INSTANCE):
        print(f"  !! ICON instance not found: {ICON_INSTANCE} — abort", flush=True)
        sys.exit(2)

    price_b, demand_b = train_fold_bounds()
    norm_full = {'Price': price_b, 'TOTALDEMAND_MW': demand_b}
    denorm_range = np.array([price_b[0], price_b[1]])
    print(f"  | price bounds (p1,p99)={price_b}  demand={demand_b}", flush=True)

    task = build_task()
    ml = ML(windows_len=WINDOW_LEN, predict_len=PREDICT_LEN, epoches=1,
            discount_factor=1.0, step_size=1, lr=1e-3, device=DEVICE,
            save_every=1, display_every=1, name='poc_eval')

    n_done = n_skip_missing = n_skip_cached = n_fail = 0
    for arm in ARMS:
        for seed in SEEDS:
            eps = saved_ml_epochs(arm, seed)
            if not eps:
                n_skip_missing += 1
                continue   # checkpoints not landed yet — re-run later
            base = run_basename(arm, seed)
            sel_ep, sel_src = find_selected_epoch(arm, seed, eps)
            fixed_ep = eps[-1]   # highest saved epoch = the fixed later checkpoint
            # kind -> epoch (dedupe if selected == fixed, e.g. only one saved epoch)
            kinds = {'selected': sel_ep}
            if fixed_ep != sel_ep:
                kinds['fixed'] = fixed_ep
            print(f"\n########## {arm} s{seed} | saved epochs={eps} | "
                  f"selected=ep{sel_ep} ({sel_src}) | fixed=ep{fixed_ep} ##########",
                  flush=True)

            for kind, epoch in kinds.items():
                head_path = f'{MODELS_DIR}/{base}-ML-epoch{epoch}.pth'
                ddpm_path = f'{MODELS_DIR}/{base}-DDPM-epoch{epoch}.pth'
                if not os.path.exists(ddpm_path):
                    ddpm_path = None  # provenance only; not loaded by test_task
                if not os.path.exists(head_path):
                    print(f"  | {kind} ep{epoch}: ML head missing -> skip", flush=True)
                    continue
                for rlabel, csv in OOD_REGIMES:
                    out_path = f'{OUT_DIR}/{arm}_s{seed}_{rlabel}_{kind}.json'
                    if os.path.exists(out_path) and not FORCE:
                        n_skip_cached += 1
                        print(f"  | {kind}/{rlabel}: cached -> skip "
                              f"({os.path.basename(out_path)})", flush=True)
                        continue
                    try:
                        res = eval_one(ml, task, head_path, ddpm_path, norm_full,
                                       denorm_range, arm, seed, rlabel, csv,
                                       kind, epoch)
                        res['selected_epoch_source'] = sel_src
                        res['saved_epochs'] = eps
                        json.dump(res, open(out_path, 'w'), indent=2, default=str)
                        n_done += 1
                        print(f"  | {kind}/{rlabel}: NormRegret="
                              f"{res['norm_regret']}  n_win={res['n_windows']}  "
                              f"timeouts={res['mip_timeouts']}/{res['mip_solves']}  "
                              f"greedy={res['greedy_fallback']}  -> "
                              f"{os.path.basename(out_path)}", flush=True)
                    except Exception as e:
                        n_fail += 1
                        # The error JSON takes out_path, so a rerun skips this
                        # checkpoint unless POC_EVAL_FORCE=1.
                        err = {
                            'arm': arm, 'seed': seed, 'regime': rlabel,
                            'checkpoint_kind': kind, 'epoch': epoch,
                            'head_path': head_path,
                            'error': f'{type(e).__name__}: {e}',
                            'traceback': traceback.format_exc(),
                        }
                        json.dump(err, open(out_path, 'w'), indent=2, default=str)
                        print(f"  | {kind}/{rlabel}: FAILED "
                              f"({type(e).__name__}: {e}) -> wrote error JSON",
                              flush=True)

    print(f"\n======== POC OOD EVAL DONE ========", flush=True)
    print(f"  | results written : {n_done}", flush=True)
    print(f"  | cached (skipped) : {n_skip_cached}", flush=True)
    print(f"  | runs not yet available (skipped): {n_skip_missing}", flush=True)
    print(f"  | failed checkpoints: {n_fail}", flush=True)
    print(f"  | output dir       : {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
