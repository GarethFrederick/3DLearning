"""Multivariate energy-scheduling driver: 3D-Learning on the 1D DDPM.

The DDPM checkpoint and its _meta.json are read-only here.

Stages (STAGE_MV env): genmv | 3dl_mv | cond_mv | all
  genmv   — sample z0 (price+demand, real units) + reverse trajectory (model space)
  3dl_mv  — F=2 3D-Learning with the mv DDPM as the worst-case generator
  cond_mv — F=8 conditional 3D-Learning (the production path)

Epoch/iteration budgets are env-overridable for a cheap smoke run:
  MV_GEN_ITERS, MV_DRO_ITERS, MV_DRO_INNER, MV_ML_INNER, MV_ADJUST
"""
import sys, os, json
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Utils')
# Utils/ must be on sys.path: Utils/diffusion_utils.py does bare
# `import utils` / `import datasets`.
sys.path.append(project_root)
import numpy as np
import pandas as pd
import torch

from Utils.diffusion_utils import Inference_DFU_MV, diffusion_name_mv
from Utils.Algos.ddd_learning import DDD_Learning, algo_name as ddd_name
from Utils.Algos.ml import ML, algo_name as ml_name
from Utils.Downstream.energy_scheduling import Energy_Scheduling
from Utils.utils import Utils, set_global_seed, Log_Redirect, Tee
from Utils.datasets import Data_Loader, build_ml_features
from Utils.config import Config


def _make_live_plot_cb(log_folder, **name_kwargs):
    """callback(**series) that redraws the 3DL training-timeline PNG each outer epoch."""
    def _cb(**series):
        Utils().draw_3dl_training_plot(log_folder_path=log_folder, **name_kwargs, **series)
    return _cb

torch.cuda.empty_cache()
# CPU intra-op threads. Default 1 so torch does not fight Gurobi's workers.
if not torch.cuda.is_available():
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
cfg = Config()
SEED = int(os.environ.get('MV_SEED', '42'))
set_global_seed(SEED)
print(f"  | SEED | {SEED} (MV_SEED env)", flush=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
STAGE = os.environ.get('STAGE_MV', 'all').lower()
_VALID = {'genmv', '3dl_mv', 'cond_mv', 'all'}
if STAGE not in _VALID:
    raise ValueError(f"STAGE_MV must be one of {sorted(_VALID)}, got {STAGE!r}")
COND = (STAGE == 'cond_mv')

#   MV_WARM_SOURCE — classical checkpoint to warm-start the ML head from
#                    (default DFL; COLD* = intentional cold start).
#   MV_RUN_SUFFIX  — appended to RUN_NAME / metadata tag / log folder so a
#                    variant never overwrites another run's files.
WARM_SOURCE = os.environ.get('MV_WARM_SOURCE', 'DFL')
DRO_LR      = 1e-6
BUDGET      = 0.030  # ambiguity radius
RUN_SUFFIX  = os.environ.get('MV_RUN_SUFFIX', '')
# DDPM tag ('' = production, e.g. '_wide'). It tags the checkpoint, meta AND
# z0 pool so the generator and its samples come from one model.
DDPM_TAG    = os.environ.get('MV_DDPM_TAG', '')
print(f"  | DEVICE | {DEVICE}  | STAGE_MV | {STAGE}  | COND | {COND}"
      f"  | WARM_SOURCE | {WARM_SOURCE}  | DRO_LR | {DRO_LR}  | BUDGET | {BUDGET}"
      f"  | RUN_SUFFIX | {RUN_SUFFIX!r}", flush=True)

# --- paths / static config ---
MODEL_SAVE    = f'{project_root}/Models'
GEN_DATA_SAVE = f'{project_root}/Gen_Data'
RESULT_SAVE   = f'{project_root}/Results'

# build_log_file reads STAGE (not STAGE_MV) to name the folder
# Results/{device}/{STAGE}_{JOBID}/.
os.environ['STAGE'] = ('3DL_MV' if STAGE == '3dl_mv'
                       else '3DL_MV_COND' if STAGE == 'cond_mv'
                       else STAGE) + RUN_SUFFIX.upper()
log_folder_path, log_file_path = Log_Redirect().build_log_file(result_path=RESULT_SAVE, cuda=DEVICE)
sys.stdout = Tee(log_file_path)
print(f"  | log + plots -> {log_folder_path}", flush=True)
# DATA_TRAIN / DATA_OODVAL_A / DATA_OODVAL_B / PRICE_RANGE_LO / PRICE_RANGE_HI
# let a second domain reuse this driver. Every default is the AEMO value.
TRAIN_FILE    = os.environ.get('DATA_TRAIN',
                               './Data/AEMO_VIC1_train_2015_2020.csv')
ML_SEQ_LEN    = 192
WINDOW_LEN, PREDICT_LEN = 48, 48
BATCH_SIZE    = 64
CAL_COLS      = ['sin_h', 'cos_h', 'sin_dow', 'cos_dow', 'sin_m', 'cos_m']
FEATURE_COLS  = (['Price', 'TOTALDEMAND_MW'] + CAL_COLS) if COND else ['Price', 'TOTALDEMAND_MW']
RUN_NAME      = ('3DLearning_mv_cond' if COND else '3DLearning_mv') + RUN_SUFFIX

# F=8 uses the conditional DDPM (n_cond=6) from train_ddpm_mv_cond.py.
if COND:
    MV_CKPT      = f'{MODEL_SAVE}/model_DDPM_mv_F8_L192_T500{DDPM_TAG}_best.pth'
    MV_META_PATH = f'{MODEL_SAVE}/model_DDPM_mv_F8_L192_T500{DDPM_TAG}_meta.json'
    N_FEATURES, MV_SEQ_LEN = 8, 192
else:
    MV_CKPT      = f'{MODEL_SAVE}/model_DDPM_mv_F2_L192_T500_best.pth'
    MV_META_PATH = f'{MODEL_SAVE}/model_DDPM_mv_F2_L192_T500_meta.json'
    N_FEATURES, MV_SEQ_LEN = 2, 192
T, BETA_1, BETA_T = 500, 5e-2, 1e-1

# Per-stage metadata file: parallel jobs read-modify-write it, and a shared
# file would let them overwrite each other's keys.
_meta_tag = ('' if STAGE == 'all' else f'_{STAGE}') + RUN_SUFFIX
META_PATH = os.path.join(MODEL_SAVE, f'training_metadata_mv{_meta_tag}.json')

GEN_ITERS = int(os.environ.get('MV_GEN_ITERS', '4'))
DRO_ITERS = int(os.environ.get('MV_DRO_ITERS', '15'))
DRO_INNER = int(os.environ.get('MV_DRO_INNER', '10'))
ML_INNER  = int(os.environ.get('MV_ML_INNER', '5'))
ADJUST    = int(os.environ.get('MV_ADJUST', '10'))
PLOT_JSM  = False


def _load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def _save_meta(m):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, 'w') as f:
        json.dump(m, f, indent=2)


# Train-fold normalisation bounds.
_train_full = pd.read_csv(TRAIN_FILE, parse_dates=['TIMESTAMP'], date_format='mixed', dayfirst=False)
_n = int(0.70 * len(_train_full))
_price_fold  = _train_full['Price'].iloc[:_n]
_demand_fold = _train_full['TOTALDEMAND_MW'].iloc[:_n]
_train_norm_bounds  = (float(_price_fold.quantile(0.01)), float(_price_fold.quantile(0.99)))
_demand_norm_bounds = (float(_demand_fold.min()), float(_demand_fold.max()))
_DENORM_RANGE = np.array([_train_norm_bounds[0], _train_norm_bounds[1]])
_MV_LOADER_NORM_BOUNDS = {'Price': _train_norm_bounds, 'TOTALDEMAND_MW': _demand_norm_bounds}
demand_stats = {
    'mean_mwh':  float(_demand_fold.mean() * 0.5),
    'price_p25': float(_price_fold.quantile(0.25)),
    'price_p75': float(_price_fold.quantile(0.75)),
}
print(f"  | price bounds (p1,p99) | {_train_norm_bounds}  | demand bounds (min,max) | {_demand_norm_bounds}", flush=True)

# Calendar cols have no norm_bounds entry, so they pass through unnormalised.
# The train-fold slice feeds the 3DL DRO s0.
_feat2      = build_ml_features(_train_full)[FEATURE_COLS]
_feat2_fold = _feat2.iloc[:_n]

# Mean $/MWh by half-hour over the train fold, length PREDICT_LEN. Guided
# sampling (GUIDE_DIVERGENCE=1) steers scenarios away from w*(this profile).
_hh = (_train_full['TIMESTAMP'].dt.hour * 2
       + _train_full['TIMESTAMP'].dt.minute // 30).iloc[:_n]
_prior_price_profile = (
    _train_full['Price'].iloc[:_n].groupby(_hh).mean()
    .reindex(range(PREDICT_LEN)).interpolate().bfill().ffill().to_numpy()
)
print(f"  | prior price profile (night-cheap) | "
      f"min={_prior_price_profile.min():.1f} max={_prior_price_profile.max():.1f} "
      f"argmin_hh={int(_prior_price_profile.argmin())} "
      f"argmax_hh={int(_prior_price_profile.argmax())} |", flush=True)

ml_dl = Data_Loader(df=_feat2)
ml_train_loader, ml_val_loader, ml_test_loader = ml_dl.ml_set2loader_mv(
    split_test=True, val_size=0.15, normlized=True, device=DEVICE,
    batch_size=BATCH_SIZE, pic_size=ML_SEQ_LEN, norm_bounds=_MV_LOADER_NORM_BOUNDS,
)

# The SPO+ loss is divided by (HI - LO) in energy_scheduling.py. It is a LOSS
# SCALE, not a denormaliser (that uses the train fold's p1/p99). A domain whose
# target spans ~500, not ~18500, must set these or its gradient is ~40x small.
_PRICE_RANGE = np.array([float(os.environ.get('PRICE_RANGE_LO', '-1000.0')),
                         float(os.environ.get('PRICE_RANGE_HI', '17500.0'))])
print(f'  | price_range (SPO+ loss scale) = {_PRICE_RANGE.tolist()}', flush=True)

# Downstream task: ICON job-scheduling MIP.
energy_task = Energy_Scheduling.from_icon_instance(
    instance_path=os.environ.get('ICON_INSTANCE',
                                 'Data/sampleinstances/sample01/instance.txt'),
    price_range=_PRICE_RANGE,
    T_external=PREDICT_LEN, seed=SEED,
    # ICON_AGG=1 (default): aggregated T=48 LP for training. ICON_AGG=0:
    # native T=288 MIP for scoring.
    aggregate_to_external=(os.environ.get('ICON_AGG', '1') == '1'),
)

with open(MV_META_PATH) as f:
    mv_meta = json.load(f)

z0_path      = f'{GEN_DATA_SAVE}/z0_mv_F{N_FEATURES}_L{MV_SEQ_LEN}_T{T}{DDPM_TAG}.csv'
z0_traj_path = f'{GEN_DATA_SAVE}/z0_mv_F{N_FEATURES}_L{MV_SEQ_LEN}_T{T}{DDPM_TAG}_trajectory.npy'


# Stage: generate mv z0 + trajectory.
if STAGE in ('genmv', 'all', 'cond_mv'):
    if os.path.exists(z0_path) and os.path.exists(z0_traj_path):
        print(f"  [skip] z0 already exists at {z0_path}", flush=True)
    else:
        os.makedirs(GEN_DATA_SAVE, exist_ok=True)
        _n_cond = int(mv_meta.get('n_cond', 0))
        _cond_pool = None
        if _n_cond > 0:
            _cond_pool = np.load(mv_meta['cond_pool_path']).astype(np.float32)
            print(f"  | cond_pool | shape={_cond_pool.shape}", flush=True)
        Inference_DFU_MV(
            T=T, beta_1=BETA_1, beta_T=BETA_T, seq_len=MV_SEQ_LEN,
            n_features=N_FEATURES, batch_size=BATCH_SIZE, device=DEVICE, meta=mv_meta,
            n_cond=_n_cond, cond_pool=_cond_pool,
        ).generate_wholeset_data(
            model_path=MV_CKPT, total_iteration=GEN_ITERS,
            save_gen_data_path=z0_path, save_trajectory_path=z0_traj_path,
        )


# Stage: 3D-Learning. A run whose <RUN>_final_epoch key exists is skipped.
if STAGE in ('3dl_mv', 'cond_mv', 'all'):
    meta = _load_meta()
    _epoch_key = f'{RUN_NAME}_final_epoch'
    if _epoch_key in meta:
        print(f"  [skip] {RUN_NAME} (final_epoch={meta[_epoch_key]})", flush=True)
    else:
        if not (os.path.exists(z0_path) and os.path.exists(z0_traj_path)):
            raise SystemExit("ERROR: z0 artefacts missing — run STAGE_MV=genmv first.")
        z0_df   = pd.read_csv(z0_path)[FEATURE_COLS]
        # The array is (n_iter, T=500, B, F, L), but build_a0/r_theta read only
        # axis-1 indices 0..ADJUST-1 (the read [-1] at idx=0 is overwritten by
        # the `if idx == 0` guard). Slicing cuts ~2 GB to ~40 MB.
        z0_traj = torch.tensor(
            np.ascontiguousarray(np.load(z0_traj_path, mmap_mode='r')[:, :ADJUST]),
            dtype=torch.float32, device=DEVICE)
        print(f"  | z0 rows | {len(z0_df)}  | trajectory shape | {tuple(z0_traj.shape)} "
              f"(sliced axis-1 to ADJUST={ADJUST})", flush=True)
        z0_timesteps = list(range(0, T, 1))

        ddd = DDD_Learning(
            device=DEVICE, discount_factor=0.5, batch_size=BATCH_SIZE, pic_size=28,
            epoches=DRO_ITERS, ppo_clip=0.4, display_every=1, dro_lr=DRO_LR, step_size=10,
            adjust_timesteps=ADJUST, mu=1, eta=0.01, budget=BUDGET, p_s0=0.3,
            ml_save_every=2, dfu_save_every=2, batch_repeat=4,
            beta_1=BETA_1, beta_T=BETA_T, T=T, dfu_name=diffusion_name_mv,
            selected_timesteps=z0_timesteps, windows_len=WINDOW_LEN, predict_len=PREDICT_LEN,
            # Starting head LR. StepLR(step_size=10, gamma=0.5) steps once per
            # INNER epoch: with MV_ML_INNER=5 it halves every 2 outer rounds.
            ml_lr=1e-3,
            ml_seq_len=ML_SEQ_LEN, run_name=RUN_NAME,
            dro_inner_epochs=DRO_INNER, ml_inner_epochs=ML_INNER,
            n_features=N_FEATURES, mv_seq_len=MV_SEQ_LEN, mv_meta=mv_meta,
            mv_loader_norm_bounds=_MV_LOADER_NORM_BOUNDS, mv_stride=4,
        )
        # Warm-start the ML head from {MV_WARM_SOURCE}_final_epoch in the
        # classical metadata. COLD* means an intentional cold start; ANY other
        # value MUST resolve, so a typo crashes instead of running cold.
        _ml_warm = None
        _is_cold = WARM_SOURCE.upper().startswith('COLD')
        _cls_meta_path = os.path.join(MODEL_SAVE, 'training_metadata.json')
        if not _is_cold:
            assert os.path.exists(_cls_meta_path), (
                f"3DL warm-start: MV_WARM_SOURCE='{WARM_SOURCE}' but "
                f"{_cls_meta_path} is missing. Use MV_WARM_SOURCE=COLD for an "
                f"intentional cold start.")
            with open(_cls_meta_path) as _f:
                _ws_ep = json.load(_f).get(f'{WARM_SOURCE}_final_epoch')
            _cand = (os.path.join(MODEL_SAVE, f'{WARM_SOURCE}-epoch{_ws_ep}.pth')
                     if _ws_ep is not None else None)
            assert _cand is not None and os.path.exists(_cand), (
                f"3DL warm-start: MV_WARM_SOURCE='{WARM_SOURCE}' did not resolve "
                f"(no '{WARM_SOURCE}_final_epoch' in metadata or ckpt missing). "
                f"Fix the source name, or use MV_WARM_SOURCE=COLD for a cold start.")
            _ml_warm = _cand
        print(f"  | 3DL ML-head warm-start | source={WARM_SOURCE}  | "
              f"{_ml_warm or 'COLD (explicit)'}", flush=True)

        # VAL_OOD_REGRET=1: pick the epoch on worst-case oodval regret, with the
        # in-dist constraint OOD_VAL_EPSILON, as in ml.py. oodtest stays held
        # out. Default: in-dist selection.
        _val_regret_loaders = None
        if os.environ.get('VAL_OOD_REGRET') == '1':
            _val_regret_loaders = []
            for _lbl, _p in [
                    (os.environ.get('DATA_OODVAL_A_LBL', 'oodval2122'),
                     os.environ.get('DATA_OODVAL_A',
                                    './Data/AEMO_VIC1_oodval_2021_2022.csv')),
                    (os.environ.get('DATA_OODVAL_B_LBL', 'oodval2325'),
                     os.environ.get('DATA_OODVAL_B',
                                    './Data/AEMO_VIC1_oodval_2023_2025.csv'))]:
                _ofeat = build_ml_features(pd.read_csv(_p, parse_dates=['TIMESTAMP']))[FEATURE_COLS]
                _oloader = Data_Loader(df=_ofeat).ml_set2loader_mv(
                    split_test=False, normlized=True, device=DEVICE,
                    batch_size=BATCH_SIZE, pic_size=ML_SEQ_LEN,
                    norm_bounds=_MV_LOADER_NORM_BOUNDS,
                )
                _val_regret_loaders.append((_lbl, _oloader))
            print(f"  | VAL_OOD_REGRET=1 | 3DL worst-case epoch selection over "
                  f"{[l for l, _ in _val_regret_loaders]} (ε={os.environ.get('OOD_VAL_EPSILON','unset')}; oodtest held out)",
                  flush=True)

        fe = ddd.train(
            ml_model_path=_ml_warm, dfu_model_path=MV_CKPT,
            z0=z0_df, s0=_feat2_fold, z0_trajectory=z0_traj,
            model_save_folder_path=MODEL_SAVE, task=energy_task,
            val_task=energy_task, val_data_loader=ml_val_loader,
            val_regret_loaders=_val_regret_loaders,
            data_range=_DENORM_RANGE,
            prior_price_profile=_prior_price_profile,
            live_plot_callback=_make_live_plot_cb(
                log_folder_path, dro_name=RUN_NAME,
                diffusion_name=diffusion_name_mv, ml_name=ml_name, show_jsm=PLOT_JSM),
        )
        Utils().draw_3dl_training_plot(
            log_folder_path=log_folder_path,
            dro_x=ddd.dro_x, dro_y=ddd.dro_y, dro_labels=ddd.dro_labels, jsm_y=ddd.jsm_y,
            ml_x=ddd.ml_x, ml_y=ddd.ml_y, ml_labels=ddd.ml_labels,
            val_loss_list=ddd.epoches_val_losses_list,
            val_regret_list=ddd.epoches_val_regret_list,
            oodval_loss_list=ddd.epoches_oodval_losses_list,
            oodval_regret_list=ddd.epoches_oodval_regret_list,
            boundaries=ddd.outer_boundaries,
            dro_name=RUN_NAME, diffusion_name=diffusion_name_mv, ml_name=ml_name,
            show_jsm=PLOT_JSM,
        )
        meta = _load_meta(); meta[_epoch_key] = fe; _save_meta(meta)

print("\n  | main_energy_mv.py done |", flush=True)
