
import sys
import os
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Utils/Exp_Scripts', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch

from Utils.diffusion_utils import Train_DFU, Inference_DFU, diffusion_name
from Utils.Algos.ddd_learning import DDD_Learning
from Utils.Algos.ddd_learning import algo_name as ddd_name

from Utils.Algos.ml import ML
from Utils.Algos.ml import algo_name as ml_name
from Utils.Algos.wdro import WassersteinDRO
from Utils.Downstream.energy_scheduling import Energy_Scheduling
from Utils.Net.task_net import ConstantMeanPredictor
from Utils.utils import Utils, Evaluation, Log_Redirect, Tee, set_global_seed
from Utils.datasets import Data_Loader, build_ml_features
from Utils.config import Config

torch.cuda.empty_cache()

# On CPU partitions, cap PyTorch intra-op threads at 1 so torch does not compete
# with the Gurobi LP workers for the same cores. No effect on GPU nodes.
if not torch.cuda.is_available():
    torch.set_num_threads(1)

cfg = Config()


def _make_live_plot_cb(log_folder, **name_kwargs):
    """Return a callback(loss_list, **extra) that redraws the training-loss PNG each epoch.

    Overwrites one PNG per run. Extra kwargs (val_loss_list, loss_label) pass to
    draw_loss_plot.
    """
    def _cb(loss_list, **extra):
        Utils().draw_loss_plot(loss_list=loss_list, log_folder_path=log_folder,
                               **name_kwargs, **extra)
    return _cb


# Pin every RNG before any model construction or data loading so re-runs are
# byte-identical. MV_SEED also seeds the ICON instance (SEED below).
_GLOBAL_SEED = int(os.environ.get('MV_SEED', '42'))
set_global_seed(_GLOBAL_SEED)
print(f"  | SEED | {_GLOBAL_SEED} (MV_SEED env)", flush=True)

# Seed for the synthetic-OOD price noise (ε-tune val set, Gaussian-OOD test sets).
# Fixed and independent of MV_SEED so every model seed scores on the same noisy set.
_OOD_NOISE_SEED = 42

# Shared experiment parameters
shared_params = {
    # CPU fallback for partitions without a GPU.
    'DEVICE': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'PIC_SIZE': 28,
    # ML loader row length; MUST be >= WINDOW_LEN + PREDICT_LEN (48 + 48 = 96).
    # Independent of PIC_SIZE=28, which the DDPM image architecture fixes.
    'ML_SEQ_LEN': 192,

    'MODEL_SAVE':    rf'{project_root}/Models',
    'GEN_DATA_SAVE': rf'{project_root}/Gen_Data',
    'RESULT_SAVE':   rf'{project_root}/Results',

    # Training distribution: AEMO VIC1 prices 2015-2020
    'TRAIN_FILE': rf'./Data/AEMO_VIC1_train_2015_2020.csv',

    # Test sets: in-distribution, natural OOD, synthetic Gaussian OOD.
    'TEST_SET': {
        # In-distribution (held-out portion of training years)
        'AEMO_train_2015_2020':    rf'./Data/AEMO_VIC1_train_2015_2020.csv',
        # OOD: mild natural shift (2021-2022, ~15% negative price incidence)
        'AEMO_OOD_2021_2022':      rf'./Data/AEMO_VIC1_test_2021_2022.csv',
        # OOD: strong natural shift (2023-2025, ~23% negative price incidence)
        'AEMO_OOD_2023_2025':      rf'./Data/AEMO_VIC1_test_2023_2025.csv',
        # Locked OOD-test halves: month-interleaved, disjoint from the oodval splits that
        # select epochs. TEST_BUCKETS=oodtest scores only these.
        'AEMO_oodtest_2021_2022':  rf'./Data/AEMO_VIC1_oodtest_2021_2022.csv',
        'AEMO_oodtest_2023_2025':  rf'./Data/AEMO_VIC1_oodtest_2023_2025.csv',
        # Synthetic Gaussian OOD, sigma = 0.1/0.25/0.5/1.0 × mean train price; built
        # inline from the training data in the test loop.
        'AEMO_gauss_sigma0.10':    None,
        'AEMO_gauss_sigma0.25':    None,
        'AEMO_gauss_sigma0.50':    None,
        'AEMO_gauss_sigma1.00':    None,
    },

    'TEST_TASK_SET': {
        'ENERGY_SCHEDULING': {
            # AUD/MWh clip range; MUST match PRICE_FLOOR / PRICE_CAP in Data/preprocess_aemo.py.
            # Sets the SPO+ loss scale and clips the Gaussian-OOD prices; tighter bounds
            # would squash the OOD test sets into the training range.
            'PRICE_RANGE': np.array([-1000.0, 17500.0]),
            # ICON Challenge 2014 instance.txt, native 5-min grid (T=288); AEMO 30-min
            # prices are step-replicated 6x. None -> synthetic instance below. MUST match
            # main_energy_eval.py. Env ICON_INSTANCE overrides; unset = sample01.
            'ICON_INSTANCE': os.environ.get('ICON_INSTANCE',
                                            'Data/sampleinstances/sample01/instance.txt'),
            # ICON_AGG=1 (default) coarsens T=288 -> T=48 for a tractable SPO+ training LP.
            # Trap: aggregation makes the integer MIP infeasible, so eval silently falls
            # back to the greedy heuristic. Use ICON_AGG=0 (native T=288) for test/eval.
            'ICON_AGGREGATE_TO_EXTERNAL': bool(int(os.environ.get('ICON_AGG', '1'))),
            # Number of schedulable tasks in the SYNTHETIC ICON instance
            # (ignored when ICON_INSTANCE is set).
            'N_TASKS':     10,
            'N_MACHINES':  3,
            # Full-day time horizon (30-min resolution, |T| = 48 periods).
            # Ignored when ICON_INSTANCE is set (T comes from the file, typically 288).
            'T':           48,
            # Synthetic-instance seed, tied to MV_SEED. With ICON_INSTANCE set it only
            # fills None-fields.
            'SEED':        _GLOBAL_SEED,
        }
    },

    'TRAIN_DIFFUSION':    True,
    'GENERATE_FAKE_DATA': True,
    'EVALUATE_FAKE_DATA': True,
    'TRAIN_PTO':          True,   # Predict-Then-Optimise: MSE loss only
    'TRAIN_ML':           True,   # DFL: SPO+ loss (decision-focused)
    'TRAIN_DDDRO':        True,   # 3D-Learning: worst-case dist + SPO+
    'TRAIN_DRO':          True,   # DRO baseline: worst-case dist + MSE inner loss
    # Grid-search EPSILON before TRAIN_DRO: keep the ε with the lowest forecast MSE on a
    # noise-injected val set. False -> wdro_params['EPSILON'].
    'TUNE_DRO_EPSILON':   True,
    'TEST_METHODS':       True,
}

# STAGE env runs one pipeline phase per sbatch job. Dependencies:
#   ddpm: none   gen: ddpm   pto/dfl/wdro: none   3dl: retired (raises)   test: all
# 'all' (default) runs the full sequential pipeline. 'test' enables every flag so each
# block resolves its paths; _stage_done() skips stages already complete.
_STAGE = os.environ.get('STAGE', 'all').lower()
_STAGE_FLAGS = {
    'ddpm':       ['TRAIN_DIFFUSION'],
    'gen':        ['GENERATE_FAKE_DATA', 'EVALUATE_FAKE_DATA'],
    'pto':        ['TRAIN_PTO'],
    'dfl':        ['TRAIN_ML'],
    'wdro':       ['TRAIN_DRO', 'TUNE_DRO_EPSILON'],
    '3dl':        ['TRAIN_DDDRO'],
    'test':       ['TRAIN_DIFFUSION', 'GENERATE_FAKE_DATA', 'EVALUATE_FAKE_DATA',
                   'TRAIN_PTO', 'TRAIN_ML', 'TRAIN_DRO', 'TUNE_DRO_EPSILON',
                   'TRAIN_DDDRO', 'TEST_METHODS'],
    # Pure eval: scores existing checkpoints; a method without one skips with a note.
    'test_only':  ['TEST_METHODS'],
}
_ALL_STAGE_FLAGS = {
    'TRAIN_DIFFUSION', 'GENERATE_FAKE_DATA', 'EVALUATE_FAKE_DATA',
    'TRAIN_PTO', 'TRAIN_ML', 'TRAIN_DDDRO', 'TRAIN_DRO',
    'TUNE_DRO_EPSILON', 'TEST_METHODS',
}
if _STAGE != 'all':
    if _STAGE not in _STAGE_FLAGS:
        raise ValueError(
            f"Invalid STAGE={_STAGE!r}. Valid: {sorted(_STAGE_FLAGS)} or 'all'."
        )
    _enable = set(_STAGE_FLAGS[_STAGE])
    for _flag in _ALL_STAGE_FLAGS:
        shared_params[_flag] = (_flag in _enable)
    print(f"  | STAGE | {_STAGE}  enabled: {sorted(_enable)}", flush=True)

# Diffusion model hyperparameters
dfu_params = {
    'PUBLIC': {
        'NAME': diffusion_name,
        'SELECTED_EPOCH': 7000,
    },
    'TRAIN': {
        'T': 500,
        'BETA_1': 5e-2,
        'BETA_T': 1e-1,
        'BATCH_SIZE': 64,
        'LR': 1e-4,
        'ITERATION': 7000,
        'SAVE_EVERY': 1000,
    },
    'GENERATE': {
        'T': 500,
        'BATCH_REPEAT': 4,
        'BETA_1': 5e-2,
        'BETA_T': 1e-1,
        'BATCH_SIZE': 64,
        'SAVE_TRAJECTORY': True,
    },
}

# ML predictor hyperparameters. PREDICT_LEN=48 = one day of 30-min periods, matching
# the T=48 scheduling horizon.
ml_params = {
    'PUBLIC': {
        # Trains the DFL baseline (SPO+). DFL_TAG appends a suffix so a variant run saves
        # to DFL{tag}-epoch*.pth and DFL{tag}_final_epoch in training_metadata.json.
        'NAME': 'DFL' + os.environ.get('DFL_TAG', ''),
        # 48 slots = 24 h of history, so the LSTM sees the daily cycle. At 7 slots every
        # method collapsed to near-mean prediction.
        'WINDOW_LEN':   48,
        'PREDICT_LEN':  48,     # full-day prediction for ICON MIP
        'SELECTED_EPOCH': 100,
        'DISPLAY_EVERY':  5,
        'SAVE_EVERY':     10,
    },
    'TRAIN': {
        # ~13.5 min/epoch on CPU: 100 epochs ≈ 22 h, inside the 48 h walltime.
        'ITERATION':       100,
        # LR halves at epochs 30/60/90. A harsher decay (step 10, gamma 0.05) froze
        # training by epoch 30.
        'DISCOUNT_FACTOR': 0.5,
        'STEP_SIZE':       30,
        # At LR=1e-6 every method was still underfit at epoch 100.
        'LR':              1e-4,
        'BATCH_SIZE':      64,
    },
}

# Wasserstein DRO baseline (Mohajerin Esfahani & Kuhn 2018). EPSILON = Wasserstein-1
# ball radius; the inner max uses PGD with projection onto ||δ|| <= ε.
wdro_params = {
    # WDRO_SPO on sample01 takes ~58 min/epoch, so 100 epochs ≈ 4 days: set
    # run_wdro.sh --time to 5-00:00:00.
    'ITERATION':       100,
    'DISCOUNT_FACTOR': 0.5,
    'STEP_SIZE':       30,
    # Matches ml_params LR.
    'LR':              1e-4,
    'BATCH_SIZE':      64,
    'EPSILON':         0.1,    # Wasserstein-1 ball radius (normalised input space)
    # PGD inner steps: 1 outer + 3 inner = 4 LP batches per WDRO_SPO minibatch.
    # The 2.5·ε/N step (Madry-style) offsets the small step count.
    'INNER_STEPS':     3,
    'INNER_LR':        0.01,   # step size for PGD gradient ascent
    'SELECTED_EPOCH':  100,
    'SAVE_EVERY':      10,
    'DISPLAY_EVERY':   5,
    # ε grid. ε lives in NORMALISED input space: real radius = ε * (norm_max - norm_min)
    # AUD/MWh. The grid is in fractions of mean train price, converted at runtime once
    # the norm bounds are known. Mean ≈ $71 and P1/P99 span ≈ $233 give normalised
    # ε ≈ 0.015-0.307; above ~0.6 the perturbation swamps the input.
    'EPSILON_GRID_FRAC_OF_MEAN_PRICE': [0.05, 0.1, 0.25, 0.5, 1.0],
    # Reduced training budget per ε (full-budget retrain happens at the chosen ε).
    'TUNE_ITERATION':  30,
    # Val-price noise σ as a fraction of mean train price. Matches the strongest
    # Gaussian-OOD test set, so the ε calibration is conservative.
    'TUNE_VAL_NOISE_SIGMA_FRAC': 0.5,
}

# 3D-Learning (DDD-DRO) hyperparameters
ddd_params = {
    'PUBLIC': {
        'NAME': ddd_name,
        'BATCH_SIZE': 64,
        'DISPLAY_EVERY': 1,
        # 15 outer epochs: step=10/gamma=0.5 halves LR once, near the end.
        'DISCOUNT_FACTOR': 0.5,
        'STEP_SIZE': 10,
        'PPO_CLIP': 0.4,
    },
    'DRO': {
        'ITERATION': 15,
        'ETA': 0.01,
        'MU': 1,
        'BUDGET': 0.030,
        # Diffusion adversarial LR stays at 1e-6: the DDPM fine-tunes from a pretrained
        # checkpoint with PPO clip 0.4, and a larger LR risks unstable dfu_theta updates.
        'LR': 1e-6,
        'ML_SAVE_EVERY': 2,
        'DFU_SAVE_EVERY': 2,
        'ADJUST_TIMESTEPS': 10,
        'P_S0': 0.3,
    },
    'DIFFUSION': {
        'PUBLIC': {
            'NAME': diffusion_name,
            'BETA_1': 5e-2,
            'BETA_T': 1e-1,
            'T': 500,
        },
        'TRAIN': {
            'BATCH_REPEAT': 4,
            'ITERATION': 10,
        },
        'GENERATE': {},
    },
    'ML': {
        'PUBLIC': {
            'WINDOW_LEN':  48,    # must match ml_params
            'PREDICT_LEN': 48,    # must match ml_params
        },
        'TRAIN': {
            # SPO+'s sparse {-2, 0, +2} subgradient needs LR=1e-3 to escape the
            # cold-start basin; it beat LR=1e-4 in the DFL ablation.
            'LR':        1e-3,
            # 5 inner epochs per outer so the head adapts to each new s_theta sample.
            'ITERATION': 5,
        },
    },
}

# Logging
log_folder_path, log_file_path = Log_Redirect().build_log_file(
    result_path=shared_params['RESULT_SAVE'],
    cuda=shared_params['DEVICE']
)
sys.stdout = Tee(log_file_path)

# training_metadata.json holds final epochs and tuned hyperparameters;
# main_energy_eval.py reads it to resolve checkpoint epochs.
_TRAIN_META_PATH = os.path.join(shared_params['MODEL_SAVE'], 'training_metadata.json')

def _load_train_meta():
    if os.path.exists(_TRAIN_META_PATH):
        with open(_TRAIN_META_PATH, 'r') as _f:
            return json.load(_f)
    return {}

def _save_train_meta(meta):
    os.makedirs(os.path.dirname(_TRAIN_META_PATH), exist_ok=True)
    with open(_TRAIN_META_PATH, 'w') as _f:
        json.dump(meta, _f, indent=2)


def _stage_done(meta_key):
    """True iff `meta_key` is in training_metadata.json.

    The model file is written before its key, so a present key implies the file
    existed. Lets a resubmitted job skip finished stages ([Resume] log line)."""
    return meta_key in _load_train_meta()


def _log_skip(stage_name, detail=''):
    suffix = f' — {detail}' if detail else ''
    print(f'\n  [Resume] skipping {stage_name}{suffix} (training_metadata.json already records completion)')

with open(log_file_path, 'a') as f:
    print(f'  | Log Path |  {log_file_path}')

    # Load training data (AEMO electricity prices)
    dfu_train_params    = dfu_params['TRAIN']
    dfu_public_params   = dfu_params['PUBLIC']
    dfu_generate_params = dfu_params['GENERATE']

    _train_full = pd.read_csv(
        shared_params['TRAIN_FILE'],
        parse_dates=['TIMESTAMP'],
        date_format='mixed',
        dayfirst=False,
    )
    train_data_df = _train_full['Price']   # Series used by all downstream code

    # Train fold = first 70% (matches the 70/15/15 ML split). DDPM training, norm stats
    # and demand calibration use only this fold, so val/test rows never leak.
    _train_fold_n = int(0.70 * len(train_data_df))
    train_fold_df = train_data_df.iloc[:_train_fold_n]

    # Demand stats calibrate the synthetic ICON instance (train fold only).
    demand_stats = {
        'mean_mwh':  float(_train_full['TOTALDEMAND_MW'].iloc[:_train_fold_n].mean() * 0.5),
        'price_p25': float(train_fold_df.quantile(0.25)),
        'price_p75': float(train_fold_df.quantile(0.75)),
    }
    train_mean_price = float(train_fold_df.mean())
    # Norm bounds = train-fold P1/P99, shared by every loader (train, val, test, OOD).
    # Held-out and OOD inputs may fall outside [0, 1]; that is intended.
    # P1/P99, not min/max: AEMO spikes (max ≈ $14,500) put the mean at 0.04 under
    # min/max, deep in sigmoid saturation. P1/P99 (span ≈ $250) puts it near 0.28.
    _train_norm_bounds = (float(train_fold_df.quantile(0.01)),
                          float(train_fold_df.quantile(0.99)))
    # Demand norm bounds (train-fold only, no leakage) — used by the
    # multivariate ML loader to normalise TOTALDEMAND_MW alongside Price.
    _train_demand_fold = _train_full['TOTALDEMAND_MW'].iloc[:_train_fold_n]
    _demand_norm_bounds = (float(_train_demand_fold.min()), float(_train_demand_fold.max()))
    # The one denormalisation range for every path; MUST equal the loader norm bounds.
    # A full-series range here shifted ConstantMean's prediction by ~$108.
    _DENORM_RANGE = np.array([_train_norm_bounds[0], _train_norm_bounds[1]])

    print(f'  | Training price mean: {train_mean_price:.2f} AUD/MWh'
          f'  P25: {demand_stats["price_p25"]:.2f}'
          f'  P75: {demand_stats["price_p75"]:.2f}')
    print(f'  | Mean demand: {demand_stats["mean_mwh"] / 0.5:.0f} MW'
          f'  ({demand_stats["mean_mwh"]:.1f} MWh per 30-min interval)')

    # Downstream task: ICON Challenge 2014 MIP
    es_params = shared_params['TEST_TASK_SET']['ENERGY_SCHEDULING']
    if es_params.get('ICON_INSTANCE'):
        # Faithful ICON Challenge 2014 instance — T=288 internal, T_external=48
        energy_task = Energy_Scheduling.from_icon_instance(
            instance_path=es_params['ICON_INSTANCE'],
            price_range=es_params['PRICE_RANGE'],
            T_external=ml_params['PUBLIC']['PREDICT_LEN'],
            seed=es_params['SEED'],
            aggregate_to_external=es_params.get('ICON_AGGREGATE_TO_EXTERNAL', False),
        )
    else:
        energy_task = Energy_Scheduling(
            price_range=es_params['PRICE_RANGE'],
            demand_stats=demand_stats,
            n_tasks=es_params['N_TASKS'],
            n_machines=es_params['N_MACHINES'],
            T=es_params['T'],
            seed=es_params['SEED'],
        )

    # Optional per-solve MIP time-limit override (env MIP_TL). Native T=288 MIP
    # needs >~100s to produce an incumbent (else the silent SolCount==0 greedy
    # fallback fires); MIP_GAP is read inside Energy_Scheduling.__init__.
    if os.environ.get('MIP_TL'):
        energy_task.MIP_TIMELIMIT = float(os.environ['MIP_TL'])
        print(f'  | MIP_TIMELIMIT override | {energy_task.MIP_TIMELIMIT}s'
              f'   MIP_GAP | {energy_task.MIP_GAP}')

    # Diffusion pre-training
    if shared_params['TRAIN_DIFFUSION']:
        if _stage_done('DDPM_done'):
            _log_skip('DDPM pretraining')
        else:
            dfu_train = Train_DFU(
                T=dfu_train_params['T'],
                beta_1=dfu_train_params['BETA_1'],
                beta_T=dfu_train_params['BETA_T'],
                batch_size=dfu_train_params['BATCH_SIZE'],
                lr=dfu_train_params['LR'],
                total_iteration=dfu_train_params['ITERATION'],
                picture_size=shared_params['PIC_SIZE'],
                device=shared_params['DEVICE'],
                data_df=train_fold_df,    # train fold only — no leakage from val/test
                save_every=dfu_train_params['SAVE_EVERY'],
                model_save_folder_path=shared_params['MODEL_SAVE'],
            )
            dfu_train.train()
            _meta = _load_train_meta()
            _meta['DDPM_done'] = True
            _save_train_meta(_meta)

    # Model and generated-data paths
    dfu_model_path = cfg.diffusion_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name=dfu_public_params['NAME'],
        cuda=shared_params['DEVICE'],
        epoch=dfu_public_params['SELECTED_EPOCH'],
        batch_size=dfu_train_params['BATCH_SIZE'],
        pic_size=shared_params['PIC_SIZE'],
        T=dfu_train_params['T'],
        lr=dfu_train_params['LR'],
        beta_1=dfu_train_params['BETA_1'],
        beta_T=dfu_train_params['BETA_T'],
    )

    z0_data_path = cfg.generate_data_path(
        folder=shared_params['GEN_DATA_SAVE'],
        model_name=dfu_public_params['NAME'],
        cuda=shared_params['DEVICE'],
        epoch=dfu_public_params['SELECTED_EPOCH'],
        batch_size=dfu_generate_params['BATCH_SIZE'],
        pic_size=shared_params['PIC_SIZE'],
        T=dfu_generate_params['T'],
        beta_1=dfu_generate_params['BETA_1'],
        beta_T=dfu_generate_params['BETA_T'],
    )

    z0_trajectory_path = cfg.trajectory_data_path(
        folder=shared_params['GEN_DATA_SAVE'],
        model_name=dfu_public_params['NAME'],
        cuda=shared_params['DEVICE'],
        epoch=dfu_public_params['SELECTED_EPOCH'],
        batch_size=dfu_generate_params['BATCH_SIZE'],
        pic_size=shared_params['PIC_SIZE'],
        T=dfu_generate_params['T'],
        beta_1=dfu_generate_params['BETA_1'],
        beta_T=dfu_generate_params['BETA_T'],
    )

    # Generate synthetic price data via diffusion
    if shared_params['GENERATE_FAKE_DATA']:
        if os.path.exists(z0_data_path) and (
            not dfu_generate_params['SAVE_TRAJECTORY'] or os.path.exists(z0_trajectory_path)
        ):
            print(f'\n  [Resume] skipping fake-data generation — {z0_data_path} already exists')
        else:
            test_data_df = train_fold_df    # mirror DDPM training input

            inference = Inference_DFU(
                T=dfu_generate_params['T'],
                beta_1=dfu_generate_params['BETA_1'],
                beta_T=dfu_generate_params['BETA_T'],
                batch_size=dfu_generate_params['BATCH_SIZE'],
                device=shared_params['DEVICE'],
                picture_size=shared_params['PIC_SIZE'],
                sampling_number=dfu_generate_params['BATCH_SIZE'],
            )
            inference.generate_wholeset_data(
                model_path=dfu_model_path,
                batch_repeat=dfu_generate_params['BATCH_REPEAT'],
                data_df=test_data_df,
                save_trajectory=dfu_generate_params['SAVE_TRAJECTORY'],
                save_gen_data_path=z0_data_path,
                title='Price',
            )

    # Evaluate synthetic data quality
    if shared_params['EVALUATE_FAKE_DATA']:
        print(f'\n=============== Evaluating Generated Data Start ===============')
        print(f"  | Selected Epoch | {dfu_public_params['SELECTED_EPOCH']}")
        print(f'  | File Name | {z0_data_path}')

        z0_data_df = pd.read_csv(z0_data_path)['Price']

        Evaluation().fid_evaluation(
            fake_df=z0_data_df,
            true_df=train_fold_df,    # DDPM was trained on train fold; compare like-for-like
            picture_size=shared_params['PIC_SIZE'],
            device=shared_params['DEVICE'],
            isshow=False,
        )
        dist = Utils().compute_normalized_wasserstein(z0_data_df, train_fold_df)
        print(f'  | Wasserstein Distance | {dist}')
        print(f'=============== Evaluating Generated Data End ===============')

    # Shared ML predictor settings
    ml_train_params  = ml_params['TRAIN']
    ml_public_params = ml_params['PUBLIC']

    # 70/15/15 train/val/test split; val drives early stopping, test is the in-dist set.
    # Features: price + demand + sin/cos calendar (hour, dow, month). Price MUST stay at
    # column 0: the persistence-residual head reads x[:, -1, 0].
    _ml_feature_df = build_ml_features(_train_full)
    ml_dl = Data_Loader(df=_ml_feature_df)
    ml_train_loader, ml_val_loader, ml_test_loader = ml_dl.ml_set2loader_mv(
        split_test=True,
        val_size=0.15,
        normlized=True,
        device=shared_params['DEVICE'],
        batch_size=ml_train_params['BATCH_SIZE'],
        pic_size=shared_params['ML_SEQ_LEN'],
        norm_bounds={
            'Price':          _train_norm_bounds,
            'TOTALDEMAND_MW': _demand_norm_bounds,
            # sin/cos features pass through (already in [-1, 1]).
        },
    )

    # VAL_OOD_REGRET=1: select the epoch by MAX val regret over the two OOD-val splits
    # (month-interleaved halves; the OOD-test halves stay locked). Same train norm
    # bounds. Default off: in-dist val selection.
    _val_regret_loaders = None
    if os.environ.get('VAL_OOD_REGRET') == '1':
        _val_regret_loaders = []
        for _lbl, _p in [('oodval2122', './Data/AEMO_VIC1_oodval_2021_2022.csv'),
                         ('oodval2325', './Data/AEMO_VIC1_oodval_2023_2025.csv')]:
            _ofeat = build_ml_features(pd.read_csv(_p, parse_dates=['TIMESTAMP']))
            _oloader = Data_Loader(df=_ofeat).ml_set2loader_mv(
                split_test=False, normlized=True, device=shared_params['DEVICE'],
                batch_size=ml_train_params['BATCH_SIZE'], pic_size=shared_params['ML_SEQ_LEN'],
                norm_bounds={'Price': _train_norm_bounds, 'TOTALDEMAND_MW': _demand_norm_bounds},
            )
            _val_regret_loaders.append((_lbl, _oloader))
        print(f'  | VAL_OOD_REGRET=1 | worst-case epoch selection over '
              f'{[l for l, _ in _val_regret_loaders]} (oodtest held out)')

    # Univariate price-only loaders: the DDPM is univariate, so a 3DL head has
    # input_size=1. Same 70/15/15 split and Price norm bounds.
    ml_dl_uv = Data_Loader(df=train_data_df)
    ml_train_loader_uv, ml_val_loader_uv, ml_test_loader_uv = ml_dl_uv.ml_set2loader(
        split_test=True,
        val_size=0.15,
        normlized=True,
        device=shared_params['DEVICE'],
        batch_size=ml_train_params['BATCH_SIZE'],
        pic_size=shared_params['ML_SEQ_LEN'],
        norm_bounds=_train_norm_bounds,
    )

    # Sanity baseline: ConstantMean predicts the normalised train-mean price for every
    # slot. Trained methods must beat it, else the LSTM is underfit and rankings are noise.
    _const_mean_norm = (train_mean_price - _train_norm_bounds[0]) / max(
        _train_norm_bounds[1] - _train_norm_bounds[0], 1e-12
    )
    print(f'\n  [ ConstantMean baseline: predicting normalised price = {_const_mean_norm:.6f}'
          f'  (= ${train_mean_price:.2f} real) for every slot ]')
    _const_predictor = ConstantMeanPredictor(
        predict_len=ml_public_params['PREDICT_LEN'],
        value=_const_mean_norm,
    ).to(shared_params['DEVICE'])
    const_model_path = cfg.ml_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name='ConstantMean',
        cuda=shared_params['DEVICE'],
        epoch=1,
        lr=0.0,
    )
    os.makedirs(os.path.dirname(const_model_path), exist_ok=True)
    torch.save(_const_predictor, const_model_path)
    print(f'  | ConstantMean baseline saved | {const_model_path}')

    # This ML wrapper only scores the constant predictor: test_task calls torch.load +
    # forward, so any nn.Module mapping (B, W, F) → (B, PREDICT_LEN) works.
    ml_const = ML(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=1,
        discount_factor=ml_train_params['DISCOUNT_FACTOR'],
        step_size=ml_train_params['STEP_SIZE'],
        lr=ml_train_params['LR'],
        device=shared_params['DEVICE'],
        display_every=ml_public_params['DISPLAY_EVERY'],
        save_every=ml_public_params['SAVE_EVERY'],
        name='ConstantMean',
    )

    # PTO baseline: Predict-Then-Optimise, MSE loss only (task=None).
    # PTO_TAG appends a suffix so a variant run leaves the production PTO checkpoint and
    # metadata alone. The head range comes from PRED_NORM_LO/HI (read in task_net.py).
    _pto_name = 'PTO' + os.environ.get('PTO_TAG', '')
    _pto_meta_key = f'{_pto_name}_final_epoch'
    ml_pto = ML(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=ml_train_params['ITERATION'],
        discount_factor=ml_train_params['DISCOUNT_FACTOR'],
        step_size=ml_train_params['STEP_SIZE'],
        lr=ml_train_params['LR'],
        device=shared_params['DEVICE'],
        display_every=ml_public_params['DISPLAY_EVERY'],
        save_every=ml_public_params['SAVE_EVERY'],
        name=_pto_name,   # save to PTO-prefixed model files, not ML-prefixed
    )

    # Epoch from metadata (SELECTED_EPOCH only before metadata exists), so later stages
    # with TRAIN_PTO=False load a file that exists.
    ml_pto_model_path = cfg.ml_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name=_pto_name,
        cuda=shared_params['DEVICE'],
        epoch=_load_train_meta().get(_pto_meta_key, ml_public_params['SELECTED_EPOCH']),
        lr=ml_train_params['LR'],
    )

    if shared_params['TRAIN_PTO']:
        if _stage_done(_pto_meta_key):
            _pto_final_epoch = _load_train_meta()[_pto_meta_key]
            _log_skip(f'{_pto_name} baseline', detail=f'final_epoch={_pto_final_epoch}')
        else:
            print('\n  [ Training PTO baseline (MSE loss, no task) ]')
            _pto_final_epoch = ml_pto.train(
                task=None,           # MSE loss only — no decision-focused training
                val_task=None,       # no val-regret eval
                model_path=None,
                model_save_folder_path=shared_params['MODEL_SAVE'],
                train_data_loader=ml_train_loader,
                val_data_loader=ml_val_loader,
                data_range=_DENORM_RANGE,
                live_plot_callback=_make_live_plot_cb(log_folder_path, ml_name='PTO'),
            )
            Utils().draw_loss_plot(
                loss_list=ml_pto.epoches_losses_list,
                log_folder_path=log_folder_path,
                val_loss_list=ml_pto.epoches_val_losses_list,
                val_regret_list=ml_pto.epoches_val_regret_list,
                loss_label='MSE Loss',
                ml_name='PTO',
            )
            _meta = _load_train_meta()
            _meta[_pto_meta_key] = _pto_final_epoch
            _save_train_meta(_meta)
        ml_pto_model_path = cfg.ml_model_path(
            folder=shared_params['MODEL_SAVE'],
            model_name=_pto_name,
            cuda=shared_params['DEVICE'],
            epoch=_pto_final_epoch,
            lr=ml_train_params['LR'],
        )

    # DFL baseline: SPO+ task loss, cold start (random LSTM weights, PTO architecture) at
    # LR=1e-3. Cold start beat the PTO/WDRO warm-start variants on regret.
    DFL_FINETUNE_LR = 1e-3
    ml = ML(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=ml_train_params['ITERATION'],
        discount_factor=ml_train_params['DISCOUNT_FACTOR'],
        step_size=ml_train_params['STEP_SIZE'],
        lr=DFL_FINETUNE_LR,
        device=shared_params['DEVICE'],
        display_every=ml_public_params['DISPLAY_EVERY'],
        save_every=ml_public_params['SAVE_EVERY'],
        name=ml_public_params['NAME'],  # 'DFL' — keep ALGO_NAME aligned with file path
    )

    # Epoch from metadata, as for PTO above.
    ml_model_path = cfg.ml_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name=ml_public_params['NAME'],
        cuda=shared_params['DEVICE'],
        epoch=_load_train_meta().get(f'{ml_public_params["NAME"]}_final_epoch', ml_public_params['SELECTED_EPOCH']),
        lr=ml_train_params['LR'],
    )

    if shared_params['TRAIN_ML']:
        _dfl_meta_key = f'{ml_public_params["NAME"]}_final_epoch'
        if _stage_done(_dfl_meta_key):
            _ml_final_epoch = _load_train_meta()[_dfl_meta_key]
            _log_skip(f'{ml_public_params["NAME"]} baseline', detail=f'final_epoch={_ml_final_epoch}')
        else:
            # Cold start: PTO architecture (persistence-residual head + zero_init_fc2 +
            # avg_delta_init), random LSTM weights, no checkpoint load.
            print(f'\n  [ Training DFL baseline (SPO+ task loss) ]')
            print(f'  | Cold-start (PTO architecture, random weights)', flush=True)
            # val_task=energy_task tracks val regret per epoch and selects the best
            # checkpoint by regret. DFL_NO_VAL_REGRET=1 skips it (best by val loss).
            _dfl_val_task = None if os.environ.get('DFL_NO_VAL_REGRET') == '1' else energy_task
            _ml_final_epoch = ml.train(
                task=energy_task,
                val_task=_dfl_val_task,
                # DFL_WARM_CKPT: optional checkpoint to warm-start or continue from.
                model_path=os.environ.get('DFL_WARM_CKPT') or None,
                model_save_folder_path=shared_params['MODEL_SAVE'],
                train_data_loader=ml_train_loader,
                val_data_loader=ml_val_loader,
                val_regret_loaders=_val_regret_loaders,
                data_range=_DENORM_RANGE,
                live_plot_callback=_make_live_plot_cb(log_folder_path, ml_name=ml_name),
            )
            Utils().draw_loss_plot(
                loss_list=ml.epoches_losses_list,
                log_folder_path=log_folder_path,
                val_loss_list=ml.epoches_val_losses_list,
                val_regret_list=ml.epoches_val_regret_list,
                loss_label='SPO+ Loss',
                ml_name=ml_name,
            )
            # In-dist test eval runs in the TEST_METHODS loop below, not here.
            _meta = _load_train_meta()
            _meta[_dfl_meta_key] = _ml_final_epoch
            _save_train_meta(_meta)
        ml_model_path = cfg.ml_model_path(
            folder=shared_params['MODEL_SAVE'],
            model_name=ml_public_params['NAME'],
            cuda=shared_params['DEVICE'],
            epoch=_ml_final_epoch,
            lr=ml_train_params['LR'],
        )

    # z0_timesteps feeds the DDD_Learning constructor below.
    z0_timesteps = list(range(0, dfu_generate_params['T'], 1))

    # DDD-DRO and DRO baseline setup
    ddd_public_params = ddd_params['PUBLIC']
    ddd_dro_params    = ddd_params['DRO']
    ddd_dfu_params    = ddd_params['DIFFUSION']
    ddd_ml_params     = ddd_params['ML']

    _ddd_kwargs = dict(
        beta_1=ddd_dfu_params['PUBLIC']['BETA_1'],
        beta_T=ddd_dfu_params['PUBLIC']['BETA_T'],
        T=ddd_dfu_params['PUBLIC']['T'],
        ppo_clip=ddd_public_params['PPO_CLIP'],
        adjust_timesteps=ddd_dro_params['ADJUST_TIMESTEPS'],
        batch_repeat=ddd_dfu_params['TRAIN']['BATCH_REPEAT'],
        batch_size=ddd_public_params['BATCH_SIZE'],
        pic_size=shared_params['PIC_SIZE'],
        ml_seq_len=shared_params['ML_SEQ_LEN'],
        discount_factor=ddd_public_params['DISCOUNT_FACTOR'],
        step_size=ddd_public_params['STEP_SIZE'],
        dro_lr=ddd_dro_params['LR'],
        epoches=ddd_dro_params['ITERATION'],
        dro_inner_epochs=ddd_dfu_params['TRAIN']['ITERATION'],
        ml_inner_epochs=ddd_ml_params['TRAIN']['ITERATION'],
        p_s0=ddd_dro_params['P_S0'],
        eta=ddd_dro_params['ETA'],
        mu=ddd_dro_params['MU'],
        budget=ddd_dro_params['BUDGET'],
        windows_len=ddd_ml_params['PUBLIC']['WINDOW_LEN'],
        predict_len=ddd_ml_params['PUBLIC']['PREDICT_LEN'],
        display_every=ddd_public_params['DISPLAY_EVERY'],
        ml_lr=ddd_ml_params['TRAIN']['LR'],
        selected_timesteps=z0_timesteps,
        ml_save_every=ddd_dro_params['ML_SAVE_EVERY'],
        dfu_save_every=ddd_dro_params['DFU_SAVE_EVERY'],
        dfu_name=ddd_dfu_params['PUBLIC']['NAME'],
        device=shared_params['DEVICE'],
    )

    # 3D-Learning: worst-case distribution + decision-focused (SPO+) inner loss
    ddd = DDD_Learning(**_ddd_kwargs, run_name='3DLearning')

    ddd_dro_selected_epoch = (ddd_dro_params['ITERATION'] // ddd_dro_params['ML_SAVE_EVERY']) * \
                              ddd_dro_params['ML_SAVE_EVERY']

    selected_ddd_dro_ml_model_path = cfg.dro_save_model_path(
        folder=shared_params['MODEL_SAVE'],
        dro_name='3DLearning',
        model_name=ml_name,
        cuda=shared_params['DEVICE'],
        epoch=ddd_dro_selected_epoch,
        lr=ddd_ml_params['TRAIN']['LR'],
    )

    # Classical Wasserstein DRO: no diffusion, MSE outer loss. WDRO_TAG appends a suffix
    # to WDRO, WDRO_SPO and their ε-grid metadata keys, so a variant run re-tunes ε from
    # scratch and leaves production checkpoints untouched.
    _WDRO_TAG = os.environ.get('WDRO_TAG', '')
    _wdro_name = 'WassersteinDRO' + _WDRO_TAG
    wdro_baseline = WassersteinDRO(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=wdro_params['ITERATION'],
        discount_factor=wdro_params['DISCOUNT_FACTOR'],
        step_size=wdro_params['STEP_SIZE'],
        lr=wdro_params['LR'],
        device=shared_params['DEVICE'],
        save_every=wdro_params['SAVE_EVERY'],
        display_every=wdro_params['DISPLAY_EVERY'],
        epsilon=wdro_params['EPSILON'],
        inner_steps=wdro_params['INNER_STEPS'],
        inner_lr=wdro_params['INNER_LR'],
        name=_wdro_name,
    )

    wdro_model_path = cfg.ml_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name=_wdro_name,
        cuda=shared_params['DEVICE'],
        epoch=_load_train_meta().get(f'{_wdro_name}_final_epoch', wdro_params['SELECTED_EPOCH']),
        lr=wdro_params['LR'],
    )

    # WDRO_SPO: same ε and settings as WDRO, but SPO+ in both the inner max and the outer
    # min. Separates robustness alone (WDRO) from robustness × DFL; the closest
    # non-diffusion comparison to 3D-Learning. Cold start at LR=1e-3, as for DFL.
    _wdro_spo_name = 'WassersteinDRO_SPO' + _WDRO_TAG
    _wdro_spo_lr = 1e-3

    wdro_spo_baseline = WassersteinDRO(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=wdro_params['ITERATION'],
        discount_factor=wdro_params['DISCOUNT_FACTOR'],
        step_size=wdro_params['STEP_SIZE'],
        lr=_wdro_spo_lr,
        device=shared_params['DEVICE'],
        save_every=wdro_params['SAVE_EVERY'],
        display_every=wdro_params['DISPLAY_EVERY'],
        epsilon=wdro_params['EPSILON'],
        inner_steps=wdro_params['INNER_STEPS'],
        inner_lr=wdro_params['INNER_LR'],
        name=_wdro_spo_name,
    )

    wdro_spo_model_path = cfg.ml_model_path(
        folder=shared_params['MODEL_SAVE'],
        model_name=_wdro_spo_name,
        cuda=shared_params['DEVICE'],
        epoch=_load_train_meta().get(f'{_wdro_spo_name}_final_epoch', wdro_params['SELECTED_EPOCH']),
        lr=wdro_params['LR'],
    )

    # Multivariate F=8 3D-Learning, trained by main_energy_mv.py, scored here in the same
    # native-MIP test loop. A plain DeepLSTM(input_size=8), so it uses the multivariate
    # loader. MV3DL_ONLY=1 scores only this model. Epoch comes from mv-cond metadata.
    _MV3DL_ONLY = os.environ.get('MV3DL_ONLY', '0') == '1'
    #   MV3DL_SUFFIX — run suffix ('' = champion); resolves metadata key + checkpoint.
    #   MV3DL_CKPT   — explicit checkpoint path override.
    #   MV3DL_LABEL  — results-table row name (default: the run name).
    _mv3dl_suffix = os.environ.get('MV3DL_SUFFIX', '')
    _mv3dl_label  = os.environ.get('MV3DL_LABEL', f'3DLearning_mv_cond{_mv3dl_suffix}')
    _mv3dl_path   = os.environ.get('MV3DL_CKPT') or None
    if _mv3dl_path is None:
        _mv3dl_meta_path = os.path.join(
            shared_params['MODEL_SAVE'],
            f'training_metadata_mv_cond_mv{_mv3dl_suffix}.json')
        if os.path.exists(_mv3dl_meta_path):
            with open(_mv3dl_meta_path) as _f:
                _mv3dl_ep = json.load(_f).get(f'3DLearning_mv_cond{_mv3dl_suffix}_final_epoch')
            if _mv3dl_ep is not None:
                _cand = os.path.join(
                    shared_params['MODEL_SAVE'],
                    f'3DLearning_mv_cond{_mv3dl_suffix}-ML-epoch{_mv3dl_ep}.pth')
                if os.path.exists(_cand):
                    _mv3dl_path = _cand
    if _mv3dl_path is not None and not os.path.exists(_mv3dl_path):
        print(f'  [ {_mv3dl_label} ] MV3DL_CKPT not found: {_mv3dl_path}')
        _mv3dl_path = None
    ml_mv3dl = ML(
        windows_len=ml_public_params['WINDOW_LEN'],
        predict_len=ml_public_params['PREDICT_LEN'],
        epoches=1,
        discount_factor=ml_train_params['DISCOUNT_FACTOR'],
        step_size=ml_train_params['STEP_SIZE'],
        lr=ml_train_params['LR'],
        device=shared_params['DEVICE'],
        display_every=ml_public_params['DISPLAY_EVERY'],
        save_every=ml_public_params['SAVE_EVERY'],
        name=_mv3dl_label,
    )

    # 3D-Learning training
    if shared_params['TRAIN_DDDRO']:
        if _stage_done('3DLearning_final_epoch'):
            _ddd_final_epoch = _load_train_meta()['3DLearning_final_epoch']
            _log_skip('3D-Learning training', detail=f'final_epoch={_ddd_final_epoch}')
        else:
            # Univariate 3D-Learning training is retired (confirmed defects). Train via
            # main_energy_mv.py; STAGE=test_only here still scores the MV checkpoint.
            raise RuntimeError(
                'Univariate STAGE=3dl training has been retired (defective + '
                'unused). Train 3D-Learning via `main_energy_mv.py` (cond_mv, '
                'F=8) and evaluate it through STAGE=test_only here.')
        selected_ddd_dro_ml_model_path = cfg.dro_save_model_path(
            folder=shared_params['MODEL_SAVE'],
            dro_name='3DLearning',
            model_name=ml_name,
            cuda=shared_params['DEVICE'],
            epoch=_ddd_final_epoch,
            lr=ddd_ml_params['TRAIN']['LR'],
        )

    # DRO ε grid search: train a short WDRO (TUNE_ITERATION epochs) per ε and score
    # forecast MSE on a noise-injected val set. The lowest MSE wins, so ε is tuned on
    # forecast error, never on scheduling cost.
    if shared_params['TRAIN_DRO'] and shared_params.get('TUNE_DRO_EPSILON', False):
        # PGD runs in normalised space, so convert the real-space grid to normalised ε.
        _norm_range = max(_train_norm_bounds) - min(_train_norm_bounds)
        _epsilon_grid = [
            (frac * train_mean_price) / _norm_range
            for frac in wdro_params['EPSILON_GRID_FRAC_OF_MEAN_PRICE']
        ]

        if _stage_done(f'{_wdro_name}_epsilon_chosen'):
            _best_eps = _load_train_meta()[f'{_wdro_name}_epsilon_chosen']
            _log_skip('WDRO ε grid search', detail=f'chosen ε={_best_eps:.5f}')
            wdro_params['EPSILON'] = _best_eps
            wdro_baseline.EPSILON = _best_eps
        else:
            print('\n  [ Tuning Wasserstein DRO epsilon via grid search on noise-injected val ]')

            # Val set: train prices plus Gaussian noise at the strongest synthetic-OOD
            # level, normalised with _train_norm_bounds.
            _tune_sigma_frac = wdro_params['TUNE_VAL_NOISE_SIGMA_FRAC']
            _tune_sigma_abs  = _tune_sigma_frac * train_mean_price
            np.random.seed(_OOD_NOISE_SEED)   # fixed test set across model seeds
            _val_clean_prices = train_data_df.values.copy()
            _val_noise = np.random.normal(0.0, _tune_sigma_abs, len(_val_clean_prices))
            # Noise on Price only; demand and calendar stay clean. The full 8-feature
            # frame matches the trainers' input_size.
            _val_noisy_full = _train_full.copy()
            _val_noisy_full['Price'] = (_val_clean_prices + _val_noise).clip(
                shared_params['TEST_TASK_SET']['ENERGY_SCHEDULING']['PRICE_RANGE'][0],
                shared_params['TEST_TASK_SET']['ENERGY_SCHEDULING']['PRICE_RANGE'][1],
            )
            _val_noisy_feat_df = build_ml_features(_val_noisy_full)
            # Same 70/15/15 split as the trainers: the tuning set is the noisy val fold,
            # never the test fold.
            _, _tune_val_loader, _ = Data_Loader(df=_val_noisy_feat_df).ml_set2loader_mv(
                split_test=True,
                val_size=0.15,
                normlized=True,
                device=shared_params['DEVICE'],
                batch_size=ml_train_params['BATCH_SIZE'],
                pic_size=shared_params['ML_SEQ_LEN'],
                norm_bounds={
                    'Price':          _train_norm_bounds,
                    'TOTALDEMAND_MW': _demand_norm_bounds,
                },
            )

            _epsilon_results = {}   # ε (normalised) -> val MSE
            _data_range = _DENORM_RANGE
            _tune_save_dir = os.path.join(shared_params['MODEL_SAVE'], 'wdro_tune')
            os.makedirs(_tune_save_dir, exist_ok=True)

            print(f'  [ ε grid (real-space radius / normalised ε): ', end='')
            for _frac, _eps_n in zip(wdro_params['EPSILON_GRID_FRAC_OF_MEAN_PRICE'], _epsilon_grid):
                print(f'${_frac * train_mean_price:.1f}/{_eps_n:.5f}  ', end='')
            print(']')

            # Per-ε resume: each finished ε is saved under {name}_epsilon_grid_partial as
            # {f'{ε:.5f}': val_mse}; a restart skips those.
            _partial_key = f'{_wdro_name}_epsilon_grid_partial'
            _meta = _load_train_meta()
            _partial = dict(_meta.get(_partial_key, {}))
            if _partial:
                for _k, _v in _partial.items():
                    _epsilon_results[float(_k)] = _v
                print(f'  [Resume] loaded {len(_partial)} previously-evaluated ε from {_partial_key}')

            for _eps in _epsilon_grid:
                _eps_key = f'{_eps:.5f}'
                if _eps_key in _partial:
                    print(f'  [Resume] skipping ε = {_eps:.5f} (val MSE = {_partial[_eps_key]:.6f})')
                    continue
                print(f'\n  [ ε grid search: training WDRO with ε = {_eps:.5f} (normalised) for {wdro_params["TUNE_ITERATION"]} epochs ]')
                _tune_wdro = WassersteinDRO(
                    windows_len=ml_public_params['WINDOW_LEN'],
                    predict_len=ml_public_params['PREDICT_LEN'],
                    epoches=wdro_params['TUNE_ITERATION'],
                    discount_factor=wdro_params['DISCOUNT_FACTOR'],
                    step_size=wdro_params['STEP_SIZE'],
                    lr=wdro_params['LR'],
                    device=shared_params['DEVICE'],
                    save_every=max(1, wdro_params['TUNE_ITERATION']),  # only one save at end
                    display_every=max(1, wdro_params['TUNE_ITERATION'] // 3),
                    epsilon=_eps,
                    inner_steps=wdro_params['INNER_STEPS'],
                    inner_lr=wdro_params['INNER_LR'],
                    # 5 decimals so each normalised ε gets a distinct filename.
                    name=f'{_wdro_name}_tune_eps{_eps:.5f}',
                )
                _tune_final_epoch = _tune_wdro.train(
                    task=None,
                    model_path=None,
                    model_save_folder_path=_tune_save_dir,
                    train_data_loader=ml_train_loader,
                    val_data_loader=ml_val_loader,
                    data_range=_data_range,
                    live_plot_callback=_make_live_plot_cb(
                        log_folder_path, ml_name=f'{_wdro_name}_tune_eps{_eps:.5f}'),
                )
                import torch.nn as _nn
                # Reload the best checkpoint that .train() wrote, the file eval would load.
                _eval_model_path = cfg.ml_model_path(
                    folder=_tune_save_dir,
                    model_name=f'{_wdro_name}_tune_eps{_eps:.5f}',
                    cuda=shared_params['DEVICE'],
                    epoch=_tune_final_epoch,
                    lr=wdro_params['LR'],
                )
                _model = torch.load(_eval_model_path, weights_only=False, map_location=shared_params['DEVICE'])
                # _eval_loss returns (val_loss, val_norm_regret); regret is None here.
                _val_mse, _ = _tune_wdro._eval_loss(
                    _model, _tune_val_loader, task=None,
                    loss_function=_nn.SmoothL1Loss(beta=0.1).to(shared_params['DEVICE']),
                    data_range=_data_range,
                )
                _epsilon_results[_eps] = _val_mse
                print(f'  [ ε = {_eps:.5f} (≈ ${_eps * _norm_range:.1f} real)  →  noise-injected val MSE = {_val_mse:.6f} ]')

                # Persist this ε's result before moving on, so a walltime hit
                # mid-grid only loses at most the in-progress ε.
                _partial[_eps_key] = _val_mse
                _meta = _load_train_meta()
                _meta[_partial_key] = _partial
                _save_train_meta(_meta)

            # Pick the ε that minimises worst-case forecast MSE.
            _best_eps = min(_epsilon_results, key=_epsilon_results.get)
            print(f'\n  [ ε grid search complete — chosen ε = {_best_eps:.5f} (≈ ${_best_eps * _norm_range:.1f} real) ]')
            for _e, _v in sorted(_epsilon_results.items()):
                _marker = '  ←  selected' if _e == _best_eps else ''
                print(f'      ε = {_e:.5f} (≈ ${_e * _norm_range:.1f} real)  val MSE = {_v:.6f}{_marker}')

            wdro_params['EPSILON'] = _best_eps
            wdro_baseline.EPSILON = _best_eps
            # Clear the partial key so later skip checks fire on {name}_epsilon_chosen.
            _meta = _load_train_meta()
            _meta[f'{_wdro_name}_epsilon_grid'] = {f'{k:.5f}': v for k, v in _epsilon_results.items()}
            _meta[f'{_wdro_name}_epsilon_chosen'] = _best_eps
            _meta.pop(_partial_key, None)
            _save_train_meta(_meta)

    # Wasserstein DRO baseline (classical, no diffusion): PGD inner max in an L2 ball,
    # MSE outer loss (task=None) by design. It targets worst-case forecast error only,
    # so the comparison with 3D-Learning isolates decision-focused training.
    if shared_params['TRAIN_DRO']:
        if _stage_done(f'{_wdro_name}_final_epoch'):
            _wdro_final_epoch = _load_train_meta()[f'{_wdro_name}_final_epoch']
            _log_skip('WDRO baseline', detail=f'final_epoch={_wdro_final_epoch}')
        else:
            print('\n  [ Training Wasserstein DRO baseline (classical, no diffusion, MSE loss) ]')
            _wdro_final_epoch = wdro_baseline.train(
                task=None,           # MSE loss — forecast robustness, not decision-focused
                val_task=None,       # no val-regret eval
                model_path=None,
                model_save_folder_path=shared_params['MODEL_SAVE'],
                train_data_loader=ml_train_loader,
                val_data_loader=ml_val_loader,
                data_range=_DENORM_RANGE,
                live_plot_callback=_make_live_plot_cb(log_folder_path, ml_name=_wdro_name),
            )
            Utils().draw_loss_plot(
                loss_list=wdro_baseline.epoches_losses_list,
                log_folder_path=log_folder_path,
                val_loss_list=wdro_baseline.epoches_val_losses_list,
                val_regret_list=wdro_baseline.epoches_val_regret_list,
                loss_label='MSE Loss',
                ml_name=_wdro_name,
            )
            _meta = _load_train_meta()
            _meta[f'{_wdro_name}_final_epoch'] = _wdro_final_epoch
            _meta[f'{_wdro_name}_epsilon'] = wdro_params['EPSILON']
            _save_train_meta(_meta)
        wdro_model_path = cfg.ml_model_path(
            folder=shared_params['MODEL_SAVE'],
            model_name=_wdro_name,
            cuda=shared_params['DEVICE'],
            epoch=_wdro_final_epoch,
            lr=wdro_params['LR'],
        )

        # WDRO_SPO: tuned ε, SPO+ outer loss, cold start.
        if _stage_done(f'{_wdro_spo_name}_final_epoch'):
            _wdro_spo_final_epoch = _load_train_meta()[f'{_wdro_spo_name}_final_epoch']
            _log_skip(f'{_wdro_spo_name} variant', detail=f'final_epoch={_wdro_spo_final_epoch}')
        else:
            print(f'\n  [ Training {_wdro_spo_name} variant (classical, no diffusion, SPO+ loss) ]')
            wdro_spo_baseline.EPSILON = wdro_params['EPSILON']  # use the tuned ε
            print(f'  | Cold-start (PTO architecture, random weights, LR={_wdro_spo_lr})', flush=True)
            # val_task=energy_task: select WDRO_SPO by val regret, as for DFL, so the
            # comparison is fair. WDRO_SPO_NO_VAL_REGRET=1 opts out. MSE WDRO stays
            # val_task=None by design.
            _wdro_spo_val_task = None if os.environ.get('WDRO_SPO_NO_VAL_REGRET') == '1' else energy_task
            _wdro_spo_final_epoch = wdro_spo_baseline.train(
                task=energy_task,    # SPO+ loss — both inner-max and outer-min are decision-focused
                val_task=_wdro_spo_val_task,
                model_path=None,
                model_save_folder_path=shared_params['MODEL_SAVE'],
                train_data_loader=ml_train_loader,
                val_data_loader=ml_val_loader,
                val_regret_loaders=_val_regret_loaders,
                data_range=_DENORM_RANGE,
                live_plot_callback=_make_live_plot_cb(log_folder_path, ml_name=_wdro_spo_name),
            )
            Utils().draw_loss_plot(
                loss_list=wdro_spo_baseline.epoches_losses_list,
                log_folder_path=log_folder_path,
                val_loss_list=wdro_spo_baseline.epoches_val_losses_list,
                val_regret_list=wdro_spo_baseline.epoches_val_regret_list,
                loss_label='SPO+ Loss',
                ml_name=_wdro_spo_name,
            )
            _meta = _load_train_meta()
            _meta[f'{_wdro_spo_name}_final_epoch'] = _wdro_spo_final_epoch
            _meta[f'{_wdro_spo_name}_epsilon'] = wdro_params['EPSILON']
            _save_train_meta(_meta)
        wdro_spo_model_path = cfg.ml_model_path(
            folder=shared_params['MODEL_SAVE'],
            model_name=_wdro_spo_name,
            cuda=shared_params['DEVICE'],
            epoch=_wdro_spo_final_epoch,
            lr=wdro_params['LR'],
        )

    # Evaluation across all test sets. Gaussian sigmas are fractions of mean train price.
    _GAUSS_SIGMAS = {
        'AEMO_gauss_sigma0.10': 0.10,
        'AEMO_gauss_sigma0.25': 0.25,
        'AEMO_gauss_sigma0.50': 0.50,
        'AEMO_gauss_sigma1.00': 1.00,
    }

    if shared_params['TEST_METHODS']:
        print(f'\n=============== Testing Methods Start ===============')

        # TEST_BUCKETS: comma-separated substrings; only matching test sets run.
        # Unset = all.
        _bucket_filter = [s.strip() for s in os.environ.get('TEST_BUCKETS', '').split(',') if s.strip()]
        _test_set = shared_params['TEST_SET']
        if _bucket_filter:
            _test_set = {k: v for k, v in _test_set.items()
                         if any(f in k for f in _bucket_filter)}
            print(f'  | TEST_BUCKETS filter | {_bucket_filter} → {list(_test_set)}')

        # (dataset, model) -> metrics, for the summary table after the loop.
        _results_records = []

        for test_set_name, test_set_path in _test_set.items():
            print('\033[92m\n *****************************************************\n\033[0m')
            print(f'  | Test Dataset | \033[93m{test_set_name}\033[0m')

            # Models take 8 input features; eval loaders must match or forward() raises.
            if test_set_path is not None:
                # Real CSV: needs TIMESTAMP / Price / TOTALDEMAND_MW for build_ml_features.
                print(f'  | Dataset Path | {test_set_path}')
                try:
                    _eva_raw_df = pd.read_csv(
                        test_set_path,
                        parse_dates=['TIMESTAMP'],
                        date_format='mixed',
                        dayfirst=False,
                    )
                except Exception:
                    _eva_raw_df = pd.read_csv(test_set_path)
                    _eva_raw_df['TIMESTAMP'] = pd.to_datetime(_eva_raw_df['TIMESTAMP'])
                eva_data_df = build_ml_features(_eva_raw_df)
            else:
                # Synthetic Gaussian OOD: noise on the Price column of the full train
                # frame, but the loader below scores ONLY the held-out test fold, so no
                # train rows leak into the scored set.
                sigma_frac = _GAUSS_SIGMAS[test_set_name]
                sigma_abs  = sigma_frac * train_mean_price
                np.random.seed(_OOD_NOISE_SEED)   # fixed test set across model seeds
                noise      = np.random.normal(0.0, sigma_abs, len(train_data_df))
                _eva_noisy_full = _train_full.copy()
                _eva_noisy_full['Price'] = (train_data_df.values + noise).clip(
                    shared_params['TEST_TASK_SET']['ENERGY_SCHEDULING']['PRICE_RANGE'][0],
                    shared_params['TEST_TASK_SET']['ENERGY_SCHEDULING']['PRICE_RANGE'][1],
                )
                eva_data_df = build_ml_features(_eva_noisy_full)
                print(f'  | Gaussian noise | sigma={sigma_frac} × mean_price'
                      f' = {sigma_abs:.2f} AUD/MWh (on Price column only; '
                      f'held-out TEST fold only)')

            # Gaussian-OOD sets (path None) reuse the train frame, so they also need
            # the split to score only the held-out test fold.
            split_test = (test_set_path == shared_params['TRAIN_FILE']
                          or test_set_path is None)

            # No len_limit: must keep all rows so the 70/15/15 split below
            # produces the SAME indices as the trainer's ml_dl. Truncating
            # would shift the test fold and overlap the trainer's val fold.
            eva_dl = Data_Loader(df=eva_data_df)
            _mv_norm_bounds = {
                'Price':          _train_norm_bounds,
                'TOTALDEMAND_MW': _demand_norm_bounds,
            }
            if split_test:
                # Keep only the held-out test fold; the val fold drove early stopping.
                _train_fold, _val_fold, eva_loader = eva_dl.ml_set2loader_mv(
                    split_test=True,
                    val_size=0.15,
                    normlized=True,
                    device=shared_params['DEVICE'],
                    batch_size=ml_train_params['BATCH_SIZE'],
                    pic_size=shared_params['ML_SEQ_LEN'],
                    norm_bounds=_mv_norm_bounds,
                )
            else:
                eva_loader = eva_dl.ml_set2loader_mv(
                    split_test=False,
                    normlized=True,
                    device=shared_params['DEVICE'],
                    batch_size=ml_train_params['BATCH_SIZE'],
                    pic_size=shared_params['ML_SEQ_LEN'],
                    norm_bounds=_mv_norm_bounds,
                )
            # data_range is the train-fold bounds — consistent with norm_bounds above.
            data_range = _DENORM_RANGE

            # Univariate eva loader: same rows as eva_loader, Price only.
            _eva_dl_uv = Data_Loader(df=eva_data_df[['Price']].copy())
            if split_test:
                _t_uv, _v_uv, eva_loader_uv = _eva_dl_uv.ml_set2loader(
                    split_test=True,
                    val_size=0.15,
                    normlized=True,
                    device=shared_params['DEVICE'],
                    batch_size=ml_train_params['BATCH_SIZE'],
                    pic_size=shared_params['ML_SEQ_LEN'],
                    norm_bounds=_train_norm_bounds,
                )
            else:
                eva_loader_uv = _eva_dl_uv.ml_set2loader(
                    split_test=False,
                    normlized=True,
                    device=shared_params['DEVICE'],
                    batch_size=ml_train_params['BATCH_SIZE'],
                    pic_size=shared_params['ML_SEQ_LEN'],
                    norm_bounds=_train_norm_bounds,
                )

            # Each method runs only if its checkpoint exists; missing ones skip with a note.
            def _test_or_skip(label, runner, path, loader=None):
                # TEST_METHOD_ONLY: comma-separated substrings; a method whose label
                # matches none skips. Unset = every method runs.
                _only = [x.strip() for x in
                         os.environ.get('TEST_METHOD_ONLY', '').split(',') if x.strip()]
                if _only and not any(x in label for x in _only):
                    print(f'  [ {label} ] SKIP — TEST_METHOD_ONLY={_only}')
                    return
                if not os.path.exists(path):
                    print(f'  [ {label} ] SKIP — checkpoint not found: {path}')
                    return
                print(f'  [ {label} ]')
                # Reset the greedy-fallback counters so the fraction read below covers
                # THIS method × test set only; _solve_batch accumulates them.
                energy_task._greedy_solves = 0
                energy_task._total_solves  = 0
                energy_task._greedy_warned = False
                _metrics = runner.test_task(
                    model_path=path,
                    test_data_loader=loader if loader is not None else eva_loader,
                    task=energy_task,
                    data_range=data_range,
                )
                if _metrics:
                    _g = getattr(energy_task, '_greedy_solves', 0)
                    _tot = getattr(energy_task, '_total_solves', 0)
                    _gfrac = (_g / _tot) if _tot else 0.0
                    if _gfrac > 0:
                        print(f'  [ {label} ] WARNING — {_g}/{_tot} '
                              f'({100*_gfrac:.1f}%) solves were GREEDY fallback; '
                              f'this row\'s regret is CONTAMINATED.')
                    _results_records.append(
                        {'dataset': test_set_name, 'model': label,
                         'greedy_frac': _gfrac, **_metrics})

            if not _MV3DL_ONLY:
                _test_or_skip('ConstantMean',  ml_const,          const_model_path)
                _test_or_skip('PTO',           ml_pto,            ml_pto_model_path)
                _test_or_skip('DFL',           ml,                ml_model_path)
                _test_or_skip('DRO',           wdro_baseline,     wdro_model_path)
                _test_or_skip('DRO_SPO',       wdro_spo_baseline, wdro_spo_model_path)
                # No univariate 3D-Learning row: its training is retired, and a stale
                # checkpoint would score against fresh baselines.
            # Multivariate F=8 3DL variant — multivariate loader (input_size=8).
            if _mv3dl_path is not None:
                _test_or_skip(_mv3dl_label, ml_mv3dl, _mv3dl_path, loader=eva_loader)

        print('\033[92m\n *****************************************************\n\033[0m')
        print(f'=============== Testing Methods End ===============')

        # Summary: rows = model, cols = dataset, cell = Norm Regret. Writes the pivot
        # CSV and a long-format CSV with every metric.
        if _results_records:
            import csv as _csv
            _models = []
            for _r in _results_records:
                if _r['model'] not in _models:
                    _models.append(_r['model'])
            _datasets = list(_test_set.keys())
            _norm = {(_r['model'], _r['dataset']): _r['Norm Regret']
                     for _r in _results_records}

            _cw = 16                       # column width for the printed table
            _mw = max((len(m) for m in _models), default=8)
            _hdr = ' ' * _mw + '  ' + '  '.join(f'{d[:_cw]:>{_cw}}' for d in _datasets)
            _lines = ['', '  ===== Norm Regret (rows=model, cols=dataset) =====', _hdr]
            for m in _models:
                _cells = '  '.join(
                    (f'{_norm[(m, d)]:>{_cw}.6f}' if (m, d) in _norm else f'{"—":>{_cw}}')
                    for d in _datasets)
                _lines.append(f'{m:<{_mw}}  {_cells}')
            print('\n'.join(_lines))

            _pivot_csv = os.path.join(log_folder_path, 'results_norm_regret.csv')
            with open(_pivot_csv, 'w', newline='') as _f:
                _w = _csv.writer(_f)
                _w.writerow(['model'] + _datasets)
                for m in _models:
                    _w.writerow([m] + [(f'{_norm[(m, d)]:.6f}' if (m, d) in _norm else '')
                                       for d in _datasets])

            _long_csv = os.path.join(log_folder_path, 'results_all_metrics.csv')
            _metric_cols = ['MSE Loss', 'Actual Cost', 'Task Regret', 'Norm Regret']
            with open(_long_csv, 'w', newline='') as _f:
                _w = _csv.writer(_f)
                _w.writerow(['dataset', 'model'] + _metric_cols)
                for _r in _results_records:
                    _w.writerow([_r['dataset'], _r['model']]
                                + [_r.get(c, '') for c in _metric_cols])

            print(f'\n  | Norm-regret table | {_pivot_csv}')
            print(f'  | All-metrics CSV   | {_long_csv}')

# Restore stdout
print(f'\n  | Log Saved | {log_file_path}')
sys.stdout = sys.__stdout__
