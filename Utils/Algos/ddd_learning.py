import os
import re
import math
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn   # `F` shadowed by _unfold_seq_batch's
                                   # feature-count return value (see ml.py).
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from Utils.datasets import Data_Loader, inverse_mv_sample, signed_log, signed_expm1
from Utils.diffusion_utils import DiffusionProcess,diffusion_name,diffusion_name_mv
from Utils.Algos.ml import algo_name as ml_name
from Utils.Algos.ml import LAMBDA_MSE_ANCHOR, _unfold_seq_batch, ML, _instr_head_augment_price
from Utils.Net.task_net import DeepLSTM
from Utils.config import Config
from Utils.utils import timer_with_memory, cleanup_epoch_checkpoints
from Utils.Downstream.opt_oracle import build_oracle


from Utils.Algos.ddd._common import _TRAIN_YR_LO, _TRAIN_YR_HI, algo_name, cfg
from Utils.Algos.ddd.guidance import GuidanceMixin
from Utils.Algos.ddd.trend import TrendMixin
from Utils.Algos.ddd.realdata import RealDataMixin
from Utils.Algos.ddd.foldin import FoldinMixin
from Utils.Algos.ddd.instructor import InstructorMixin
from Utils.Algos.ddd.dro import DROMixin
from Utils.Algos.ddd.inner_ml import InnerMLMixin



class DDD_Learning(GuidanceMixin, TrendMixin, RealDataMixin, FoldinMixin, InstructorMixin, DROMixin, InnerMLMixin):
    def __init__(self, device,discount_factor,batch_size,pic_size,epoches,ppo_clip,display_every,
                 dro_lr,step_size,adjust_timesteps,mu,eta,budget,p_s0,ml_save_every,dfu_save_every,
                 batch_repeat,beta_1,beta_T,T,dfu_name,selected_timesteps,windows_len,predict_len,ml_lr,
                 ml_seq_len=None,run_name=None,early_stopping_patience=25,**kwargs):

        self.device = device
        self.DISCOUNT = discount_factor
        self.BATCH_SIZE = batch_size
        self.PIC_SIZE = pic_size
        self.EPOCHES = epoches
        self.DRO_INNER_EPOCHES = kwargs['dro_inner_epochs']
        self.ML_INNER_EPOCHES = kwargs['ml_inner_epochs']
        self.PPO_CLIP = ppo_clip
        self.DISPLAY_EVERY = display_every
        self.DFU_NAME = dfu_name

        self.DRO_LR = dro_lr
        self.STEP_SIZE = step_size
        self.ADJUST_TIMESTEPS = adjust_timesteps
        self.MU = mu
        self.ETA = eta
        self.BUDGET = budget
        self.P_S0 = p_s0
        self.ML_SAVE_EVERY = ml_save_every
        self.DFU_SAVE_EVERY = dfu_save_every

        self.BATCH_REPEAT = batch_repeat
        self.BETA_1 = beta_1
        self.BETA_T = beta_T
        self.T = T

        self.SELECTED_TIMESTEPS = selected_timesteps[:self.ADJUST_TIMESTEPS]

        self.WINDOWS_LEN = windows_len
        self.PREDICT_LEN = predict_len
        # ML loader row length; must be >= WINDOWS_LEN + PREDICT_LEN. Defaults to PIC_SIZE.
        self.ML_SEQ_LEN = ml_seq_len if ml_seq_len is not None else pic_size
        self.ML_LR = ml_lr
        self.RUN_NAME = run_name if run_name else algo_name
        # Cap patience at half the epoch count so early stopping can fire on short runs.
        self.EARLY_STOPPING_PATIENCE = max(1, min(early_stopping_patience, max(1, int(0.5 * epoches))))

        self.betas = torch.linspace(start=self.BETA_1, end=self.BETA_T, steps=self.T).to(device=self.device)
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=self.BETA_1, end=self.BETA_T, steps=self.T), dim=0).to(device=self.device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=self.device), self.alpha_bars[:-1]])
        self.alphas = 1 - self.betas

        self.sigmas = self.sigmas()
        # Plot timeline: _plot_step counts every inner epoch on one x-axis, in
        # training order per outer epoch: DRO inner, then ML inner, then val.
        self._plot_step = 0
        self.outer_boundaries = []
        self.dro_x, self.dro_y, self.dro_labels = [], [], []
        self.jsm_y = []
        self.ml_x, self.ml_y, self.ml_labels = [], [], []
        # Val curves hold (global_step, value), anchored at each outer epoch's last ML step.
        self.epoches_val_losses_list = []   # (global_step, val SPO+ loss)  — IN-DIST
        self.epoches_val_regret_list = []   # (global_step, norm regret)    — IN-DIST
        # Worst case over the oodval splits; filled only when val_regret_loaders is passed.
        self.epoches_oodval_losses_list = []   # (global_step, worst-case oodval SPO+ loss)
        self.epoches_oodval_regret_list = []   # (global_step, worst-case oodval norm regret)
        self.cur_r_theta = None
        self.next_r_theta = None

        # Decision-divergence guidance for s_theta sampling; built in train().
        # None => plain ancestral sampling (GUIDE_DIVERGENCE unset; see guidance.py).
        self._guidance_fn = None

        # n_features > 1: ModelMV on (B, F, L) sequences, not the (B, 1, P, P) image net.
        #   MV_META: meta.json dict used to inverse-transform samples to real $/MWh.
        #   MV_DDPM_NORM_BOUNDS/LOG_FEATURES: the DDPM's native (signed-log) scaling for s0 windows.
        #   MV_LOADER_NORM_BOUNDS: train-fold ML bounds, so the head sees val/test scaling.
        self.N_FEATURES = kwargs.get('n_features', 1)
        self.MV = self.N_FEATURES > 1
        self.MV_SEQ_LEN = kwargs.get('mv_seq_len', self.ML_SEQ_LEN)
        self.MV_STRIDE = kwargs.get('mv_stride', 4)
        self.MV_META = kwargs.get('mv_meta', None)
        self.MV_LOADER_NORM_BOUNDS = kwargs.get('mv_loader_norm_bounds', None)
        if self.MV:
            _m = self.MV_META or {}
            self.MV_DDPM_NORM_BOUNDS = {k: tuple(v) for k, v in _m.get('norm_bounds', {}).items()}
            self.MV_LOG_FEATURES = set(_m.get('log_features', []))
            self.MV_FEATURE_NAMES = _m.get('feature_names', ['Price', 'TOTALDEMAND_MW'])
            # The last n_cond channels are clean calendar conditioning; noise and
            # mu_t apply to the leading n_stoch channels only. Cond windows come from MV_COND_POOL.
            self.MV_N_COND = int(_m.get('n_cond', 0))
            self.MV_N_STOCH = self.N_FEATURES - self.MV_N_COND
            self.MV_COND_POOL = None
            _cp_path = _m.get('cond_pool_path')
            if self.MV_N_COND > 0 and _cp_path:
                self.MV_COND_POOL = np.load(_cp_path).astype(np.float32)

    @timer_with_memory
    def train(self ,ml_model_path ,dfu_model_path , z0,s0, z0_trajectory ,model_save_folder_path,task=None,val_data_loader=None,**kwargs):
        # Live plot callback (see ml.py); fires once per outer epoch.
        _live_plot_cb = kwargs.pop('live_plot_callback', None)

        def forward_fn(z_t, t_tensor):
            t = int(t_tensor.item())
            return dfu_theta(z_t, t)

        dfu_0 = torch.load(dfu_model_path, weights_only=False, map_location=self.device)
        if ml_model_path is not None:
            print(f'  | Warm-start ML head from | {ml_model_path}', flush=True)
            hw = torch.load(ml_model_path, weights_only=False, map_location=self.device)
            # Keep the checkpoint's own pred_lo/pred_hi; PRED_NORM_LO/HI override only
            # when set. Trap: forcing the [0,1] default clips a negative-capable head.
            # Checkpoints without the attrs fall back to [0,1], as task_net does.
            _lo_env = os.environ.get('PRED_NORM_LO')
            _hi_env = os.environ.get('PRED_NORM_HI')
            if _lo_env is not None:
                hw.pred_lo = float(_lo_env)
            elif not hasattr(hw, 'pred_lo'):
                hw.pred_lo = 0.0
            if _hi_env is not None:
                hw.pred_hi = float(_hi_env)
            elif not hasattr(hw, 'pred_hi'):
                hw.pred_hi = 1.0
            _src = ('env override' if (_lo_env is not None or _hi_env is not None)
                    else 'persisted from checkpoint')
            print(f'  | Head range on warm-start | [{hw.pred_lo}, {hw.pred_hi}] ({_src})', flush=True)
        else:
            # Cold start with the same init as ml.py / wdro.py; input_size and
            # avg_delta come from z0, the series that seeds s_theta.
            if self.MV:
                _z0_loader = Data_Loader(df=z0).ml_set2loader_mv(
                    split_test=False,
                    device=self.device,
                    batch_size=self.BATCH_SIZE,
                    normlized=True,
                    pic_size=self.ML_SEQ_LEN,
                    norm_bounds=self.MV_LOADER_NORM_BOUNDS,
                )
            else:
                _z0_loader = Data_Loader(df=z0).ml_set2loader(
                    split_test=False,
                    device=self.device,
                    batch_size=self.BATCH_SIZE,
                    normlized=True,
                    pic_size=self.ML_SEQ_LEN,
                )
            _peek = next(iter(_z0_loader))[0]
            _F = _peek.shape[-1] if _peek.dim() == 3 else 1
            print(f'  | Cold-start ML head (PTO architecture, random weights)', flush=True)
            print(f'  | LSTM input_size | {_F}', flush=True)
            avg_delta = ML._compute_avg_delta_norm(self, _z0_loader)
            print(f'  | Empirical avg_delta (normalised)  '
                  f'mean={float(avg_delta.mean()):.6f}  '
                  f'std={float(avg_delta.std()):.6f}  '
                  f'min={float(avg_delta.min()):.6f}  '
                  f'max={float(avg_delta.max()):.6f}', flush=True)
            hw = DeepLSTM(input_size=_F,
                          output_size=self.PREDICT_LEN,
                          avg_delta_init=avg_delta,
                          zero_init_fc2=True).to(self.device)
        dfu_theta = copy.deepcopy(dfu_0)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            dfu_theta = nn.DataParallel(dfu_theta)
            hw = nn.DataParallel(hw)
            print(f"  | Using {n_gpu} GPUs with DataParallel |")

        print(f'\n=============== {algo_name} Training Start ===============')
        print(f'  | Selected Timesteps | {self.SELECTED_TIMESTEPS}')
        if val_data_loader is None:
            print('  | [Warning] no val_data_loader — early stopping uses inner ML loss on diffusion-generated worst-case data |')

        if self.MV and task is not None:
            self._guidance_fn = self._build_divergence_guidance(
                task, kwargs.get('prior_price_profile'))

        # Fold-in co-evolution (FOLD_IN=1; default off). State:
        #   _foldin_buffer : growing native-space (B,F,L) worst-case windows
        #   _foldin_w_ref  : oracle.sensitivity(c_ref), next round's guidance signal
        #   _foldin_rows   : per-round JSON metric rows
        self._foldin_on = (os.environ.get('FOLD_IN') == '1')
        self._foldin_buffer = None
        self._foldin_w_ref = None
        self._foldin_rows = []
        # Rows added per round; the flat buffer holds no round boundaries.
        self._foldin_buffer_round_sizes = []
        self._gen_transport = False
        # train_ml() reads these every outer epoch, so they must exist with
        # fold-in off. The _foldin_on block below overrides them.
        self._instr_head_aug = False
        self._instr_head_dir = None
        self._instr_head_accum = False
        self._instr_head_aim_bank = []
        self._instr_tie_enabled = False
        self._instr_tie_pct = 0.05
        self._instr_tie_max = 20
        self._instr_tie_memory = False
        self._instr_tie_bank = []          # this ROUND's (centre,sign,label,width,depth)
        self._instr_head_width = 2.0
        self._instr_head_centroid = False
        self._instr_head_wave = False
        self._instr_head_harm = 2
        self._wave_bank_k = 0
        self._wave_refresh = 1
        self._wave_bank = None
        self._instr_head_regime = False
        self._instr_head_aim_trend = False
        self._trend_aim_cache = None
        self._instr_head_depth_trend = False
        self._trend_depth_cache = None
        self._trend_depth_loocv_cache = None
        self._instr_head_aim_filter = False
        self._instr_head_aim_filter_stat = 'p5'
        self._head_rows_mult = 1
        self._gen_rows_mult = 1
        self._tailwass_cache = None
        self._instr_head_discover_width = False
        self._instr_real_windows = False
        self._instr_head_discover_depth = False
        self._instr_head_depth_cands = [40.0,80.0,120.0,200.0,300.0]
        self._instr_head_width_cands = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        # HEAD_TRAIN_REAL=1: the head trains on real 2015-2020 days, not s_theta.
        self._head_train_real = os.environ.get('HEAD_TRAIN_REAL', '0') == '1'
        # INSTR_REAL_WINDOWS=1: the instructor judges edits on real windows.
        # With HEAD_TRAIN_REAL=1 no generator output reaches the head.
        self._instr_real_windows = os.environ.get('INSTR_REAL_WINDOWS', '0') == '1'
        self._real_base_feats = None
        if self._foldin_on:
            self._foldin_replay_frac = float(os.environ.get('FOLDIN_REPLAY_FRAC', '0.3'))
            # FOLDIN_HEAD_ONLY=1: no s0 replay into the generator; the buffer
            # feeds the decision head only.
            self._foldin_head_only = os.environ.get('FOLDIN_HEAD_ONLY', '0') == '1'
            # INSTR_HEAD_AUG=1: the instructor's direction reaches the head as a broad
            # augmentation of every training day; buffer->head replay is off.
            # Needs GEN_TRANSPORT_MODE=instructor_typical and FOLDIN_HEAD_ONLY=1.
            self._instr_head_aug = os.environ.get('INSTR_HEAD_AUG', '0') == '1'
            self._instr_head_amax = float(os.environ.get('INSTR_HEAD_ALPHA_MAX', '120.0'))
            self._instr_head_prob = 0.5
            # INSTR_HEAD_DEPTH_TREND=1: depth comes from the training years' price
            # trend, not INSTR_HEAD_ALPHA_MAX. Off: no CSV is opened.
            self._instr_head_depth_trend = os.environ.get('INSTR_HEAD_DEPTH_TREND', '0') == '1'
            # Deployment horizon in calendar years; 2024.0 = midpoint of the 2023-2025 test period.
            self._instr_head_depth_horizon = float(
                os.environ.get('INSTR_HEAD_DEPTH_HORIZON', '2024.0'))
            # INSTR_HEAD_DEPTH_OBSERVED=1: depth = the observed last-year fall of the
            # aimed block's floor (p5[2019] - p5[2020]), not the trend at the horizon.
            # Neither branch draws a random number.
            self._instr_head_depth_observed = os.environ.get('INSTR_HEAD_DEPTH_OBSERVED', '0') == '1'
            assert not (self._instr_head_depth_observed and not self._instr_head_depth_trend), (
                'INSTR_HEAD_DEPTH_OBSERVED requires INSTR_HEAD_DEPTH_TREND=1 — it replaces '
                'the trend depth, so without it the knob would do nothing')
            self._instr_head_width = 2.0
            self._instr_head_aim_trend = os.environ.get('INSTR_HEAD_AIM_TREND', '0') == '1'
            # Multiplier on the head's real row count: more batches and optimiser
            # steps per inner epoch. Cost: about 41 s per batch (one solve per sequence).
            self._head_rows_mult = int(os.environ.get('HEAD_ROWS_MULT', '1'))
            self._gen_rows_mult = 1
            assert self._head_rows_mult >= 1, (
                f'HEAD_ROWS_MULT must be >= 1, got {self._head_rows_mult}')
            assert not (self._head_rows_mult > 1
                        and os.environ.get('HEAD_TRAIN_REAL', '0') != '1'), (
                'HEAD_ROWS_MULT > 1 requires HEAD_TRAIN_REAL=1 — the knob only '
                'scales the REAL row count and would otherwise do nothing.')
            # The trend depth is measured on the block the trend aim picks, so a
            # divergence-chosen aim would put the depth at the wrong hours.
            assert not (self._instr_head_depth_trend and not self._instr_head_aim_trend), (
                'INSTR_HEAD_DEPTH_TREND requires INSTR_HEAD_AIM_TREND=1 — the depth '
                'is fitted on the aimed block, so both must come from the same trend.')
            self._instr_head_discover_width = os.environ.get('INSTR_HEAD_DISCOVER_WIDTH', '0') == '1'
            self._instr_head_width_cands = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
            # Never tune these trend knobs on an oodval or oodtest score.
            self._trend_block_hours = float(os.environ.get('INSTR_TREND_BLOCK_HOURS', '4'))
            self._trend_quantile    = 0.05
            self._trend_degree      = int(os.environ.get('INSTR_TREND_DEGREE', '2'))
            assert 24.0 % self._trend_block_hours == 0, (
                f'INSTR_TREND_BLOCK_HOURS={self._trend_block_hours} does not divide 24')
            assert 0.0 < self._trend_quantile < 1.0, 'INSTR_TREND_QUANTILE must be in (0,1)'
            assert self._trend_degree >= 1, 'INSTR_TREND_DEGREE must be >= 1'
            self._trend_aim_multi_cache = None
            self._trend_multi_aims = []
            # (block, sign, label) from the instructor; while None the augmentation
            # is a logged no-op.
            self._instr_head_dir = None
            self._foldin_keep_frac   = float(os.environ.get('FOLDIN_KEEP_FRAC', '0.2'))
            self._foldin_objective   = os.environ.get('FOLDIN_OBJECTIVE', 'divergence')
            self._foldin_ref         = os.environ.get('FOLDIN_REF', 'live-head')
            # FOLDIN_REF_POOL=real builds the live-head reference from real training
            # windows; 'generated' uses z0. Neither branch draws a random number.
            self._foldin_ref_pool    = os.environ.get('FOLDIN_REF_POOL', 'generated')
            assert self._foldin_ref_pool in ('generated', 'real'), (
                f'FOLDIN_REF_POOL={self._foldin_ref_pool!r} must be generated or real')
            self._foldin_real_ref_loader_cache = None
            # Fail fast on a typo: the string dispatch would otherwise fall through silently.
            _VALID_FOLDIN_OBJ = ('divergence',)
            _VALID_FOLDIN_REF = ('live-head', 'night-prior')
            assert self._foldin_objective in _VALID_FOLDIN_OBJ, (
                f"FOLD-IN: FOLDIN_OBJECTIVE='{self._foldin_objective}' is not one of "
                f"{_VALID_FOLDIN_OBJ} (a typo would silently run 'divergence').")
            assert self._foldin_ref in _VALID_FOLDIN_REF, (
                f"FOLD-IN: FOLDIN_REF='{self._foldin_ref}' is not one of "
                f"{_VALID_FOLDIN_REF} (a typo would silently run 'night-prior').")
            # CONSOL_EVERY feeds only the OFF banner: consolidation sessions no
            # longer exist. Asking for them without FOLDIN_HEAD_ONLY=1 must crash.
            self._consol_every        = int(os.environ.get('CONSOL_EVERY', '0'))
            assert self._consol_every >= 0, (
                f'CONSOL_EVERY must be >= 0 (0 = off), got {self._consol_every}')
            assert not (self._consol_every > 0 and not self._foldin_head_only), (
                f'CONSOL_EVERY={self._consol_every} without FOLDIN_HEAD_ONLY=1 asks for '
                f'consolidation sessions, which no longer exist; set CONSOL_EVERY=0 '
                f'or FOLDIN_HEAD_ONLY=1')
            if self._consol_every > 0 and self._foldin_head_only:
                print(f'  | [FOLD-IN][CONSOL] periodic re-teach OFF | CONSOL_EVERY='
                      f'{self._consol_every} is set but FOLDIN_HEAD_ONLY=1 skips every '
                      f'session — no consolidation will run |', flush=True)
            # Seeded from MV_SEED so each seed draws its own stream; no wall-clock.
            _fold_seed = int(os.environ.get('MV_SEED', '42')) * 100003
            self._foldin_rng         = np.random.default_rng(_fold_seed)
            # Real-$ LP relaxation, the same adapter the guidance uses.
            self._foldin_oracle = build_oracle(task, device=self.device) if task is not None else None
            # Night-cheap prior in real $/MWh on the external grid: round-0
            # c_ref fallback and the night-prior REF mode.
            _ppp = kwargs.get('prior_price_profile')
            self._foldin_prior = (np.asarray(_ppp, dtype=np.float32)
                                  if _ppp is not None else None)
            # Per-round JSON goes to ./gate_out/foldin_coevo_<RUN_NAME>.json.
            self._foldin_out = os.path.join(os.getcwd(), 'gate_out')
            os.makedirs(self._foldin_out, exist_ok=True)
            self._foldin_random_dir = None   # fixed unit dir for OBJECTIVE=random
            # GEN_TRANSPORT=1: edit each selected window's price channel before it
            # enters the buffer, within the floor/cap/ramp leash. Off: windows go in unedited.
            self._gen_transport = (os.environ.get('GEN_TRANSPORT') == '1')
            if self._gen_transport:
                self._gt_alpha_max = float(os.environ.get('GEN_TRANSPORT_ALPHA', '120.0'))
                self._gt_alpha_min = 0.0
                self._gt_floor     = -1000.0
                self._gt_cap       = 15000.0
                _gt_ramp           = os.environ.get('GEN_TRANSPORT_RAMP')
                self._gt_ramp      = float(_gt_ramp) if _gt_ramp else None
                # GEN_TRANSPORT_LEASH=0 drops the validity projection (physics-invalid control).
                self._gt_leash     = (os.environ.get('GEN_TRANSPORT_LEASH', '1') == '1')
                self._gt_sunrise   = 6.0
                self._gt_sunset    = 18.0
                assert self._gt_alpha_max > 0.0, (
                    'GEN_TRANSPORT_ALPHA must be > 0 (alpha=0 is a silent no-op that '
                    'looks like an active transport arm — use GEN_TRANSPORT unset for a '
                    'no-transport control instead)')
                assert 0.0 <= self._gt_alpha_min <= self._gt_alpha_max, (
                    'GEN_TRANSPORT_ALPHA_MIN must satisfy 0 <= MIN <= ALPHA '
                    '(MIN == ALPHA gives a fixed per-window alpha)')
                assert self._gt_cap > self._gt_floor, 'GEN_TRANSPORT_CAP must exceed FLOOR'
                assert self._gt_sunset > self._gt_sunrise, 'GEN_TRANSPORT sunset must exceed sunrise'
                # Only 'instructor_typical' exists; the unset default 'solar' must fail fast.
                self._gt_mode = os.environ.get('GEN_TRANSPORT_MODE', 'solar')
                if self._gt_mode not in ('instructor_typical',):
                    raise ValueError(
                        f"GEN_TRANSPORT_MODE='{self._gt_mode}' is not one of "
                        f"('instructor_typical',) — the solar, instructor and legalnoise "
                        f"modes were removed; set GEN_TRANSPORT_MODE explicitly.")
                # Edit depths in $/MWh; the token 'floor' pushes the block to self._gt_floor.
                self._gt_edit_down = [120.0, 500.0, 'floor']
                self._gt_edit_up = [500.0, 5000.0]
                print(f'  | [FOLD-IN][GEN_TRANSPORT] MODE={self._gt_mode} | '
                      f'down={self._gt_edit_down} up={self._gt_edit_up} '
                      f'floor={self._gt_floor} cap={self._gt_cap} ramp={self._gt_ramp} |',
                      flush=True)
                print(f'  | [FOLD-IN][GEN_TRANSPORT] ON (A3 stall-breaker) | '
                      f'mode={self._gt_mode} '
                      f'alpha=[{self._gt_alpha_min},{self._gt_alpha_max}] leash={self._gt_leash} '
                      f'floor={self._gt_floor} cap={self._gt_cap} ramp={self._gt_ramp} '
                      f'sun=[{self._gt_sunrise},{self._gt_sunset}] |', flush=True)
            print(f'  | [FOLD-IN] CO-EVOLUTION ON | objective={self._foldin_objective} '
                  f'ref={self._foldin_ref} replay_frac={self._foldin_replay_frac} '
                  f'keep_frac={self._foldin_keep_frac} | '
                  f'oracle={"yes" if self._foldin_oracle is not None else "NO"} '
                  f'prior={"yes" if self._foldin_prior is not None else "NO"} |', flush=True)
            if self._foldin_head_only:
                print('  | [FOLD-IN] HEAD-ONLY mode ON | s0-replay fold-in + '
                      'consolidation sessions SKIPPED — buffer feeds the ML '
                      'head only (E3) |', flush=True)
            if self._instr_head_aug:
                _wl = ('NARROW(1 block, ±2h)' if self._instr_head_width <= 2.0
                       else f'WIDE(±{self._instr_head_width:g}h daytime bell)')
                print(f'  | [FOLD-IN] INSTR-HEAD-AUG mode ON | E3 buffer->head '
                      f'replay DISABLED; instructor direction delivered as '
                      f'broad head augmentation | alpha_max='
                      f'${self._instr_head_amax:g} prob={self._instr_head_prob} '
                      f'width={_wl} '
                      f'aim=ARGMAX(block) |',
                      flush=True)
            if not self.MV:
                print('  | [FOLD-IN][WARNING] FOLD_IN=1 only supported in MV mode '
                      '— fold-in will be inert (univariate path untouched) |', flush=True)
            if os.path.exists(os.path.join(model_save_folder_path,
                                           f'{self.RUN_NAME}_resume.pt')):
                print('  | [FOLD-IN][WARNING] a resume checkpoint exists — fold-in '
                      'buffer/w_ref/rows are restored if present, else REBUILT from '
                      'empty (depth lost). See E-resume note. |', flush=True)
            # Fail at setup, not hours in, on a config that degrades fold-in.
            if self.MV:
                assert int(self.ML_SEQ_LEN) == int(self.MV_SEQ_LEN), (
                    'FOLD-IN head re-window requires ML_SEQ_LEN==MV_SEQ_LEN')
                assert 0.0 < self._foldin_replay_frac < 1.0, (
                    f'FOLD-IN: FOLDIN_REPLAY_FRAC must be in (0,1), got '
                    f'{self._foldin_replay_frac} (=0 or >=1 breaks the weighted '
                    f'replay sampler weight rf/(1-rf)).')
                if self._foldin_prior is not None:
                    _T_ext = int(np.asarray(self._foldin_prior).shape[0])
                    assert _T_ext > 0, 'FOLD-IN: prior price profile is empty'
                    assert self.MV_SEQ_LEN % _T_ext == 0, (
                        f'FOLD-IN: generator seq len MV_SEQ_LEN={self.MV_SEQ_LEN} '
                        f'is not divisible by the task/prior grid T_ext={_T_ext}; '
                        f'the slot-of-day reshape (N*days, T_ext) would be ragged.')
                    if self._foldin_oracle is not None:
                        _T_task = int(getattr(self._foldin_oracle, 'T_external', _T_ext))
                        assert _T_ext == _T_task, (
                            f'FOLD-IN: prior length ({_T_ext}) != oracle '
                            f'T_external ({_T_task}); divergence/regret would '
                            f'mis-align the price grid against the decision grid.')
                # Without the prior and the oracle, _foldin_select falls back to
                # random while the JSON still names the objective.
                if self._foldin_objective in ('divergence', 'regret'):
                    assert self._foldin_prior is not None, (
                        f"FOLD-IN: FOLDIN_OBJECTIVE='{self._foldin_objective}' "
                        f'requires a prior_price_profile (else selection silently '
                        f'falls back to RANDOM). Pass prior_price_profile.')
                    assert self._foldin_oracle is not None, (
                        f"FOLD-IN: FOLDIN_OBJECTIVE='{self._foldin_objective}' "
                        f'requires the optimization oracle (else selection silently '
                        f'falls back to RANDOM). build_oracle returned None.')

        # Trap: both schedulers step once per INNER epoch, so with step_size=10
        # the LRs decay by DISCOUNT every 10 inner steps, not every 10 outer epochs.
        _dfu_gamma = self.DISCOUNT
        dfu_opt = torch.optim.Adam(dfu_theta.parameters(), lr=self.DRO_LR)
        dfu_scheduler = StepLR(dfu_opt, step_size=self.STEP_SIZE, gamma=_dfu_gamma)

        # Build the head optimiser once and reuse it across outer epochs; a fresh
        # Adam per epoch discards its moment estimates.
        hw_opt = torch.optim.Adam(hw.parameters(), lr=self.ML_LR)
        _ml_gamma = self.DISCOUNT
        _held = 'schedulers held constant' if (_ml_gamma == 1.0 and _dfu_gamma == 1.0) else (
            f'DECAYING: StepLR(step_size={self.STEP_SIZE}) steps once per INNER epoch, '
            f'so with MV_ML_INNER inner epochs per round the head LR halves every '
            f'{self.STEP_SIZE} inner steps')
        print(f'  | LR-SCHEDULE | head_gamma={_ml_gamma} (lr={self.ML_LR:.1e})  '
              f'gen_gamma={_dfu_gamma} (lr={self.DRO_LR:.1e}) — {_held} |', flush=True)
        hw_scheduler = StepLR(hw_opt, step_size=self.STEP_SIZE, gamma=_ml_gamma)

        best_outer_loss = float('inf')
        best_outer_epoch = 0
        best_hw_state = None
        best_dfu_state = None
        no_improve_count = 0
        final_epoch = self.EPOCHES
        _stopping_name = 'Val Norm Regret' if val_data_loader is not None else 'Inner ML Loss'

        # <RUN_NAME>_resume.pt: written atomically after every outer epoch and
        # removed only when train() completes, so its presence means a killed run.
        resume_path = os.path.join(model_save_folder_path, f'{self.RUN_NAME}_resume.pt')
        start_epoch = 0
        if os.path.exists(resume_path):
            print(f'  | [Resume] Loading 3D-Learning outer-loop checkpoint from {resume_path} |')
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            _src_dfu = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
            _src_hw  = hw.module if isinstance(hw, nn.DataParallel) else hw
            _src_dfu.load_state_dict(ckpt['dfu_state'])
            _src_hw.load_state_dict(ckpt['hw_state'])
            dfu_opt.load_state_dict(ckpt['dfu_opt_state'])
            hw_opt.load_state_dict(ckpt['hw_opt_state'])
            dfu_scheduler.load_state_dict(ckpt['dfu_scheduler_state'])
            hw_scheduler.load_state_dict(ckpt['hw_scheduler_state'])
            start_epoch       = ckpt['epoch_1'] + 1
            best_outer_loss   = ckpt['best_outer_loss']
            best_outer_epoch  = ckpt['best_outer_epoch']
            best_dfu_state    = ckpt['best_dfu_state']
            best_hw_state     = ckpt['best_hw_state']
            no_improve_count  = ckpt['no_improve_count']
            self.MU           = ckpt['MU']
            # Plot state; missing keys start the plot fresh (training unaffected).
            self._plot_step          = ckpt.get('_plot_step', 0)
            self.outer_boundaries    = ckpt.get('outer_boundaries', [])
            self.dro_x               = ckpt.get('dro_x', [])
            self.dro_y               = ckpt.get('dro_y', [])
            self.dro_labels          = ckpt.get('dro_labels', [])
            self.jsm_y               = ckpt.get('jsm_y', [])
            self.ml_x                = ckpt.get('ml_x', [])
            self.ml_y                = ckpt.get('ml_y', [])
            self.ml_labels           = ckpt.get('ml_labels', [])
            self.epoches_val_losses_list = ckpt.get('epoches_val_losses_list', [])
            self.epoches_val_regret_list = ckpt.get('epoches_val_regret_list', [])
            self.epoches_oodval_losses_list = ckpt.get('epoches_oodval_losses_list', [])
            self.epoches_oodval_regret_list = ckpt.get('epoches_oodval_regret_list', [])
            self._best_indist_regret = ckpt.get('best_indist_regret', float('inf'))
            # Fold-in state; checkpoints without these keys keep the empty init.
            if self._foldin_on:
                _fb = ckpt.get('foldin_buffer', None)
                _fw = ckpt.get('foldin_w_ref', None)
                # The buffer stays on CPU (it is concatenated with the CPU _s0_ds);
                # w_ref goes to the device for the guidance math.
                self._foldin_buffer = (_fb.cpu() if _fb is not None else None)
                self._foldin_w_ref  = (_fw.to(self.device) if _fw is not None else None)
                self._foldin_rows   = ckpt.get('foldin_rows', [])
                self._foldin_buffer_round_sizes = ckpt.get(
                    'foldin_buffer_round_sizes', [])
                # Continue the fold-in selection/alpha stream, not restart it.
                _frs = ckpt.get('foldin_rng_state', None)
                if _frs is not None:
                    self._foldin_rng.bit_generator.state = _frs
                self._foldin_have_live_ref = bool(ckpt.get('foldin_have_live_ref', False))
            try:
                if ckpt.get('torch_rng_state', None) is not None:
                    torch.set_rng_state(ckpt['torch_rng_state'].cpu())
                if ckpt.get('numpy_rng_state', None) is not None:
                    np.random.set_state(ckpt['numpy_rng_state'])
            except Exception as _rng_e:
                print(f'  | [Resume][Warning] rng-state restore failed ({_rng_e}) '
                      f'— continuing with fresh streams |', flush=True)
                # The on-disk JSON can hold more rows than the checkpoint; keep the longer list.
                try:
                    _gp = os.path.join(os.getcwd(), 'gate_out',
                                       f'foldin_coevo_{self.RUN_NAME}.json')
                    if os.path.exists(_gp):
                        import json as _json
                        with open(_gp) as _gf:
                            _disk_rows = _json.load(_gf).get('rows', [])
                        if len(_disk_rows) > len(self._foldin_rows):
                            print(f'  | [Resume][FOLD-IN] on-disk JSON has '
                                  f'{len(_disk_rows)} rows > checkpoint '
                                  f'{len(self._foldin_rows)} — loading disk rows |')
                            self._foldin_rows = _disk_rows
                except Exception as _re:
                    print(f'  | [Resume][FOLD-IN][Warning] could not reconcile '
                          f'on-disk JSON rows: {_re} |')
                if 'foldin_buffer' in ckpt:
                    print(f'  | [Resume][FOLD-IN] restored buffer='
                          f'{0 if self._foldin_buffer is None else self._foldin_buffer.shape[0]} '
                          f'rows={len(self._foldin_rows)} '
                          f'w_ref={"yes" if self._foldin_w_ref is not None else "no"} |')
                else:
                    print('  | [Resume][FOLD-IN][WARNING] checkpoint predates fold-in '
                          '— buffer/w_ref/rows REBUILT from empty (fold-in depth lost) |')
            if ckpt.get('torch_rng_state') is not None:
                torch.set_rng_state(ckpt['torch_rng_state'].cpu())
            if ckpt.get('numpy_rng_state') is not None:
                np.random.set_state(ckpt['numpy_rng_state'])
            print(f'  | [Resume] Resuming at outer epoch {start_epoch + 1}/{self.EPOCHES} '
                  f'(best so far: outer epoch {best_outer_epoch}, loss {best_outer_loss:.6f}, '
                  f'no_improve={no_improve_count}/{self.EARLY_STOPPING_PATIENCE}, MU={self.MU:.4f}) |')

        if self.MV:
            z0_loader = Data_Loader(df=z0).ml_set2loader_mv(
                split_test=False, device=self.device, batch_size=self.BATCH_SIZE,
                normlized=True, pic_size=self.ML_SEQ_LEN,
                norm_bounds=self.MV_LOADER_NORM_BOUNDS,
            )
            # s0 windows feed ModelMV.loss_fn: use the DDPM's native (signed-log)
            # scaling, NOT the ML loader bounds.
            _s0_ds = Data_Loader(df=s0).dfu_dataset_mv(
                seq_len=self.MV_SEQ_LEN, stride=self.MV_STRIDE, normlized=True,
                norm_bounds=self.MV_DDPM_NORM_BOUNDS, log_features=self.MV_LOG_FEATURES,
            )
            self._s0_loader_mv = torch.utils.data.DataLoader(
                _s0_ds, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)
            # Keep the original s0 set; each outer epoch rebuilds the replay sampler over it.
            self._foldin_s0_orig_ds = _s0_ds if self._foldin_on else None
            # Must equal the number of saved trajectory iterations: build_a0 and
            # r_theta index z0_trajectory[i] for i in range(s0_iteration).
            s0_iteration = int(z0_trajectory.shape[0])
            s0_dl = None
        else:
            z0_loader = Data_Loader(df=z0).ml_set2loader(
                                                         split_test=False,
                                                         device=self.device,
                                                         batch_size=self.BATCH_SIZE,
                                                         normlized=True,
                                                         pic_size=self.ML_SEQ_LEN
                                                         )
            s0_dl = Data_Loader(df=s0)
            s0_iteration = math.ceil(len(s0_dl.data_group) / (self.PIC_SIZE * self.PIC_SIZE * self.BATCH_SIZE)) * self.BATCH_REPEAT
            s0_dataset = s0_dl.dfu_dataset(
                                            batch_size=self.BATCH_SIZE,
                                            total_iteration=s0_iteration,
                                            normlized=True,
                                            pic_size=self.PIC_SIZE
                                            )

        a0_T = self.build_a0(
                             total_iteration=s0_iteration,
                             z0_trajectory=z0_trajectory,
                             dfu=dfu_0,
                            )

        for epoch_1 in range(start_epoch, self.EPOCHES):

            # Replay the buffer into the generator's s0 set. num_samples == N_orig
            # keeps the batch count, so _cycle() still feeds s0_iteration steps.
            if self.MV and self._foldin_on and self._foldin_buffer is not None \
                    and self._foldin_s0_orig_ds is not None \
                    and not self._foldin_head_only:
                self._s0_loader_mv, _no, _nb, _wb = self._foldin_make_weighted_loader(
                    self._foldin_s0_orig_ds, self._foldin_buffer, self.BATCH_SIZE,
                    lambda ds, samp: torch.utils.data.DataLoader(
                        ds, batch_size=self.BATCH_SIZE, sampler=samp, drop_last=True),
                )
                print(f'  | [FOLD-IN] ep{epoch_1 + 1} s0 replay loader | '
                      f'orig={_no} buffer={_nb} w_buf={_wb:.3f} '
                      f'(num_samples={_no}) |', flush=True)

            loss_hw = self.loss_hw_task(test_data_loader=z0_loader, model=hw, task=task, **kwargs)

            # loss_hw is (B, n_z0_batches): column j = generation batch j's regret,
            # since z0_loader is unshuffled and row-aligned with z0_trajectory[j].
            # ppo(batch_i) must pair r_theta(batch_i) with column batch_i. The cyclic
            # tiling below mixes batches, so only the univariate path runs it.
            if not self.MV:
                required_cols = s0_iteration * self.PIC_SIZE
                if loss_hw.shape[1] < required_cols:
                    repeat_n = math.ceil(required_cols / loss_hw.shape[1])
                    loss_hw = loss_hw.repeat(1, repeat_n)[:, :required_cols]
            else:
                # n_z0_batches must equal s0_iteration (= GEN_ITERS); a z0 regen with
                # a different row count would misalign ppo()'s columns in silence.
                assert loss_hw.shape[1] == s0_iteration, (
                    f'loss_hw columns ({loss_hw.shape[1]}) != s0_iteration '
                    f'({s0_iteration}); z0 loader batches and trajectory '
                    f'iterations are misaligned — ppo() per-batch pairing would '
                    f'be wrong. Check the z0 CSV / trajectory regen + drop_last.')

            self.outer_boundaries.append(self._plot_step + 1)  # first global step of this outer
            # MV_SKIP_GEN_TRAIN=1 skips generator training (dfu_theta, MU, dfu_scheduler)
            # for runs whose head sees real windows only. An `if`, not `continue`:
            # the fold-in step and head training below must still run.
            if os.environ.get('MV_SKIP_GEN_TRAIN', '0') != '1':
              dfu_theta.train()
              for epoch_2 in range(self.DRO_INNER_EPOCHES):
                print(f'  | Epoch [{epoch_1 + 1}/{self.EPOCHES}] - [{epoch_2 + 1}/{self.DRO_INNER_EPOCHES}] | Training DRO')
                if self.MV:
                    # Cycle the loader so DRO iteration i lines up with z0_trajectory[i].
                    def _cycle(loader):
                        while True:
                            for b in loader:
                                yield b
                    s0_dataiterator = _cycle(self._s0_loader_mv)
                else:
                    s0_dataiterator = s0_dl.dfu_set2inter(dfu_dataset=s0_dataset,batch_size=self.BATCH_SIZE)
                progress_bar = tqdm(range(s0_iteration), desc="    DRO Inner Iterations")
                jsm_loss_total = 0
                jsm_loss_count = 0     # valid (non-nan/inf-skipped) jsm steps
                dro_loss_total = 0.0   # for the timeline plot (mean over valid steps)
                dro_loss_count = 0

                for i in progress_bar:
                    s0_data = s0_dataiterator.__next__()
                    s0_data = s0_data[0].to(device=self.device)
                    if isinstance(dfu_theta, nn.DataParallel):
                        out, eps, _ = dfu_theta(s0_data, get_target=True, T_prime=self.ADJUST_TIMESTEPS)
                        jsm_loss = (out - eps).square().mean()
                    else:
                        jsm_loss = dfu_theta.loss_fn(x=s0_data, idx=None, T_prime=self.ADJUST_TIMESTEPS)

                    r_theta = self.r_theta(z0_trajectory=z0_trajectory, a0_T=a0_T,batch_i=i,forward_fn=forward_fn)
                    ppo_loss = self.ppo(r=r_theta, loss_hw=loss_hw, batch_i=i,detach_part='hw')
                    jsm_adjust_loss = self.MU * jsm_loss
                    dro_inner_loss = -(ppo_loss - jsm_adjust_loss)

                    if torch.isnan(dro_inner_loss) or torch.isinf(dro_inner_loss):
                        print("[Warning] DRO inner loss is nan or inf, skipping backward")
                        continue

                    # Count jsm on valid steps only; the MU update averages it.
                    jsm_loss_total += jsm_loss.item()
                    jsm_loss_count += 1

                    progress_bar.set_postfix(loss=dro_inner_loss.item())
                    dro_loss_total += dro_inner_loss.item()
                    dro_loss_count += 1
                    dfu_opt.zero_grad()
                    dro_inner_loss.backward()
                    # The reward is price-scaled regret; without a clip, extreme
                    # generated prices can collapse dfu_theta in one step.
                    torch.nn.utils.clip_grad_norm_(
                        dfu_theta.parameters(),
                        max_norm=float(os.environ.get('MV_DRO_GRAD_CLIP', '1.0')))
                    dfu_opt.step()

                # All steps skipped => jsm_avg = 0, not a divide by zero.
                jsm_avg_loss = jsm_loss_total / max(jsm_loss_count, 1)
                # MU multiplies an inequality (J(θ) ≤ BUDGET), so it must stay >= 0;
                # a negative MU flips the JSM sign and rewards unrealistic data.
                self.MU = max(0.0, self.MU + self.ETA * (jsm_avg_loss - self.BUDGET))
                dfu_scheduler.step()
                self._plot_step += 1
                self.dro_x.append(self._plot_step)
                self.dro_y.append(dro_loss_total / max(dro_loss_count, 1))
                self.jsm_y.append(float(jsm_avg_loss))
                self.dro_labels.append(f'{epoch_1 + 1}-D{epoch_2 + 1}')


            if self.MV:
                # Fold-in: (a) derive c_ref for next round's guidance, (b) select
                # worst-case windows, (c) grow the replay buffer.
                if self._foldin_on and self._foldin_objective != 'off':
                    # Snapshot the greedy-fallback counter before this round's oracle work.
                    _g0 = getattr(task, '_greedy_solves', 0) if task is not None else 0
                    _ref_mode = self._foldin_ref
                    # The flag keeps the live head as reference on the first resumed
                    # epoch, where epoch_1 > start_epoch is False. Round 0 uses the night prior.
                    _use_live = (_ref_mode == 'live-head'
                                 and (epoch_1 > start_epoch
                                      or getattr(self, '_foldin_have_live_ref', False)))
                    _live_curve_ok = False
                    if _use_live:
                        _ref_ld = (self._foldin_real_ref_loader()
                                   if self._foldin_ref_pool == 'real' else z0_loader)
                        if self._foldin_ref_pool == 'real':
                            print(f'  | [REF-POOL] round {epoch_1 + 1} c_ref from the REAL pool '
                                  '(not z0) |', flush=True)
                        _c_ref = self._foldin_head_belief_curve(
                            hw, _ref_ld, kwargs.get('data_range'))
                        if _c_ref is not None:
                            self._foldin_have_live_ref = True
                            _live_curve_ok = True
                        else:
                            _c_ref = self._foldin_prior
                    else:   # 'night-prior' or round-0 fallback
                        _c_ref = self._foldin_prior
                    # A constant c_ref gives a degenerate LP vertex; the flag makes
                    # the guidance closure push nothing.
                    self._foldin_ref_is_flat = bool(
                        _c_ref is not None
                        and np.asarray(_c_ref, dtype=np.float32).std() < 1e-6)
                    # w*(c_ref) for next round's guidance. Commit it to _foldin_w_ref
                    # only if this round shows no greedy fallback (check below).
                    _w_ref_new = self._foldin_compute_w_ref(_c_ref)

                    # MV_SKIP_GEN_SAMPLE=1 never samples the diffusion. The assert
                    # limits it to the config where no sample reaches the head or instructor.
                    # Rows = gen_iterations * BATCH_SIZE * MV_SEQ_LEN, as gen_s_theta_mv gives.
                    # RNG: sampling consumes the global torch stream, so skipping it
                    # shifts every later draw; this is not byte-identical.
                    if os.environ.get('MV_SKIP_GEN_SAMPLE', '0') == '1':
                        assert (self._head_train_real and self._instr_real_windows
                                and self._instr_head_aug), (
                            'MV_SKIP_GEN_SAMPLE=1 needs HEAD_TRAIN_REAL=1, '
                            'INSTR_REAL_WINDOWS=1 and INSTR_HEAD_AUG=1; without all '
                            'three a diffusion output still reaches the head or the '
                            'instructor, and blanking it would corrupt the run')
                        _n_seq = int(self._gen_rows_mult) * int(self.BATCH_SIZE)
                        _gl = int(self.MV_SEQ_LEN)
                        _gen_rows = _n_seq * _gl
                        _native = torch.zeros((_n_seq, self.N_FEATURES, _gl))
                        _price_real = np.zeros((_n_seq, _gl), dtype=np.float64)
                        s_theta_df = None
                        if epoch_1 == start_epoch:
                            print('  | [NO-GEN] diffusion never sampled and never '
                                  f'trained | head rows={_gen_rows} x '
                                  f'{self._head_rows_mult} (formula, not measured) |',
                                  flush=True)
                    else:
                        s_theta_df, _native, _price_real = self.gen_s_theta_mv(
                            diffusion_model=dfu_theta,
                            gen_iterations=self._gen_rows_mult,
                            return_native=True,
                        )
                        _gen_rows = len(s_theta_df)
                    # Swap the head's base data to real days after the diffusion call,
                    # so the instructor's stream is untouched.
                    if self._head_train_real:
                        s_theta_df = self._real_base_sample(
                            _gen_rows * self._head_rows_mult)
                    if self._instr_real_windows:
                        _price_real = self._real_price_windows(
                            _price_real.shape[0], _price_real.shape[1])
                        if epoch_1 == start_epoch:
                            print('  | [INSTR-REAL-WIN] instructor judging on REAL '
                                  'windows — NO generator output used anywhere |',
                                  flush=True)
                    _kept, _div_kept, _div_all = self._foldin_select(
                        _native, _price_real, _c_ref)
                    # A greedy fallback this round poisons the selection and w_ref:
                    # keep last round's w_ref, flag the row, warn. ASSERT_NO_GREEDY=1 hard-fails.
                    _g1 = getattr(task, '_greedy_solves', 0) if task is not None else 0
                    _greedy_contaminated = (_g1 > _g0)
                    if _greedy_contaminated:
                        print(f'  | [FOLD-IN][WARNING] ep{epoch_1 + 1} GREEDY FALLBACK '
                              f'during oracle work ({_g1 - _g0} solves) — selection + '
                              f'w_ref are CONTAMINATED. KEEPING previous round w_ref '
                              f'(not steering on this c_ref). Set ASSERT_NO_GREEDY=1 '
                              f'to hard-fail instead. |', flush=True)
                    else:
                        self._foldin_w_ref = _w_ref_new
                    self._foldin_last_greedy_contaminated = _greedy_contaminated
                    if _kept is not None and _kept.shape[0] > 0:
                        self._foldin_buffer = (_kept if self._foldin_buffer is None
                                               else torch.cat([self._foldin_buffer, _kept], dim=0))
                        self._foldin_buffer_round_sizes.append(int(_kept.shape[0]))
                    self._foldin_last_div_kept = _div_kept
                    self._foldin_last_div_all = _div_all
                    self._foldin_last_c_ref = _c_ref
                    # Real-$ price (N, L) of the folded batch; the metric row reads
                    # midday_p5 from it, not from a fresh sample.
                    self._foldin_last_price_real = _price_real
                    print(f'  | [FOLD-IN] ep{epoch_1 + 1} ref={_ref_mode} '
                          f'kept={0 if _kept is None else _kept.shape[0]} '
                          f'buffer={0 if self._foldin_buffer is None else self._foldin_buffer.shape[0]} '
                          f'div_kept={_div_kept} div_all={_div_all} '
                          f'greedy_contaminated={_greedy_contaminated} |', flush=True)
                    # SMOKE=1: the buffer must grow, and a live c_ref must be non-flat
                    # and differ from the night prior.
                    if os.environ.get('SMOKE') == '1':
                        if self._foldin_objective != 'off':
                            assert self._foldin_buffer is not None and \
                                self._foldin_buffer.shape[0] > 0, \
                                'SMOKE: fold-in buffer did not grow'
                        # Only when the belief curve returned; on fallback _c_ref IS the prior.
                        if _ref_mode == 'live-head' and _live_curve_ok \
                                and _c_ref is not None and self._foldin_prior is not None:
                            _cr = np.asarray(_c_ref, dtype=np.float32)
                            assert float(_cr.std()) > 1e-6, \
                                'SMOKE: live-head c_ref is FLAT (no decision structure)'
                            _diff = float(np.abs(_cr - self._foldin_prior[:_cr.shape[0]]).mean())
                            assert _diff > 1e-6, \
                                'SMOKE: live-head c_ref equals the night prior (not live)'
                            print(f'  | [FOLD-IN][SMOKE] c_ref_live OK | std={_cr.std():.4f} '
                                  f'mean|c_ref-prior|={_diff:.4f} |', flush=True)
                else:
                    # Twin of the branch above: keep both in step. gen_iterations is 1
                    # here, so the row formula uses 1.
                    if os.environ.get('MV_SKIP_GEN_SAMPLE', '0') == '1':
                        _gen_rows = int(self.BATCH_SIZE) * int(self.MV_SEQ_LEN)
                        s_theta_df = None
                    else:
                        s_theta_df = self.gen_s_theta_mv(
                            diffusion_model=dfu_theta,
                            gen_iterations=1,
                        )
                        _gen_rows = len(s_theta_df)
                    if self._head_train_real:
                        s_theta_df = self._real_base_sample(
                            _gen_rows * self._head_rows_mult)
                    # No batch was folded: _foldin_log_row must not reuse a stale one.
                    self._foldin_last_price_real = None
                # Head replay: weight the buffer to replay_frac; num_samples = N_orig
                # keeps the batch count. Off under INSTR_HEAD_AUG.
                _st_ds = Data_Loader(df=s_theta_df).ml_set2loader_mv(
                    split_test=False, device=self.device,
                    batch_size=self.BATCH_SIZE, normlized=True,
                    pic_size=self.ML_SEQ_LEN, norm_bounds=self.MV_LOADER_NORM_BOUNDS,
                    return_dataset=True,
                ) if (self._foldin_on and self._foldin_buffer is not None
                      and not self._instr_head_aug) else None
                if _st_ds is not None:
                    # Decode native -> real $ so the buffer gets the s_theta ML scaling.
                    _buf_df = self._foldin_native_to_real_df(self._foldin_buffer)
                    _buf_ds = Data_Loader(df=_buf_df).ml_set2loader_mv(
                        split_test=False, device=self.device,
                        batch_size=self.BATCH_SIZE, normlized=True,
                        pic_size=self.ML_SEQ_LEN, norm_bounds=self.MV_LOADER_NORM_BOUNDS,
                        return_dataset=True,
                    )
                    s_theta_loader, _hno, _hnb, _hwb = self._foldin_make_weighted_loader(
                        _st_ds, _buf_ds.tensors[0], self.BATCH_SIZE,
                        lambda ds, samp: torch.utils.data.DataLoader(
                            ds, batch_size=self.BATCH_SIZE, sampler=samp, drop_last=True),
                    )
                    # Contract: scripts/arms/gates/banner_check_a2*.sh grep this line.
                    print(f'  | [FOLD-IN->HEAD] ep{epoch_1 + 1} head replay loader | '
                          f'orig={_hno} buffer={_hnb} w_buf={_hwb:.3f} '
                          f'rf={self._foldin_replay_frac} |', flush=True)
                else:
                    s_theta_loader = Data_Loader(df=s_theta_df).ml_set2loader_mv(
                        split_test=False,
                        device=self.device,
                        batch_size=self.BATCH_SIZE,
                        normlized=True,
                        pic_size=self.ML_SEQ_LEN,
                        norm_bounds=self.MV_LOADER_NORM_BOUNDS,
                    )
            else:
                s_theta = self.gen_s_theta(
                    diffusion_model=dfu_theta,
                    gen_iterations=1,
                )

                s_theta_df = pd.DataFrame(s_theta)
                s_theta_loader = Data_Loader(df=s_theta_df).ml_set2loader(
                    split_test=False,
                    device=self.device,
                    batch_size=self.BATCH_SIZE,
                    normlized=True,
                    pic_size=self.ML_SEQ_LEN
                )


            print(f'  | Epoch [{epoch_1 + 1}/{self.EPOCHES} | Training ML based on S_theta')

            hw, epoches_losses = self.train_ml(
                model=hw,
                optimizer=hw_opt,
                scheduler=hw_scheduler,
                epoches=self.ML_INNER_EPOCHES,
                train_data_loader=s_theta_loader,
                task=task,
                **kwargs
            )

            _ml_losses = epoches_losses if epoches_losses else [float('inf')]
            for _inner_idx, _inner_loss in enumerate(_ml_losses, start=1):
                self._plot_step += 1
                self.ml_x.append(self._plot_step)
                self.ml_y.append(float(_inner_loss))
                self.ml_labels.append(f'{epoch_1 + 1}-M{_inner_idx}')

            if (epoch_1+1) % self.DFU_SAVE_EVERY == 0:
                save_path = cfg.dro_save_model_path(
                    folder=model_save_folder_path,
                    dro_name=self.RUN_NAME,
                    model_name=diffusion_name,
                    cuda=self.device,
                    epoch=(epoch_1+1),
                    lr=self.DRO_LR,
                )
                _save_dfu = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
                torch.save(_save_dfu, save_path)
                print(f"  | {algo_name}-{diffusion_name} Model Saved | {save_path}")

            if (epoch_1+1) % self.ML_SAVE_EVERY == 0:
                save_path = cfg.dro_save_model_path(
                    folder=model_save_folder_path,
                    dro_name=self.RUN_NAME,
                    model_name=ml_name,
                    cuda=self.device,
                    epoch=epoch_1+1,
                    lr=self.ML_LR,
                )
                _save_hw = hw.module if isinstance(hw, nn.DataParallel) else hw
                torch.save(_save_hw, save_path)
                print(f"  | {self.RUN_NAME}-{ml_name} Model Saved | {save_path}")

            # Early-stop signal: val regret. Without a val loader it falls back to the
            # inner ML loss on generated data, which can stop early by chance.
            outer_norm_regret = None
            outer_oodval_regret = None
            outer_oodval_loss = None
            if val_data_loader is not None:
                _hw_for_eval = hw.module if isinstance(hw, nn.DataParallel) else hw
                # Same held-out eval as ml.py: SPO+ (+Huber anchor) loss and norm regret.
                _loss_fn = nn.SmoothL1Loss(beta=0.1).to(self.device)
                outer_val_loss, outer_norm_regret = ML._eval_loss(
                    self, _hw_for_eval, val_data_loader, task=task,
                    loss_function=_loss_fn, **kwargs,
                )
                # val_regret_loaders = [(label, loader), ...] over the oodval splits;
                # the score is the MAX regret across them.
                _val_regret_loaders = kwargs.get('val_regret_loaders', None)
                if _val_regret_loaders:
                    _ood_regrets, _ood_losses = [], []
                    for _lbl, _ldr in _val_regret_loaders:
                        _ol, _or = ML._eval_loss(
                            self, _hw_for_eval, _ldr, task=task,
                            loss_function=_loss_fn,
                            val_regret_use_mip=(os.environ.get('VAL_OOD_REGRET_MIP', '0') == '1'),
                            **kwargs,
                        )
                        if _or is not None:
                            _ood_regrets.append(_or)
                        if _ol is not None:
                            _ood_losses.append(_ol)
                    if _ood_regrets:
                        outer_oodval_regret = max(_ood_regrets)   # worst-case
                    if _ood_losses:
                        outer_oodval_loss = max(_ood_losses)
            else:
                outer_val_loss = float(np.mean(epoches_losses)) if epoches_losses else float('inf')

            # Selection score. Default: in-dist norm regret. With val_regret_loaders and
            # OOD_VAL_EPSILON: minimise worst-case oodval regret, subject to in-dist
            # regret <= (1+ε) × the best in-dist regret so far.
            _eps = os.environ.get('OOD_VAL_EPSILON')
            if outer_oodval_regret is not None and _eps is not None:
                _eps = float(_eps)
                _indist = outer_norm_regret if outer_norm_regret is not None else float('inf')
                self._best_indist_regret = min(getattr(self, '_best_indist_regret', float('inf')), _indist)
                _eligible = _indist <= (1.0 + _eps) * self._best_indist_regret
                # +1e3 keeps ineligible epochs from winning selection.
                outer_loss = outer_oodval_regret if _eligible else (outer_oodval_regret + 1e3)
                _stopping_name = f'OODval Regret (ε={_eps})'
            else:
                outer_loss = outer_norm_regret if outer_norm_regret is not None else outer_val_loss

            _regret_str = (f'  Val Norm Regret = {outer_norm_regret:.6f}'
                           if outer_norm_regret is not None else '')
            _ood_str = (f'  OODval Regret(wc) = {outer_oodval_regret:.6f}'
                        if outer_oodval_regret is not None else '')
            print(f'  | Outer epoch {epoch_1 + 1} | Val SPO+ Loss = {outer_val_loss:.6f}{_regret_str}{_ood_str}  '
                  f'(best {_stopping_name}: {best_outer_loss:.6f})')

            self.epoches_val_losses_list.append((self._plot_step, float(outer_val_loss)))
            if outer_norm_regret is not None:
                self.epoches_val_regret_list.append(
                    (self._plot_step, float(outer_norm_regret))
                )
            if outer_oodval_loss is not None:
                self.epoches_oodval_losses_list.append((self._plot_step, float(outer_oodval_loss)))
            if outer_oodval_regret is not None:
                self.epoches_oodval_regret_list.append((self._plot_step, float(outer_oodval_regret)))
            if _live_plot_cb is not None:
                try:
                    _live_plot_cb(
                        dro_x=self.dro_x, dro_y=self.dro_y,
                        dro_labels=self.dro_labels, jsm_y=self.jsm_y,
                        ml_x=self.ml_x, ml_y=self.ml_y, ml_labels=self.ml_labels,
                        val_loss_list=self.epoches_val_losses_list,
                        val_regret_list=self.epoches_val_regret_list,
                        oodval_loss_list=self.epoches_oodval_losses_list,
                        oodval_regret_list=self.epoches_oodval_regret_list,
                        boundaries=self.outer_boundaries,
                    )
                except Exception as _e:
                    print(f'  | [Warning] live plot callback failed: {_e} |')

            # Per-round fold-in row, dumped atomically each epoch to
            # gate_out/foldin_coevo_<RUN_NAME>.json.
            if self.MV and self._foldin_on:
                try:
                    self._foldin_log_row(
                        epoch_1, hw, dfu_theta, z0_loader, kwargs.get('data_range'),
                        outer_norm_regret, outer_oodval_regret, outer_val_loss)
                except Exception as _fe:
                    print(f'  | [FOLD-IN][Warning] metric row failed: {_fe} |', flush=True)

            if outer_loss < best_outer_loss:
                best_outer_loss = outer_loss
                best_outer_epoch = epoch_1 + 1
                _src_dfu = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
                _src_hw  = hw.module if isinstance(hw, nn.DataParallel) else hw
                best_dfu_state = copy.deepcopy(_src_dfu.state_dict())
                best_hw_state  = copy.deepcopy(_src_hw.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Resume checkpoint: write .tmp then os.replace, so a kill leaves the last one intact.
            _src_dfu_save = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
            _src_hw_save  = hw.module if isinstance(hw, nn.DataParallel) else hw
            _ckpt = {
                'dfu_state':           _src_dfu_save.state_dict(),
                'hw_state':            _src_hw_save.state_dict(),
                'dfu_opt_state':       dfu_opt.state_dict(),
                'hw_opt_state':        hw_opt.state_dict(),
                'dfu_scheduler_state': dfu_scheduler.state_dict(),
                'hw_scheduler_state':  hw_scheduler.state_dict(),
                'epoch_1':             epoch_1,
                'best_outer_loss':     best_outer_loss,
                'best_outer_epoch':    best_outer_epoch,
                'best_dfu_state':      best_dfu_state,
                'best_hw_state':       best_hw_state,
                'no_improve_count':    no_improve_count,
                'MU':                  self.MU,
                '_plot_step':          self._plot_step,
                'outer_boundaries':    self.outer_boundaries,
                'dro_x':               self.dro_x,
                'dro_y':               self.dro_y,
                'dro_labels':          self.dro_labels,
                'jsm_y':               self.jsm_y,
                'ml_x':                self.ml_x,
                'ml_y':                self.ml_y,
                'ml_labels':           self.ml_labels,
                'epoches_val_losses_list': self.epoches_val_losses_list,
                'epoches_val_regret_list': self.epoches_val_regret_list,
                'epoches_oodval_losses_list': self.epoches_oodval_losses_list,
                'epoches_oodval_regret_list': self.epoches_oodval_regret_list,
                'best_indist_regret':  getattr(self, '_best_indist_regret', float('inf')),
                'torch_rng_state':     torch.get_rng_state(),
                'numpy_rng_state':     np.random.get_state(),
                # Fold-in keys are None/[] with FOLD_IN off, so the dict shape is stable.
                'foldin_buffer':       (self._foldin_buffer.cpu()
                                        if self._foldin_buffer is not None else None),
                'foldin_buffer_round_sizes': self._foldin_buffer_round_sizes,
                'foldin_w_ref':        (self._foldin_w_ref.cpu()
                                        if self._foldin_w_ref is not None else None),
                'foldin_rows':         self._foldin_rows,
                # _foldin_rng is a separate Generator; numpy_rng_state does not cover it.
                'foldin_rng_state':    (self._foldin_rng.bit_generator.state
                                        if self._foldin_on else None),
                # Without this flag a resumed run falls back to the night prior for one epoch.
                'foldin_have_live_ref': getattr(self, '_foldin_have_live_ref', False),
            }
            _tmp_path = resume_path + '.tmp'
            torch.save(_ckpt, _tmp_path)
            os.replace(_tmp_path, resume_path)

            if no_improve_count >= self.EARLY_STOPPING_PATIENCE:
                print(f'  | Early stopping at outer epoch {epoch_1 + 1} (no {_stopping_name} improvement for {self.EARLY_STOPPING_PATIENCE} / {self.EPOCHES} epochs) |')
                final_epoch = epoch_1 + 1
                break

        # Restore the best pair before the final save; eval loads the best epoch, not the last.
        if best_dfu_state is not None:
            _restore_dfu = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
            _restore_hw  = hw.module if isinstance(hw, nn.DataParallel) else hw
            _restore_dfu.load_state_dict(best_dfu_state)
            _restore_hw.load_state_dict(best_hw_state)
            print(f'  | Restored best 3D-Learning checkpoint from outer epoch {best_outer_epoch} ({_stopping_name}={best_outer_loss:.6f}) |')
            final_epoch = best_outer_epoch
        else:
            print(f'  | [Warning] No best 3D-Learning checkpoint captured ({_stopping_name} never improved below inf) — saving last-epoch DFU+ML pair as "best" |')

        save_path = cfg.dro_save_model_path(
            folder=model_save_folder_path, dro_name=self.RUN_NAME,
            model_name=diffusion_name, cuda=self.device,
            epoch=final_epoch, lr=self.DRO_LR,
        )
        _save_dfu = dfu_theta.module if isinstance(dfu_theta, nn.DataParallel) else dfu_theta
        torch.save(_save_dfu, save_path)
        print(f"  | {algo_name}-{diffusion_name} Model Saved (best) | {save_path}")

        save_path = cfg.dro_save_model_path(
            folder=model_save_folder_path, dro_name=self.RUN_NAME,
            model_name=ml_name, cuda=self.device,
            epoch=final_epoch, lr=self.ML_LR,
        )
        _save_hw = hw.module if isinstance(hw, nn.DataParallel) else hw
        torch.save(_save_hw, save_path)
        print(f"  | {self.RUN_NAME}-{ml_name} Model Saved (best) | {save_path}")

        # Delete per-epoch checkpoints, keeping the best. KEEP_EPOCH_CKPTS=1 keeps
        # them all; poc_eval_ood's fixed-epoch comparison needs them.
        if os.environ.get('KEEP_EPOCH_CKPTS') == '1':
            print(f'  | [Cleanup] KEEP_EPOCH_CKPTS=1 — keeping all per-epoch checkpoints '
                  f'(trajectory probe / fixed-epoch eval) |', flush=True)
        else:
            cleanup_epoch_checkpoints(model_save_folder_path, f'{self.RUN_NAME}-{diffusion_name}', final_epoch)
            cleanup_epoch_checkpoints(model_save_folder_path, f'{self.RUN_NAME}-{ml_name}', final_epoch)

        # The caller records 3DLearning_final_epoch in training_metadata.json, so
        # a resubmission skips this stage.
        if os.path.exists(resume_path):
            try:
                os.remove(resume_path)
                print(f'  | [Resume] Cleared 3D-Learning outer-loop checkpoint at {resume_path} |')
            except OSError:
                pass

        print(f'=============== {algo_name} Training End ===============')
        return final_epoch

    def sigmas(self):
        sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars) * self.betas)
        return sigmas

    def gen_s_theta(self,diffusion_model,gen_iterations):
        diffusion_model.eval()
        s_theta = []

        for _ in range(gen_iterations):
            process = DiffusionProcess(
                beta_1=self.BETA_1,
                beta_T=self.BETA_T,
                T=self.T,
                diffusion_fn=diffusion_model,
                device=self.device,
                shape=(1, self.PIC_SIZE, self.PIC_SIZE),
            )

            sample, _ = process.sampling(
                                         sampling_number=self.BATCH_SIZE,
                                         save_trajectory=False
                                        )

            generate_data = sample.detach().cpu().numpy().reshape(-1)
            s_theta.append(generate_data)

        s_theta = np.concatenate(s_theta, axis=0)

        return s_theta

    @timer_with_memory
    def test_task(self, test_data_loader, model_path, task, data_range):
        # Score with ML.test_task, the same code (oracle MIP cache, EVAL_ORIGIN_STRIDE)
        # as every other method. It reads self.{device, WINDOWS_LEN, PREDICT_LEN, ALGO_NAME}.
        if not hasattr(self, 'ALGO_NAME'):
            self.ALGO_NAME = algo_name
        return ML.test_task(
            self,
            test_data_loader=test_data_loader,
            model_path=model_path,
            task=task,
            data_range=data_range,
        )
