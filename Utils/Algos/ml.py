import copy
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn   # `F` shadowed by _unfold_seq_batch's
                                   # feature-count return value
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from Utils.utils import timer_with_memory, cleanup_epoch_checkpoints
from Utils.config import Config
from Utils.Net.task_net import DeepLSTM


cfg = Config()
algo_name = 'ML'

# Outer-epoch resume cadence: a walltime kill loses at most N-1 epochs.
_RESUME_EVERY = 5

# SPO+ has no gradient on prediction magnitude: its subgradient is
# 2·(w*(2ŷ-y) - w*(y)), a schedule indicator that vanishes wherever the
# schedules match. Without a magnitude anchor the model drifts to noise that
# moves the LP but does not approximate y. The anchor goes in both train and
# eval of the SPO+ branch.
# At λ=0.5 the anchor carries most of the gradient. That is intended for the
# warm start from PTO: SPO+ refines PTO's weights and does not tear them up.
# Lower λ with a large LR made DFL collapse.
LAMBDA_MSE_ANCHOR = 0.5


def _unfold_seq_batch(seq_batch, windows_len, predict_len, n_steps):
    """Slide windows over the time dim of a batch and split into LSTM input
    and the price-only target.

    seq_batch: (B, L) [univariate-legacy] or (B, L, F) [multivariate].
    Returns:
      input_unfolded:  (B, n_unfolds, windows_len, F)
      target_unfolded: (B, n_unfolds, predict_len)  — price column only
      F: number of input features (1 for univariate).
    The caller may further index [:, :n_steps] or subsample as it sees fit.
    """
    if seq_batch.dim() == 2:
        seq_batch = seq_batch.unsqueeze(-1)
    B, L, F = seq_batch.shape
    inp = seq_batch.unfold(1, windows_len, 1)                 # (B, L-W+1, F, W)
    inp = inp.permute(0, 1, 3, 2).contiguous()                # (B, L-W+1, W, F)
    # Target = price (column 0), starting predict_len steps after the window end.
    price = seq_batch[..., 0]                                  # (B, L)
    tgt = price.unfold(1, predict_len, 1)[:, windows_len:windows_len + n_steps]
    return inp[:, :n_steps], tgt, F


def _phase_roll_price(seq_batch, period):
    """Phase augmentation: roll ONLY the price channel (column 0) of each
    sequence by a random offset, leaving demand and calendar fixed. This breaks
    the hour->price link, so the forecaster must read cheap hours from recent
    price history. One offset per sequence keeps each window's history->target
    order intact. `period` = daily slot count (48 @ 30-min). Training only.
    Consumes the torch RNG.
    """
    if seq_batch.dim() != 3:
        return seq_batch
    B, L, _F = seq_batch.shape
    shifts = torch.randint(0, period, (B,), device=seq_batch.device)
    base = torch.arange(L, device=seq_batch.device).unsqueeze(0).expand(B, L)
    idx = (base - shifts.unsqueeze(1)) % L                    # (B, L) per-row roll
    out = seq_batch.clone()
    out[..., 0] = torch.gather(seq_batch[..., 0], 1, idx)
    return out


def _solar_augment_price(seq_batch, data_range, alpha_max, prob=1.0, alpha=None,
                         slot_weight=None, sin_h_col=2, cos_h_col=3,
                         sunrise=6.0, sunset=18.0):
    """Solar augmentation: a clear-sky midday discount on the price channel.
    It is tied to hour of day, so the forecaster can learn the midday-cheap
    regime from its sin_h/cos_h features.

    Per sequence: subtract  alpha * solar_bell(hour)  from the REAL price, where
    solar_bell is a half-sine over daylight peaking at solar noon (0 at night).
    Applied in normalised space via the LINEAR ML data_range, so a Δ$ maps to
    Δp_norm = -alpha*bell / (hi-lo). Hour of day comes from the unnormalised
    sin_h/cos_h channels. Training only. Consumes the torch RNG.

    Args:
        data_range : (lo, hi) the ML price denorm range (linear).
        alpha_max  : max solar discount in $/MWh (penetration knob).
        prob       : fraction of sequences to augment (rest stay night-cheap, so
                     the model learns BOTH regimes).
        alpha      : None -> per-sequence U[0, alpha_max]; a scalar -> fixed.
    """
    if seq_batch.dim() != 3 or seq_batch.shape[-1] <= cos_h_col:
        return seq_batch
    B, L, _F = seq_batch.shape
    lo, hi = min(data_range), max(data_range)
    rng = max(hi - lo, 1e-6)
    # hour_frac in [0,24) from sin_h=sin(2π h/24), cos_h=cos(2π h/24).
    ang = torch.atan2(seq_batch[..., sin_h_col], seq_batch[..., cos_h_col])
    hour = (ang % (2 * math.pi)) / (2 * math.pi) * 24.0           # (B, L)
    day = (hour >= sunrise) & (hour <= sunset)
    bell = torch.sin(math.pi * (hour - sunrise) / max(sunset - sunrise, 1e-6))
    bell = torch.where(day, bell.clamp(min=0.0), torch.zeros_like(bell))  # (B, L)
    if slot_weight is not None:
        # slot_weight (B, L), mean ~1.0, puts the discount on decision slots.
        bell = bell * slot_weight
    if alpha is None:
        a = torch.rand(B, 1, device=seq_batch.device) * alpha_max
    else:
        a = torch.full((B, 1), float(alpha), device=seq_batch.device)
    if prob < 1.0:
        a = a * (torch.rand(B, 1, device=seq_batch.device) < prob).float()
    dp_norm = -(a * bell) / rng                                   # (B, L)
    out = seq_batch.clone()
    out[..., 0] = out[..., 0] + dp_norm
    # SOLAR_TRANSPORT_CLIP=1: project onto AEMO floor/cap + ramp limit. Fixed
    # market constants only, never fit to OOD prices.
    if os.environ.get('SOLAR_TRANSPORT_CLIP', '0') == '1':
        _ramp = os.environ.get('SOLAR_RAMP')
        ramp = float(_ramp) if _ramp else 0.4 * rng
        # SOLAR_RAMP<=0 disables the ramp. The ramp also filters REAL spikes on
        # un-augmented sequences; floor/cap alone is a no-op on real data.
        if ramp <= 0:
            ramp = None
        out[..., 0] = _project_validity(out[..., 0], lo, hi, ramp=ramp)
    return out


def _instr_head_augment_price(seq_batch, data_range, center_h, sign, alpha_max,
                              prob=1.0, alpha=None, width_h=2.0,
                              sin_h_col=2, cos_h_col=3, T_day=48):
    """Head augmentation along the instructor's direction. Same hour recovery,
    U[0,alpha_max] draw, `prob` mask and normalised-space subtraction as
    _solar_augment_price; the shape is a signed raised-cosine bell.

    center_h: HOUR of day [0,24) to centre the bell on.
    sign    : +1.0 if 'down' (cheaper -> subtract), -1.0 if 'up' (dearer -> add).
    alpha   : None -> per-sequence U[0, alpha_max]; scalar -> fixed.
    width_h : bell half-width in HOURS. 2.0 equals one 4-hour menu block.
    T_day   : unused.
    Training only; consumes the torch RNG.
    """
    if center_h is None or seq_batch.dim() != 3 or seq_batch.shape[-1] <= cos_h_col:
        return seq_batch
    B, L, _F = seq_batch.shape
    lo, hi = min(data_range), max(data_range)
    rng = max(hi - lo, 1e-6)
    # Hour recovery must match _solar_augment_price.
    ang = torch.atan2(seq_batch[..., sin_h_col], seq_batch[..., cos_h_col])
    hour = (ang % (2 * math.pi)) / (2 * math.pi) * 24.0           # (B, L)
    center = float(center_h) % 24.0
    # CIRCULAR distance so a bell near midnight wraps correctly (24h day).
    d = torch.abs(hour - center)
    d = torch.minimum(d, 24.0 - d)
    hw = max(float(width_h), 1e-6)
    in_span = d < hw
    bump = torch.where(in_span, 0.5 + 0.5 * torch.cos(math.pi * d / hw),
                       torch.zeros_like(hour))                    # (B, L), peak 1.0
    shape = float(sign) * bump
    if alpha is None:
        a = torch.rand(B, 1, device=seq_batch.device) * alpha_max
    else:
        a = torch.full((B, 1), float(alpha), device=seq_batch.device)
    if prob < 1.0:
        a = a * (torch.rand(B, 1, device=seq_batch.device) < prob).float()
    dp_norm = -(a * shape) / rng                                  # (B, L)
    out = seq_batch.clone()
    out[..., 0] = out[..., 0] + dp_norm
    return out


def _project_validity(price_norm, lo, hi, floor=-1000.0, cap=15000.0, ramp=None):
    """Validity projection in the normalised ML space the solar bell uses:
      1. bounds : clip REAL price to AEMO [floor, cap] (fixed, not OOD-fit).
      2. ramp   : cap |Delta price| between adjacent 30-min slots at `ramp` $/MWh.
    price_norm (B,L)=channel-0 slice; (lo,hi)=ML data_range. Returns clipped (B,L)."""
    real = price_norm * (hi - lo) + lo
    real = real.clamp(min=floor, max=cap)
    if ramp is not None:
        d = (real[:, 1:] - real[:, :-1]).clamp(min=-ramp, max=ramp)
        real = torch.cat([real[:, :1], real[:, :1] + torch.cumsum(d, dim=1)], dim=1)
    return (real - lo) / max(hi - lo, 1e-6)



class ML():
    def __init__(self,windows_len,predict_len,epoches,discount_factor,step_size,lr,device,save_every,display_every,name=None,
                 early_stopping_patience_frac=0.10, min_delta_rel=0.001):
        self.device = device
        self.LR = lr
        self.STEP_SIZE = step_size
        self.DISCOUNT_FACTOR = discount_factor
        self.EPOCHS = epoches
        self.WINDOWS_LEN = windows_len
        self.PREDICT_LEN = predict_len
        self.DISPLAY_EVERY = display_every
        self.SAVE_EVERY = save_every
        self.ALGO_NAME = name if name else algo_name
        self.epoches_losses_list = []
        # (epoch_1based, val_loss) per epoch.
        self.epoches_val_losses_list = []
        # (epoch_1based, val_norm_regret) per epoch, with
        # norm_regret = Σ(c·w*(ĉ) − c·w*(c)) / Σ|c·w*(c)|, the same formula as
        # test_task()'s "Norm Regret". Empty unless _eval_loss reaches a task.
        self.epoches_val_regret_list = []
        # Epochs without a significant val gain before stopping.
        self.EARLY_STOPPING_PATIENCE = max(1, int(early_stopping_patience_frac * epoches))
        # Only a relative gain ≥ MIN_DELTA_REL resets patience; smaller gains
        # still update the best snapshot. At 0, noise resets patience forever.
        self.MIN_DELTA_REL = min_delta_rel

    def _eval_loss(self, model, val_data_loader, task, loss_function, **kwargs):
        """Compute mean loss + normalised regret on a held-out loader.

        Returns (val_loss, val_norm_regret).
          - val_loss: scalar matching the training loss (SPO+/Huber/MSE).
            None if no loader provided.
          - val_norm_regret: scalar Σ(c·w*(ĉ) − c·w*(c)) / Σ|c·w*(c)| across
            the val set, the test_task() "Norm Regret" formula. None unless
            the caller passes val_task.

        `val_task` kwarg supplies the scheduling task for regret ONLY; the
        loss path is untouched, so PTO (task=None) can still track regret.
        `val_regret_use_mip=True` solves regret with the MIP, as test_task
        does; the default is the LP relaxation.
        """
        if val_data_loader is None:
            return None, None
        val_task = kwargs.get('val_task', None)
        val_regret_use_mip = kwargs.get('val_regret_use_mip', False)
        was_training = model.training
        model.eval()
        total = 0.0
        n = 0
        sum_regret = 0.0
        sum_abs_opt = 0.0
        with torch.no_grad():
            for batch in val_data_loader:
                seq_batch = batch[0].to(self.device)
                B = seq_batch.shape[0]
                L = seq_batch.shape[1]
                n_steps = L - self.WINDOWS_LEN - self.PREDICT_LEN + 1
                if n_steps <= 0:
                    continue
                input_unfolded, target_unfolded, F = _unfold_seq_batch(
                    seq_batch, self.WINDOWS_LEN, self.PREDICT_LEN, n_steps,
                )

                # Deterministic linspace subsample (no RNG), so val is
                # comparable across epochs.
                _SPO_WINDOW_SUBSAMPLE = 8
                if (task is not None and hasattr(task, 'spo_plus_loss')
                        and n_steps > _SPO_WINDOW_SUBSAMPLE):
                    idx = torch.linspace(0, n_steps - 1, _SPO_WINDOW_SUBSAMPLE,
                                         device=seq_batch.device).long()
                    input_unfolded  = input_unfolded[:, idx]
                    target_unfolded = target_unfolded[:, idx]
                    n_steps_used = _SPO_WINDOW_SUBSAMPLE
                else:
                    n_steps_used = n_steps

                input_seq  = input_unfolded.reshape(B * n_steps_used, self.WINDOWS_LEN, F)
                target_seq = target_unfolded.reshape(B * n_steps_used, self.PREDICT_LEN)

                y_pred = model(input_seq)

                # Outside `if task:` so PTO (task=None) can still score regret.
                if (task is not None) or (val_task is not None):
                    # No relu: it would clip negative predictions to 0. The head
                    # bounds its own output range (PRED_NORM_LO may be <0), and
                    # the task accepts negative prices (floor -$1000).
                    lo, hi = min(kwargs['data_range']), max(kwargs['data_range'])
                    y_pred_real = y_pred * (hi - lo) + lo
                    y_real = target_seq * (hi - lo) + lo

                if task:
                    if hasattr(task, 'spo_plus_loss'):
                        # Must match the training loss: early stopping reads it.
                        spo = task.spo_plus_loss(y_pred=y_pred_real, y_true=y_real)
                        anchor = Fn.smooth_l1_loss(y_pred_real, y_real, beta=0.1, reduction='mean')
                        loss = spo + LAMBDA_MSE_ANCHOR * anchor
                    else:
                        best_x, _ = task.find_best_scheduling(y_pred_real)
                        loss = torch.mean(task.loss(x=best_x, y=y_real))
                else:
                    loss = loss_function(y_pred, target_seq)

                # The LP default reads slightly low against the MIP test number.
                if val_task is not None:
                    if val_regret_use_mip:
                        _solver = getattr(val_task, 'find_best_scheduling_mip',
                                          val_task.find_best_scheduling)
                    else:
                        _solver = val_task.find_best_scheduling
                    x_hat, _ = _solver(y=y_pred_real)
                    x_opt, opt_loss = _solver(y=y_real)
                    evaluate_loss = val_task.loss(x=x_hat, y=y_real)
                    # The finiteness check must come BEFORE accumulation, or
                    # one inf/nan window poisons the selection regret.
                    if not torch.isfinite(evaluate_loss - opt_loss).all():
                        continue
                    # Regret is >= 0 by definition; clamp away solver noise.
                    sum_regret  += float((evaluate_loss - opt_loss).clamp(min=0).sum().item())
                    sum_abs_opt += float(torch.abs(opt_loss).sum().item())

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total += loss.item()
                n += 1
        if was_training:
            model.train()
        val_loss = total / max(n, 1) if n > 0 else float('inf')
        val_norm_regret = (sum_regret / sum_abs_opt) if sum_abs_opt > 0 else None
        return val_loss, val_norm_regret

    def _compute_avg_delta_norm(self, train_data_loader):
        """Empirical mean of (target - last_input_price) across all training
        sliding windows, in normalised space. Returns shape (PREDICT_LEN,)."""
        delta_sum = torch.zeros(self.PREDICT_LEN, device=self.device)
        count = 0
        with torch.no_grad():
            for batch in train_data_loader:
                seq_batch = batch[0].to(self.device)
                B = seq_batch.shape[0]
                L = seq_batch.shape[1]
                n_steps = L - self.WINDOWS_LEN - self.PREDICT_LEN + 1
                if n_steps <= 0:
                    continue
                input_unfolded, target_unfolded, _F = _unfold_seq_batch(
                    seq_batch, self.WINDOWS_LEN, self.PREDICT_LEN, n_steps,
                )
                # last_price = price (column 0) at the last input step.
                last_prices = input_unfolded[:, :, -1, 0].unsqueeze(-1)  # (B, n_steps, 1)
                delta = target_unfolded - last_prices                     # (B, n_steps, PREDICT_LEN)
                delta_sum += delta.sum(dim=(0, 1))
                count    += B * n_steps
        return delta_sum / max(count, 1)

    @timer_with_memory
    def train(self, train_data_loader, model_save_folder_path, task=None, model_path=None, val_data_loader=None, **kwargs):
        # Optional per-epoch callback that redraws the live loss plot.
        _live_plot_cb = kwargs.pop('live_plot_callback', None)
        if model_path:
            model = torch.load(model_path, weights_only=False, map_location=self.device)
        else:
            # Feature count from a batch peek: 2D batches mean F=1.
            _peek = next(iter(train_data_loader))[0]
            _F = _peek.shape[-1] if _peek.dim() == 3 else 1
            print(f'  | LSTM input_size | {_F}', flush=True)

            # Cold start for both losses: persistence-residual head, zero-init
            # fc2, empirical avg_delta bias init.
            avg_delta = self._compute_avg_delta_norm(train_data_loader)
            print(f'  | Empirical avg_delta (normalised)  '
                  f'mean={float(avg_delta.mean()):.6f}  '
                  f'std={float(avg_delta.std()):.6f}  '
                  f'min={float(avg_delta.min()):.6f}  '
                  f'max={float(avg_delta.max()):.6f}', flush=True)
            model = DeepLSTM(input_size=_F,
                             output_size=self.PREDICT_LEN,
                             avg_delta_init=avg_delta,
                             zero_init_fc2=True).to(self.device)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
            print(f"  | Using {n_gpu} GPUs with DataParallel |")
        else:
            print(f"  | Using single GPU: {self.device} |")

        print(f'\n=============== {self.ALGO_NAME} Training Start ===============')
        # PTO (task=None) uses MSE: the persistence-residual head keeps deltas
        # small, so MSE can push on rare large errors that SmoothL1 caps.
        # Other branches keep SmoothL1 so eval stays comparable across methods.
        if task is None:
            loss_function = nn.MSELoss().to(self.device)
            _loss_label = 'MSE Loss'
        else:
            loss_function = nn.SmoothL1Loss(beta=0.1).to(self.device)
            # Plot label: the loss that drives the gradient.
            _loss_label = ('SPO+ Loss' if hasattr(task, 'spo_plus_loss')
                           else 'Huber Loss')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        scheduler = StepLR(optimizer, step_size=self.STEP_SIZE, gamma=self.DISCOUNT_FACTOR)

        best_loss = float('inf')         # tracks lowest val_loss (drives val_loss patience)
        best_state = None                # snapshot at lowest val_loss OR lowest regret depending on mode
        best_epoch = 0                   # epoch the snapshot corresponds to
        no_improve_count = 0             # patience counter on val_loss
        # Regret stop state; used only when val_task is supplied.
        best_regret = float('inf')       # tracks lowest val Norm Regret
        best_regret_epoch = 0            # epoch the regret minimum corresponds to
        regret_no_improve_count = 0      # patience counter on regret
        final_epoch = self.EPOCHS
        # Phase augmentation; training only.
        _phase_aug = os.environ.get('PHASE_AUG', '0') == '1'
        _phase_period = int(os.environ.get('PHASE_AUG_PERIOD', '48'))
        if _phase_aug:
            print(f'  | PHASE_AUG ON | price-channel circular roll, period={_phase_period} slots '
                  f'(domain randomization — decorrelate price from calendar prior)', flush=True)
        #   SOLAR_AUG=1  : mix midday-cheap scenarios in (U[0,αmax], SOLAR_PROB
        #                  of sequences).
        #   SOLAR_DRO=1  : group-DRO over {night (orig), solar (αmax)}; the loss
        #                  is the worst case (or softmax-tilted) of the two.
        _solar_aug   = os.environ.get('SOLAR_AUG', '0') == '1'
        _solar_dro   = os.environ.get('SOLAR_DRO', '0') == '1'
        _solar_amax  = float(os.environ.get('SOLAR_ALPHA_MAX', '120.0'))
        _solar_prob  = float(os.environ.get('SOLAR_PROB', '0.5'))
        # SOLAR_ALPHA_FIXED: every augmented sequence gets exactly this discount
        # instead of the U[0, α_max] draw.
        _solar_afix_env = os.environ.get('SOLAR_ALPHA_FIXED')
        _solar_afix  = float(_solar_afix_env) if _solar_afix_env else None
        if _solar_afix is not None:
            assert _solar_afix > 0.0, 'SOLAR_ALPHA_FIXED must be > 0 when set'
        _solar_dro_beta = float(os.environ.get('SOLAR_DRO_BETA', '0.0'))  # 0=hard max
        if _solar_aug or _solar_dro:
            _alpha_desc = (f'α_FIXED=${_solar_afix:.0f}/MWh' if _solar_afix is not None
                           else f'α_max=${_solar_amax:.0f}/MWh')
            print(f'  | SOLAR {"DRO(group)" if _solar_dro else "AUG"} ON | '
                  f'midday clear-sky discount  {_alpha_desc}  '
                  f'prob={_solar_prob}  (calendar-correlated → learnable; '
                  f'no new data) |', flush=True)
        _stopping_metric_name = 'Val Loss' if val_data_loader is not None else 'Train Loss'
        if val_data_loader is None:
            print('  | [Warning] no val_data_loader provided — early stopping uses training loss |')

        # Resume a killed run. The file is written atomically and removed on
        # success.
        resume_path = os.path.join(model_save_folder_path, f'{self.ALGO_NAME}_resume.pt')
        start_epoch = 0
        if os.path.exists(resume_path):
            print(f'  | [Resume] Loading {self.ALGO_NAME} checkpoint from {resume_path} |')
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            _src = model.module if isinstance(model, nn.DataParallel) else model
            _src.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            start_epoch          = ckpt['epoch'] + 1
            best_loss            = ckpt['best_loss']
            best_epoch           = ckpt['best_epoch']
            best_state           = ckpt['best_state']
            no_improve_count     = ckpt['no_improve_count']
            # .get(): older checkpoints lack these keys.
            best_regret            = ckpt.get('best_regret', float('inf'))
            best_regret_epoch       = ckpt.get('best_regret_epoch', 0)
            regret_no_improve_count = ckpt.get('regret_no_improve_count', 0)
            self.epoches_losses_list = ckpt.get('epoches_losses_list', [])
            self.epoches_val_losses_list = ckpt.get('epoches_val_losses_list', [])
            self.epoches_val_regret_list = ckpt.get('epoches_val_regret_list', [])
            torch.set_rng_state(ckpt['torch_rng_state'].cpu())
            np.random.set_state(ckpt['numpy_rng_state'])
            print(f'  | [Resume] Resuming at epoch {start_epoch + 1}/{self.EPOCHS} '
                  f'(best so far: epoch {best_epoch}, {_stopping_metric_name}={best_loss:.6f}) |')

        # Keep accum=1: with few batches per epoch, accumulation leaves SPO+
        # too few optimiser steps. The clip guards single-batch blow-ups.
        _accum_steps = 1
        _grad_clip = 5.0

        # One loss path for every regime. Returns None when the batch is too
        # short to form a window. The SPO+ subsample consumes the torch RNG.
        def _seq_loss(sb):
            _B, _L = sb.shape[0], sb.shape[1]
            _n = _L - self.WINDOWS_LEN - self.PREDICT_LEN + 1
            if _n <= 0:
                return None
            _inp, _tgt, _F = _unfold_seq_batch(sb, self.WINDOWS_LEN, self.PREDICT_LEN, _n)
            _SPO_WINDOW_SUBSAMPLE = 8
            if (task is not None and hasattr(task, 'spo_plus_loss')
                    and _n > _SPO_WINDOW_SUBSAMPLE):
                _idx = torch.randperm(_n, device=sb.device)[:_SPO_WINDOW_SUBSAMPLE]
                _inp, _tgt = _inp[:, _idx], _tgt[:, _idx]
                _nu = _SPO_WINDOW_SUBSAMPLE
            else:
                _nu = _n
            _input_seq  = _inp.reshape(_B * _nu, self.WINDOWS_LEN, _F)
            _target_seq = _tgt.reshape(_B * _nu, self.PREDICT_LEN)
            _y_pred = model(_input_seq)
            if task:
                _lo, _hi = min(kwargs['data_range']), max(kwargs['data_range'])
                _ypr = _y_pred * (_hi - _lo) + _lo
                _yr  = _target_seq * (_hi - _lo) + _lo
                if hasattr(task, 'spo_plus_loss'):
                    _spo = task.spo_plus_loss(y_pred=_ypr, y_true=_yr)
                    _anchor = Fn.smooth_l1_loss(_ypr, _yr, beta=0.1, reduction='mean')
                    return _spo + LAMBDA_MSE_ANCHOR * _anchor
                _bx, _ = task.find_best_scheduling(_ypr)
                return torch.mean(task.loss(x=_bx, y=_yr))
            return loss_function(_y_pred, _target_seq)

        for epoch in range(start_epoch, self.EPOCHS):
            epoch_loss = 0
            time_step = 0
            progress_bar = tqdm(train_data_loader, desc=f"  Iteration [{epoch + 1}/{self.EPOCHS}]", leave=True)
            optimizer.zero_grad(set_to_none=True)
            n_batches = len(train_data_loader)

            for batch_idx, batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)
                if _phase_aug:
                    seq_batch = _phase_roll_price(seq_batch, _phase_period)

                _dr = kwargs.get('data_range')
                if _solar_dro and _dr is not None:
                    loss_night = _seq_loss(seq_batch)
                    _sb_solar = _solar_augment_price(seq_batch, _dr, _solar_amax,
                                                     prob=1.0, alpha=_solar_amax)
                    loss_solar = _seq_loss(_sb_solar)
                    if loss_night is None or loss_solar is None:
                        continue
                    if _solar_dro_beta > 0:           # soft group-DRO
                        # Both regimes get gradient, so the hard max cannot starve
                        # night. Standardise first so BETA is scale-free.
                        _ls = torch.stack([loss_night.detach(), loss_solar.detach()])
                        _lsn = (_ls - _ls.mean()) / (_ls.std() + 1e-6)
                        _w = torch.softmax(_solar_dro_beta * _lsn, dim=0)
                        loss = _w[0] * loss_night + _w[1] * loss_solar
                    else:                              # hard worst-case
                        loss = torch.maximum(loss_night, loss_solar)
                else:
                    _sb = seq_batch
                    if _solar_aug and _dr is not None:
                        _sb = _solar_augment_price(seq_batch, _dr, _solar_amax,
                                                   prob=_solar_prob,
                                                   alpha=_solar_afix)
                    loss = _seq_loss(_sb)
                    if loss is None:
                        continue

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[Warning] loss is nan or inf, skipping backward")
                    continue

                # Divide by accum_steps so the step sees a one-batch gradient size.
                (loss / _accum_steps).backward()

                # Also step at epoch end so a partial accumulation is not lost.
                is_step = ((batch_idx + 1) % _accum_steps == 0
                           or (batch_idx + 1) == n_batches)
                if is_step:
                    if _grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                time_step += 1

            scheduler.step()

            avg_loss = epoch_loss / max(time_step, 1)
            progress_bar.set_postfix({'  | Avg Training Loss | ': f'{avg_loss:.4f}'})
            self.epoches_losses_list.append(avg_loss)

            # Early stopping reads val loss, else train loss. With
            # val_regret_loaders=[(label, loader), ...] the selection score is
            # the MAX regret over them; val_loss still comes from the primary loader.
            _val_regret_loaders = kwargs.get('val_regret_loaders', None)
            _regret_components = None
            if _val_regret_loaders:
                # Strip val_task here, or _eval_loss runs a regret sweep on this
                # loader that is thrown away.
                _kw_noregret = {k: v for k, v in kwargs.items() if k != 'val_task'}
                val_loss, _ = self._eval_loss(model, val_data_loader, task, loss_function, **_kw_noregret)
                _regret_components = []
                for _lbl, _ldr in _val_regret_loaders:
                    _, _r = self._eval_loss(model, _ldr, task, loss_function, **kwargs)
                    _regret_components.append((_lbl, float(_r) if _r is not None else float('inf')))
                val_norm_regret = max(r for _, r in _regret_components)
            else:
                val_loss, val_norm_regret = self._eval_loss(
                    model, val_data_loader, task, loss_function, **kwargs
                )

            if val_loss is not None:
                self.epoches_val_losses_list.append((epoch + 1, float(val_loss)))
            if val_norm_regret is not None:
                self.epoches_val_regret_list.append((epoch + 1, float(val_norm_regret)))

            if _live_plot_cb is not None:
                try:
                    _live_plot_cb(
                        self.epoches_losses_list,
                        val_loss_list=self.epoches_val_losses_list,
                        val_regret_list=self.epoches_val_regret_list,
                        loss_label=_loss_label,
                    )
                except Exception as _e:
                    print(f'  | [Warning] live plot callback failed: {_e} |')

            if (epoch + 1) % self.DISPLAY_EVERY == 0:
                _regret_str = (f'  Val Norm Regret | {val_norm_regret:.6f}'
                               if val_norm_regret is not None else '')
                if _regret_components is not None:
                    _regret_str += '  [worst-case; ' + ' '.join(
                        f'{l}={r:.5f}' for l, r in _regret_components) + ']'
                if val_loss is not None:
                    print(f'  | Epoch [{epoch + 1}/{self.EPOCHS}] Train Loss | {avg_loss}  Val Loss | {val_loss}{_regret_str}')
                else:
                    print(f'  | Epoch [{epoch + 1}/{self.EPOCHS}] Avg Training Loss | {avg_loss}{_regret_str}')

            if (epoch + 1) % self.SAVE_EVERY == 0:
                save_path = cfg.ml_model_path(folder=model_save_folder_path,model_name=self.ALGO_NAME,cuda=self.device,epoch=epoch+1,lr=self.LR)
                _save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(_save, save_path)
                print(f"  | Model Saved | {save_path}")

            # Two regimes:
            #   (a) val_norm_regret is None: snapshot on val_loss.
            #   (b) val_norm_regret is set: snapshot on REGRET; the surrogate can
            #       floor while regret still moves. Stop when EITHER counter fires.
            # The val_loss counter runs in both regimes.
            if val_loss is not None:
                if val_loss < best_loss:
                    prev_best = best_loss
                    best_loss = val_loss
                    rel_improvement = (prev_best - val_loss) / max(abs(prev_best), 1e-8)
                    if rel_improvement >= self.MIN_DELTA_REL:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    # Regime (a) only.
                    if val_norm_regret is None:
                        best_epoch = epoch + 1
                        _src = model.module if isinstance(model, nn.DataParallel) else model
                        best_state = copy.deepcopy(_src.state_dict())
                else:
                    no_improve_count += 1

            # Regime (b).
            if val_norm_regret is not None:
                if val_norm_regret < best_regret:
                    best_regret = val_norm_regret
                    best_regret_epoch = epoch + 1
                    best_epoch = best_regret_epoch
                    _src = model.module if isinstance(model, nn.DataParallel) else model
                    best_state = copy.deepcopy(_src.state_dict())
                    regret_no_improve_count = 0
                else:
                    regret_no_improve_count += 1

            # Atomic write: a kill mid-write keeps the previous checkpoint.
            if (epoch + 1) % _RESUME_EVERY == 0:
                _src_save = model.module if isinstance(model, nn.DataParallel) else model
                _ckpt = {
                    'model_state':            _src_save.state_dict(),
                    'optimizer_state':        optimizer.state_dict(),
                    'scheduler_state':        scheduler.state_dict(),
                    'epoch':                  epoch,
                    'best_loss':              best_loss,
                    'best_epoch':             best_epoch,
                    'best_state':             best_state,
                    'no_improve_count':       no_improve_count,
                    'best_regret':            best_regret,
                    'best_regret_epoch':      best_regret_epoch,
                    'regret_no_improve_count': regret_no_improve_count,
                    'epoches_losses_list':    self.epoches_losses_list,
                    'epoches_val_losses_list': self.epoches_val_losses_list,
                    'epoches_val_regret_list': self.epoches_val_regret_list,
                    'torch_rng_state':        torch.get_rng_state(),
                    'numpy_rng_state':        np.random.get_state(),
                }
                _tmp_path = resume_path + '.tmp'
                torch.save(_ckpt, _tmp_path)
                os.replace(_tmp_path, resume_path)

            # ML_REGRET_ONLY_STOP=1 ignores the val_loss plateau, for when loss
            # and regret diverge.
            _regret_only_stop = os.environ.get('ML_REGRET_ONLY_STOP') == '1'
            _val_loss_plateaued = (no_improve_count >= self.EARLY_STOPPING_PATIENCE
                                   and not _regret_only_stop)
            _regret_plateaued   = (val_norm_regret is not None
                                   and regret_no_improve_count >= self.EARLY_STOPPING_PATIENCE)
            if _val_loss_plateaued or _regret_plateaued:
                _reasons = []
                if _val_loss_plateaued:
                    _reasons.append(f'val_loss plateau ({no_improve_count} epochs no ≥{self.MIN_DELTA_REL*100:.1f}% improvement)')
                if _regret_plateaued:
                    _reasons.append(f'val regret plateau ({regret_no_improve_count} epochs since last min)')
                print(f'  | Early stopping at epoch {epoch + 1} ({" + ".join(_reasons)}) |')
                final_epoch = epoch + 1
                break

        # final_epoch is the snapshot's epoch; downstream eval uses it to find
        # the checkpoint file.
        if best_state is not None:
            _restore_target = model.module if isinstance(model, nn.DataParallel) else model
            _restore_target.load_state_dict(best_state)
            if best_regret < float('inf'):
                print(f'  | Restored best checkpoint from epoch {best_epoch} (Val Norm Regret={best_regret:.6f}, Val Loss at that epoch tracked separately) |')
            else:
                print(f'  | Restored best checkpoint from epoch {best_epoch} ({_stopping_metric_name}={best_loss:.6f}) |')
            final_epoch = best_epoch
        else:
            print(f'  | [Warning] No best checkpoint captured ({_stopping_metric_name} never improved below inf) — saving last-epoch model as "best" |')

        save_path = cfg.ml_model_path(folder=model_save_folder_path,model_name=self.ALGO_NAME,cuda=self.device,epoch=final_epoch,lr=self.LR)
        _save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(_save, save_path)
        print(f"  | Model Saved (best) | {save_path}")

        # Only the best-epoch checkpoint is loaded downstream (eval, warm starts).
        cleanup_epoch_checkpoints(model_save_folder_path, self.ALGO_NAME, final_epoch)

        if os.path.exists(resume_path):
            os.remove(resume_path)
            print(f'  | [Resume] Removed resume checkpoint after successful completion |')

        print(f'=============== {self.ALGO_NAME} Training End ===============')
        return final_epoch

    @timer_with_memory
    def test_task(self, test_data_loader, model_path,task,data_range):

        model = torch.load(model_path, weights_only=False, map_location=self.device)
        # A no-op at dropout=0, but required if dropout>0 is ever set.
        model.eval()

        print(f'\n=============== {self.ALGO_NAME} Testing Start ===============')

        # Huber (beta=0.1, normalised space) so price spikes do not dominate.
        loss_function = nn.SmoothL1Loss(beta=0.1).to(self.device)

        # MIP solves are slow, so EVAL_ORIGIN_STRIDE thins forecast origins, and
        # the true-price schedule is cached on the task: it does not depend on
        # the method, so every method that shares the task reuses it.
        _origin_stride = max(1, int(os.environ.get('EVAL_ORIGIN_STRIDE', '1')))
        _oracle_cache = getattr(task, '_oracle_cache', None)
        if _oracle_cache is None:
            _oracle_cache = {}
            task._oracle_cache = _oracle_cache

        for epoch in range(1):

            epoch_loss  = 0
            epoch_loss2 = 0
            epoch_loss4 = 0   # primary metric: actual scheduling cost
            # One global ratio Σregret / Σ|opt|; a mean of batch ratios would
            # over-weight batches with small |opt_loss|.
            sum_regret = 0.0
            sum_abs_opt = 0.0
            time_step = 0
            # Raw per-instance values (real-$) for downstream intervals and
            # paired tests.
            _per_inst_regret = []
            _per_inst_absopt = []
            progress_bar = tqdm(test_data_loader, desc=f"Epoch {epoch + 1}/1", leave=True)
            for i,batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)
                if seq_batch.dim() == 2:
                    seq_batch = seq_batch.unsqueeze(-1)         # (B, L, 1)

                for t in range(self.WINDOWS_LEN, seq_batch.shape[1]-self.PREDICT_LEN+1, _origin_stride):
                    input_seq  = seq_batch[:, t - self.WINDOWS_LEN:t, :]  # (B, W, F)
                    target_seq = seq_batch[:, t:t + self.PREDICT_LEN, 0]   # (B, P) — price only
                    y_pred = model(input_seq)   # no relu: keep negative-capable head's output
                    y_pred_real = y_pred * (max(data_range)-min(data_range))+min(data_range)
                    y_real = target_seq * (max(data_range) - min(data_range)) + min(data_range)

                    # MIP when the task has one, else LP.
                    _solver = getattr(task, 'find_best_scheduling_mip', task.find_best_scheduling)
                    x_hat, lowest_loss = _solver(y=y_pred_real)
                    _ock = hash(y_real.detach().cpu().numpy().tobytes())
                    if _ock in _oracle_cache:
                        x_opt, opt_loss = _oracle_cache[_ock]
                    else:
                        # Never cache a greedy fallback (status=-1): later methods
                        # would get it silently and greedy_frac would read 0.
                        _g0 = getattr(task, '_greedy_solves', 0)
                        x_opt, opt_loss = _solver(y=y_real)
                        if getattr(task, '_greedy_solves', 0) == _g0:
                            _oracle_cache[_ock] = (x_opt, opt_loss)
                    evaluate_loss = task.loss(
                        x=x_hat,
                        y=y_real
                    )

                    # Regret is >= 0 by definition; clamp away MIP-gap noise.
                    _reg = (evaluate_loss - opt_loss).clamp(min=0)
                    regret = torch.mean(_reg)

                    loss = loss_function(y_pred, target_seq)
                    epoch_loss  += loss.item()
                    epoch_loss2 += regret.item()
                    sum_regret  += _reg.sum().item()
                    sum_abs_opt += torch.abs(opt_loss).sum().item()
                    _per_inst_regret.extend(_reg.detach().cpu().reshape(-1).tolist())
                    _per_inst_absopt.extend(
                        torch.abs(opt_loss).detach().cpu().reshape(-1).tolist())
                    # Primary metric (Simonis et al.): total actual scheduling cost
                    epoch_loss4 += torch.mean(evaluate_loss).item()
                    time_step   += 1

            avg_loss           = epoch_loss  / time_step
            avg_task_loss      = epoch_loss2 / time_step
            avg_norm_task_loss = sum_regret / max(sum_abs_opt, 1e-8)
            avg_actual_cost    = epoch_loss4 / time_step
            progress_bar.set_postfix({'Avg MSE': f'{avg_loss:.4f}'})

            print(f'\n  -------- {self.ALGO_NAME} Test Results --------')
            print(f'  | MSE Loss    | {avg_loss:.6f}')
            print(f'  | Actual Cost | {avg_actual_cost:.4f}')
            print(f'  | Task Regret | {avg_task_loss:.4f}')
            print(f'  | Norm Regret | {avg_norm_task_loss:.6f}')
            print(f'  -----------------------------------------')
        print(f'=============== {self.ALGO_NAME} Testing End ===============')
        return {
            'MSE Loss':    avg_loss,
            'Actual Cost': avg_actual_cost,
            'Task Regret': avg_task_loss,
            'Norm Regret': avg_norm_task_loss,
            'n_instances':           len(_per_inst_regret),
            'regret_per_instance':   _per_inst_regret,    # real-$, ordered by origin
            'abs_opt_per_instance':  _per_inst_absopt,    # real-$, |optimal cost|
        }


