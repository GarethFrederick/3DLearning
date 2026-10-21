"""Inner head-training loop (train_ml) for DDD_Learning."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn   # `F` shadowed by _unfold_seq_batch's
                                   # feature-count return value (see ml.py).
from tqdm import tqdm
from Utils.Algos.ml import LAMBDA_MSE_ANCHOR, _unfold_seq_batch, _instr_head_augment_price
from Utils.Downstream.opt_oracle import build_oracle


class InnerMLMixin:
    def train_ml(self, train_data_loader, model, optimizer, scheduler, epoches, task=None, **kwargs):
        # Same SPO+ recipe as ml.py / wdro.py: loss = spo + LAMBDA_MSE_ANCHOR ·
        # SmoothL1 (β=0.1), a dense anchor where the SPO+ subgradient is sparse;
        # clip_grad_norm=5.0 against single-batch blow-ups from s_theta.
        loss_function = nn.SmoothL1Loss(beta=0.1).to(self.device)
        epoches_losses = []
        _grad_clip = 5.0

        # data_range: the linear ML price bounds ($/MWh), read by the head aug.
        _solar_dr   = kwargs.get('data_range')

        # Head augmentation along the instructor's direction. It fires only once
        # self._instr_head_dir is set, by the typical instructor edit earlier in
        # the same outer epoch.
        _ih_aug = self._instr_head_aug and _solar_dr is not None
        if _ih_aug:
            _ihd = self._instr_head_dir
            _wlbl = ('NARROW(1 block)' if self._instr_head_width <= 2.0
                     else f'WIDE(±{self._instr_head_width:g}h)')
            print(f'  | [INSTR-HEAD-AUG] ON in ML inner loop | '
                  f'dir={"(pending discovery)" if _ihd is None else _ihd[2]} '
                  f'alpha_max=${self._instr_head_amax:g} prob={self._instr_head_prob} '
                  f'width={_wlbl} |', flush=True)

        # With 5 inner epochs the patience is 1, which stops after one flat
        # epoch. MV_ML_NO_EARLY_STOP=1 runs every inner epoch.
        if os.environ.get('MV_ML_NO_EARLY_STOP') == '1':
            inner_patience = epoches + 1   # never fires
        else:
            inner_patience = max(1, int(0.25 * epoches))
        best_loss = float('inf')
        no_improve_count = 0

        # Per-seq_batch SPO+/MSE loss (mirrors ml.py _seq_loss). Returns loss or None.
        def _seq_loss_3dl(sb):
            _B, _L = sb.shape[0], sb.shape[1]
            _n = _L - self.WINDOWS_LEN - self.PREDICT_LEN + 1
            if _n <= 0:
                return None
            _inp, _tgt, _F = _unfold_seq_batch(sb, self.WINDOWS_LEN, self.PREDICT_LEN, _n)
            _SUB = 8
            if (task is not None and hasattr(task, 'spo_plus_loss') and _n > _SUB):
                _idx = torch.randperm(_n, device=sb.device)[:_SUB]
                _inp, _tgt = _inp[:, _idx], _tgt[:, _idx]
                _nu = _SUB
            else:
                _nu = _n
            _iseq = _inp.reshape(_B * _nu, self.WINDOWS_LEN, _F)
            _tseq = _tgt.reshape(_B * _nu, self.PREDICT_LEN)
            _yp = model(_iseq)
            if task:
                _lo, _hi = min(kwargs['data_range']), max(kwargs['data_range'])
                _ypr = _yp * (_hi - _lo) + _lo
                _yr  = _tseq * (_hi - _lo) + _lo
                if hasattr(task, 'spo_plus_loss'):
                    _spo = task.spo_plus_loss(y_pred=_ypr, y_true=_yr)
                    _anc = Fn.smooth_l1_loss(_ypr, _yr, beta=0.1, reduction='mean')
                    return _spo + LAMBDA_MSE_ANCHOR * _anc
                _bx, _ = task.find_best_scheduling(_ypr)
                return torch.mean(task.loss(x=_bx, y=_yr))
            return loss_function(_yp, _tseq)

        for epoch in range(epoches):
            epoch_loss = 0
            time_step = 0
            _ih_n = 0                       # batches the head-aug fired on
            progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{epoches}", leave=True)

            for batch in progress_bar:
                seq_batch = batch[0].to(self.device)
                if _ih_aug and self._instr_head_dir is not None:
                    _ihc, _ihs, _ihl = self._instr_head_dir   # centre HOUR, sign, label
                    _ihw = self._instr_head_width
                    _iha = self._instr_head_amax
                    seq_batch = _instr_head_augment_price(
                        seq_batch, _solar_dr, _ihc, _ihs, _iha,
                        prob=self._instr_head_prob, width_h=_ihw)
                    _ih_n += 1
                loss = _seq_loss_3dl(seq_batch)
                if loss is None:
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[Warning] loss is nan or inf, skipping backward")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                time_step += 1

            scheduler.step()

            avg_loss = epoch_loss / max(time_step, 1)
            progress_bar.set_postfix({'  | Avg Training Loss | ': f'{avg_loss:.4f}'})
            epoches_losses.append(avg_loss)

            print(f'  | Epoch [{epoch + 1}/{epoches}] Avg Training Loss | {avg_loss}')
            if _ih_aug:
                # The true batch count; a no-op shows as 'applied to 0 batches'.
                _ihd = self._instr_head_dir
                print(f'  | [INSTR-HEAD-AUG] ep{epoch + 1} applied to {_ih_n} batches | '
                      f'{"NO DIRECTION YET" if _ihd is None else f"centre={_ihd[0]:.1f}h sign={_ihd[1]:+.0f} ({_ihd[2]})"} |',
                      flush=True)

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= inner_patience:
                print(f'  | ML inner: early stopping at epoch {epoch + 1} (no improvement for {inner_patience} / {epoches} epochs) |')
                break

        return model, epoches_losses
