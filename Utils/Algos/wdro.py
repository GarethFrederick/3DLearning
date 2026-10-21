import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn  # see ml.py: `F` collides with _unfold_seq_batch's
                                  # feature-count return.
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from Utils.utils import timer_with_memory, cleanup_epoch_checkpoints
from Utils.config import Config
from Utils.Net.task_net import DeepLSTM
from Utils.Algos.ml import ML, LAMBDA_MSE_ANCHOR, _unfold_seq_batch, _solar_augment_price

cfg = Config()
algo_name = 'WassersteinDRO'

# Outer-epoch resume cadence, so a walltime kill does not restart from epoch 0.
_RESUME_EVERY = 5


class WassersteinDRO(ML):
    """
    Wasserstein-1 ball DRO (Mohajerin Esfahani & Kuhn 2018, Math. Programming).

    Tractable dual of min_θ max_{Q: W1(Q,Pn)≤ε} E_Q[L(θ;X)]:
        min_θ (1/n) Σᵢ max_{‖δᵢ‖₂ ≤ ε} L(f_θ(xᵢ+δᵢ), yᵢ)

    The inner max is solved via PGD (projected gradient ascent):
        δ ← δ + α ∇_δ L,   then project onto L2-ball of radius ε.
    ε is the Wasserstein ball radius — directly interpretable and comparable
    across models, unlike the Lagrange multiplier γ in Sinha et al.

    At test time no perturbation is applied (standard clean inference),
    inherited unchanged from ML.test_task().
    """

    def __init__(self, epsilon=0.1, inner_steps=5, inner_lr=0.01, **kwargs):
        kwargs.setdefault('name', algo_name)
        super().__init__(**kwargs)
        self.EPSILON = epsilon
        self.INNER_STEPS = inner_steps
        self.INNER_LR = inner_lr

    def _build_inner_loss(self, task, mse_fn, **kwargs):
        """Build a callable loss(model, x, y) that matches the outer training loss.

        The duality needs the inner max and outer min to use the SAME loss, so
        the SPO+ branch uses `SPO+ + LAMBDA_MSE_ANCHOR · SmoothL1` here too.
        """
        if task is None:
            return lambda model, x, y: mse_fn(model(x), y)

        lo = min(kwargs['data_range'])
        hi = max(kwargs['data_range'])
        has_spo = hasattr(task, 'spo_plus_loss')

        def task_loss(model, x, y_normalized):
            y_pred = model(x)   # no relu: keep negative-capable head's output
            y_pred_real = y_pred * (hi - lo) + lo
            y_real = y_normalized * (hi - lo) + lo
            if has_spo:
                spo = task.spo_plus_loss(y_pred=y_pred_real, y_true=y_real)
                anchor = Fn.smooth_l1_loss(y_pred_real, y_real, beta=0.1, reduction='mean')
                return spo + LAMBDA_MSE_ANCHOR * anchor
            best_x, _ = task.find_best_scheduling(y_pred_real)
            return torch.mean(task.loss(x=best_x, y=y_real))

        return task_loss

    def _worst_case_delta(self, model, input_seq, target_seq, loss_fn, delta_init=None):
        """PGD inner loop: gradient ascent on δ with L2-ball projection (radius ε).

        Random init within the ball + normalised step (2.5·ε / steps), so δ
        reaches the boundary whatever the gradient size, and avoids the flat
        region at δ=0 (matters for SPO+).

        delta_init: reuse this minibatch's previous δ; random init when the
        shape does not match. Random init consumes the torch RNG.
        """
        if delta_init is not None and delta_init.shape == input_seq.shape:
            delta = delta_init.detach().to(input_seq.device)
        else:
            # Random init: δ uniform in B_ε (direction Gaussian-then-normalised, magnitude U[0,ε])
            delta = torch.randn_like(input_seq)
            flat = delta.view(delta.shape[0], -1)
            norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            radius = self.EPSILON * torch.rand(flat.shape[0], 1, device=delta.device)
            delta = (flat / norms * radius).view_as(input_seq)

        step = 2.5 * self.EPSILON / max(self.INNER_STEPS, 1)

        for _ in range(self.INNER_STEPS):
            delta = delta.detach().requires_grad_(True)
            obj = loss_fn(model, input_seq + delta, target_seq)
            # autograd.grad keeps the inner-max from polluting model.parameters().grad
            grad = torch.autograd.grad(obj, delta, retain_graph=False)[0]
            with torch.no_grad():
                grad_flat = grad.view(grad.shape[0], -1)
                grad_norms = grad_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                grad_dir = (grad_flat / grad_norms).view_as(delta)
                delta_new = delta + step * grad_dir
                # Project onto L2-ball of radius ε (per sample)
                flat = delta_new.view(delta_new.shape[0], -1)
                norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(self.EPSILON / norms, max=1.0)
                delta = (flat * scale).view_as(delta_new)
        return delta.detach()

    @timer_with_memory
    def train(self, train_data_loader, model_save_folder_path, task=None, model_path=None, val_data_loader=None, **kwargs):
        _live_plot_cb = kwargs.pop('live_plot_callback', None)
        if model_path:
            model = torch.load(model_path, weights_only=False, map_location=self.device)
        else:
            # Feature count from a batch peek: 2D batches mean F=1.
            _peek = next(iter(train_data_loader))[0]
            _F = _peek.shape[-1] if _peek.dim() == 3 else 1
            print(f'  | LSTM input_size | {_F}', flush=True)

            # Same cold start as ml.py for both losses: persistence-residual head,
            # zero-init fc2, empirical avg_delta bias init.
            avg_delta = ML._compute_avg_delta_norm(self, train_data_loader)
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
        # Huber loss for the non-SPO+ branch; SPO+ uses task.spo_plus_loss.
        loss_fn = nn.SmoothL1Loss(beta=0.1).to(self.device)
        # Plot-axis label: the loss that drives gradients on this branch.
        if task is None:
            _loss_label = 'Huber Loss'
        elif hasattr(task, 'spo_plus_loss'):
            _loss_label = 'SPO+ Loss'
        else:
            _loss_label = 'Huber Loss'
        # Inner-max must use the same loss as the outer-min (WDRO duality requirement)
        inner_loss_fn = self._build_inner_loss(task, loss_fn, **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        scheduler = StepLR(optimizer, step_size=self.STEP_SIZE, gamma=self.DISCOUNT_FACTOR)

        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        no_improve_count = 0
        # Regret stop state; used only when a val regret is computed.
        best_regret = float('inf')
        best_regret_epoch = 0
        regret_no_improve_count = 0
        final_epoch = self.EPOCHS
        _stopping_metric_name = 'Val Loss' if val_data_loader is not None else 'Train Loss'
        if val_data_loader is None:
            print('  | [Warning] no val_data_loader — early stopping uses training loss |')

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
            best_regret             = ckpt.get('best_regret', float('inf'))
            best_regret_epoch        = ckpt.get('best_regret_epoch', 0)
            regret_no_improve_count  = ckpt.get('regret_no_improve_count', 0)
            self.epoches_losses_list = ckpt.get('epoches_losses_list', [])
            self.epoches_val_losses_list = ckpt.get('epoches_val_losses_list', [])
            self.epoches_val_regret_list = ckpt.get('epoches_val_regret_list', [])
            torch.set_rng_state(ckpt['torch_rng_state'].cpu())
            np.random.set_state(ckpt['numpy_rng_state'])
            print(f'  | [Resume] Resuming at epoch {start_epoch + 1}/{self.EPOCHS} '
                  f'(best so far: epoch {best_epoch}, {_stopping_metric_name}={best_loss:.6f}) |')

        # δ warm-start cache keyed by minibatch index. It is not checkpointed, so
        # a resumed run takes one cold epoch and is not bit-identical to an
        # unbroken one.
        delta_cache = {}

        # SOLAR_AUG=1: mix midday-cheap scenarios in, α~U[0,αmax]; the inner max
        # perturbs the augmented window.
        _solar_aug  = os.environ.get('SOLAR_AUG', '0') == '1'
        _solar_amax = float(os.environ.get('SOLAR_ALPHA_MAX', '120.0'))
        _solar_prob = float(os.environ.get('SOLAR_PROB', '0.5'))
        _solar_dr   = kwargs.get('data_range')
        if _solar_aug and _solar_dr is not None:
            print(f'  | [WDRO] SOLAR_AUG ON | α_max=${_solar_amax:.0f} '
                  f'prob={_solar_prob} (regime coverage + Wasserstein ball) |', flush=True)

        for epoch in range(start_epoch, self.EPOCHS):
            epoch_loss = 0
            time_step = 0
            progress_bar = tqdm(
                train_data_loader,
                desc=f"  Iteration [{epoch + 1}/{self.EPOCHS}]",
                leave=True,
            )

            # L2 grad-norm cap against single-batch blow-ups; same as ml.py.
            _grad_clip = 5.0

            for i, batch in enumerate(progress_bar):
                seq_batch = batch[0].to(self.device)
                if _solar_aug and _solar_dr is not None:
                    seq_batch = _solar_augment_price(seq_batch, _solar_dr,
                                                     _solar_amax, prob=_solar_prob)
                B = seq_batch.shape[0]
                L = seq_batch.shape[1]
                n_steps = L - self.WINDOWS_LEN - self.PREDICT_LEN + 1
                if n_steps <= 0:
                    continue

                input_unfolded, target_unfolded, F = _unfold_seq_batch(
                    seq_batch, self.WINDOWS_LEN, self.PREDICT_LEN, n_steps,
                )

                # SPO+ is LP-bound: subsample windows as ml.py does (torch RNG).
                # The MSE baseline keeps every window.
                _SPO_WINDOW_SUBSAMPLE = 8
                if (task is not None and hasattr(task, 'spo_plus_loss')
                        and n_steps > _SPO_WINDOW_SUBSAMPLE):
                    idx = torch.randperm(n_steps, device=seq_batch.device)[:_SPO_WINDOW_SUBSAMPLE]
                    input_unfolded  = input_unfolded[:, idx]
                    target_unfolded = target_unfolded[:, idx]
                    n_steps_used = _SPO_WINDOW_SUBSAMPLE
                else:
                    n_steps_used = n_steps

                input_seq  = input_unfolded.reshape(B * n_steps_used, self.WINDOWS_LEN, F)
                target_seq = target_unfolded.reshape(B * n_steps_used, self.PREDICT_LEN)

                # Inner max with the outer loss, warm-started from this minibatch's
                # previous δ.
                delta = self._worst_case_delta(
                    model, input_seq, target_seq, inner_loss_fn,
                    delta_init=delta_cache.get(i),
                )
                delta_cache[i] = delta
                perturbed_input = (input_seq + delta).detach()

                # Outer min: model update on worst-case perturbed inputs
                y_pred = model(perturbed_input)

                if task:
                    # no relu: keep negative-capable head's output
                    lo = min(kwargs['data_range'])
                    hi = max(kwargs['data_range'])
                    y_pred_real = y_pred * (hi - lo) + lo
                    y_real = target_seq * (hi - lo) + lo
                    if hasattr(task, 'spo_plus_loss'):
                        # Must match ml.py, whose _eval_loss drives early stopping.
                        spo = task.spo_plus_loss(y_pred=y_pred_real, y_true=y_real)
                        anchor = Fn.smooth_l1_loss(y_pred_real, y_real, beta=0.1, reduction='mean')
                        loss = spo + LAMBDA_MSE_ANCHOR * anchor
                    else:
                        best_x, _ = task.find_best_scheduling(y_pred_real)
                        loss = torch.mean(task.loss(x=best_x, y=y_real))
                else:
                    loss = loss_fn(y_pred, target_seq)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("[Warning] loss is nan or inf, skipping")
                    continue

                optimizer.zero_grad()
                loss.backward()
                if _grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                time_step += 1

            scheduler.step()
            avg_loss = epoch_loss / max(time_step, 1)
            progress_bar.set_postfix({'Avg Training Loss': f'{avg_loss:.4f}'})
            self.epoches_losses_list.append(avg_loss)

            # Validation on clean inputs (no inner max). With
            # val_regret_loaders=[(label, loader), ...] the selection score is
            # the MAX regret across them.
            _val_regret_loaders = kwargs.get('val_regret_loaders', None)
            _regret_components = None
            if _val_regret_loaders:
                val_loss, _ = self._eval_loss(model, val_data_loader, task, loss_fn, **kwargs)
                _regret_components = []
                for _lbl, _ldr in _val_regret_loaders:
                    _, _r = self._eval_loss(model, _ldr, task, loss_fn, **kwargs)
                    _regret_components.append((_lbl, float(_r) if _r is not None else float('inf')))
                val_norm_regret = max(r for _, r in _regret_components)
            else:
                val_loss, val_norm_regret = self._eval_loss(
                    model, val_data_loader, task, loss_fn, **kwargs
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
                save_path = cfg.ml_model_path(
                    folder=model_save_folder_path,
                    model_name=self.ALGO_NAME,
                    cuda=self.device,
                    epoch=epoch + 1,
                    lr=self.LR,
                )
                _save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(_save, save_path)
                print(f"  | Model Saved | {save_path}")

            # The snapshot follows regret when regret is computed, else val_loss.
            # The val_loss patience counter runs in both cases.
            if val_loss is not None:
                if val_loss < best_loss:
                    # Reset patience only on a ≥MIN_DELTA_REL relative gain;
                    # best_loss still tracks the running minimum.
                    prev_best = best_loss
                    best_loss = val_loss
                    rel_improvement = (prev_best - val_loss) / max(abs(prev_best), 1e-8)
                    if rel_improvement >= self.MIN_DELTA_REL:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    # Regime (a): no regret signal → snapshot on val_loss.
                    if val_norm_regret is None:
                        best_epoch = epoch + 1
                        _src = model.module if isinstance(model, nn.DataParallel) else model
                        best_state = copy.deepcopy(_src.state_dict())
                else:
                    no_improve_count += 1

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

            # ML_REGRET_ONLY_STOP=1: stop only on a val regret plateau.
            _regret_only_stop = os.environ.get('ML_REGRET_ONLY_STOP') == '1'
            _val_loss_plateaued = (no_improve_count >= self.EARLY_STOPPING_PATIENCE
                                   and not _regret_only_stop)
            _regret_plateaued   = (val_norm_regret is not None
                                   and regret_no_improve_count >= self.EARLY_STOPPING_PATIENCE)
            if _val_loss_plateaued or _regret_plateaued:
                _reasons = []
                if _val_loss_plateaued:
                    _reasons.append(f'val_loss plateau ({no_improve_count} epochs)')
                if _regret_plateaued:
                    _reasons.append(f'val regret plateau ({regret_no_improve_count} epochs since last min)')
                print(f'  | Early stopping at epoch {epoch + 1} ({" + ".join(_reasons)}) |')
                final_epoch = epoch + 1
                break

        if best_state is not None:
            _restore_target = model.module if isinstance(model, nn.DataParallel) else model
            _restore_target.load_state_dict(best_state)
            if best_regret < float('inf'):
                print(f'  | Restored best checkpoint from epoch {best_epoch} (Val Norm Regret={best_regret:.6f}) |')
            else:
                print(f'  | Restored best checkpoint from epoch {best_epoch} ({_stopping_metric_name}={best_loss:.6f}) |')
            final_epoch = best_epoch
        else:
            print(f'  | [Warning] No best checkpoint captured ({_stopping_metric_name} never improved below inf) — saving last-epoch model as "best" |')

        save_path = cfg.ml_model_path(
            folder=model_save_folder_path,
            model_name=self.ALGO_NAME,
            cuda=self.device,
            epoch=final_epoch,
            lr=self.LR,
        )
        _save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(_save, save_path)
        print(f"  | Model Saved (best) | {save_path}")

        # Only the best-epoch checkpoint is loaded downstream.
        cleanup_epoch_checkpoints(model_save_folder_path, self.ALGO_NAME, final_epoch)

        # Clean up resume checkpoint on successful completion.
        if os.path.exists(resume_path):
            os.remove(resume_path)

        print(f'=============== {self.ALGO_NAME} Training End ===============')
        return final_epoch
