"""Generator-side DRO / PPO pieces for DDD_Learning."""
import torch
from torch.utils.checkpoint import checkpoint


class DROMixin:
    def build_a0(self,total_iteration,z0_trajectory,dfu):
        a0_T = []
        _cond = self.MV and self.MV_N_COND > 0
        for i in range(total_iteration):
            a0_T_i = []

            for idx,t in enumerate(self.SELECTED_TIMESTEPS): # 4

                z_t = z0_trajectory[i][idx]
                z_t_1 = z0_trajectory[i][idx - 1]

                # Guard on idx==0, NOT t==0: with a schedule that does not start
                # at 0, `t==0` never fires and idx=0 reads [-1], the noisiest
                # step. r_theta uses the same guard.
                if idx == 0:
                    z_t_1 = z0_trajectory[i][idx]

                epsilon = dfu(z_t, t)   # (B, n_stoch, L) for cond, (B, F, L) otherwise

                # Cond mode: the backbone predicts noise on stoch channels only,
                # so mu_t uses z_t[:, :n_stoch] to match.
                if _cond:
                    z_t_s   = z_t[:, :self.MV_N_STOCH]
                    z_t_1_s = z_t_1[:, :self.MV_N_STOCH]
                    mu_dfu0_t = self.mu_t(x_t=z_t_s, t=t, epsilon=epsilon).detach()
                    a0_t = (z_t_1_s - mu_dfu0_t) ** 2
                else:
                    mu_dfu0_t = self.mu_t(x_t=z_t, t=t, epsilon=epsilon).detach()
                    a0_t = (z_t_1 - mu_dfu0_t) ** 2


                a0_T_i.append(a0_t)

            a0_T_i = torch.stack(a0_T_i, dim=0)
            a0_T.append(a0_T_i)
        a0_T = torch.stack(a0_T, dim=0)
        return a0_T

    def mu_t(self,x_t,t,epsilon):
        mu = torch.sqrt(1 / self.alphas[t]) * (x_t - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * epsilon)
        return mu

    def loss_hw_task(self, test_data_loader, model, task, return_norm_regret=False, **kwargs):
        """Per-window regret tensor over a loader; same LP eval as test_task().

        When return_norm_regret=True, also returns the global normalised regret
        scalar Σ(c·w*(ĉ) − c·w*(c)) / Σ|c·w*(c)| over every window, the same
        formula as test_task()'s "Norm Regret" line.
        """
        avg_regret_list = []
        # Deterministic linspace subsample, so the score is comparable across
        # outer epochs.
        _SPO_WINDOW_SUBSAMPLE = 8

        sum_regret = 0.0
        sum_abs_opt = 0.0

        for batch in test_data_loader:
            seq_batch = batch[0].to(self.device)
            # Some loaders yield (B, L); make it (B, L, F).
            if seq_batch.dim() == 2:
                seq_batch = seq_batch.unsqueeze(-1)
            regret_mat = []

            t_lo = self.WINDOWS_LEN
            t_hi = seq_batch.shape[1] - self.PREDICT_LEN + 1
            n_steps = t_hi - t_lo
            if n_steps > _SPO_WINDOW_SUBSAMPLE:
                _idx = torch.linspace(0, n_steps - 1, _SPO_WINDOW_SUBSAMPLE).long().tolist()
                t_iter = [t_lo + i for i in _idx]
            else:
                t_iter = list(range(t_lo, t_hi))

            for t in t_iter:
                input_seq = seq_batch[:, t - self.WINDOWS_LEN:t]    # (B, W, F)
                # Target = price column only (column 0 by build_ml_features convention).
                target_seq = seq_batch[:, t:t+self.PREDICT_LEN, 0]

                # No relu: negative prices are valid, and the head bounds its output.
                y_pred = model(input_seq)
                y_pred_real = y_pred * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])
                y_real = target_seq * (max(kwargs['data_range']) - min(kwargs['data_range'])) + min(kwargs['data_range'])

                x_hat, lowest_loss = task.find_best_scheduling(y=y_pred_real)
                x_opt, opt_loss = task.find_best_scheduling(y=y_real)
                evaluate_loss = task.loss(x=x_hat, y=y_real)
                regret = evaluate_loss - opt_loss
                regret_mat.append(regret)

                if return_norm_regret:
                    sum_regret  += float(regret.sum().item())
                    sum_abs_opt += float(torch.abs(opt_loss).sum().item())

            regret_mat_tensor = torch.stack(regret_mat, dim=1)
            batch_regret = torch.mean(regret_mat_tensor, dim=1, keepdim=True)
            avg_regret_list.append(batch_regret)

        avg_regret_tensor = torch.cat(avg_regret_list, dim=1)
        if return_norm_regret:
            norm_regret = (sum_regret / sum_abs_opt) if sum_abs_opt > 0 else None
            return avg_regret_tensor, norm_regret
        return avg_regret_tensor

    def r_theta(self,z0_trajectory,a0_T,batch_i,forward_fn):
        # Importance-ratio exponent. The mv path sums (B, F, L) over feature and
        # time to one scalar per sequence (the likelihood ratio of one sample),
        # then broadcasts across PIC_SIZE for ppo().
        if self.MV:
            a_diff = torch.zeros(self.BATCH_SIZE, device=self.device)
        else:
            a_diff = torch.zeros(self.BATCH_SIZE, self.PIC_SIZE).to(self.device)

        for idx,t in enumerate(self.SELECTED_TIMESTEPS):

            if idx == 0:
                continue

            z_t = z0_trajectory[batch_i][idx]
            z_t_1 = z0_trajectory[batch_i][idx - 1]

            t_tensor = torch.tensor(t, device=self.device)
            epsilon = checkpoint(forward_fn, z_t, t_tensor, use_reentrant=False)

            # Cond mode: slice to stoch, as build_a0 does, so a and a0 match.
            if self.MV and self.MV_N_COND > 0:
                z_t   = z_t[:, :self.MV_N_STOCH]
                z_t_1 = z_t_1[:, :self.MV_N_STOCH]

            mu_dfu_t = self.mu_t(x_t=z_t, t=t, epsilon=epsilon)

            if self.MV:
                # (B, F, L) -> (B,): sum squared deviation over feature + time.
                a  = ((z_t_1 - mu_dfu_t) ** 2).sum(dim=(1, 2))
                a0 = a0_T[batch_i][idx].to(device=self.device).sum(dim=(1, 2))
            else:
                a = (z_t_1 - mu_dfu_t) ** 2
                a = a.sum(dim=-1)
                a = a.squeeze(1)

                a0 = a0_T[batch_i][idx].to(device=self.device)
                a0 = a0.sum(dim=-1)
                a0 = a0.squeeze(1)

            a_diff += (a - a0.detach()) / (2 * (self.sigmas[t] ** 2))

        # Clamp before exp: exp(±30) ≈ 1e±13 stays in float32 range, and PPO
        # clipping discards extreme ratios anyway.
        a_diff = a_diff.clamp(min=-30.0, max=30.0)
        r = torch.exp(-a_diff)

        if self.MV:
            # per-sequence weight -> (B, PIC_SIZE), uniform across the loss_hw block.
            r = r.unsqueeze(1).expand(-1, self.PIC_SIZE)
        return r

    def ppo(self,r,loss_hw,batch_i,detach_part):

        if detach_part == 'hw':
            loss_hw = loss_hw.detach()

        elif detach_part == 'r':
            r = r.detach()

        # MV: loss_hw is (B, n_z0_batches); pair batch_i's ratio with column
        # batch_i, broadcast (B,1)→(B,PIC_SIZE) against r.
        if self.MV:
            _reg_block = loss_hw[:, batch_i:batch_i + 1]
        else:
            _reg_block = loss_hw[:, batch_i * self.PIC_SIZE:(batch_i + 1) * self.PIC_SIZE]
        unclipped = r * _reg_block
        clipped_r = torch.clamp(r, 1 - self.PPO_CLIP, 1 + self.PPO_CLIP)
        clipped = clipped_r * _reg_block
        ppo_loss = torch.min(unclipped, clipped).mean()
        return ppo_loss
