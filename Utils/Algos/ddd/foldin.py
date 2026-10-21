"""Fold-in selection, weighting and logging helpers for DDD_Learning."""
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils.datasets import inverse_mv_sample, signed_log


class FoldinMixin:
    def _foldin_head_belief_curve(self, hw, z0_loader, data_range):
        """Distill the CURRENT (live) head's belief to a 48-pt real-$ curve.

        Runs the head outside the sampling loop (the sampler passes no calendar
        cond) over the loader's windows. By default that is z0: sequences sampled
        once from the pretrained diffusion model, NOT real data, even with the
        generator off. FOLDIN_REF_POOL=real passes real training windows.
        Predictions are denormalised to $/MWh with data_range and averaged to a
        (PREDICT_LEN,) slot-of-day curve. Returns None if nothing was predicted."""
        _hw = hw.module if isinstance(hw, nn.DataParallel) else hw
        lo, hi = float(min(data_range)), float(max(data_range))
        _was_training = _hw.training
        _hw.eval()
        preds = []
        with torch.no_grad():
            for batch in z0_loader:
                seq = batch[0].to(self.device)
                if seq.dim() == 2:
                    seq = seq.unsqueeze(-1)                              # (B, L, F)
                t_lo = self.WINDOWS_LEN
                t_hi = seq.shape[1] - self.PREDICT_LEN + 1
                # Deterministic stride; no random draw.
                for t in range(t_lo, t_hi, max(1, self.WINDOWS_LEN // 4)):
                    inp = seq[:, t - self.WINDOWS_LEN:t, :]
                    yp = _hw(inp) * (hi - lo) + lo                       # real $/MWh
                    preds.append(yp.detach().cpu().numpy())
        if _was_training:
            _hw.train()
        if not preds:
            return None
        return np.concatenate(preds, 0).mean(0).astype(np.float32)      # (PREDICT_LEN,)

    def _foldin_compute_w_ref(self, c_ref):
        """oracle.sensitivity(c_ref) = w*(c_ref) on the external grid, (T_ext,)."""
        if self._foldin_oracle is None or c_ref is None:
            return None
        c = torch.as_tensor(np.asarray(c_ref, dtype=np.float32),
                            device=self.device).reshape(1, -1)
        return self._foldin_oracle.sensitivity(c).detach().float().reshape(-1)

    def _foldin_select(self, native, price_real, c_ref):
        """Select top-keep_frac windows by the configured objective.

        native     : (N, F, L) native-space samples (kept verbatim into buffer)
        price_real : (N, L)    decoded real-$ price (oracle cost vectors)
        c_ref      : (T_ext,)  reference real-$ curve for divergence
        Returns (keep_native (k,F,L), mean_div_kept, mean_div_all)."""
        N, L = price_real.shape
        T_ext = self._foldin_prior.shape[0] if self._foldin_prior is not None else L
        days = max(1, L // T_ext)
        use_L = days * T_ext
        obj = self._foldin_objective
        k = max(1, int(round(self._foldin_keep_frac * N)))

        # Always compute divergence vs c_ref for the LOGGED divergence_mag, even
        # when selecting on a different objective.
        day_c = price_real[:, :use_L].reshape(N * days, T_ext)
        div_all = None
        score_div = None
        if self._foldin_oracle is not None and c_ref is not None:
            c = torch.as_tensor(day_c, dtype=torch.float32, device=self.device)
            cref = torch.as_tensor(np.asarray(c_ref, np.float32)[None, :],
                                   device=self.device).expand(c.shape[0], -1)
            div = self._foldin_oracle.divergence(c, cref).detach().cpu().numpy()
            score_div = div.sum(axis=1).reshape(N, days).sum(axis=1)     # (N,)
            div_all = float(score_div.mean())

        # FOLDIN_REF=flat: selection must be decision-blind too, so keep k
        # windows uniformly at random. The draw uses self._foldin_rng, never the
        # global np.random stream.
        if getattr(self, '_foldin_ref_is_flat', False):
            keep_idx = self._foldin_rng.choice(N, size=k, replace=False)
            keep_native = native[keep_idx]
            mean_div_kept = (float(score_div[keep_idx].mean())
                             if score_div is not None else None)
            return keep_native, mean_div_kept, div_all

        if score_div is not None:
            keep_idx = np.argsort(-score_div)[:k]
        else:
            keep_idx = self._foldin_rng.choice(N, size=k, replace=False)

        keep_native = native[keep_idx]
        # Edit the kept windows before they enter the buffer. The flat control
        # returned above and is never edited. Unknown modes must raise.
        if self._gen_transport:
            if self._gt_mode == 'instructor_typical':
                # The winning direction's mildest depth goes to ALL kept windows:
                # teach the typical-day shift, not the tail.
                keep_native = self._foldin_instructor_edit(
                    keep_native, price_real[keep_idx], c_ref, typical=True)
            else:
                raise ValueError(
                    f"GEN_TRANSPORT_MODE='{self._gt_mode}' is not 'instructor_typical'")
        mean_div_kept = (float(score_div[keep_idx].mean())
                         if score_div is not None else None)
        return keep_native, mean_div_kept, div_all

    def _foldin_regret_direction(self, c_ref):
        """argmax-hour + midday/night regret share of the schedule under c_ref,
        by single-slot price knock-down. Returns (argmax_hour, mid_share,
        night_share) or (None,None,None) when the oracle/c_ref are missing."""
        if self._foldin_oracle is None or c_ref is None:
            return None, None, None
        c_ref = np.asarray(c_ref, dtype=np.float32)
        T = c_ref.shape[0]
        hours = np.arange(T) / T * 24.0
        floor = -1000.0
        w_ref, _ = self._foldin_oracle.solve(
            torch.tensor(c_ref[None, :], dtype=torch.float32, device=self.device))
        w_ref = self._foldin_oracle._decision_to_slot(
            w_ref, torch.tensor(c_ref[None, :], device=self.device))

        def _cost(w, c):
            ct = torch.as_tensor(c, dtype=torch.float32, device=self.device)
            wt = torch.as_tensor(w, dtype=torch.float32, device=self.device)
            k = min(ct.shape[-1], wt.shape[-1])
            return float((ct[..., :k] * wt[..., :k]).sum())

        reg = np.zeros(T)
        for h in range(T):
            ch = c_ref.copy(); ch[h] = floor
            _, cs = self._foldin_oracle.solve(
                torch.tensor(ch[None, :], dtype=torch.float32, device=self.device))
            reg[h] = _cost(w_ref, ch) - float(cs.sum() if torch.is_tensor(cs) else cs)
        argmax_h = float(hours[reg.argmax()])
        tot = reg.sum() if reg.sum() != 0 else 1.0
        mid = float(reg[(hours >= 10.0) & (hours <= 16.0)].sum() / tot)
        night = float(reg[(hours >= 0.0) & (hours <= 5.0)].sum() / tot)
        return argmax_h, mid, night

    def _foldin_log_row(self, epoch_1, hw, dfu_theta, z0_loader, data_range,
                        indist_regret, oodval_regret, realism_valloss):
        """Append a per-round metric row and rewrite the running JSON.
        Other scripts read the row field names and the '[FOLD-IN] ep' line."""
        c_ref = getattr(self, '_foldin_last_c_ref', None)
        if c_ref is None:
            c_ref = self._foldin_prior
        argmax_h, mid_share, night_share = self._foldin_regret_direction(c_ref)
        # midday p5 must come from the batch that was folded this round; sample
        # fresh only when none was folded.
        midday_p5 = None
        _threaded = getattr(self, '_foldin_last_price_real', None)
        try:
            if _threaded is not None:
                price_real = _threaded
            elif os.environ.get('MV_SKIP_GEN_SAMPLE', '0') == '1':
                # MV_SKIP_GEN_SAMPLE must not touch the diffusion model here.
                # midday_p5 is a log field only, so log it null.
                price_real = None
            else:
                _, _, price_real = self.gen_s_theta_mv(
                    diffusion_model=dfu_theta, gen_iterations=1, return_native=True)
            if price_real is not None:
                L = price_real.shape[1]
                # Day grid length from the prior, not a literal 48.
                _Tday = (self._foldin_prior.shape[0]
                         if self._foldin_prior is not None else self.PREDICT_LEN)
                slot_h = ((np.arange(L) % _Tday) / _Tday) * 24.0
                mid_mask = (slot_h >= 10.0) & (slot_h <= 16.0)
                mids = price_real[:, mid_mask].ravel()
                if mids.size:
                    midday_p5 = float(np.percentile(mids, 5))
        except Exception as _e:
            print(f'  | [FOLD-IN][Warning] midday-p5 decode failed: {_e} |', flush=True)
        row = {
            'outer_epoch': int(epoch_1 + 1),
            'regret_argmax_hour': argmax_h,
            'midday_share': mid_share,
            'night_share': night_share,
            'midday_p5': midday_p5,
            'divergence_mag_kept': getattr(self, '_foldin_last_div_kept', None),
            'divergence_mag_all': getattr(self, '_foldin_last_div_all', None),
            'realism_valloss': (float(realism_valloss)
                                if realism_valloss is not None else None),
            'indist_regret': (float(indist_regret)
                              if indist_regret is not None else None),
            'oodval_regret': (float(oodval_regret)
                              if oodval_regret is not None else None),
            'buffer_size': (0 if self._foldin_buffer is None
                            else int(self._foldin_buffer.shape[0])),
            'objective': self._foldin_objective,
            'ref': self._foldin_ref,
            # True when this round's oracle fell back to greedy (status=-1).
            'greedy_contaminated': bool(
                getattr(self, '_foldin_last_greedy_contaminated', False)),
            # The instructor_* and consol_* fields are null when their path
            # did not run this round.
            'instructor_edit_hist': getattr(self, '_foldin_last_instructor_hist', None),
            'instructor_unedited': getattr(self, '_foldin_last_instructor_unedited', None),
            'instructor_div_gain_mean': getattr(
                self, '_foldin_last_instructor_div_gain_mean', None),
            'consol_session_fired': getattr(self, '_foldin_last_consol_fired', None),
            'consol_session_accepted': getattr(self, '_foldin_last_consol_accepted', None),
            'consol_pre_valloss': getattr(self, '_foldin_last_consol_pre_valloss', None),
            'consol_post_valloss': getattr(self, '_foldin_last_consol_post_valloss', None),
            'consol_pre_p5': getattr(self, '_foldin_last_consol_pre_p5', None),
            'consol_post_p5': getattr(self, '_foldin_last_consol_post_p5', None),
            'consol_reject_reason': getattr(self, '_foldin_last_consol_reject_reason', None),
        }
        # Consume-once: clear the per-round fields so a later round that skips
        # their path does not re-report stale values. A consol session that
        # fires in round N appears in round N+1's row.
        self._foldin_last_consol_fired = None
        self._foldin_last_consol_accepted = None
        self._foldin_last_consol_pre_valloss = None
        self._foldin_last_consol_post_valloss = None
        self._foldin_last_consol_pre_p5 = None
        self._foldin_last_consol_post_p5 = None
        self._foldin_last_consol_reject_reason = None
        self._foldin_last_instructor_hist = None
        self._foldin_last_instructor_unedited = None
        self._foldin_last_instructor_div_gain_mean = None
        # Non-finite floats become None, and allow_nan=False makes any that slip
        # through raise: Infinity/NaN tokens break strict JSON readers.
        def _san(v):
            if isinstance(v, bool):          # bool is an int subclass — keep as-is
                return v
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return v if math.isfinite(v) else None
            return v
        row = {k: _san(v) for k, v in row.items()}
        self._foldin_rows.append(row)
        import json as _json
        _suffix = self.RUN_NAME
        _path = os.path.join(self._foldin_out, f'foldin_coevo_{_suffix}.json')
        _tmp = _path + '.tmp'
        with open(_tmp, 'w') as _f:
            _json.dump({'run': self.RUN_NAME, 'rows': self._foldin_rows}, _f,
                       indent=2, allow_nan=False)
        os.replace(_tmp, _path)
        print(f'  | [FOLD-IN] ep{epoch_1 + 1} row | argmax_h={argmax_h} '
              f'mid_share={mid_share} midday_p5={midday_p5} '
              f'indist={indist_regret} oodval={oodval_regret} '
              f'buffer={row["buffer_size"]} -> {_path} |', flush=True)

    def _foldin_native_to_real_df(self, native):
        """Decode buffered NATIVE (B,F,L) windows -> real-$ (N, F) DataFrame the
        ML loader can window. Same inverse transform as gen_s_theta_mv so the
        head sees buffer windows in the identical real-unit space as s_theta."""
        m = self.MV_META
        norm_bounds   = {k: tuple(v) for k, v in m['norm_bounds'].items()}
        feature_names = m['feature_names']
        log_features  = set(m.get('log_features', []))
        calibration   = {k: tuple(v) for k, v in m.get('calibration', {}).items()}
        clip          = {k: tuple(v) for k, v in m.get('clip', {}).items()}
        gen = native.detach().cpu().numpy()                              # (N, F, L)
        gen_real = inverse_mv_sample(gen, feature_names, norm_bounds,
                                     log_features=log_features,
                                     calibration=calibration, clip=clip)
        # (N, F, L) -> (N, L, F) -> (N*L, F): contiguous time, price at col 0.
        flat = np.transpose(gen_real, (0, 2, 1)).reshape(-1, self.N_FEATURES)
        return pd.DataFrame(flat, columns=feature_names)

    def _price_real_to_native(self, price_real):
        """Forward-encode a real-$ price array (..., L) into the DDPM NATIVE space
        (channel 0) with the encode dfu_dataset_mv used to build s0:
            signed_log  ->  normalise with MV_DDPM_NORM_BOUNDS.
        NO calibration and NO clip: those belong to the decode only. The
        generator trained on the uncalibrated encode, so an encoded window must
        match it or it lands at a constant native offset. Price is feature 0."""
        name = self.MV_META['feature_names'][0]
        lo, hi = self.MV_DDPM_NORM_BOUNDS[name]
        v = np.asarray(price_real, dtype=np.float64)
        if name in self.MV_LOG_FEATURES:       # matches dfu_dataset_mv's signed_log
            v = signed_log(v)
        if hi != lo:                           # matches dfu_dataset_mv's linear normalise
            v = (v - lo) / (hi - lo)
        return v

    def _foldin_make_weighted_loader(self, orig_ds, buf_windows, batch_size,
                                     to_loader_fn):
        """Concat original dataset + buffer windows and build a
        WeightedRandomSampler that upweights the buffer to replay_frac of the
        sampling mass, with num_samples == N_ORIGINAL so the per-epoch iteration
        count (and the s0_iteration assert) is unchanged.

        orig_ds      : a torch Dataset whose tensors[0] is (N_orig, F, L)
        buf_windows  : (N_buf, F, L) tensor in the SAME space as orig_ds
        to_loader_fn : callable(concat_ds, sampler) -> DataLoader
        """
        x_orig = orig_ds.tensors[0]
        N_orig = x_orig.shape[0]
        N_buf = buf_windows.shape[0]
        concat = torch.utils.data.ConcatDataset([
            orig_ds,
            torch.utils.data.TensorDataset(buf_windows.to(x_orig.dtype)),
        ])
        rf = self._foldin_replay_frac
        # buffer weight so that (N_buf * w_buf) / (N_orig*1 + N_buf*w_buf) == rf.
        w_buf = (rf / (1.0 - rf)) * (N_orig / max(N_buf, 1))
        weights = torch.cat([
            torch.ones(N_orig, dtype=torch.double),
            torch.full((N_buf,), float(w_buf), dtype=torch.double),
        ])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=N_orig, replacement=True)
        return to_loader_fn(concat, sampler), N_orig, N_buf, w_buf
