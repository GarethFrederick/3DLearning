"""Decision-divergence guidance builder for DDD_Learning."""
import os
import math
import numpy as np
import torch
from Utils.datasets import inverse_mv_sample
from Utils.Downstream.opt_oracle import build_oracle


class GuidanceMixin:
    def _build_divergence_guidance(self, task, prior_price_profile):
        """Construct the decision-divergence guidance callback for guided sampling.

        Returns a closure guidance_fn(idx, x_stoch, x0_hat)->delta, or None when
        disabled. The fixed direction is g[t] ∝ centred w*(prior)[t], where
        w*(prior) is the optimal schedule under the night-cheap training price.
        Adding +g to the normalised price channel raises price where that
        schedule consumes, which pushes the sample's optimal schedule away from it.
        """
        # FOLD_IN=1 turns guidance on even when GUIDE_DIVERGENCE is unset.
        _foldin_on = (os.environ.get('FOLD_IN') == '1')
        _foldin_obj = os.environ.get('FOLDIN_OBJECTIVE', 'divergence')
        if os.environ.get('GUIDE_DIVERGENCE') != '1' and not _foldin_on:
            return None
        if _foldin_on and _foldin_obj == 'off':
            print('  | [Guidance] FOLD_IN=1 FOLDIN_OBJECTIVE=off — guidance OFF |', flush=True)
            return None
        if prior_price_profile is None:
            print('  | [Guidance] GUIDE_DIVERGENCE/FOLD_IN set but no '
                  'prior_price_profile passed — guidance OFF |', flush=True)
            return None
        # Fold-in forces the adaptive path so it can read self._foldin_w_ref each
        # round. The 'random' control takes the same adaptive schedule, so only
        # the direction differs; the live-head w_ref applies to divergence/regret only.
        _foldin_random = _foldin_on and (_foldin_obj == 'random')
        _foldin_adaptive = _foldin_on and (_foldin_obj in ('divergence', 'regret'))
        oracle = build_oracle(task, device=self.device)
        prior = torch.as_tensor(np.asarray(prior_price_profile, dtype=np.float32),
                                device=self.device).reshape(1, -1)
        # Sensitivity = w*(prior) on the external price grid (1, T_external).
        s_prior = oracle.sensitivity(prior).detach().float().reshape(-1)
        g_daily = s_prior - s_prior.mean()
        _max = g_daily.abs().max()
        if _max <= 0:
            print('  | [Guidance] prior schedule is flat (no decision structure) '
                  '— guidance OFF |', flush=True)
            return None
        g_daily = g_daily / _max                                   # unit max-abs
        # Tile the daily decision shape across the diffusion window length L.
        L = self.MV_SEQ_LEN
        T_ext = g_daily.shape[0]
        reps = int(math.ceil(L / T_ext))
        g_L = g_daily.repeat(reps)[:L].contiguous()                # (L,)

        # GUIDE_SCALE is the TOTAL shift on the normalised price channel over the
        # guided phase, not a per-step nudge; applying it every step saturates
        # prices. The clamp keeps the inverse signed-log transform off inf.
        # GUIDE_WASSERSTEIN_RHO>0 overrides GUIDE_SCALE.
        _rho     = float(os.environ.get('GUIDE_WASSERSTEIN_RHO', '0'))
        scale    = _rho if _rho > 0 else float(os.environ.get('GUIDE_SCALE', '0.3'))
        last_k   = int(os.environ.get('GUIDE_LAST_K', '150'))
        price_ch = int(os.environ.get('GUIDE_PRICE_CH', '0'))
        clamp_lo = float(os.environ.get('GUIDE_CLAMP_LO', '-1.0'))
        clamp_hi = float(os.environ.get('GUIDE_CLAMP_HI', '2.0'))
        per_step = scale / max(1, last_k)

        print(f'  | [Guidance] decision-divergence guided sampling ON | '
              f'scale(total)={scale}  per_step={per_step:.5f}  last_k={last_k}  '
              f'price_ch={price_ch}  clamp=[{clamp_lo},{clamp_hi}]  '
              f'g_daily min={float(g_daily.min()):.2f} '
              f'max={float(g_daily.max()):.2f}  L={L} T_ext={T_ext} |', flush=True)

        def guidance_fn(idx, x_stoch, x0_hat):
            # Only the last low-noise reverse steps; earlier pushes do not stick.
            if idx >= last_k:
                return None
            cur = x_stoch[:, price_ch, :]
            new = (cur + per_step * g_L).clamp(min=clamp_lo, max=clamp_hi)
            delta = torch.zeros_like(x_stoch)
            delta[:, price_ch, :] = new - cur                      # nudge + clamp
            return delta

        # Adaptive guidance: re-solve w*(x0_hat) per sample and set
        # Δprice ∝ (w_ref − w_now), which moves the schedule away from w_ref.
        # It costs B×days LP solves per application, so it runs on few late steps.

        if os.environ.get('GUIDE_ADAPTIVE') == '1' or _foldin_adaptive or _foldin_random:
            _m = self.MV_META or {}
            _pname = (_m.get('feature_names', ['Price']))[price_ch]
            _pnb   = {_pname: tuple(_m.get('norm_bounds', {}).get(_pname, (0.0, 1.0)))}
            _plog  = set(_m.get('log_features', [])) & {_pname}
            _pcal  = {k: tuple(v) for k, v in _m.get('calibration', {}).items() if k == _pname}
            _pclip = {k: tuple(v) for k, v in _m.get('clip', {}).items() if k == _pname}
            days   = max(1, L // T_ext)
            a_last = int(os.environ.get('GUIDE_ADAPT_LAST_K', '40'))
            a_every = max(1, int(os.environ.get('GUIDE_ADAPT_EVERY', '10')))
            a_count = max(1, len([i for i in range(a_last) if i % a_every == 0]))
            a_step  = scale / a_count
            w_prior_row = s_prior.reshape(1, -1)                    # (1, T_ext)
            print(f'  | [Guidance] ADAPTIVE per-sample divergence | '
                  f'scale(total)={scale}  a_step={a_step:.4f}  last_k={a_last}  '
                  f'every={a_every}  apps≈{a_count}  days/window={days} |', flush=True)

            def guidance_fn_adaptive(idx, x_stoch, x0_hat):
                if idx >= a_last or (idx % a_every) != 0:
                    return None
                # FOLDIN_REF=flat must not steer: w*(flat) is a degenerate LP
                # vertex, not a decision direction.
                if getattr(self, '_foldin_ref_is_flat', False):
                    return None
                B = x0_hat.shape[0]
                # Decode the Tweedie clean price estimate -> real $/MWh.
                p_norm = x0_hat[:, price_ch, :].unsqueeze(1).detach().cpu().numpy()  # (B,1,L)
                p_real = inverse_mv_sample(p_norm, [_pname], _pnb,
                                           log_features=_plog,
                                           calibration=_pcal, clip=_pclip)[:, 0, :]  # (B,L)
                day = torch.as_tensor(p_real.reshape(B * days, T_ext),
                                      dtype=torch.float32, device=self.device)
                w_now = oracle.sensitivity(day).detach().float()   # (B*days, T_ext)
                # w_ref comes from the live head each outer epoch; unset falls
                # back to the night-prior schedule.
                _wref = getattr(self, '_foldin_w_ref', None)
                if _foldin_adaptive and _wref is not None:
                    w_ref_row = _wref.reshape(1, -1).to(w_now)
                else:
                    w_ref_row = w_prior_row
                g = w_ref_row - w_now                              # (B*days, T_ext)
                gmax = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
                g = (g / gmax).reshape(B, days * T_ext)            # (B, L)
                cur = x_stoch[:, price_ch, :]
                new = (cur + a_step * g.to(cur)).clamp(min=clamp_lo, max=clamp_hi)
                delta = torch.zeros_like(x_stoch)
                delta[:, price_ch, :] = new - cur
                return delta

            return guidance_fn_adaptive

        return guidance_fn
