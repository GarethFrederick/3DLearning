"""Real-price windows, real base samples and the DDPM sampler for DDD_Learning."""
import os
import numpy as np
import pandas as pd
import torch
from Utils.datasets import Data_Loader, inverse_mv_sample
from Utils.diffusion_utils import DiffusionProcess


class RealDataMixin:
    def _real_price_windows(self, n, L):
        """n real price windows of length L from the train series, in real $/MWh.
        Start rows come from the global np.random stream."""
        if self._real_base_feats is None:
            self._real_base_sample(1)                     # triggers the lazy load
        _pr = self._real_base_feats['Price'].to_numpy(np.float64)
        N = len(_pr)
        out = np.empty((int(n), int(L)), dtype=np.float64)
        for i in range(int(n)):
            st = int(np.random.randint(0, max(1, N - int(L))))
            out[i] = _pr[st:st + int(L)]
        return out

    def _real_base_sample(self, n_rows):
        """A random contiguous slice of real train rows as the head's base data,
        size-matched to the s_theta it replaces, with the same feature columns.
        The start row comes from the global np.random stream."""
        if self._real_base_feats is None:
            from Utils.datasets import build_ml_features
            _csv = os.environ.get('HEAD_REAL_CSV', 'Data/AEMO_VIC1_train_2015_2020.csv')
            _tr = pd.read_csv(_csv, parse_dates=['TIMESTAMP'], date_format='mixed')
            # Filter before build_ml_features so lag features see retained rows
            # only. n_rows is unchanged: this swaps source years, not the dose.
            _fy = os.environ.get('HEAD_REAL_FROM_YEAR', '')
            if _fy:
                _n_all = len(_tr)
                _tr = _tr[_tr['TIMESTAMP'].dt.year >= int(_fy)].reset_index(drop=True)
                print(f'  | [HEAD-REAL-RECENCY] pool restricted to years >= {_fy}: '
                      f'{_n_all} -> {len(_tr)} rows |', flush=True)
                assert len(_tr) > 0, f'HEAD_REAL_FROM_YEAR={_fy} left no rows'
            _feat = build_ml_features(_tr)[self.MV_META['feature_names']]
            self._real_base_feats = _feat.reset_index(drop=True)
            print(f'  | [HEAD-REAL] loaded {len(self._real_base_feats)} real train '
                  f'rows as head base data (diffusion s_theta REPLACED for the head) |',
                  flush=True)
        N = len(self._real_base_feats)
        n = int(max(1, min(n_rows, N)))
        # Print the real row count every round so a pool that caps at N shows.
        if getattr(self, '_head_rows_mult', 1) > 1:
            # Divide by MV_SEQ_LEN, not PIC_SIZE (the head's shorter frame).
            # This is an estimate; the [INSTR-HEAD-AUG] "applied to N batches"
            # line is the true count.
            _seqs = n // max(int(getattr(self, 'MV_SEQ_LEN', self.PIC_SIZE)), 1)
            print(f'  | [HEAD-ROWS] asked {n_rows} rows, got {n} of {N} available '
                  f'-> {_seqs} sequences -> ~{_seqs // max(int(self.BATCH_SIZE), 1)} '
                  f'batches per inner epoch (mult={self._head_rows_mult}) '
                  f'| ESTIMATE ONLY — the authoritative count is the '
                  f'[INSTR-HEAD-AUG] "applied to N batches" line |', flush=True)
        start = int(np.random.randint(0, max(1, N - n + 1)))
        return self._real_base_feats.iloc[start:start + n].reset_index(drop=True)

    def gen_s_theta_mv(self, diffusion_model, gen_iterations, return_native=False,
                       midday_cond=False):
        """Multivariate worst-case data: sample ModelMV (B, F, L), inverse the
        signed-log+calibration+clip transform to real $/MWh + MW, and flatten to
        a (N, F) DataFrame the ML loader can window, in time order.

        return_native: also return the native-space samples (N,F,L) and the
        decoded real-$ price (N,L), so fold-in can append to its replay buffer
        without re-encoding. False keeps the single-return contract.

        RNG: the cond draws consume the global np.random stream."""
        diffusion_model.eval()
        m = self.MV_META
        norm_bounds   = {k: tuple(v) for k, v in m['norm_bounds'].items()}
        feature_names = m['feature_names']
        log_features  = set(m.get('log_features', []))
        calibration   = {k: tuple(v) for k, v in m.get('calibration', {}).items()}
        clip          = {k: tuple(v) for k, v in m.get('clip', {}).items()}

        seqs = []
        _native_list = []          # (B,F,L) full native tensors (cond re-attached)
        _price_real_list = []      # (B,L) decoded real-$ price
        _cond = self.MV_N_COND > 0
        for _ in range(gen_iterations):
            process = DiffusionProcess(
                beta_1=self.BETA_1, beta_T=self.BETA_T, T=self.T,
                diffusion_fn=diffusion_model, device=self.device,
                shape=(self.MV_N_STOCH if _cond else self.N_FEATURES, self.MV_SEQ_LEN),
                n_cond=self.MV_N_COND,
            )
            if _cond:
                if midday_cond:
                    # Retention depth gate only: midday-centred cond windows,
                    # to match the instrument that set its thresholds.
                    _mid = self._midday_cond_pool_idx()
                    pool_idx = np.random.choice(_mid, self.BATCH_SIZE)
                else:
                    pool_idx = np.random.randint(0, self.MV_COND_POOL.shape[0], self.BATCH_SIZE)
                cond_t = torch.from_numpy(self.MV_COND_POOL[pool_idx]).float().to(self.device)
            else:
                cond_t = None
            sample, _ = process.sampling(sampling_number=self.BATCH_SIZE,
                                         save_trajectory=False, cond=cond_t,
                                         guidance_fn=self._guidance_fn)
            gen = sample.detach().cpu().numpy()                          # (B, F, L) model space
            gen_real = inverse_mv_sample(gen, feature_names, norm_bounds,
                                         log_features=log_features,
                                         calibration=calibration, clip=clip)
            if return_native:
                _native_list.append(sample.detach().cpu())              # (B, F, L) native
                _price_real_list.append(gen_real[:, 0, :])              # (B, L) real-$ price
            # (B, F, L) -> (B, L, F) -> (B*L, F): contiguous time, price at col 0.
            seqs.append(np.transpose(gen_real, (0, 2, 1)).reshape(-1, self.N_FEATURES))

        df = pd.DataFrame(np.concatenate(seqs, axis=0), columns=feature_names)
        if return_native:
            native = torch.cat(_native_list, dim=0)                     # (N, F, L) native
            price_real = np.concatenate(_price_real_list, axis=0)       # (N, L) real-$
            return df, native, price_real
        return df

    def _midday_cond_pool_idx(self):
        """Indices of MV_COND_POOL windows whose CENTRE slot is midday (hours
        10-16), from the sin_h/cos_h cond channels. Must match
        generator_alone_probe.midday_cond_idx, which calibrated the retention
        depth thresholds. Cached. The all-cond draw does not see taught midday
        depth, so the gate must condition here."""
        if getattr(self, '_midday_idx_cache', None) is None:
            cond_names = self.MV_META.get('cond_names', [])
            try:
                i_sin = cond_names.index('sin_h'); i_cos = cond_names.index('cos_h')
            except (ValueError, AttributeError):
                i_sin, i_cos = 0, 1                     # probe's same fallback
            pool = self.MV_COND_POOL                    # (W, n_cond, L)
            L = pool.shape[2]
            sin_h = pool[:, i_sin, L // 2]; cos_h = pool[:, i_cos, L // 2]
            hour = (np.arctan2(sin_h, cos_h) % (2 * np.pi)) / (2 * np.pi) * 24.0
            idx = np.where((hour >= 10.0) & (hour <= 16.0))[0]
            if idx.size == 0:
                raise RuntimeError('midday depth gate: no midday-centred cond '
                                   'windows in MV_COND_POOL')
            self._midday_idx_cache = idx
            # Proves the midday-cond path is live; fires once per run.
            print(f'  | [DEPTH GATE] midday-cond instrument ON | '
                  f'{idx.size} midday-centred cond windows |', flush=True)
        return self._midday_idx_cache

    def _foldin_real_ref_loader(self):
        """Real training windows for the live-head reference curve, built like
        z0_loader (same features, ML norm bounds, sequence length, batch size,
        no shuffle). It does not call _real_base_sample, which draws from the
        global random stream."""
        if self._foldin_real_ref_loader_cache is None:
            from Utils.datasets import build_ml_features
            _csv = os.environ.get('HEAD_REAL_CSV', 'Data/AEMO_VIC1_train_2015_2020.csv')
            _tr = pd.read_csv(_csv, parse_dates=['TIMESTAMP'], date_format='mixed')
            _feat = build_ml_features(_tr)[self.MV_META['feature_names']].reset_index(drop=True)
            _nrows = len(_feat)
            # 512 midnight-start sequences, drawn once without replacement by
            # their own RandomState (main stream untouched). Midnight starts keep
            # the averaged curve a time-of-day profile. Row 0 is stamp 00:30
            # (period-ending), so day d starts at row 48*d.
            _L, _T, _n = int(self.ML_SEQ_LEN), 48, 512
            assert _tr['TIMESTAMP'].iloc[0].strftime('%H:%M') == '00:30', 'row 0 is not midnight'
            _starts = np.arange(0, len(_feat) - _L + 1, _T)
            _pick = np.sort(np.random.RandomState(20261001).choice(_starts, _n, replace=False))
            _feat = pd.concat([_feat.iloc[s:s + _L] for s in _pick], ignore_index=True)
            _first, _last = _tr['TIMESTAMP'].iloc[_pick[0]], _tr['TIMESTAMP'].iloc[_pick[-1]]
            self._foldin_real_ref_loader_cache = Data_Loader(df=_feat).ml_set2loader_mv(
                split_test=False, device=self.device, batch_size=self.BATCH_SIZE,
                normlized=True, pic_size=self.ML_SEQ_LEN,
                norm_bounds=self.MV_LOADER_NORM_BOUNDS)
            _ns = sum(int(b[0].shape[0]) for b in self._foldin_real_ref_loader_cache)
            assert _ns > 0, 'FOLDIN_REF_POOL=real built an empty loader'
            print(f'  | [REF-POOL] REAL reference pool: {_nrows} rows from {_csv}, {_n} midnight-start '
                  f'sequences drawn by RandomState(20261001) from {len(_starts)} start days '
                  f'(first {_first}, last {_last}) -> '
                  f'{_ns} sequences of {self.ML_SEQ_LEN} (z0 NOT used for c_ref) |', flush=True)
        return self._foldin_real_ref_loader_cache
