"""Per-block trend rules (aim and depth) for DDD_Learning."""
import os
import numpy as np
import pandas as pd
from Utils.Algos.ddd._common import _TRAIN_YR_LO, _TRAIN_YR_HI


class TrendMixin:
    def _trend_blocks(self, _tr):
        """Block index per row, and the block count, under the configured grid.
        Keep this the only copy of the block expression."""
        _h = float(getattr(self, '_trend_block_hours', 4.0))
        _nb = max(1, int(round(24.0 / _h)))
        _b = np.clip(((_tr['TIMESTAMP'].dt.hour
                       + _tr['TIMESTAMP'].dt.minute / 60.0) // _h).astype(int), 0, _nb - 1)
        return _b, _nb

    def _trend_aim(self):
        """The aim (centre hour) from the training years' own price trend; no OOD
        data. Regress each block's 5th-percentile price on year and aim at the
        block that falls fastest. The median is not used: every block's median
        rose in the gas era.

        Returns (centre_hour, block, slopes). The sign is DOWN: falling train
        prices mean the augmentation makes those hours cheaper.
        """
        if self._trend_aim_cache is None:
            _csv = os.environ.get('HEAD_REAL_CSV',
                                  'Data/AEMO_VIC1_train_2015_2020.csv')
            _tr = pd.read_csv(_csv, parse_dates=['TIMESTAMP'], date_format='mixed')
            _yr = _tr['TIMESTAMP'].dt.year
            # Timestamps are period-ENDING so one 2021-01-01 00:00 row closes 2020;
            # drop out-of-range rows but ABORT if it is more than that boundary row,
            # which would mean the "train" CSV really carries OOD years.
            _n0 = len(_tr)
            _tr = _tr[(_yr >= _TRAIN_YR_LO) & (_yr <= _TRAIN_YR_HI)]
            assert _n0 - len(_tr) <= 2, (
                f'{_csv} carries {_n0 - len(_tr)} rows outside {_TRAIN_YR_LO}-{_TRAIN_YR_HI} — more than '
                'the period-ending boundary row; refusing to derive the aim from it')
            _blk, _nb = self._trend_blocks(_tr)
            _years = np.array(sorted(_tr['TIMESTAMP'].dt.year.unique()), dtype=float)
            _xc = _years - _years.mean()
            _slopes = np.zeros(_nb, dtype=np.float64)
            for _b in range(_nb):
                _p5 = np.array([_tr['Price'][(_blk == _b)
                                             & (_tr['TIMESTAMP'].dt.year == int(_y))
                                             ].quantile(self._trend_quantile) for _y in _years])
                _slopes[_b] = float((_xc * (_p5 - _p5.mean())).sum() / (_xc ** 2).sum())
            _bi = int(np.argmin(_slopes))          # most NEGATIVE slope = falling fastest
            # Random-aim control. The draw excludes the trend block, or some seeds
            # would run the treatment under the control's name. It uses its own
            # RandomState so the main random stream stays paired with the treatment.
            if os.environ.get('INSTR_HEAD_AIM_RANDOM') == '1':
                _rs = np.random.RandomState(
                    20260906 + int(os.environ.get('MV_SEED', '1')))
                _cands = [_b2 for _b2 in range(_nb) if _b2 != _bi]
                _bi_rand = int(_rs.choice(_cands))
                print(f'  | [RANDOM-AIM] CONTROL ON | trend block {_bi} REPLACED '
                      f'by random block {_bi_rand} (drawn from {_cands}, '
                      f'seed-derived) | depth and width rules unchanged |',
                      flush=True)
                _bi = _bi_rand
            # INSTR_HEAD_AIM_BLOCK=<b> forces the aim to block b. It draws no
            # random number.
            _fixed = os.environ.get('INSTR_HEAD_AIM_BLOCK')
            if _fixed is not None:
                assert os.environ.get('INSTR_HEAD_AIM_RANDOM') != '1', (
                    'INSTR_HEAD_AIM_BLOCK and INSTR_HEAD_AIM_RANDOM both set -- '
                    'the forced block would silently replace the random draw. Pick one.')
                _bf = int(_fixed)
                assert 0 <= _bf < _nb, f'INSTR_HEAD_AIM_BLOCK={_bf} not in 0..{_nb - 1}'
                print(f'  | [FIXED-AIM] CONTROL ON | trend block {_bi} REPLACED '
                      f'by fixed block {_bf} | depth and width rules unchanged |',
                      flush=True)
                _bi = _bf
            # Centre and label must use the block width, not a literal 4 h.
            _bh = float(getattr(self, '_trend_block_hours', 4.0))
            self._trend_aim_cache = (_bh * _bi + _bh / 2.0, _bi, _slopes)
            print(f'  | [TREND-AIM] train-only p5 slope/yr per block=['
                  + ','.join(f'{s:+.2f}' for s in _slopes)
                  + f'] -> block {_bi} ({int(_bh * _bi):02d}-{int(_bh * _bi + _bh):02d}h) '
                  f'centre={_bh * _bi + _bh / 2.0:.1f}h | derived from {_csv} '
                  f'({len(_tr)} rows, {int(_years[0])}-{int(_years[-1])}), NO OOD data |',
                  flush=True)
        return self._trend_aim_cache

    def _trend_depth(self):
        """The discount depth from the training years' own price trend; no OOD
        data. On the aimed block, fit the 5th-percentile price against year
        (default degree 2: the fall accelerates, and a line is unstable when one
        year is dropped), extend it to the deployment horizon, and take the depth
        as the fall below the last training year's value.

        Returns a POSITIVE depth in $/MWh. A non-positive extrapolation
        contradicts the aim rule, so it crashes rather than apply a zero edit.
        """
        if self._trend_depth_cache is None:
            _ch, _bi, _ = self._trend_aim()   # depth belongs to the aimed block
            _csv = os.environ.get('HEAD_REAL_CSV',
                                  'Data/AEMO_VIC1_train_2015_2020.csv')
            _tr = pd.read_csv(_csv, parse_dates=['TIMESTAMP'], date_format='mixed')
            _yr = _tr['TIMESTAMP'].dt.year
            # Same period-ending boundary allowance as _trend_aim.
            _n0 = len(_tr)
            _tr = _tr[(_yr >= _TRAIN_YR_LO) & (_yr <= _TRAIN_YR_HI)]
            assert _n0 - len(_tr) <= 2, (
                f'{_csv} carries {_n0 - len(_tr)} rows outside {_TRAIN_YR_LO}-{_TRAIN_YR_HI} — more than '
                'the period-ending boundary row; refusing to derive the depth from it')
            _blk, _nb = self._trend_blocks(_tr)
            _years = np.array(sorted(_tr['TIMESTAMP'].dt.year.unique()), dtype=float)
            _p5 = np.array([_tr['Price'][(_blk == _bi)
                                         & (_tr['TIMESTAMP'].dt.year == int(_y))
                                         ].quantile(self._trend_quantile) for _y in _years])
            _quad = np.polyfit(_years, _p5, self._trend_degree)
            _base = float(_p5[-1])
            _hz = float(self._instr_head_depth_horizon)
            _depth = _base - float(np.polyval(_quad, _hz))
            if getattr(self, '_instr_head_depth_observed', False):
                _fit_depth = _depth
                _depth = float(_p5[-2]) - _base
                print(f'  | [DEPTH-OBSERVED] CONTROL ON | block {_bi} fitted depth '
                      f'${_fit_depth:.2f} REPLACED by observed {int(_years[-2])}->'
                      f'{int(_years[-1])} fall ${_depth:.2f} |', flush=True)
            assert _depth > 0.0, (
                f'trend depth is {_depth:.2f} on block {_bi} — the aimed block is '
                'not falling, which contradicts the aim rule; refusing to run a '
                'zero-or-negative-strength edit')
            self._trend_depth_cache = _depth
            _bh = float(getattr(self, '_trend_block_hours', 4.0))
            print(f'  | [TREND-DEPTH] block {_bi} ({int(_bh * _bi):02d}-{int(_bh * _bi + _bh):02d}h) '
                  f'p5 by year=[' + ','.join(f'{v:+.2f}' for v in _p5) + '] '
                  f'| quadratic accel={2 * _quad[0]:+.2f} $/yr^2 '
                  f'| {int(_years[-1])} tail=${_base:.2f} -> horizon {_hz:.1f} '
                  f'tail=${float(np.polyval(_quad, _hz)):.2f} '
                  f'| DEPTH=${_depth:.2f} (was ${self._instr_head_amax:.0f} hand-set) '
                  f'| derived from {_csv} ({len(_tr)} rows, {int(_years[0])}-'
                  f'{int(_years[-1])}), NO OOD data |', flush=True)
        return self._trend_depth_cache
