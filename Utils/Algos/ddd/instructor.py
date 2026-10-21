"""Instructor edit and validity leash for DDD_Learning."""
import os
import numpy as np
import torch


class InstructorMixin:
    def _foldin_instructor_leash_project(self, p):
        """Validity projection for instructor edits: clip[floor,cap] -> ramp via
        clip(diff,±ramp)+cumsum -> FINAL re-clip[floor,cap] + assert. The final
        clip is required: the ramp's cumsum can drift below the floor after a
        clipped up-spike.
        p : (M, L) real-$ price array, already edited. Returns the projected
        array, floor/cap-valid with or without a ramp."""
        p = np.clip(p, self._gt_floor, self._gt_cap)
        if self._gt_ramp is not None:
            d = np.clip(np.diff(p, axis=1), -self._gt_ramp, self._gt_ramp)
            p = np.concatenate([p[:, :1], p[:, :1] + np.cumsum(d, axis=1)], axis=1)
        p = np.clip(p, self._gt_floor, self._gt_cap)          # FINAL re-clip
        assert p.min() >= self._gt_floor - 1e-9, (
            f'INSTRUCTOR leash failed: min={p.min():.2f} below floor={self._gt_floor} '
            f'after projection (should be structurally impossible after the final clip)')
        return p

    def _foldin_instructor_edit(self, keep_native, keep_price_real, c_ref,
                                typical=False):
        """Sweep+judge edit. Build a direction-free menu of smooth price edits,
        the same at every 4-hour block (down and up depths), and judge each by
        the oracle.divergence() that selection uses. The menu must stay
        symmetric: it then encodes no belief about where the worst case lives.

        keep_native     : (k, F, L) native windows selected by _foldin_select
        keep_price_real : (k, L)    decoded real-$ price for the same windows
        c_ref           : (T_ext,)  reference curve — identical role to selection's
        Returns keep_native with channel 0 replaced by the edited price. Only
        typical=True has a return path; typical=False returns None.

        RNG: draws nothing from self._foldin_rng or np.random; the menu is
        deterministic."""
        k, L = keep_price_real.shape
        if self._foldin_oracle is None or c_ref is None:
            # Nothing to judge against: ship unedited, do not invent a winner.
            self._foldin_last_instructor_hist = {}
            self._foldin_last_instructor_unedited = k
            self._foldin_last_instructor_div_gain_mean = 0.0
            return keep_native
        T_day = (self._foldin_prior.shape[0]
                 if self._foldin_prior is not None else self.PREDICT_LEN)
        days = max(1, L // T_day)
        use_L = days * T_day
        cref_t = torch.as_tensor(np.asarray(c_ref, np.float32)[None, :], device=self.device)

        def _score(price_arr):
            """_foldin_select's divergence reduction for an (M, L) real-$ array.
            Must stay identical to it, so edits and selection share one ruler."""
            M = price_arr.shape[0]
            day_c = price_arr[:, :use_L].reshape(M * days, T_day)
            c = torch.as_tensor(day_c, dtype=torch.float32, device=self.device)
            cref = cref_t.expand(c.shape[0], -1)
            div = self._foldin_oracle.divergence(c, cref).detach().cpu().numpy()
            return div.sum(axis=1).reshape(M, days).sum(axis=1)             # (M,)

        # The unedited windows go through the same leash, so base_score uses the
        # same footing as the candidates and every output window is valid.
        price0 = self._foldin_instructor_leash_project(
            np.asarray(keep_price_real, dtype=np.float64))
        base_score = _score(price0)                                        # (k,)

        # Raised-cosine bump per block (0 at its edges, 1 at its centre), the same
        # every day; hours come from T_day, not a literal 48.
        slot_h = ((np.arange(L) % T_day) / T_day) * 24.0
        # Label format 'block-dir-depth' is a contract: the typical path parses
        # block, direction and depth back out of it.
        variant_specs = []      # [(label, edit_fn(price0)->edited), ...]
        for b in range(6):
            h0, h1 = 4.0 * b, 4.0 * b + 4.0
            in_block = (slot_h >= h0) & (slot_h < h1)
            pos = np.where(in_block, (slot_h - h0) / (h1 - h0), 0.0)
            bump = np.where(in_block, 0.5 - 0.5 * np.cos(2.0 * np.pi * pos), 0.0)[None, :]
            for _dep in self._gt_edit_down:
                if _dep == 'floor':
                    variant_specs.append((
                        f'{b}-down-floor',
                        lambda p, _bump=bump: p - _bump * (p - self._gt_floor)))
                else:
                    variant_specs.append((
                        f'{b}-down-{_dep:g}',
                        lambda p, _bump=bump, _d=_dep: p - _bump * _d))
            for _dep in self._gt_edit_up:
                variant_specs.append((
                    f'{b}-up-{_dep:g}',
                    lambda p, _bump=bump, _d=_dep: p + _bump * _d))

        if not variant_specs:
            # Both GEN_TRANSPORT_EDIT_DOWN/UP set empty — no menu to judge.
            self._foldin_last_instructor_hist = {}
            self._foldin_last_instructor_unedited = k
            self._foldin_last_instructor_div_gain_mean = 0.0
            print(f'  | [INSTRUCTOR] edit menu empty (EDIT_DOWN/UP both blank) | '
                  f'unedited x{k} |', flush=True)
            return keep_native

        n_variants = len(variant_specs)
        edited_stack = np.empty((n_variants, k, L), dtype=np.float64)
        for v, (_lbl, _fn) in enumerate(variant_specs):
            edited_stack[v] = self._foldin_instructor_leash_project(_fn(price0))

        # One batched oracle call for all k*n_variants candidates.
        flat_score = _score(edited_stack.reshape(n_variants * k, L)).reshape(n_variants, k)
        best_v = flat_score.argmax(axis=0)                                  # (k,)
        best_score = flat_score[best_v, np.arange(k)]                       # (k,)
        edited_wins = best_score > base_score                               # strict only

        if typical:
            # Typical dose: pick one (block, direction) by MEAN divergence gain
            # over all kept windows, then apply its MILDEST depth to all of them.
            # The per-window argmax would pick extreme depths, which teach the tail.
            mean_gain = flat_score.mean(axis=1) - float(base_score.mean())  # (V,)
            _gain_sel = mean_gain
            win_v = int(_gain_sel.argmax())
            win_lbl = variant_specs[win_v][0]                               # 'b-dir-depth'
            _b, _dir = win_lbl.split('-')[0], win_lbl.split('-')[1]
            # mildest sibling in the SAME block+direction (menu order is not
            # guaranteed sorted; pick by numeric depth, floor = deepest down)
            def _depth_of(_l):
                _tok = _l.split('-', 2)[2]          # maxsplit: depth token intact
                return float('inf') if _tok == 'floor' else float(_tok)
            _sibs = [(i, _depth_of(l)) for i, (l, _f) in enumerate(variant_specs)
                     if l.startswith(f'{_b}-{_dir}-')]
            # fail-loud: a floor-only EDIT_DOWN override would make 'floor' the
            # "mildest" — the deepest edit under a mode named typical.
            assert any(np.isfinite(d) for _i, d in _sibs), (
                f"instructor_typical: no numeric depth in direction '{_dir}' "
                f"(menu floor-only?) — typical dose undefined")
            mild_v = min(_sibs, key=lambda t: t[1])[0]
            mild_lbl = variant_specs[mild_v][0]
            out_price = edited_stack[mild_v].copy()                         # ALL k windows
            hist = {mild_lbl: k}

            self._instr_tie_bank = []
            _bm = float(base_score.mean())
            _mg = (float(mean_gain[win_v] / abs(_bm) * 100.0) if _bm != 0.0 else 0.0)
            # _mg is the judged winner's gain; the log field stores the gain of
            # the edit that was applied.
            _ag = (float(mean_gain[mild_v] / abs(_bm) * 100.0) if _bm != 0.0 else 0.0)
            self._foldin_last_instructor_hist = hist
            self._foldin_last_instructor_unedited = 0
            self._foldin_last_instructor_div_gain_mean = _ag
            print(f'  | [INSTRUCTOR-TYP] direction={win_lbl} (mean-judged, '
                  f'gain={_mg:+.1f}%) applied={mild_lbl} x{k} | '
                  f'applied_gain={_ag:+.1f}% |', flush=True)
            if self._instr_head_aug:
                # Store the aim for the head augmentation. Block b spans
                # [4b,4b+4)h; sign +1 = down/cheaper (subtract), -1 = up (add),
                # as _instr_head_augment_price applies it.
                _blk = int(_b)
                _sgn = 1.0 if _dir == 'down' else -1.0
                _center_h = 4.0 * _blk + 2.0            # argmax block mid-hour
                _aim_note = f'argmax block={_blk}'
                if self._instr_head_aim_trend:
                    # Train-only trend aim; direction forced DOWN because the
                    # trend is a fall.
                    _tc, _tb, _tsl = self._trend_aim()
                    _center_h, _sgn = float(_tc), 1.0
                    _aim_note = (f'TREND-AIM block={_tb} centre={_center_h:.1f}h '
                                 f'(divergence argmax was block={_blk}) '
                                 'p5slope/yr=[' +
                                 ','.join(f'{v:+.2f}' for v in _tsl) + ']')
                if self._instr_head_depth_trend:
                    # Must run BEFORE the width search: the width is scored with
                    # a bell of _instr_head_amax depth.
                    _td = self._trend_depth()
                    _aim_note += (f' | TREND-DEPTH=${_td:.0f} '
                                  f'(was ${self._instr_head_amax:.0f})')
                    self._instr_head_amax = float(_td)
                if self._instr_head_discover_width:
                    # Score each candidate width by the same divergence; a very
                    # wide bell is a level shift with low divergence, so the
                    # widest does not win by default.
                    _Td_w = (self._foldin_prior.shape[0]
                             if self._foldin_prior is not None else self.PREDICT_LEN)
                    _slot_w = ((np.arange(L) % _Td_w) / _Td_w) * 24.0
                    _dcw = np.abs(_slot_w - _center_h)
                    _dcw = np.minimum(_dcw, 24.0 - _dcw)
                    _wsc = []
                    for _w in self._instr_head_width_cands:
                        _hw = max(float(_w), 1e-6)
                        _bellw = np.where(_dcw < _hw,
                                          0.5 + 0.5 * np.cos(np.pi * _dcw / _hw), 0.0)
                        _wamp = self._instr_head_amax
                        _edw = price0 - _sgn * _bellw[None, :] * _wamp
                        _wsc.append(float(_score(np.asarray(_edw, np.float64)).mean()))
                    _wi = int(np.argmax(_wsc))
                    self._instr_head_width = float(self._instr_head_width_cands[_wi])
                    _aim_note += (' | WIDTH-DISC=' + f'{self._instr_head_width:g}h '
                                  + '[' + ','.join(f'{w:g}h:{sc:+.0f}'
                                    for w, sc in zip(self._instr_head_width_cands, _wsc)) + ']')
                self._instr_head_dir = (_center_h, _sgn, win_lbl)
                print(f'  | [INSTR-HEAD-AUG] aim centre={_center_h:.1f}h dir={_dir} '
                      f'(sign={_sgn:+.0f}) from {win_lbl} | {_aim_note} | '
                      f'alpha<=${self._instr_head_amax:g} prob={self._instr_head_prob} |',
                      flush=True)
            native_price = self._price_real_to_native(out_price)
            out = keep_native.clone()
            out[:, 0, :] = torch.as_tensor(native_price, dtype=out.dtype,
                                           device=out.device)
            return out
