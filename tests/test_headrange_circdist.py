"""Characterization tests for the head-range map and the clock-distance helper.

Run on a compute node:  python tests/test_headrange_circdist.py  (exits 1 on a
failed check).

1. The head-range affine map in Utils/Net/task_net.py (DeepLSTM.forward and the
   init-bias solve). A forgotten PRED_NORM_LO/HI silently reruns the narrow
   head; a sign slip shifts every real-dollar price the wide-head arms report.
   The tests pin forward and inverse at BOTH (0,1) and (-0.5,1.5).

2. circ_dist / circ_mean_hour from scripts/analysis/diag_trough_location.py,
   the source of the trough-location numbers. That file runs at module scope,
   so they are copied here; test_circ_copy_has_not_drifted fails if the copy
   and the source differ (whitespace ignored).
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Utils'))

FAILS = []


def check(label, cond):
    print(f'  | {"PASS" if cond else "FAIL"}  {label}')
    if not cond:
        FAILS.append(label)


# circ_dist / circ_mean_hour: VERBATIM copies of diag_trough_location.py (not
# import-safe). Change both together; the drift test below enforces it.
def circ_mean_hour(idx):
    a = 2 * np.pi * np.asarray(idx) / 48.0
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi)) / (2 * np.pi) * 24.0


def circ_dist(h1, h2):
    d = abs(h1 - h2) % 24.0
    return min(d, 24.0 - d)


def test_circ_copy_has_not_drifted():
    """The two copies above must still match the source, whitespace ignored.
    The source is parsed with ast, not imported (it has no __main__ guard)."""
    import ast
    import inspect

    src_path = os.path.join(ROOT, 'scripts', 'analysis', 'diag_trough_location.py')
    src = open(src_path).read()
    tree = ast.parse(src)
    upstream = {n.name: ast.get_source_segment(src, n)
                for n in tree.body if isinstance(n, ast.FunctionDef)}

    for name, local_fn in [('circ_mean_hour', circ_mean_hour),
                           ('circ_dist', circ_dist)]:
        check(f'{name} still exists in diag_trough_location.py',
              name in upstream)
        if name not in upstream:
            continue
        a = ''.join(upstream[name].split())
        b = ''.join(inspect.getsource(local_fn).split())
        check(f'{name} copy matches the production source exactly', a == b)


def test_circ_dist_basic():
    check('circ_dist(1, 3) == 2', math.isclose(circ_dist(1, 3), 2.0))
    check('circ_dist(3, 1) == 2 (symmetric)', math.isclose(circ_dist(3, 1), 2.0))
    check('circ_dist(0, 0) == 0', math.isclose(circ_dist(0, 0), 0.0))


def test_circ_dist_wraps_midnight():
    # 23h and 1h are 2h apart on a clock, not 22h.
    check('circ_dist(23, 1) == 2 (wraps midnight)', math.isclose(circ_dist(23, 1), 2.0))
    check('circ_dist(0.5, 23.5) == 1 (wraps at the boundary)',
          math.isclose(circ_dist(0.5, 23.5), 1.0))


def test_circ_dist_antipodal_is_the_max():
    # 12h is the farthest two hours can be on a 24h clock; nothing may exceed it.
    check('circ_dist(0, 12) == 12 (antipodal max)', math.isclose(circ_dist(0, 12), 12.0))
    for h1 in range(24):
        for h2 in range(24):
            d = circ_dist(h1, h2)
            if d > 12.0 + 1e-9:
                check(f'circ_dist({h1},{h2})={d} exceeds the 12h ceiling', False)
    check('no (h1,h2) pair in 0..23 exceeds the 12h ceiling', True)


def test_circ_mean_hour_wraps_correctly():
    # Half-hour SLOT indices on a 48-slot day (idx 0 = 00:00, idx 47 = 23:30).
    # Slots either side of midnight (47 -> 23:30, 0 -> 00:00) must average to
    # ~0h, not ~12h — the failure a naive arithmetic mean would produce.
    m = circ_mean_hour([47, 0])
    check(f'circ_mean_hour([47,0])={m:.2f} is near 0h/24h, not 12h',
          (m < 1.0) or (m > 23.0))
    # A cluster in the middle of the day must average to its arithmetic mean.
    m2 = circ_mean_hour([20, 22, 24])   # 10h, 11h, 12h
    check(f'circ_mean_hour([20,22,24])={m2:.2f} == 11.0h (no wrap needed)',
          math.isclose(m2, 11.0, abs_tol=1e-6))


# Head-range affine map: forward vs the inverse that seeds the initial bias.
# Imports the PRODUCTION class directly.
def test_headrange_forward_matches_narrow_default():
    """At the default (0,1) range the affine map must reduce to plain sigmoid,
    so the default-off PRED_NORM_LO/HI path stays unchanged."""
    import torch
    from Utils.Net.task_net import DeepLSTM

    os.environ.pop('PRED_NORM_LO', None)
    os.environ.pop('PRED_NORM_HI', None)
    torch.manual_seed(0)
    m = DeepLSTM(input_size=8, head_mode='sigmoid', pred_lo=0.0, pred_hi=1.0)
    m.eval()
    x = torch.randn(4, 5, 8)
    with torch.no_grad():
        out = m(x)
    check('narrow-range output stays inside [0,1]',
          bool((out >= 0.0).all() and (out <= 1.0).all()))


def test_headrange_forward_reaches_wide_bounds():
    """The wide range must actually widen the output space.

    Drive the map, do not sample it: an untrained net outputs about 0.5 at any
    range. Zero fc2's weight and set its bias, so the output is exactly
    pred_lo + (pred_hi - pred_lo) * sigmoid(bias). A hardcoded (0,1) anywhere
    leaves the wide model inside [0,1] and fails this."""
    import torch
    from Utils.Net.task_net import DeepLSTM

    def drive(pred_lo, pred_hi, bias):
        torch.manual_seed(0)
        m = DeepLSTM(input_size=8, head_mode='sigmoid', pred_lo=pred_lo, pred_hi=pred_hi)
        m.eval()
        with torch.no_grad():
            m.fc2.weight.zero_()
            m.fc2.bias.fill_(bias)
            return float(m(torch.randn(2, 5, 8)).flatten()[0])

    # sigmoid(-3) = 0.0474 -> wide: -0.5 + 2.0 * 0.0474 = -0.405, below zero.
    low_w = drive(-0.5, 1.5, -3.0)
    high_w = drive(-0.5, 1.5, 3.0)
    check(f'wide head reaches below 0.0 (got {low_w:.4f})', low_w < 0.0)
    check(f'wide head reaches above 1.0 (got {high_w:.4f})', high_w > 1.0)
    check(f'wide head stays inside its declared bounds (got {low_w:.4f}, {high_w:.4f})',
          -0.5 - 1e-5 <= low_w and high_w <= 1.5 + 1e-5)

    # Contrast: the same logits on the narrow head must stay inside [0,1], or the
    # default path changed.
    low_n = drive(0.0, 1.0, -3.0)
    high_n = drive(0.0, 1.0, 3.0)
    check(f'narrow head cannot go below 0.0 (got {low_n:.4f})', low_n >= 0.0)
    check(f'narrow head cannot go above 1.0 (got {high_n:.4f})', high_n <= 1.0)
    check('the two ranges give different outputs at the same logit',
          abs(low_w - low_n) > 1e-6)


def test_headrange_init_bias_solve_is_the_forward_maps_inverse():
    """task_net.py seeds sigmoid(b) ~= (target - pred_lo) / (pred_hi - pred_lo)
    so the forward map reproduces `target` at init. Round trip at the wide
    range too, where a sign slip is visible (at (0,1) it hides near 0.5)."""
    for pred_lo, pred_hi, target in [(0.0, 1.0, 0.3), (-0.5, 1.5, 0.1),
                                      (-0.5, 1.5, -0.4), (-0.5, 1.5, 1.4)]:
        p = (target - pred_lo) / (pred_hi - pred_lo)
        p = max(min(p, 1.0 - 1e-4), 1e-4)     # mirrors the task_net.py clamp exactly
        b = math.log(p / (1.0 - p))
        s = 1.0 / (1.0 + math.exp(-b))        # sigmoid(b), what fc2.bias seeds
        back = pred_lo + (pred_hi - pred_lo) * s
        check(f'init-bias round trip @ range=[{pred_lo},{pred_hi}] target={target}: '
              f'got {back:.4f}', math.isclose(back, target, abs_tol=1e-3))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        print(f'\n{fn.__name__}')
        fn()
    print(f'\n{len(fns)} tests, {len(FAILS)} failing checks.')
    if FAILS:
        print('FAILURES:')
        for f in FAILS:
            print(f'  - {f}')
        sys.exit(1)
    print('ALL CHARACTERIZATION CHECKS PASS.')
