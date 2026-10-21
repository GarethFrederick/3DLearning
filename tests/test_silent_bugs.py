"""Fast, CPU-only regression tests for the project's recurring SILENT-bug classes
(the ones a refactor most easily reintroduces). Gurobi-free, GPU-free, no IO.

Covers:
  1. signed_log / signed_expm1 round-trip exactness over the real AEMO price range
     — the load-bearing TRANSFORMED<->REAL-$ boundary. A sign/branch/clamp slip
     here silently shifts every real-$ price the thesis reports.
  2. The 700-clamp tames overflow WITHOUT touching the in-range values.
  3. set_global_seed determinism — the MV_SEED no-op class (a "multi-seed" run
     that is silently identical).

Run:  python -m pytest tests/test_silent_bugs.py
  or:  python tests/test_silent_bugs.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'Utils'))

from Utils.datasets import signed_log, signed_expm1   # noqa: E402
from Utils.utils import set_global_seed                # noqa: E402

# Real VIC1 AEMO operating range: market floor -$1000, historical cap ~$16,600,
# code clip 17500. Include the deep-negative solar trough and the evening spike.
_REAL_PRICES = np.array(
    [-1000.0, -676.0, -95.0, -9.0, 0.0, 5.0, 66.0, 247.0, 14700.0, 17500.0])


def test_signed_log_roundtrip_is_identity():
    """signed_expm1(signed_log(x)) == x across the real-$ range (REAL-$ -> in -> REAL-$)."""
    back = signed_expm1(signed_log(_REAL_PRICES))
    assert np.allclose(back, _REAL_PRICES, atol=1e-6), \
        f"round-trip drifted: {back - _REAL_PRICES}"


def test_signed_log_preserves_sign():
    """The transform must never flip the sign of a price (a -$95 must stay negative)."""
    y = signed_log(_REAL_PRICES)
    nz = _REAL_PRICES != 0.0
    assert np.all(np.sign(y[nz]) == np.sign(_REAL_PRICES[nz]))


def test_signed_expm1_clamp_tames_overflow_but_not_inrange():
    """The 700-clamp prevents inf on absurd inputs yet leaves real-range values exact."""
    # In-range: signed_log($17500) ~= 9.77 << 700, so the clamp must be a no-op.
    in_range = signed_log(_REAL_PRICES)
    assert np.all(np.abs(in_range) < 700.0)
    assert np.allclose(signed_expm1(in_range), _REAL_PRICES, atol=1e-6)
    # Absurd guided value: would overflow expm1 (np.expm1(710)=inf) -> must stay finite.
    assert np.isfinite(signed_expm1(np.array([710.0, -710.0]))).all()


def test_set_global_seed_is_reproducible_and_seed_dependent():
    """Same seed -> identical draws (torch+numpy); different seed -> different.

    Locks the MV_SEED-no-op class: a 'multi-seed' run that is silently identical.
    """
    import torch

    set_global_seed(7)
    a_t, a_n = torch.randn(8), np.random.rand(8)
    set_global_seed(7)
    b_t, b_n = torch.randn(8), np.random.rand(8)
    assert torch.equal(a_t, b_t), "torch RNG not reproducible under same seed"
    assert np.allclose(a_n, b_n), "numpy RNG not reproducible under same seed"

    set_global_seed(8)
    c_t = torch.randn(8)
    assert not torch.equal(a_t, c_t), "different seed produced identical draws (no-op seed)"


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} silent-bug regression tests passed.")
