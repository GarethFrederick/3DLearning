"""Re-runnable unit test for the per-row schedule cache used by
energy_scheduling.spo_plus_loss and the B1 greedy-contamination guard.

Imports the PRODUCTION helper `cache_solve_per_row` directly (no mirror copy),
so edits to the production cache block are exercised here. A fake solve_fn keeps
the test Gurobi-free and runnable anywhere.

Asserts: (1) reassembled output preserves batch row order; (2) only distinct
rows are solved (misses), repeats served from cache; (3) cross-"epoch" reuse;
(4) B1 — a solve that increments the greedy counter is used but NOT cached, so
the next call re-solves (no permanent laundering).

Run:  python -m pytest tests/test_spo_row_cache.py
  or:  python tests/test_spo_row_cache.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.Downstream.energy_scheduling import cache_solve_per_row


def test_order_dedup_and_cross_epoch_reuse():
    solve_calls = []

    def fake_solve_fn(arr):                # pretend LP: w = row * 10
        solve_calls.append(arr.shape[0])
        return arr * 10.0

    cache = {}
    A = np.array([[1., 2.], [3., 4.], [1., 2.]])   # row0 == row2 (dup in batch)
    B = np.array([[3., 4.], [9., 9.], [1., 2.]])   # reuses A's rows + one new

    rA = cache_solve_per_row(A, cache, fake_solve_fn)
    rB = cache_solve_per_row(B, cache, fake_solve_fn)

    assert np.allclose(rA, A * 10.0)               # (1) order + values
    assert np.allclose(rB, B * 10.0)
    assert len(cache) == 3                         # (2) distinct rows only
    # A is fully uncached → miss_idx [0,1,2]; the within-batch dup is solved
    # twice in that one solve (harmless). B then solves only its 1 new row.
    assert solve_calls == [3, 1]                   # (3) cross-epoch reuse


def test_b1_greedy_results_not_cached():
    """A solve that bumps the greedy counter must be USED but not CACHED."""
    greedy = {'n': 0}

    def greedy_count_fn():
        return greedy['n']

    def contaminating_solve(arr):
        greedy['n'] += arr.shape[0]                # simulate greedy fallback
        return arr * 10.0

    def clean_solve(arr):
        return arr * 100.0                         # different value to detect re-solve

    cache = {}
    X = np.array([[1., 2.], [3., 4.]])

    # First call: solver "falls back to greedy" → results used, NOT cached.
    r1 = cache_solve_per_row(X, cache, contaminating_solve, greedy_count_fn)
    assert np.allclose(r1, X * 10.0)
    assert len(cache) == 0, "greedy results must not be cached (B1)"

    # Second call with a clean solver re-solves (cache was empty) → fresh value.
    r2 = cache_solve_per_row(X, cache, clean_solve, greedy_count_fn)
    assert np.allclose(r2, X * 100.0), "must re-solve, not serve laundered greedy"
    assert len(cache) == 2, "clean results SHOULD be cached"


if __name__ == '__main__':
    test_order_dedup_and_cross_epoch_reuse()
    test_b1_greedy_results_not_cached()
    print('test_spo_row_cache: PASS')
