"""Regression test for the PPO block-pairing fix (MV path).

The bug: loss_hw is (B, n_z0_batches) — one regret column per generation batch.
The old cyclic tiling repeated columns into PIC-wide blocks, so ppo(batch_i)
multiplied r_theta(batch_i) by a MIX of all batches' regrets, destroying
per-sample credit assignment. The fix: in MV, ppo(batch_i) indexes COLUMN
batch_i directly.

This test drives DDD_Learning.ppo with a hand-built loss_hw where column j is
the constant j, so we can assert the block ppo selects is exactly column
batch_i (all == batch_i), not a mixture.

Run:  python tests/test_ppo_pairing.py
"""
import os
import sys
import types
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Utils'))  # bare `from utils import` in diffusion_utils
from Utils.Algos.ddd_learning import DDD_Learning


def _make_stub():
    """Minimal DDD_Learning shell exercising only ppo() — bypass __init__."""
    obj = DDD_Learning.__new__(DDD_Learning)
    obj.MV = True
    obj.PIC_SIZE = 28
    obj.PPO_CLIP = 0.4
    return obj


def test_mv_ppo_pairs_with_own_column():
    obj = _make_stub()
    B, n_batches = 64, 8
    # Column j is the constant j → trivially detect which column ppo selected.
    loss_hw = torch.arange(n_batches, dtype=torch.float32).repeat(B, 1)  # (B, 8)
    # r = ones (B, PIC_SIZE) so unclipped == clipped == the selected regret block.
    r = torch.ones(B, obj.PIC_SIZE)

    for batch_i in range(n_batches):
        ppo_loss = obj.ppo(r=r, loss_hw=loss_hw, batch_i=batch_i, detach_part='hw')
        # With r==1 and a constant column == batch_i, min(unclipped,clipped).mean()
        # == batch_i exactly. A mixed block would average several columns.
        assert abs(ppo_loss.item() - batch_i) < 1e-5, (
            f"batch_i={batch_i}: ppo paired with value {ppo_loss.item()}, "
            f"expected pure column {batch_i} (mixing → wrong credit assignment)")


if __name__ == '__main__':
    test_mv_ppo_pairs_with_own_column()
    print('test_ppo_pairing: PASS')
