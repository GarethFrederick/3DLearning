"""Optimisation oracle for the 3D-Learning adversary.

The adversary never touches a task object; it talks to this oracle, so new
downstream problems need only a new adapter.

Contract (all operate on a batch of cost vectors c of shape (B, T)):

    solve(c)               -> (w_star, cost_star)
        w_star    : (B, D) optimal decision encoding (D = task decision dim)
        cost_star : (B,)   optimal objective value

    sensitivity(c)         -> s            (B, T)
        Per-slot sensitivity of the optimal cost to the cost coefficient at slot t.
        For a linear objective min c·w, the LP/LP-relaxation envelope theorem gives
        ∂cost*/∂c[t] = w*[t], so the sensitivity IS the optimal primal decision on
        the cost-bearing slots. This is the structural signal the adversary is
        conditioned on: where the decision "spends", small price changes move cost.

    divergence(c, c_ref)   -> d            (B, T)
        Per-slot decision shift |w*(c) - w*(c_ref)| between a candidate regime c
        and a reference regime c_ref (e.g. the training-prior price). This is the
        decision-focused distance the adversary is rewarded for maximising: a
        scenario that merely scales prices has d≈0; a scenario that FLIPS the
        cheap-hour ordering (the OOD failure mode) has large d.

A backend adapter wraps a concrete task. EnergyOracle wraps Energy_Scheduling;
the only task-specific knowledge is how to read w* out of the task's schedule
encoding.
"""
import numpy as np
import torch


class OptimizationOracle:
    """Abstract base. Subclass per downstream problem; the adversary only ever
    sees this interface, never the concrete task."""

    # --- subclasses must provide ---------------------------------------
    def solve(self, c):
        raise NotImplementedError

    # --- derived signals (generic, defined in terms of solve) ----------
    def sensitivity(self, c):
        """∂cost*/∂c[t] = w*[t] (envelope theorem, linear objective)."""
        w_star, _ = self.solve(c)
        return self._decision_to_slot(w_star, c)

    def divergence(self, c, c_ref):
        """Per-slot decision shift |w*(c) - w*(c_ref)|."""
        w_c,   _ = self.solve(c)
        w_ref, _ = self.solve(c_ref)
        s_c   = self._decision_to_slot(w_c,   c)
        s_ref = self._decision_to_slot(w_ref, c_ref)
        return (s_c - s_ref).abs()

    def _decision_to_slot(self, w_star, c):
        """Map the task's decision encoding to a per-slot (B, T) sensitivity that
        lines up with the cost vector c. Default: identity-on-leading-T (the
        decision's first T entries are the cost-bearing slots). Override if the
        encoding differs."""
        T = c.shape[-1]
        return w_star[..., :T]


class EnergyOracle(OptimizationOracle):
    """Adapter for Utils.Downstream.energy_scheduling.Energy_Scheduling.

    The task encodes a schedule as x of shape (B, T+1): x[:, :T] = energy profile
    w_t (the cost-bearing decision, sensitivity ∂cost*/∂price[t]) and x[:, T] = a
    price-independent fixed startup/shutdown cost. So _decision_to_slot drops the
    trailing fixed-cost column and (when the LP grid is finer than the price grid)
    coarsens w from the internal T back to the external T_external the prices live
    on, so sensitivity/divergence align with the generated price vector.

    solve() uses the LP relaxation (find_best_scheduling): the envelope identity
    holds at the LP optimum, so the adversary does not need a MIP-exact decision.
    """

    def __init__(self, task, device=None, use_mip=False):
        self.task = task
        self.device = device
        # LP relaxation by default; the MIP is much slower.
        self._solver = (getattr(task, 'find_best_scheduling_mip', None)
                        if use_mip else None) or task.find_best_scheduling
        self.T_internal = int(getattr(task, 'T', 0))
        self.T_external = int(getattr(task, 'T_external', self.T_internal))
        self.upsample_factor = int(getattr(task, 'upsample_factor', 1))

    def solve(self, c):
        if not torch.is_tensor(c):
            c = torch.as_tensor(c, dtype=torch.float32,
                                 device=self.device)
        x_star, cost_star = self._solver(y=c)
        return x_star, cost_star

    def _decision_to_slot(self, w_star, c):
        # Drop the trailing fixed-cost column -> (B, T_internal).
        w = w_star[..., :self.T_internal]
        # The task upsamples prices with repeat_interleave; its adjoint (sum over
        # each block) maps w back to the price grid (B, T_external).
        if self.upsample_factor > 1 and w.shape[-1] == self.T_internal:
            B = w.shape[0]
            w = w.reshape(B, self.T_external, self.upsample_factor).sum(dim=-1)
        return w


def build_oracle(task, device=None, **kwargs):
    """Factory: pick the adapter for a task instance."""
    cls_name = type(task).__name__
    # EnergyOracle needs only T / T_external / upsample_factor and
    # find_best_scheduling. Battery_Scheduling meets that contract, and its
    # sensitivity dcost*/dp_t = net*_t is the slot decision.
    if cls_name in ('Energy_Scheduling', 'Battery_Scheduling'):
        return EnergyOracle(task, device=device, **kwargs)
    raise ValueError(
        f'No OptimizationOracle adapter registered for task type {cls_name!r}. '
        f'Add an adapter in Utils/Downstream/opt_oracle.py.')
