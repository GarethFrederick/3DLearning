"""Shim for the old misspelled package name.

On 2026-10-21 the package Utils/Dowstream was renamed to Utils/Downstream.
This shim keeps old imports working (old checkpoints, old scripts, and the
untouched second_domain/ code). It maps each Utils.Dowstream.<name> entry in
sys.modules to the SAME module object as Utils.Downstream.<name>, so module
state (caches, globals) is shared and never loaded twice.
"""
import importlib, sys
for _m in ('energy_scheduling', 'opt_oracle', 'icon_loader'):
    _mod = importlib.import_module(f'Utils.Downstream.{_m}')
    sys.modules[f'Utils.Dowstream.{_m}'] = _mod
    globals()[_m] = _mod
