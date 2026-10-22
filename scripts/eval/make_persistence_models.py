"""Write the two persistence baselines as scorer-readable model files.

persist        forecast = the previous 48 half-hours of price, unchanged.
persistdip120  the same, minus a fixed raised-cosine dip of $120 centred at
               14:00 with a 2 h half-width (no training, no projection).

Both are written as Utils/Models/3DLearning_mv_cond_gt_<arm>_s1-ML-epoch1.pth,
so run_poc_eval_measure.sh scores them with POC_ARMS=<arm>,POC_SEEDS=1. The
price bounds are the same 70%-train-fold bounds the scorer uses.
"""
import os
import sys
from pathlib import Path

import torch

ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import poc_eval_ood as P                      # noqa: E402
from Utils.Net.task_net import PersistencePredictor  # noqa: E402

price_b, _ = P.train_fold_bounds()
for arm, dip in (('persist', 0.0), ('persistdip120', 120.0)):
    path = f'{P.MODELS_DIR}/3DLearning_mv_cond_gt_{arm}_s1-ML-epoch1.pth'
    if os.path.exists(path):
        sys.exit(f'FATAL: {path} exists; move it aside first')
    torch.save(PersistencePredictor(price_b[0], price_b[1], dip_dollars=dip,
                                    dip_center_h=14.0, dip_halfwidth_h=2.0), path)
    print(f'wrote {path}  (price bounds {price_b}, dip ${dip:g})')
