"""Does each model put its cheap hours where the cheap hours really are?

Measures every window of both held-out eras, seed-1 heads from
results/scores/poc_eval_measure. Forward passes only, no solver; prints only
(plus DIAG_DUMP_JSON if set).

For each window it takes the 6 cheapest half-hours (3 hours) of the true day and
of each model's prediction. Two measures:
  overlap  how many of the 6 match, out of 6. Ties at the output floor are
           common, so a set overlap is safer than an argmin.
  centre   circular distance in hours between the middle of each model's cheapest
           6 and the middle of the true cheapest 6.
"""
import json, numpy as np, pandas as pd, torch, os
import sys
from pathlib import Path
ROOT = os.environ.get('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
import poc_eval_ood as P
from Utils.datasets import Data_Loader, build_ml_features

# DIAG_ARMS="ours=<arm>,3DL=<arm>"; the 'ours' key is required. Unset = the
# narrow pair (a34_honestdose vs a1_stall).
_da = os.environ.get('DIAG_ARMS', '')
ARMS = (dict(kv.split('=', 1) for kv in _da.split(',')) if _da
        else {'ours': 'a34_honestdose', '3DL': 'a1_stall'})
K = 6

def circ_mean_hour(idx):
    a = 2*np.pi*np.asarray(idx)/48.0
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2*np.pi))/(2*np.pi)*24.0

def circ_mean_hour_of(hours):
    a = 2*np.pi*np.asarray(hours)/24.0
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2*np.pi))/(2*np.pi)*24.0


def circ_dist(h1, h2):
    d = abs(h1-h2) % 24.0
    return min(d, 24.0-d)

price_b, demand_b = P.train_fold_bounds()
lo, hi = price_b
norm_full = {'Price': price_b, 'TOTALDEMAND_MW': demand_b}

for regime, csv in P.OOD_REGIMES:
    heads = {}
    for lab, arm in ARMS.items():
        j = json.load(open(f'{P.ROOT}/results/scores/poc_eval_measure/{arm}_s1_{regime}_selected.json'))
        heads[lab] = torch.load(j['head_path'], weights_only=False, map_location='cpu'); heads[lab].eval()
    F = P.detect_F(json.load(open(f'{P.ROOT}/results/scores/poc_eval_measure/{ARMS["ours"]}_s1_{regime}_selected.json'))['head_path'])
    cols = P.feature_cols_for(F or 8)
    feat = build_ml_features(pd.read_csv(csv, parse_dates=['TIMESTAMP']))[cols]
    nb = {k: v for k, v in norm_full.items() if k in cols}
    loader = Data_Loader(df=feat).ml_set2loader_mv(split_test=False, normlized=True,
        device='cpu', batch_size=P.BATCH_SIZE, pic_size=P.ML_SEQ_LEN, norm_bounds=nb)
    W = Pl = 48
    res = {lab: {'ov': [], 'cd': [], 'hr': []} for lab in ARMS}
    true_hr = []
    with torch.no_grad():
        for batch in loader:
            sb = batch[0]
            if sb.dim() == 2: sb = sb.unsqueeze(-1)
            for t in range(W, sb.shape[1]-Pl+1, 24):
                inp, tgt = sb[:, t-W:t, :], sb[:, t:t+Pl, 0]
                true = (tgt*(hi-lo)+lo).numpy()
                for lab, m in heads.items():
                    pred = (m(inp)*(hi-lo)+lo).numpy()
                    for b in range(true.shape[0]):
                        ti = set(np.argsort(true[b])[:K].tolist())
                        pi = set(np.argsort(pred[b])[:K].tolist())
                        res[lab]['ov'].append(len(ti & pi))
                        _ph = circ_mean_hour(sorted(pi)); _th = circ_mean_hour(sorted(ti))
                        res[lab]['hr'].append(_ph)
                        if lab == list(ARMS)[0]: true_hr.append(_th)
                        res[lab]['cd'].append(circ_dist(_ph, _th))
    n = len(res['ours']['ov'])
    print(f"\n=== {regime}   n={n} windows ===")
    print(f"  TRUE cheap-hour centre, circular mean over windows = {circ_mean_hour_of(true_hr):.2f}h")
    print(f"  {'model':6s} {'cheap-hours matched /6':>24s} {'hours off centre':>20s} {'within 2h':>11s} {'its own centre':>15s}")
    for lab in ARMS:
        ov = np.array(res[lab]['ov']); cd = np.array(res[lab]['cd'])
        print(f"  {lab:6s} {ov.mean():>24.2f} {cd.mean():>20.2f} {100*(cd<=2).mean():>10.1f}% "
              f"{circ_mean_hour_of(res[lab]['hr']):>14.2f}h")
    # DIAG_DUMP_JSON=<path>: write per-window arrays for the thesis figure. Off by
    # default (stdout unchanged). The file is rewritten each era, holding all eras so far.
    if os.environ.get('DIAG_DUMP_JSON'):
        _dump = globals().setdefault('_DUMP', {})
        _dump[regime] = {'true_hr': [float(x) for x in true_hr],
                         **{lab: {k: [float(x) for x in v] for k, v in res[lab].items()}
                            for lab in ARMS}}
        with open(os.environ['DIAG_DUMP_JSON'], 'w') as _f:
            json.dump(_dump, _f)
        print(f"  [DUMP] per-window arrays for {regime} ({n} windows) -> {os.environ['DIAG_DUMP_JSON']}")
