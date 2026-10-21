"""Validation-only sampler for the trained CONDITIONAL mv DDPM (F=8, n_cond=6).

Samples with random train-fold calendar windows, fits offset-only signed-log
calibration on price, and writes calibration+clip into the meta sidecar so the
3DL z0 loader applies them automatically.
"""
import sys, os, json
sys.path.insert(0, '.')
sys.path.insert(0, 'Utils')
import numpy as np
import pandas as pd
import torch

from datasets import build_ml_features, inverse_mv_sample, signed_log
from Utils.Diffusion.DDPM import DiffusionProcess

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available() and os.environ.get('ALLOW_CPU') != '1':
    raise SystemExit("ERROR: CUDA not visible. Set ALLOW_CPU=1 to override.")
print(f"  | DEVICE | {DEVICE}")

_TAG = os.environ.get('DDPM_NAME_TAG', '')   # must match train_ddpm_mv_cond.py
CKPT = f'Utils/Models/model_DDPM_mv_F8_L192_T500{_TAG}_best.pth'
META = f'Utils/Models/model_DDPM_mv_F8_L192_T500{_TAG}_meta.json'
VAL_N = int(os.environ.get('VAL_N', '1024'))

with open(META) as f:
    meta = json.load(f)
norm_bounds   = {k: tuple(v) for k, v in meta['norm_bounds'].items()}
feature_names = meta['feature_names']
log_features  = set(meta['log_features'])
SEQ_LEN, N_FEATURES, T = meta['seq_len'], meta['n_features'], meta['T']
N_COND = meta['n_cond']
N_STOCH = N_FEATURES - N_COND
BETA_1, BETA_T = meta['beta_1'], meta['beta_T']
cond_pool = np.load(meta['cond_pool_path']).astype(np.float32)            # (N_pool, 6, L)
print(f"  | meta | n_cond={N_COND}  cond_pool={cond_pool.shape}  bounds={norm_bounds}")

model = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.eval()
proc = DiffusionProcess(beta_1=BETA_1, beta_T=BETA_T, T=T, diffusion_fn=model,
                        device=DEVICE, shape=(N_STOCH, SEQ_LEN), n_cond=N_COND)

chunks = []
remaining = VAL_N
while remaining > 0:
    n = min(256, remaining)
    pool_idx = np.random.randint(0, cond_pool.shape[0], n)
    cond_t = torch.from_numpy(cond_pool[pool_idx]).float().to(DEVICE)
    s, _ = proc.sampling(sampling_number=n, save_trajectory=False, cond=cond_t)
    chunks.append(s.detach().cpu().numpy())
    remaining -= n
gen = np.concatenate(chunks, axis=0)                                      # (VAL_N, 8, L)

# Read the same DATA_TRAIN as train_ddpm_mv_cond.py, or a non-AEMO generator
# gets calibrated against AEMO prices.
raw = pd.read_csv(os.environ.get('DATA_TRAIN', './Data/AEMO_VIC1_train_2015_2020.csv'))
feat = build_ml_features(raw)
feat_train = feat.iloc[:int(0.70 * len(feat))]
real = {'Price': feat_train['Price'].values,
        'Demand': feat_train['TOTALDEMAND_MW'].values}

# The clip is the target's PHYSICAL range. Defaults are the AEMO floor/cap;
# carbon intensity uses floor 0 (observed max 459).
PRICE_FLOOR = float(os.environ.get('DDPM_CLIP_LO', '-1000.0'))
PRICE_CAP = float(os.environ.get('DDPM_CLIP_HI', '15000.0'))
lo, hi = norm_bounds['Price']
gen_sl = gen[:, 0, :].reshape(-1) * (hi - lo) + lo
# ONLY log the reference if the generator was trained on logged prices (the
# meta's log_features says so); else the offset mixes real and log space.
real_sl = signed_log(real['Price']) if 'Price' in log_features else real['Price']
g50 = float(np.percentile(gen_sl, 50))
r50 = float(np.percentile(real_sl, 50))
a, b = 1.0, float(r50 - g50)
calibration = {'Price': [a, b]}
clip = {'Price': [PRICE_FLOOR, PRICE_CAP]}
print(f"  | price calibration (signed-log offset) | a={a:.4f} b={b:.4f}  | clip | [{PRICE_FLOOR}, {PRICE_CAP}]")

gen_uncal = inverse_mv_sample(gen, feature_names, norm_bounds, log_features=log_features)
gen_cal   = inverse_mv_sample(gen, feature_names, norm_bounds,
                              log_features=log_features, calibration=calibration, clip=clip)


def lag1(arrs):
    acs = [pd.Series(w).autocorr(1) for w in arrs if pd.Series(w).std() > 1e-8]
    return float(np.nanmean(acs))


def stats(name, g, rv):
    print(f"  [{name:6s}] gen  mean={g.mean():8.1f} median={np.median(g):8.1f} "
          f"std={g.std():8.1f} min={g.min():9.1f} max={g.max():9.1f} lag1={lag1(g):.3f}")
    print(f"           real mean={rv.mean():8.1f} median={np.median(rv):8.1f} "
          f"std={rv.std():8.1f} min={rv.min():9.1f} max={rv.max():9.1f} lag1={pd.Series(rv).autocorr(1):.3f}")


print(f"\n=============== Cond MV DDPM VALIDATION (N={VAL_N}, real units) ===============")
print("  --- UNCALIBRATED ---")
stats('Price', gen_uncal[:, 0, :], real['Price'])
print("  --- CALIBRATED (bulk-matched) ---")
stats('Price',  gen_cal[:, 0, :], real['Price'])
stats('Demand', gen_cal[:, 1, :], real['Demand'])
gpc = gen_cal[:, 0, :]
for thr in (300, 1000, 5000):
    print(f"  | calibrated gen price >${thr:5d} | {float((gpc > thr).mean()*100):.4f}%  "
          f"(real >${thr}: {float((real['Price'] > thr).mean()*100):.4f}%)")
print("=========================================================================")

# REACH gate on the CALIBRATED generator: only write the meta (and let the
# afterok 3DL job fire) if the generator's support covers the OOD regime.
# Reach, not frequency — the adversary amplifies frequency itself.
_gc = gpc.reshape(-1)
_gmin, _gmax = float(_gc.min()), float(_gc.max())
_neg = float(os.environ.get('DDPM_GATE_NEG_MIN', '-120'))
_pos = float(os.environ.get('DDPM_GATE_POS_MAX', '550'))
_ok = (_gmin < _neg) and (_gmax > _pos)
print(f"  | DISPERSION GATE (reach, calibrated) | min=${_gmin:.1f} (need <${_neg:.0f})  "
      f"max=${_gmax:.1f} (need >${_pos:.0f}) | {'PASS' if _ok else 'FAIL'}")
if not _ok:
    raise SystemExit("REACH GATE FAILED on calibrated generator — meta NOT written; "
                     "afterok 3DL will not launch. Raise DDPM_MV_TAIL_ALPHA and retrain.")

meta['calibration'] = calibration
meta['clip'] = clip
with open(META, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  | calibration+clip written to | {META}")
