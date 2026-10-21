"""Conditional multivariate (1D-over-time) DDPM training driver — F=8.

Stoch channels (denoised): Price, TOTALDEMAND_MW.
Cond  channels (clean):    sin_h, cos_h, sin_dow, cos_dow, sin_m, cos_m.

The backbone reads all 8 channels but predicts noise only on the first 2.
At sampling time the cond channels come from a train-fold calendar pool
(saved beside the checkpoint) and are re-attached every reverse step.

Saves ({TAG} = DDPM_NAME_TAG, default ''):
  Utils/Models/model_DDPM_mv_F8_L192_T500{TAG}_best.pth
  Utils/Models/model_DDPM_mv_F8_L192_T500{TAG}_meta.json
  Utils/Models/model_DDPM_mv_F8_L192_T500{TAG}_cond_pool.npy
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'Utils')
import json
import numpy as np
import pandas as pd
import torch

from datasets import Data_Loader, build_ml_features, signed_log, inverse_mv_sample
from Utils.diffusion_utils import Train_DFU_MV
from Utils.Diffusion.DDPM import DiffusionProcess

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available() and os.environ.get('ALLOW_CPU') != '1':
    raise SystemExit("ERROR: CUDA not visible — DDPM training on CPU is ~18s/iter. "
                     "Set ALLOW_CPU=1 to override.")
print(f"  | DEVICE | {DEVICE}  | GPU | "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

SEQ_LEN = 192
N_FEATURES = 8
N_COND = 6
N_STOCH = N_FEATURES - N_COND   # 2: Price, Demand
T = 500
BETA_1, BETA_T = 5e-2, 1e-1
torch.manual_seed(42); np.random.seed(42)

# F=8 feature df (Price, Demand, sin_h, cos_h, sin_dow, cos_dow, sin_m, cos_m).
# build_ml_features orders cols this way: stoch leads, cond tails.
# DATA_TRAIN lets a second domain train its own generator; the default
# reproduces the production checkpoint.
_TRAIN_CSV = os.environ.get('DATA_TRAIN', './Data/AEMO_VIC1_train_2015_2020.csv')
raw = pd.read_csv(_TRAIN_CSV)
print(f'  | ddpm train csv | {_TRAIN_CSV}', flush=True)

# DDPM_MIDDAY puts the midday-cheap regime, absent from 2015-2020, into the
# DDPM's training data. off (default) = production model.
#   solar : subtract a clear-sky solar bell from daytime price on some days.
#   real  : append the oodval halves. Uses ONLY oodval, never oodtest, so the
#           held-out test claim stays intact.
# Give each variant its own DDPM_NAME_TAG.
_midday = os.environ.get('DDPM_MIDDAY', 'off').lower()
if _midday == 'solar':
    _amax  = float(os.environ.get('DDPM_SOLAR_ALPHA_MAX', '120.0'))  # $/MWh peak discount
    _prob  = float(os.environ.get('DDPM_SOLAR_PROB', '0.5'))         # fraction of days augmented
    _ts    = pd.to_datetime(raw['TIMESTAMP'])
    _hour  = (_ts.dt.hour + _ts.dt.minute / 60.0).values
    # Clear-sky half-sine bell over daylight [6,18], peak at solar noon, 0 at night.
    _bell  = np.where((_hour >= 6.0) & (_hour <= 18.0),
                      np.clip(np.sin(np.pi * (_hour - 6.0) / 12.0), 0.0, None), 0.0)
    # Random depth per day; only `prob` of days change, so night-cheap stays.
    _rng   = np.random.RandomState(42)
    _day_id = _ts.dt.normalize().values
    _udays  = pd.unique(_day_id)
    _day_alpha = {d: (_rng.uniform(0.0, _amax) if _rng.rand() < _prob else 0.0)
                  for d in _udays}
    _alpha_vec = np.array([_day_alpha[d] for d in _day_id])
    raw = raw.copy()
    raw['Price'] = raw['Price'].values - _alpha_vec * _bell
    print(f"  | DDPM_MIDDAY=solar | α_max=${_amax:.0f} prob={_prob} | "
          f"augmented {sum(v>0 for v in _day_alpha.values())}/{len(_udays)} days | "
          f"price mean ${raw['Price'].mean():.1f} (was ${pd.read_csv(_TRAIN_CSV)['Price'].mean():.1f})")
elif _midday == 'real':
    _ood_paths = [os.environ.get('DATA_OODVAL_A',
                                 './Data/AEMO_VIC1_oodval_2021_2022.csv'),
                  os.environ.get('DATA_OODVAL_B',
                                 './Data/AEMO_VIC1_oodval_2023_2025.csv')]
    _extra = [pd.read_csv(p) for p in _ood_paths]
    _n0 = len(raw)
    raw = pd.concat([raw] + _extra, ignore_index=True)
    print(f"  | DDPM_MIDDAY=real | +oodval (DISJOINT from oodtest): "
          f"{[len(e) for e in _extra]} rows | total {_n0} -> {len(raw)} | "
          f"price mean ${raw['Price'].mean():.1f}")
elif _midday != 'off':
    raise SystemExit(f"DDPM_MIDDAY must be off|solar|real, got {_midday!r}")

feat = build_ml_features(raw)
FEATURE_NAMES = list(feat.columns)
assert FEATURE_NAMES[:N_STOCH] == ['Price', 'TOTALDEMAND_MW'], FEATURE_NAMES
COND_NAMES = FEATURE_NAMES[N_STOCH:]
print(f"  | feature_names | {FEATURE_NAMES}")
print(f"  | stoch | {FEATURE_NAMES[:N_STOCH]}  | cond | {COND_NAMES}")

# Train fold: chronological 70% for off/solar. For real, the oodval rows sit
# at the END, so a :70% split would drop them all; shuffle whole days instead
# (the DDPM learns a distribution, not a forward forecast).
if _midday == 'real':
    _day = pd.to_datetime(raw['TIMESTAMP']).dt.normalize().values
    _udays = pd.unique(_day)
    _rng2 = np.random.RandomState(42)
    _perm = _rng2.permutation(len(_udays))
    _train_days = set(_udays[_perm[:int(0.70 * len(_udays))]])
    _mask = np.array([d in _train_days for d in _day])
    feat_train = feat.iloc[_mask].reset_index(drop=True)
    print(f"  | real-variant train fold | {_mask.sum()}/{len(feat)} rows "
          f"(day-shuffled 70%, both 2015-2020 + oodval regimes represented)")
else:
    n_train = int(0.70 * len(feat))
    feat_train = feat.iloc[:n_train]
price, demand = feat_train['Price'], feat_train['TOTALDEMAND_MW']

# signed-log suits AEMO prices (-1000 to 15000). DDPM_LOG_PRICE=0 turns it
# off for a narrow positive target; default 1 = production checkpoint.
LOG_FEATURES = {'Price'} if os.environ.get('DDPM_LOG_PRICE', '1') == '1' else set()
price_sl = signed_log(price.values)
# THE BOUNDS MUST LIVE IN THE SAME SPACE AS THE DATA, or the no-log path
# normalises raw prices against log-space bounds.
_price_for_bounds = price_sl if 'Price' in LOG_FEATURES else price.values
# Price bounds: train-fold P1/P99 by default. DDPM_PRICE_FLOOR/CAP (real $,
# both needed) widen them; NOTE they go through signed_log unconditionally.
_pfloor = os.environ.get('DDPM_PRICE_FLOOR')
_pcap   = os.environ.get('DDPM_PRICE_CAP')
if _pfloor is not None and _pcap is not None:
    _price_bounds = (float(signed_log(np.array([float(_pfloor)]))[0]),
                     float(signed_log(np.array([float(_pcap)]))[0]))
    print(f"  | WIDE price bounds | real [${float(_pfloor):.0f}, ${float(_pcap):.0f}] "
          f"-> signed-log {_price_bounds}")
else:
    _price_bounds = (float(np.quantile(_price_for_bounds, 0.01)),
                     float(np.quantile(_price_for_bounds, 0.99)))
norm_bounds = {
    'Price':          _price_bounds,
    'TOTALDEMAND_MW': (float(demand.quantile(0.01)), float(demand.quantile(0.99))),
}
NAME_TAG = os.environ.get('DDPM_NAME_TAG', '')
# Calendar features (sin/cos) already live in [-1, 1] → no entry in norm_bounds,
# they pass through ml_set2loader_mv / inverse_mv_sample unchanged.
print(f"  | log_features | {sorted(LOG_FEATURES)}")
print(f"  | norm_bounds  | {norm_bounds}")

trainer = Train_DFU_MV(
    T=T, beta_1=BETA_1, beta_T=BETA_T,
    batch_size=int(os.environ.get('DDPM_MV_BATCH', '64')),
    lr=2e-4,
    total_iteration=int(os.environ.get('DDPM_MV_ITERS', '20000')),
    save_every=1000,
    seq_len=SEQ_LEN,
    device=DEVICE,
    model_save_folder_path='Utils/Models',
    data_df=feat_train,
    n_features=N_FEATURES,
    n_cond=N_COND,
    norm_bounds=norm_bounds,
    stride=int(os.environ.get('DDPM_MV_STRIDE', '4')),
    val_frac=0.1,
    eval_every=250,
    patience=10,
    ch=int(os.environ.get('DDPM_MV_CH', '128')),
    num_res_blocks=int(os.environ.get('DDPM_MV_RES', '2')),
    log_features=LOG_FEATURES,
    tail_weight_alpha=float(os.environ.get('DDPM_MV_TAIL_ALPHA', '3.0')),
    tail_weight_channel=0,  # Price is channel 0
    name_tag=NAME_TAG,      # e.g. '_wide'; keeps production intact
)
best_path = trainer.train()   # → Utils/Models/model_DDPM_mv_F8_L192_T500{TAG}_best.pth

# --- cond pool: train-fold calendar windows, same stride as DDPM training ---
# val_frac=0.0 → dfu_dataset_mv returns a single TensorDataset (not a tuple).
print(f"\n=============== Building cond pool from train fold ===============")
loader = Data_Loader(df=feat_train)
train_ds = loader.dfu_dataset_mv(
    seq_len=SEQ_LEN, stride=int(os.environ.get('DDPM_MV_STRIDE', '4')),
    normlized=True, norm_bounds=norm_bounds, val_frac=0.0,
    log_features=LOG_FEATURES,
)
all_windows = train_ds.tensors[0].numpy()      # (N, F=8, L)
cond_pool = all_windows[:, N_STOCH:, :].astype(np.float32)   # (N, 6, L)
print(f"  | cond_pool shape | {cond_pool.shape}")

# --- meta sidecar + cond pool save ---
meta_path = best_path.replace('_best.pth', '_meta.json')
cond_pool_path = best_path.replace('_best.pth', '_cond_pool.npy')
np.save(cond_pool_path, cond_pool)
print(f"  | cond_pool saved | {cond_pool_path}")
with open(meta_path, 'w') as f:
    json.dump({
        'norm_bounds':   norm_bounds,
        'log_features':  sorted(LOG_FEATURES),
        'feature_names': FEATURE_NAMES,
        'cond_names':    COND_NAMES,
        'seq_len':       SEQ_LEN,
        'n_features':    N_FEATURES,
        'n_cond':        N_COND,
        'T':             T,
        'beta_1':        BETA_1,
        'beta_T':        BETA_T,
        'cond_pool_path': cond_pool_path,
    }, f, indent=2)
print(f"  | meta saved | {meta_path}")

# --- validate: sample with random cond windows, check lag-1 and spikes ---
model = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.eval()
proc = DiffusionProcess(beta_1=BETA_1, beta_T=BETA_T, T=T, diffusion_fn=model,
                        device=DEVICE, shape=(N_STOCH, SEQ_LEN), n_cond=N_COND)
_val_n = int(os.environ.get('DDPM_MV_VAL_N', '512'))
_chunks = []
_rem = _val_n
while _rem > 0:
    _n = min(256, _rem)
    pool_idx = np.random.randint(0, cond_pool.shape[0], _n)
    cond_t = torch.from_numpy(cond_pool[pool_idx]).float().to(DEVICE)
    _s, _ = proc.sampling(sampling_number=_n, save_trajectory=False, cond=cond_t)
    _chunks.append(_s.detach().cpu().numpy())
    _rem -= _n
gen = np.concatenate(_chunks, axis=0)   # (val_n, 8, L)

def lag1(arrs):
    acs = []
    for w in arrs:
        s = pd.Series(w)
        if s.std() > 1e-8:
            acs.append(s.autocorr(1))
    return float(np.nanmean(acs))

gen_real = inverse_mv_sample(gen, FEATURE_NAMES, norm_bounds, log_features=LOG_FEATURES)

print(f"\n=============== Cond MV DDPM VALIDATION (N={_val_n}, real units) ===============")
for j, (name, rv) in enumerate([('Price', price.values), ('Demand', demand.values)]):
    g = gen_real[:, j, :]
    print(f"  [{name:6s}] gen  mean={g.mean():8.1f} median={np.median(g):8.1f} "
          f"std={g.std():8.1f} min={g.min():9.1f} max={g.max():9.1f} lag1={lag1(g):.3f}")
    print(f"           real mean={rv.mean():8.1f} median={np.median(rv):8.1f} "
          f"std={rv.std():8.1f} min={rv.min():9.1f} max={rv.max():9.1f} lag1={pd.Series(rv).autocorr(1):.3f}")
gate = lag1(gen_real[:, 0, :])
gp = gen_real[:, 0, :]
print(f"  | gen price spike%>$300 | {float((gp > 300).mean()*100):.3f}  (real train ~0.23%)")
print(f"  | GATE (price lag-1 autocorr > 0.3) | {'PASS' if gate > 0.3 else 'FAIL'} ({gate:.3f})")

# Dispersion gate: reject a generator that does not reach both OOD tails, so
# the afterok 3DL job never runs on it.
_flat = gp.reshape(-1)
_gmin, _gmax = float(_flat.min()), float(_flat.max())
_p05, _p995 = float(np.percentile(_flat, 0.5)), float(np.percentile(_flat, 99.5))
# REACH gate, NOT a frequency gate: the adversary needs only the SUPPORT to
# cover the OOD regime; it raises the frequency of extremes itself.
_neg_gate = float(os.environ.get('DDPM_GATE_NEG_MIN', '-120'))   # 2023-25 p0.5=-$105, head -$107
_pos_gate = float(os.environ.get('DDPM_GATE_POS_MAX', '550'))    # 2021-22 p99=$507
_disp_ok = (_gmin < _neg_gate) and (_gmax > _pos_gate)
print(f"  | DISPERSION GATE (reach) | gen min=${_gmin:.1f} (need <${_neg_gate:.0f})  "
      f"max=${_gmax:.1f} (need >${_pos_gate:.0f})  | p0.5=${_p05:.1f} p99.5=${_p995:.1f} | "
      f"{'PASS' if _disp_ok else 'FAIL'}")
print("================================================================")
if not (gate > 0.3 and _disp_ok):
    raise SystemExit("GATE FAILED — generator rejected (under-dispersed or low lag-1). "
                     "Raise DDPM_MV_TAIL_ALPHA / adjust bounds and re-run; the "
                     "afterok 3DL job will not launch.")
