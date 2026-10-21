import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from perlin_noise import PerlinNoise


# Signed-log transform for AEMO price (heavy tails, negative values). Min-max
# alone leaves spikes ~60x the bulk and the DDPM drops the tail. The sign()
# branch handles negatives (log1p(x) is NaN for x < -1). signed_expm1 is the
# exact inverse: apply it to generated samples to get back to $/MWh.
def signed_log(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(y):
    y = np.asarray(y, dtype=np.float64)
    # Clamp before expm1: np.expm1(710) overflows float64. 700 is far above any
    # real signed-log price (~9.6 for $14.7k), so normal values do not change.
    mag = np.clip(np.abs(y), None, 700.0)
    return np.sign(y) * np.expm1(mag)


def inverse_mv_sample(samples, feature_names, norm_bounds, log_features=None,
                      calibration=None, clip=None):
    """Invert dfu_dataset_mv's transform on generated DDPM samples → real units.

    samples: array (..., F, L) in normalised (and signed-log, for log features)
             space — the space ModelMV is trained on / generates in.
    feature_names: ordered list matching axis -2 (channel F).
    norm_bounds: {name: (lo, hi)} used at training time.
    log_features: set of feature names that had signed_log applied before
                  normalisation; their inverse re-applies signed_expm1.
    calibration: optional {name: (a, b)} affine map applied in the de-normalised
                 (signed-log, for log features) space BEFORE signed_expm1, fit
                 to match the generated median to the real one. Use a=1
                 (offset only): a multiplicative widening here is a power law in
                 price space and explodes the tail.
    clip: optional {name: (lo, hi)} hard bounds applied in REAL units after the
          inverse transform — e.g. the AEMO market price floor/cap, since the
          generator can overshoot the physical price ceiling.

    Returns an array of the same shape in real units (e.g. $/MWh for price).
    """
    log_features = set(log_features or ())
    calibration = calibration or {}
    clip = clip or {}
    out = np.asarray(samples, dtype=np.float64).copy()
    for j, name in enumerate(feature_names):
        bounds = norm_bounds.get(name) if norm_bounds else None
        v = out[..., j, :]
        if bounds is not None:
            lo, hi = bounds
            if hi != lo:
                v = v * (hi - lo) + lo
        if name in calibration:
            a, b = calibration[name]
            v = a * v + b
        if name in log_features:
            v = signed_expm1(v)
        if name in clip:
            clo, chi = clip[name]
            v = np.clip(v, clo, chi)
        out[..., j, :] = v
    return out


# ML input features. Price must be column 0: the persistence-residual head
# reads last_input_price = x[:, -1, 0:1].
ML_FEATURE_ORDER = [
    'Price',           # target & autoregressive — MUST stay at index 0
    'TOTALDEMAND_MW',  # exogenous, normalised with train-fold bounds
    'sin_h', 'cos_h',  # half-hour-resolution cyclic encoding of time-of-day
    'sin_dow', 'cos_dow',  # day-of-week cycle
    'sin_m',   'cos_m',    # month-of-year cycle
]


def build_ml_features(df_raw):
    """Build a (length, 8) feature DataFrame from a raw AEMO CSV DataFrame.
    df_raw must have TIMESTAMP (datetime-parseable), Price, TOTALDEMAND_MW.
    Calendar features are bounded in [-1, 1] by construction; price and demand
    are normalised downstream by ml_set2loader_mv.
    """
    ts = pd.to_datetime(df_raw['TIMESTAMP'])
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    return pd.DataFrame({
        'Price':           df_raw['Price'].values,
        'TOTALDEMAND_MW':  df_raw['TOTALDEMAND_MW'].values,
        'sin_h':   np.sin(2 * np.pi * hour_frac / 24.0).values,
        'cos_h':   np.cos(2 * np.pi * hour_frac / 24.0).values,
        'sin_dow': np.sin(2 * np.pi * ts.dt.dayofweek / 7.0).values,
        'cos_dow': np.cos(2 * np.pi * ts.dt.dayofweek / 7.0).values,
        'sin_m':   np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0).values,
        'cos_m':   np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0).values,
    })[ML_FEATURE_ORDER]





class Data_Loader():
    def __init__(self,df,len_limit=None):
        # DataFrame -> data_group_mv (length, n_features); data_group stays 1D
        # price for dfu_dataset. Series -> data_group only.
        if isinstance(df, pd.DataFrame):
            self.data_group_mv = df.to_numpy()
            self.feature_names = list(df.columns)
            self.data_group = df.iloc[:, 0].to_numpy()   # price column for DDPM compat
        else:
            self.data_group = df.to_numpy()
            self.feature_names = [getattr(df, 'name', None) or 'feature_0']
            self.data_group_mv = self.data_group.reshape(-1, 1)
        if len_limit:
            self.data_group    = self.data_group[:min(len_limit, len(self.data_group))]
            self.data_group_mv = self.data_group_mv[:min(len_limit, len(self.data_group_mv))]
        self.n_features = self.data_group_mv.shape[1]
        self.data_len = 0

    def data_transfer(self, normlized=True, pic_size=28, noise_type=None, noise_param=0.05, norm_bounds=None):

        flattened_data = np.concatenate([arr.flatten() for arr in self.data_group])

        has_nan = np.isnan(flattened_data).any()
        if has_nan:
            mean_value = np.nanmean(flattened_data)
            flattened_data = np.nan_to_num(flattened_data, nan=mean_value)

        if normlized:
            if norm_bounds is not None:
                min_value, max_value = norm_bounds
            else:
                max_value = np.max(flattened_data)
                min_value = np.min(flattened_data)
            if max_value == min_value:
                normalized_data = np.zeros_like(flattened_data)
            else:
                normalized_data = (flattened_data - min_value) / (max_value - min_value)
        else:
            normalized_data = flattened_data


        if noise_type:

            if noise_type == 'gaussian':
                print(f'  | Noise | {noise_type} noise - {noise_param}')
                noise = np.random.normal(loc=0.0, scale=noise_param, size=normalized_data.shape)
                normalized_data = normalized_data + noise
                normalized_data = np.clip(normalized_data, 0.0, 1.0)

            elif noise_type == 'perlin':
                print(f'  | Noise | {noise_type} noise - {noise_param}')

                noise = PerlinNoise(octaves=8)
                perlin = np.array([noise(i / len(normalized_data)) for i in range(len(normalized_data))])
                perlin = perlin - perlin.mean()
                perlin = perlin / (np.abs(perlin).max() + 1e-8)
                normalized_data = normalized_data + noise_param * perlin
                normalized_data = np.clip(normalized_data, 0.0, 1.0)

            elif noise_type == 'cutout':
                print(f'  | Noise | {noise_type} noise - {noise_param}')
                length = int(noise_param * len(normalized_data))
                start = np.random.randint(0, len(normalized_data) - length)
                normalized_data[start:start + length] = 1

        remainder = len(normalized_data) % pic_size
        if remainder != 0:
            normalized_data = normalized_data[:-remainder]

        dataset = normalized_data.reshape(-1, pic_size)
        self.data_len = len(dataset)
        return dataset

    def ml_dataset(self,normlized=True,pic_size=28,noise_type=None, noise_param=0.05, norm_bounds=None, val_size=None):
        """
        Split rows of the prepared dataset into 2 (train/test) or 3 (train/val/test) folds.

        Noise (if requested) is applied to the train fold only — val and test
        stay clean so early-stopping signals and held-out metrics aren't biased
        by augmentation.

        val_size: if None, returns (train, test) with test_size=0.3 (legacy behaviour).
                  if a fraction, returns (train, val, test) honouring train/val/test
                  proportions of (1 - 0.3) / val_size / (0.3 - val_size). For example,
                  val_size=0.15 yields a 70/15/15 split.
        """
        # Build the clean base dataset; noise (if any) is added post-split below.
        dataset = self.data_transfer(normlized=normlized,pic_size=pic_size, noise_type=None, norm_bounds=norm_bounds)
        train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42, shuffle=False)
        if val_size is not None:
            # Carve val out of the test fold so the 70% train fold does not
            # change. val_size is a fraction of the full dataset.
            rel = val_size / 0.3
            if not (0 < rel < 1):
                raise ValueError(f"val_size must be in (0, 0.3); got {val_size}")
            val_data, test_data = train_test_split(test_data, test_size=(1 - rel), random_state=42, shuffle=False)

        if noise_type is not None:
            train_data = self._apply_noise(train_data, noise_type, noise_param)

        if val_size is None:
            return train_data, test_data
        return train_data, val_data, test_data

    def _apply_noise(self, data, noise_type, noise_param):
        """Inject noise into a 2D (rows × pic_size) array; returns a new array."""
        if noise_type == 'gaussian':
            print(f'  | Noise (train fold only) | gaussian - {noise_param}')
            noise = np.random.normal(loc=0.0, scale=noise_param, size=data.shape)
            return np.clip(data + noise, 0.0, 1.0)
        if noise_type == 'perlin':
            print(f'  | Noise (train fold only) | perlin - {noise_param}')
            flat = data.reshape(-1)
            noise_obj = PerlinNoise(octaves=8)
            perlin = np.array([noise_obj(i / len(flat)) for i in range(len(flat))])
            perlin = perlin - perlin.mean()
            perlin = perlin / (np.abs(perlin).max() + 1e-8)
            return np.clip((flat + noise_param * perlin).reshape(data.shape), 0.0, 1.0)
        if noise_type == 'cutout':
            print(f'  | Noise (train fold only) | cutout - {noise_param}')
            flat = data.reshape(-1).copy()
            length = max(1, int(noise_param * len(flat)))
            start = np.random.randint(0, max(1, len(flat) - length))
            flat[start:start + length] = 1
            return flat.reshape(data.shape)
        return data

    def dfu_dataset(self,batch_size, total_iteration,normlized=True,pic_size=28):
        data_transfer = self.data_transfer(normlized=normlized,pic_size=pic_size)
        num_days, num_features = data_transfer.shape
        split_size = pic_size

        num_full_groups = num_days // split_size
        remaining_days = num_days % split_size

        grouped_data = np.array(
            np.split(data_transfer[:num_full_groups * split_size], num_full_groups))
        if remaining_days > 0:
            missing_days = split_size - remaining_days
            last_group = np.vstack(
                [data_transfer[num_full_groups * split_size:], data_transfer[:missing_days]])
            grouped_data = np.vstack([grouped_data, last_group[np.newaxis, :, :]])

        grouped_data = np.expand_dims(grouped_data, axis=1)

        total_images = batch_size * total_iteration

        num_original_images = grouped_data.shape[0]

        repeat_times = (total_images // num_original_images) + 1

        expanded_data = np.tile(grouped_data, (repeat_times, 1, 1, 1))
        expanded_data = expanded_data[:total_images]

        expanded_data_tensor = torch.tensor(expanded_data, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(expanded_data_tensor)

        return dataset

    def ml_set2loader(self,split_test,device,batch_size,normlized=True,pic_size=28,noise_type=None, noise_param=0.05, norm_bounds=None, val_size=None):
        """
        If val_size is None: legacy behaviour.
            split_test=True  → (train_loader, test_loader)
            split_test=False → data_loader
        If val_size is a fraction: 3-way split.
            split_test=True  → (train_loader, val_loader, test_loader)
            (val_size is ignored when split_test=False)
        """
        if split_test:
            if val_size is None:
                train_data, test_data = self.ml_dataset(pic_size=pic_size,normlized=normlized,noise_type=noise_type, noise_param=noise_param, norm_bounds=norm_bounds)
                print('\n  | Train Dataset | ', train_data.shape)
                print('  | Test  Dataset | ', test_data.shape)
                train_data = torch.FloatTensor(train_data).to(device)
                test_data = torch.FloatTensor(test_data).to(device)
                train_dataset = TensorDataset(train_data)
                test_dataset = TensorDataset(test_data)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
                return train_loader, test_loader
            else:
                train_data, val_data, test_data = self.ml_dataset(pic_size=pic_size,normlized=normlized,noise_type=noise_type, noise_param=noise_param, norm_bounds=norm_bounds, val_size=val_size)
                print('\n  | Train Dataset | ', train_data.shape)
                print('  | Val   Dataset | ', val_data.shape)
                print('  | Test  Dataset | ', test_data.shape)
                train_data = torch.FloatTensor(train_data).to(device)
                val_data   = torch.FloatTensor(val_data).to(device)
                test_data  = torch.FloatTensor(test_data).to(device)
                train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=False, drop_last=True)
                val_loader   = DataLoader(TensorDataset(val_data),   batch_size=batch_size, shuffle=False, drop_last=True)
                test_loader  = DataLoader(TensorDataset(test_data),  batch_size=batch_size, shuffle=False, drop_last=True)
                return train_loader, val_loader, test_loader
        else:
            data = self.data_transfer(pic_size=pic_size,normlized=normlized,noise_type=noise_type, noise_param=noise_param, norm_bounds=norm_bounds)
            print('\n  | Whole Dataset | ', data.shape)
            data = torch.FloatTensor(data).to(device)
            dataset = TensorDataset(data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            return data_loader

    def _data_transfer_mv(self, normlized=True, pic_size=28, norm_bounds=None):
        """Multivariate equivalent of data_transfer. Output shape:
           (N_rows, pic_size, n_features). NaNs imputed with per-column mean.
           norm_bounds: dict {feature_name: (lo, hi)} or None. Features whose
                        name is absent are left untouched (e.g. sin/cos already
                        in [-1, 1]); features whose entry is None likewise pass
                        through.
        """
        data = self.data_group_mv.copy().astype(np.float64)
        # Per-column NaN handling
        for j in range(self.n_features):
            col = data[:, j]
            if np.isnan(col).any():
                data[:, j] = np.nan_to_num(col, nan=np.nanmean(col))

        if normlized and norm_bounds:
            for j, name in enumerate(self.feature_names):
                bounds = norm_bounds.get(name)
                if bounds is None:
                    continue
                lo, hi = bounds
                if hi != lo:
                    data[:, j] = (data[:, j] - lo) / (hi - lo)

        # Trim to multiple of pic_size and reshape to (N_rows, pic_size, F).
        remainder = len(data) % pic_size
        if remainder != 0:
            data = data[:-remainder]
        dataset = data.reshape(-1, pic_size, self.n_features)
        self.data_len = len(dataset)
        return dataset

    def ml_set2loader_mv(self, split_test, device, batch_size,
                          normlized=True, pic_size=28, norm_bounds=None,
                          val_size=None, return_dataset=False, drop_last=True):
        """Multivariate counterpart of ml_set2loader. Batches yield 3D tensors
        of shape (B, pic_size, n_features). The first feature (column 0) is
        the prediction target — downstream code in ml.py / wdro.py extracts
        it via target[..., 0]. Chronological 70/15/15 split (shuffle=False).

        return_dataset (fold-in, whole-dataset path): return the TensorDataset
        so the caller can build a WeightedRandomSampler over (orig + buffer).

        drop_last (whole-dataset path only): True yields floor(N/batch_size)
        full batches and discards the chronological tail. Default True keeps
        every reported AEMO number. Only poc_eval_ood.py passes False, under
        EVAL_FULL_COVERAGE.
        """
        dataset = self._data_transfer_mv(normlized=normlized, pic_size=pic_size,
                                          norm_bounds=norm_bounds)
        if not split_test:
            print('\n  | Whole Dataset (mv) | ', dataset.shape)
            data = torch.FloatTensor(dataset).to(device)
            _ds = TensorDataset(data)
            if return_dataset:
                return _ds
            return DataLoader(_ds, batch_size=batch_size,
                              shuffle=False, drop_last=drop_last)
        train_data, test_data = train_test_split(dataset, test_size=0.3,
                                                  random_state=42, shuffle=False)
        if val_size is not None:
            rel = val_size / 0.3
            if not (0 < rel < 1):
                raise ValueError(f"val_size must be in (0, 0.3); got {val_size}")
            val_data, test_data = train_test_split(test_data, test_size=(1 - rel),
                                                    random_state=42, shuffle=False)
            print('\n  | Train Dataset | ', train_data.shape)
            print('  | Val   Dataset | ', val_data.shape)
            print('  | Test  Dataset | ', test_data.shape)
            train_data = torch.FloatTensor(train_data).to(device)
            val_data   = torch.FloatTensor(val_data).to(device)
            test_data  = torch.FloatTensor(test_data).to(device)
            return (
                DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=False, drop_last=True),
                DataLoader(TensorDataset(val_data),   batch_size=batch_size, shuffle=False, drop_last=True),
                DataLoader(TensorDataset(test_data),  batch_size=batch_size, shuffle=False, drop_last=True),
            )
        print('\n  | Train Dataset | ', train_data.shape)
        print('  | Test  Dataset | ', test_data.shape)
        train_data = torch.FloatTensor(train_data).to(device)
        test_data  = torch.FloatTensor(test_data).to(device)
        return (
            DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=False, drop_last=True),
            DataLoader(TensorDataset(test_data),  batch_size=batch_size, shuffle=False, drop_last=True),
        )

    def dfu_dataset_mv(self, seq_len, stride=1, normlized=True, norm_bounds=None,
                       val_frac=0.0, log_features=None):
        """Multivariate DDPM training set: SLIDING windows over time.

        Output shape (N, F, seq_len): each sample is a contiguous window,
        `stride` steps apart, so time order is preserved.

        norm_bounds: dict {feature_name: (lo, hi)}; features absent or mapped to
                     None pass through unnormalised (e.g. calendar sin/cos).
        val_frac: if > 0, the LAST val_frac of windows is held out (chronological,
                  no shuffle) and returned separately for DDPM early stopping.
        log_features: set/iterable of feature names to signed_log BEFORE
                      normalisation (heavy-tailed price). norm_bounds for these
                      features must therefore be quantiles of signed_log(values),
                      and inverse_mv_sample re-applies signed_expm1 on generation.

        Returns:
          val_frac == 0 → TensorDataset of all windows
          val_frac > 0  → (train_dataset, val_dataset)
        """
        log_features = set(log_features or ())
        data = self.data_group_mv.copy().astype(np.float64)   # (length, F)
        for j in range(self.n_features):
            col = data[:, j]
            if np.isnan(col).any():
                data[:, j] = np.nan_to_num(col, nan=np.nanmean(col))

        for j, name in enumerate(self.feature_names):
            if name in log_features:
                data[:, j] = signed_log(data[:, j])

        if normlized and norm_bounds:
            for j, name in enumerate(self.feature_names):
                bounds = norm_bounds.get(name)
                if bounds is None:
                    continue
                lo, hi = bounds
                if hi != lo:
                    data[:, j] = (data[:, j] - lo) / (hi - lo)

        length = len(data)
        if length < seq_len:
            raise ValueError(f"series length {length} < seq_len {seq_len}")
        starts = list(range(0, length - seq_len + 1, stride))
        # (N, seq_len, F) → transpose to (N, F, seq_len) for Conv1d channels=F
        windows = np.stack([data[s:s + seq_len] for s in starts], axis=0)
        windows = np.transpose(windows, (0, 2, 1)).astype(np.float32)

        if val_frac and val_frac > 0:
            n_val = max(1, int(val_frac * len(windows)))
            train_w = windows[:-n_val]
            val_w = windows[-n_val:]
            return (
                torch.utils.data.TensorDataset(torch.tensor(train_w)),
                torch.utils.data.TensorDataset(torch.tensor(val_w)),
            )
        return torch.utils.data.TensorDataset(torch.tensor(windows))

    def dfu_set2inter(self,dfu_dataset,batch_size):
        dataloader = DataLoader(dfu_dataset, batch_size=batch_size, drop_last=True)
        dataiterator = iter(dataloader)
        return dataiterator

class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_fn,corrupt_label, normalization,severity=5):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.corrupt_label = corrupt_label
        self.normalization = normalization

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        if isinstance(image, torch.Tensor):
            image = T.ToPILImage()(image)

        corrupted_np = self.corruption_fn(image)

        corrupted_img = T.ToTensor()(Image.fromarray(corrupted_np.astype(np.uint8)))
        if self.normalization:

            corrupted_img = (corrupted_img + 0.1307) * 0.3081

        if self.corrupt_label:
            noisy_label = label + np.random.normal(loc=0.0, scale=1.0)
            label = int(np.round(noisy_label))
            label = np.clip(label, 0, 9)

        return corrupted_img, label









