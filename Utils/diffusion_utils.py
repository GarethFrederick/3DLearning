import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import AverageMeter,ProgressMeter,Utils
from datasets import Data_Loader, inverse_mv_sample
from Utils.Diffusion.DDPM import Model,DiffusionProcess,ModelMV

diffusion_name = 'DDPM'
diffusion_name_mv = 'DDPM_mv'



class Train_DFU():
    def __init__(self,T,beta_1, beta_T, batch_size, lr, total_iteration, save_every, picture_size,
                 device, model_save_folder_path, data_df, **kwargs):
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.batch_size = batch_size
        self.lr = lr
        self.total_iteration = total_iteration
        self.save_every = save_every
        self.picture_size = picture_size
        self.device = device
        self.model_save_folder_path = model_save_folder_path
        self.data_df = data_df

        self.only_final = True
        self.current_iteration = 0

        self.losses = AverageMeter('Loss', ':.4f')
        self.progress = ProgressMeter(self.total_iteration, [self.losses], prefix='  Iteration ')

        self.env_dataset = Data_Loader(df=self.data_df)

        if not os.path.exists(self.model_save_folder_path):
            os.makedirs(self.model_save_folder_path)


    def train_dataset(self):
        train_dataset = self.env_dataset.dfu_dataset(batch_size=self.batch_size,total_iteration=self.total_iteration,pic_size=self.picture_size)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)
        dataiterator = iter(dataloader)
        return dataiterator,dataloader

    def train(self):
        print('\n=============== Diffusion Training Start ===============')
        dataiterator, dataloader = self.train_dataset()
        self.model = Model(device=self.device,
                           beta_1 = self.beta_1,
                           beta_T = self.beta_T,
                           T = self.T)

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            print(f"  | Using {n_gpu} GPUs with DataParallel |")
        else:
            print(f"  | Used Single GPU | {self.device}")

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        while self.current_iteration != self.total_iteration:
            self.model.train()

            try:
                data = dataiterator.__next__()
            except:
                dataiterator = iter(dataloader)
                data = dataiterator.__next__()

            data = data[0].to(device=self.device)

            if isinstance(self.model, nn.DataParallel):
                out, eps, _ = self.model(data, get_target=True)
                loss = (out - eps).square().mean()
            else:
                loss = self.model.loss_fn(x=data)

            # Skip the step on a NaN/inf loss so one bad batch cannot corrupt
            # Adam's running moments.
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Warning] DDPM iter {self.current_iteration}: nan/inf loss, skipping step")
                self.current_iteration += 1
                continue

            optim.zero_grad()
            loss.backward()
            # Global L2 grad-norm clip at 1.0, as in Ho et al. 2020.
            _real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            torch.nn.utils.clip_grad_norm_(_real_model.parameters(), max_norm=1.0)
            optim.step()

            self.losses.update(loss.item())
            self.progress.display(self.current_iteration)

            self.current_iteration += 1

            if self.current_iteration % self.save_every == 0:
                save_path = rf'{self.model_save_folder_path}/model_{diffusion_name}_cuda-{self.device.split(":")[1]}_epoch-{self.current_iteration}_batch-{self.batch_size}_picsize-{self.picture_size}_T-{self.T}_lr-{self.lr}_beta1-{self.beta_1}_betaT-{self.beta_T}.pth'

                _save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save(_save, save_path)
                print(f"  | Model Saved | {save_path}")

        print('=============== Diffusion Training End ===============')


class Train_DFU_MV():
    """Multivariate DDPM trainer on sliding (F, seq_len) windows.

    Early stopping: holds out the last val_frac of windows (chronological),
    computes mean denoising loss on them every eval_every iters, and stops when
    it hasn't improved by >= min_delta_rel for `patience` consecutive evals.
    Best-by-val checkpoint is saved as model_DDPM_mv_*.pth.
    """
    def __init__(self, T, beta_1, beta_T, batch_size, lr, total_iteration,
                 save_every, seq_len, device, model_save_folder_path, data_df,
                 n_features, norm_bounds, stride=8, val_frac=0.1,
                 eval_every=250, patience=10, min_delta_rel=1e-3,
                 ch=64, num_res_blocks=1, log_features=None,
                 tail_weight_alpha=0.0, tail_weight_channel=0, n_cond=0,
                 name_tag='', **kwargs):
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.batch_size = batch_size
        self.lr = lr
        self.total_iteration = total_iteration
        self.save_every = save_every
        self.seq_len = seq_len
        self.device = device
        self.model_save_folder_path = model_save_folder_path
        self.data_df = data_df
        self.n_features = n_features
        self.norm_bounds = norm_bounds
        self.stride = stride
        self.val_frac = val_frac
        self.eval_every = eval_every
        self.patience = patience
        self.min_delta_rel = min_delta_rel
        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.log_features = set(log_features or ())
        self.tail_weight_alpha = tail_weight_alpha
        self.tail_weight_channel = tail_weight_channel
        self.n_cond = n_cond
        # Filename tag so a re-pretrain (e.g. wider bounds) doesn't overwrite the
        # production model_DDPM_mv_F8_L192_T500_best.pth.
        self.name_tag = name_tag

        self.env_dataset = Data_Loader(df=self.data_df)
        os.makedirs(self.model_save_folder_path, exist_ok=True)

    def _val_loss(self, model, val_loader):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                l = model.loss_fn(x)
                if not (torch.isnan(l) or torch.isinf(l)):
                    total += l.item(); n += 1
        model.train()
        return total / max(n, 1) if n else float('inf')

    def train(self):
        print('\n=============== MV Diffusion Training Start ===============')
        train_ds, val_ds = self.env_dataset.dfu_dataset_mv(
            seq_len=self.seq_len, stride=self.stride, normlized=True,
            norm_bounds=self.norm_bounds, val_frac=self.val_frac,
            log_features=self.log_features,
        )
        print(f'  | train windows | {len(train_ds)}  | val windows | {len(val_ds)}'
              f'  | shape | (F={self.n_features}, L={self.seq_len})')
        # Tail-weighted sampling: spikes (0.075% of points) get almost no
        # gradient under plain MSE. Weight windows by max |x - median| in
        # SIGNED-LOG price space: in normalised [0,1] space wide bounds squash
        # the tails toward the bulk and the weights lose contrast.
        if self.tail_weight_alpha and self.tail_weight_alpha > 0:
            xw = train_ds.tensors[0][:, self.tail_weight_channel, :]   # (N, L) normalised
            _pb = self.norm_bounds.get('Price')
            if _pb is not None:
                sl = xw * (_pb[1] - _pb[0]) + _pb[0]                   # signed-log price
                med = sl.median()
                peak = (sl - med).abs().max(dim=1).values             # two-sided extremity
            else:
                peak = xw.max(dim=1).values.clamp(min=0.0)            # fallback: positive-only
            weights = (peak + 1e-3).pow(self.tail_weight_alpha).double()
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(train_ds), replacement=True)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=self.batch_size, sampler=sampler, drop_last=True)
            eff = float((weights.sum() ** 2) / (weights.pow(2).sum()))
            print(f'  | tail-weighted sampling | alpha={self.tail_weight_alpha} '
                  f'ch={self.tail_weight_channel}  | effective N | {eff:.0f}/{len(train_ds)}')
        else:
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        model = ModelMV(device=self.device, beta_1=self.beta_1, beta_T=self.beta_T,
                        T=self.T, n_features=self.n_features, seq_len=self.seq_len,
                        ch=self.ch, num_res_blocks=self.num_res_blocks,
                        n_cond=self.n_cond)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  | ModelMV params | {n_params/1e6:.2f}M  | device | '
              f'{next(model.parameters()).device}', flush=True)

        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_val = float('inf'); best_iter = 0; no_improve = 0
        best_path = (f'{self.model_save_folder_path}/'
                     f'model_{diffusion_name_mv}_F{self.n_features}_L{self.seq_len}'
                     f'_T{self.T}{self.name_tag}_best.pth')

        model.train()
        it = iter(train_loader); step = 0
        while step < self.total_iteration:
            try:
                x = next(it)[0]
            except StopIteration:
                it = iter(train_loader); x = next(it)[0]
            x = x.to(self.device)
            loss = model.loss_fn(x)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'  [warn] iter {step} nan/inf loss, skip'); step += 1; continue
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % self.eval_every == 0:
                vl = self._val_loss(model, val_loader)
                improved = vl < best_val * (1 - self.min_delta_rel)
                tag = ''
                if vl < best_val:
                    best_val = vl; best_iter = step
                    torch.save(model, best_path)
                    tag = '  <- best (saved)'
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                print(f'  iter {step:6d}  train {loss.item():.5f}  '
                      f'val {vl:.5f}  (patience {no_improve}/{self.patience}){tag}',
                      flush=True)
                if no_improve >= self.patience:
                    print(f'  | Early stopping at iter {step} '
                          f'(best val {best_val:.5f} @ iter {best_iter}) |')
                    break
            step += 1

        # Ensure a checkpoint exists even if val never improved below inf.
        if not os.path.exists(best_path):
            torch.save(model, best_path)
        print(f'  | MV DDPM best checkpoint | {best_path}  (val {best_val:.5f} @ iter {best_iter}) |')
        print('=============== MV Diffusion Training End ===============')
        return best_path


class Inference_DFU():
    def __init__(self,T, beta_1, beta_T, picture_size,batch_size,sampling_number,device,**kwargs):
        self.T = T
        self.BETA_1 = beta_1
        self.BETA_T = beta_T
        self.PICTURE_SIZE = picture_size

        self.BATCH_SIZE = batch_size
        self.SAMPLING_NUMBER = sampling_number
        self.DEVICE = device

        self.utils = Utils()
        self.device = torch.device(self.DEVICE) if torch.cuda.is_available() else torch.device('cpu')



    def generate_wholeset_data(self, save_gen_data_path=None, data_df=None, model_path=None,save_trajectory=False,batch_repeat=1,title='ContextTokens'):
        if save_gen_data_path:
            if os.path.exists(save_gen_data_path):
                os.remove(save_gen_data_path)

        self.env_dataset = Data_Loader(df=data_df)
        n = len(self.env_dataset.data_group)

        total_iteration = math.ceil(n / (self.PICTURE_SIZE * self.PICTURE_SIZE * self.BATCH_SIZE)) * batch_repeat

        print(f'\n=============== Dataset Generating Start ===============')
        print(f'  | Model File | {model_path}')
        print(f"  | Used GPU | {self.device}")
        print(f"  | Number | {n}")
        print(f"  | Totaol Iteration | {total_iteration}")
        print(f"\n  ***** Sampling *****")

        if model_path:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)

        else:
            self.model = Model(device=self.device,
                               beta_1=self.BETA_1,
                               beta_T=self.BETA_T,
                               T=self.T)
        self.model.eval()

        all_sample_trajectory_data = []
        for i in range(total_iteration):

            process = DiffusionProcess(
                                       beta_1=self.BETA_1,
                                       beta_T=self.BETA_T,
                                       T=self.T,
                                       diffusion_fn=self.model,
                                       device=self.device,
                                       shape=(1, self.PICTURE_SIZE, self.PICTURE_SIZE),
                                       )


            sample,sample_trajectory = process.sampling(sampling_number=self.SAMPLING_NUMBER, save_trajectory=save_trajectory)

            generate_data = sample.detach().cpu().numpy().reshape(-1)
            if save_gen_data_path:
                self.utils.write_dataset(data=generate_data, gen_data_path=save_gen_data_path,title=title)
                if i == total_iteration - 1:
                    print(f"  | Generate Data Saved | {save_gen_data_path}")

            if save_trajectory:
                sample_trajectory = sample_trajectory.detach().cpu().numpy()
                sample_trajectory = np.flip(sample_trajectory, axis=0)
                all_sample_trajectory_data.append(sample_trajectory)

        if save_trajectory:
            all_sample_trajectory_data = np.stack(all_sample_trajectory_data, axis=0)
            print('  | Trajectory Shape | ', all_sample_trajectory_data.shape)
            save_trajectory_filename = os.path.splitext(os.path.basename(save_gen_data_path))[0]
            parent_dir = os.path.dirname(save_gen_data_path)
            save_trajectory_path = os.path.join(parent_dir, f'{save_trajectory_filename}_trajectory.npy')
            np.save(file=save_trajectory_path, arr=all_sample_trajectory_data)
            print(f"  | Generate Trajectory Saved | {save_trajectory_path}")

        print("=============== Dataset Generating End ===============")


class Inference_DFU_MV():
    """Generate multivariate z0 data + reverse-diffusion trajectory from ModelMV.

    Two artefacts, in DIFFERENT spaces:
      - z0 CSV (Price, TOTALDEMAND_MW) in REAL units — consumed by the 3DL ML
        loaders, which re-normalise with train-fold bounds. Price is inverse-
        transformed through the meta (signed-log + calibration + clip).
      - trajectory .npy (n_iter, T, B, F, L) in the model's NATIVE normalised
        (signed-log for price) space — fed straight back through the diffusion
        model by ddd_learning's build_a0 / r_theta, so it must NOT be inverted.
    """
    def __init__(self, T, beta_1, beta_T, seq_len, n_features, batch_size,
                 device, meta, n_cond=0, cond_pool=None, **kwargs):
        self.T = T
        self.BETA_1 = beta_1
        self.BETA_T = beta_T
        self.SEQ_LEN = seq_len
        self.N_FEATURES = n_features
        self.BATCH_SIZE = batch_size
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.meta = meta
        # Conditional sampling: n_cond>0 means the last n_cond channels of every
        # sample come from cond_pool (np array, shape (N_pool, n_cond, L)) rather
        # than being denoised. Each iter draws BATCH_SIZE cond windows uniformly.
        self.n_cond = n_cond
        self.N_STOCH = n_features - n_cond
        self.cond_pool = cond_pool   # np array or None
        if n_cond > 0:
            assert cond_pool is not None, 'n_cond>0 requires cond_pool'
            assert cond_pool.shape[1] == n_cond and cond_pool.shape[2] == seq_len, (
                f'cond_pool shape {cond_pool.shape} != (*, {n_cond}, {seq_len})'
            )

    def generate_wholeset_data(self, model_path, total_iteration,
                               save_gen_data_path, save_trajectory_path=None):
        norm_bounds   = {k: tuple(v) for k, v in self.meta['norm_bounds'].items()}
        feature_names = self.meta['feature_names']
        log_features  = set(self.meta.get('log_features', []))
        calibration   = {k: tuple(v) for k, v in self.meta.get('calibration', {}).items()}
        clip          = {k: tuple(v) for k, v in self.meta.get('clip', {}).items()}

        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.eval()

        print(f'\n=============== MV Dataset Generating Start ===============')
        print(f'  | Model | {model_path}')
        print(f'  | iterations | {total_iteration}  | shape | (F={self.N_FEATURES}, L={self.SEQ_LEN})  | save_traj | {save_trajectory_path is not None}')

        all_real, all_traj = [], []
        for i in range(total_iteration):
            process = DiffusionProcess(
                beta_1=self.BETA_1, beta_T=self.BETA_T, T=self.T,
                diffusion_fn=model, device=self.device,
                shape=(self.N_STOCH, self.SEQ_LEN),
                n_cond=self.n_cond,
            )
            if self.n_cond > 0:
                pool_idx = np.random.randint(0, self.cond_pool.shape[0], self.BATCH_SIZE)
                cond_np = self.cond_pool[pool_idx]                       # (B, n_cond, L)
                cond_t = torch.from_numpy(cond_np).float().to(self.device)
            else:
                cond_t = None
            sample, traj = process.sampling(
                sampling_number=self.BATCH_SIZE,
                save_trajectory=save_trajectory_path is not None,
                cond=cond_t,
            )
            gen = sample.detach().cpu().numpy()                          # (B, F, L) model space
            gen_real = inverse_mv_sample(gen, feature_names, norm_bounds,
                                         log_features=log_features,
                                         calibration=calibration, clip=clip)   # (B, F, L) real
            # (B, F, L) -> (B, L, F) -> (B*L, F): contiguous time, price at col 0.
            all_real.append(np.transpose(gen_real, (0, 2, 1)).reshape(-1, self.N_FEATURES))
            if save_trajectory_path is not None:
                t = traj.detach().cpu().numpy()                          # (T, B, F, L) noisy->clean
                all_traj.append(np.flip(t, axis=0))                      # clean(t=0)->noisy
            print(f'  | iter {i+1}/{total_iteration} sampled |', flush=True)

        real_df = pd.DataFrame(np.concatenate(all_real, axis=0), columns=feature_names)
        real_df.to_csv(save_gen_data_path, index=False)
        print(f'  | MV z0 saved | {save_gen_data_path}  rows={len(real_df)}')
        print(f'  | z0 price  mean={real_df[feature_names[0]].mean():.1f} '
              f'median={real_df[feature_names[0]].median():.1f} '
              f'max={real_df[feature_names[0]].max():.1f}')

        if save_trajectory_path is not None:
            arr = np.stack(all_traj, axis=0)                             # (n_iter, T, B, F, L)
            np.save(save_trajectory_path, arr)
            print(f'  | MV trajectory saved | {save_trajectory_path}  shape={arr.shape}')
        print('=============== MV Dataset Generating End ===============')



