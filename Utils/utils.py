import os
import sys
import time
import torch
import shutil
import random
import psutil
import datetime
import threading
import numpy as np
import pandas as pd
from functools import wraps
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import wasserstein_distance
from torch_fidelity import calculate_metrics
from sklearn.preprocessing import MinMaxScaler





def cleanup_epoch_checkpoints(folder: str, prefix: str, keep_epoch: int) -> int:
    """Delete all `<folder>/<prefix>-epoch*.pth` files except the kept-epoch one.

    Returns the number of files removed. Does NOT touch resume files (different
    naming) or files from other prefixes.
    """
    import glob
    keep_path = os.path.join(folder, f'{prefix}-epoch{keep_epoch}.pth')
    pattern = os.path.join(folder, f'{prefix}-epoch*.pth')
    removed = 0
    for p in glob.glob(pattern):
        if os.path.abspath(p) == os.path.abspath(keep_path):
            continue
        try:
            os.remove(p)
            removed += 1
        except OSError as e:
            print(f'  | [Cleanup] failed to remove {p}: {e} |')
    if removed:
        print(f'  | [Cleanup] removed {removed} intermediate {prefix}-epoch*.pth files, kept epoch {keep_epoch} |')
    return removed


def set_global_seed(seed: int) -> None:
    """Make every Python / NumPy / PyTorch RNG deterministic for the run.

    Call ONCE at the top of the main script, before any model is constructed
    or any DataLoader iterates. nn.DataParallel's multi-GPU gradient reduction
    stays non-deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: ops without a deterministic kernel warn instead of crashing.
    import os as _os
    _os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class Utils():
    def __init__(self):
        pass

    def write_dataset(self, data, gen_data_path, title='ContextTokens'):
        dir_path = os.path.dirname(gen_data_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        df = pd.DataFrame({title: data})
        write_header = True

        if os.path.exists(gen_data_path) and os.path.getsize(gen_data_path) > 0:
            try:
                existing_df = pd.read_csv(gen_data_path, nrows=0)
                existing_columns = existing_df.columns.tolist()

                if title in existing_columns:
                    write_header = False
            except Exception as e:
                print(f"Warning: failed to read existing CSV header. Will write header anyway. Error: {e}")

        df.to_csv(
            gen_data_path,
            mode='a' if not write_header else 'w',
            header=write_header,
            index=False,
            encoding='utf-8'
        )

    def compute_normalized_wasserstein(self,df1, df2):
        x = df1.to_numpy().reshape(-1, 1)
        y = df2.to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x).flatten()
        y_scaled = scaler.fit_transform(y).flatten()
        distance = wasserstein_distance(x_scaled, y_scaled)
        return distance

    def draw_loss_plot(self, loss_list, log_folder_path,
                       val_loss_list=None, val_regret_list=None,
                       loss_label='Loss', x_labels=None, **kwargs):
        """Draw train/val loss curves (and optionally val normalised regret) to a PNG.

        loss_list: list of per-epoch train losses, 1-indexed by position.
        val_loss_list: optional list of (epoch_1based, val_loss_value) tuples.
            When non-empty, both train and val curves are overlaid on the
            top axes with a legend. When omitted or empty, only the train
            curve is plotted.
        val_regret_list: optional list of (epoch_1based, norm_regret) tuples.
            norm_regret matches the test_task() "Norm Regret" formula
            Σ(c·w*(ĉ) − c·w*(c)) / Σ|c·w*(c)|. When non-empty, a second
            subplot is added below the loss curve. Compare methods on this
            curve, NOT the surrogate (SPO+, MSE and Huber differ in scale).
        loss_label: y-axis label suffix on the top subplot.
        """
        dro_name = None
        diffusion_name = None
        ml_name = None
        for model_name, v in kwargs.items():
            if model_name == 'dro_name':
                dro_name = v
            elif model_name == 'diffusion_name':
                diffusion_name = v
            elif model_name == 'ml_name':
                ml_name = v
            else:
                continue

        partial_title = ''
        for name in [dro_name, diffusion_name, ml_name]:
            if name is None:
                continue
            partial_title += f'-{name}'
        partial_title = partial_title[1:]

        plot_path = os.path.join(log_folder_path, f'{partial_title}_training_loss.png')
        has_val    = val_loss_list   is not None and len(val_loss_list)   > 0
        has_regret = val_regret_list is not None and len(val_regret_list) > 0

        if has_regret:
            fig, (ax_loss, ax_regret) = plt.subplots(
                2, 1, figsize=(8, 8), sharex=True,
                gridspec_kw={'height_ratios': [3, 2]},
            )
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(8, 5))
            ax_regret = None

        ax_loss.plot(range(1, len(loss_list) + 1), loss_list,
                     marker='o', markersize=3, color='C0', label=f'Train {loss_label}')
        if has_val:
            val_epochs = [e for (e, _) in val_loss_list]
            val_values = [v for (_, v) in val_loss_list]
            ax_loss.plot(val_epochs, val_values,
                         marker='o', markersize=4, color='C1', label=f'Val {loss_label}')
            ax_loss.legend()
        ax_loss.set_title(f'{partial_title} Loss Over Epochs')
        ax_loss.set_ylabel(loss_label)
        ax_loss.grid(True)
        if ax_regret is None:
            ax_loss.set_xlabel('Epoch')

        if has_regret:
            regret_epochs = [e for (e, _) in val_regret_list]
            regret_values = [v for (_, v) in val_regret_list]
            ax_regret.plot(regret_epochs, regret_values,
                           marker='o', markersize=4, color='C2',
                           label='Val Norm Regret')
            ax_regret.axhline(0.0, color='gray', linestyle='--', linewidth=0.8,
                              label='Optimal (regret=0)')
            ax_regret.set_xlabel('Epoch')
            ax_regret.set_ylabel('Norm Regret\n(Σ regret / Σ |opt|)')
            ax_regret.grid(True)
            ax_regret.legend()

        # 3DL only: x_labels carry '<outer>-<inner>' tags, one per inner epoch.
        if x_labels:
            n = len(loss_list)
            positions = list(range(1, n + 1))
            outers = [str(lbl).split('-')[0] for lbl in x_labels]
            boundaries = [positions[i] for i in range(n)
                          if i == 0 or outers[i] != outers[i - 1]]
            axes = [ax_loss] if ax_regret is None else [ax_loss, ax_regret]
            for axx in axes:
                for bx in boundaries:
                    axx.axvline(bx - 0.5, color='gray', alpha=0.25,
                                linewidth=0.6, zorder=0)
            bottom_ax = ax_regret if ax_regret is not None else ax_loss
            if n <= 80:
                tick_pos, tick_lab = positions, list(x_labels)
            else:
                # Too dense: tick only outer starts.
                tick_pos = boundaries
                tick_lab = [x_labels[b - 1] for b in boundaries]
            bottom_ax.set_xticks(tick_pos)
            bottom_ax.set_xticklabels(tick_lab, rotation=90, fontsize=5)
            bottom_ax.set_xlabel('Outer–Inner epoch')

        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)

    def draw_3dl_training_plot(self, log_folder_path,
                               dro_x=None, dro_y=None, dro_labels=None, jsm_y=None,
                               ml_x=None, ml_y=None, ml_labels=None,
                               val_loss_list=None, val_regret_list=None,
                               oodval_loss_list=None, oodval_regret_list=None,
                               boundaries=None, show_jsm=False, **kwargs):
        """Unified 3DL training-timeline plot (everything on ONE shared x-axis).

        The x-axis is the GLOBAL inner-step counter in true training order: for
        each outer epoch, the DRO-inner steps come first, then the ML-inner
        steps, then the per-outer val/regret. Rows share the x-axis:
          Row 1 — Adversary (DRO inner): mean DRO loss per DRO-inner-epoch.
          Row 2 — Forecaster (ML inner): SPO+ train loss per ML-inner-epoch +
                  Val SPO+ loss (one per outer).
          Row 3 — Decision quality: Val Norm Regret (one per outer).
        When ``show_jsm=True`` (debug), an extra panel is inserted after Row 1
        showing the JSM budget term (generator score-matching loss) per
        DRO-inner-epoch.
        x ticks are '<outer>-D<k>' (DRO inner) / '<outer>-M<k>' (ML inner);
        faint vlines mark each outer-epoch boundary.
        """
        dro_x = dro_x or []; dro_y = dro_y or []; dro_labels = dro_labels or []
        ml_x  = ml_x  or []; ml_y  = ml_y  or []; ml_labels  = ml_labels  or []
        boundaries = boundaries or []

        # Title from the same name kwargs draw_loss_plot accepts.
        partial_title = '-'.join(kwargs[k] for k in ('dro_name', 'diffusion_name', 'ml_name')
                                 if kwargs.get(k))
        plot_path = os.path.join(log_folder_path, f'{partial_title}_training_timeline.png')

        # JSM is a debug-only panel; default layout is 3 rows (adversary /
        # forecaster / decision quality).
        if show_jsm:
            fig, (ax_dro, ax_jsm, ax_ml, ax_reg) = plt.subplots(
                4, 1, figsize=(11, 12), sharex=True,
                gridspec_kw={'height_ratios': [2, 1.5, 3, 2]},
            )
        else:
            fig, (ax_dro, ax_ml, ax_reg) = plt.subplots(
                3, 1, figsize=(11, 10), sharex=True,
                gridspec_kw={'height_ratios': [2, 3, 2]},
            )
            ax_jsm = None

        # Row 1 — adversary (DRO inner loss).
        if dro_x:
            ax_dro.plot(dro_x, dro_y, marker='o', markersize=3, color='C3',
                        label='DRO inner loss (adversary)')
        ax_dro.set_ylabel('DRO loss', color='C3')
        ax_dro.set_title(f'{partial_title} — training timeline (DRO inner → ML inner per outer)')
        ax_dro.legend(loc='upper right', fontsize=8)

        # Debug row: JSM budget term, which bounds how far the adversary shifts
        # the data.
        if ax_jsm is not None:
            if dro_x and jsm_y:
                ax_jsm.plot(dro_x, jsm_y, marker='.', markersize=3, color='C4',
                            label='JSM (budget term)')
            ax_jsm.set_ylabel('JSM', color='C4')
            ax_jsm.tick_params(axis='y', labelcolor='C4')
            ax_jsm.legend(loc='upper right', fontsize=8)

        # Row 2 — forecaster SPO+ train (per ML inner) + val (per outer).
        if ml_x:
            ax_ml.plot(ml_x, ml_y, marker='o', markersize=3, color='C0',
                       label='Train SPO+ (ML inner)')
        if val_loss_list:
            ax_ml.plot([e for e, _ in val_loss_list], [v for _, v in val_loss_list],
                       marker='s', markersize=4, color='C1', label='Val SPO+ (in-dist)')
        if oodval_loss_list:
            ax_ml.plot([e for e, _ in oodval_loss_list], [v for _, v in oodval_loss_list],
                       marker='s', markersize=4, color='C5', linestyle='--',
                       label='Val SPO+ (OOD worst-case)')
        ax_ml.set_ylabel('SPO+ loss')
        ax_ml.legend(loc='upper right', fontsize=8)

        # Row 3 — decision quality.
        if val_regret_list:
            ax_reg.plot([e for e, _ in val_regret_list], [v for _, v in val_regret_list],
                        marker='D', markersize=4, color='C2', label='Val Norm Regret (in-dist)')
        if oodval_regret_list:
            ax_reg.plot([e for e, _ in oodval_regret_list], [v for _, v in oodval_regret_list],
                        marker='D', markersize=4, color='C5', linestyle='--',
                        label='Val Norm Regret (OOD worst-case)')
        if val_regret_list or oodval_regret_list:
            ax_reg.axhline(0.0, color='gray', linestyle='--', linewidth=0.8,
                           label='Optimal (regret=0)')
        ax_reg.set_ylabel('Norm Regret\n(Σ regret / Σ |opt|)')
        ax_reg.legend(loc='upper right', fontsize=8)

        # Outer-epoch boundary lines + grid on every row.
        _axes = [ax_dro, ax_ml, ax_reg] + ([ax_jsm] if ax_jsm is not None else [])
        for ax in _axes:
            for b in boundaries:
                ax.axvline(b - 0.5, color='gray', alpha=0.25, linewidth=0.6, zorder=0)
            ax.grid(True, alpha=0.3)

        # Shared x ticks: '<outer>-D/M<k>' labels, ordered by global step.
        _pts = sorted(zip(dro_x + ml_x, dro_labels + ml_labels))
        if _pts:
            positions = [p for p, _ in _pts]
            labels = [l for _, l in _pts]
            if len(positions) <= 90:
                ax_reg.set_xticks(positions)
                ax_reg.set_xticklabels(labels, rotation=90, fontsize=5)
            else:
                # Too dense — tick the outer boundaries, labelled by outer index.
                ax_reg.set_xticks(boundaries)
                ax_reg.set_xticklabels([f'O{i + 1}' for i in range(len(boundaries))],
                                       rotation=0, fontsize=8)
        ax_reg.set_xlabel('training step   (<outer>-D<k> = DRO inner,  <outer>-M<k> = ML inner)')

        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch+1)]
        entries += [str(meter) for meter in self.meters]

        print('\r' + '\t'.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



class TensorDatasetWrapper(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.shape[0]

def compute_fid_from_tensors(true_images, fake_images, device='cuda:0'):
    def prepare(tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)

        tensor = tensor.clone()
        min_val = tensor.min()
        max_val = tensor.max()
        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        tensor = tensor.to(torch.uint8)
        return tensor

    true_tensor = prepare(true_images)
    fake_tensor = prepare(fake_images)

    dataset1 = TensorDatasetWrapper(true_tensor)
    dataset2 = TensorDatasetWrapper(fake_tensor)

    metrics = calculate_metrics(
        input1=dataset1,
        input2=dataset2,
        fid=True,
        isc=False,
        kid=False,
        input1_model_input_transform=torch.nn.Identity(),
        input2_model_input_transform=torch.nn.Identity(),
        batch_size=64,
        device=torch.device(device),
        input1_cache_name=None,
        input2_cache_name=None,
    )
    return metrics['frechet_inception_distance']


class Evaluation():
    def __init__(self):
        pass

    def fid_evaluation(self,fake_df, true_df,picture_size,device,isshow=True):
        fake_data = fake_df.to_numpy()
        true_data = true_df.to_numpy()

        true_num_images = true_data.shape[0] // (picture_size * picture_size)
        fake_num_images = fake_data.shape[0] // (picture_size * picture_size)

        fake_data = fake_data[:fake_num_images * picture_size * picture_size]
        true_data = true_data[:true_num_images * picture_size * picture_size]
        combined_data = np.concatenate([fake_data, true_data], axis=0)

        combined_data_2d = combined_data.reshape(-1, picture_size)

        combined_data_2d = normalize_rows(X=combined_data_2d)
        combined_data = combined_data_2d.reshape(-1)
        fake_data = combined_data[:len(fake_data)]
        true_data = combined_data[len(fake_data):]

        if isshow:
            plt.plot(true_data[:1000], label='True Data')
            plt.plot(fake_data[:1000],label='Generated Data')
            plt.legend()
            plt.show()

        fake_images = fake_data.reshape(fake_num_images, picture_size, picture_size)
        true_images = true_data.reshape(true_num_images, picture_size, picture_size)
        print(f'  | Fake Images | {fake_images.shape}')
        print(f'  | True Images | {true_images.shape}')

        compute_fid_from_tensors(fake_images=fake_images, true_images=true_images,device=device)



def normalize_rows(X):
    X = np.array(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


class Tee:
    def __init__(self, filepath, mode='a'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class Log_Redirect():
    def __init__(self):
        pass

    def build_log_file(self,result_path,cuda):
        result_path = os.path.join(result_path,cuda.replace(':', '_'))

        if os.path.exists(result_path) == False:
            os.makedirs(result_path, exist_ok=True)

        # Folder name <STAGE>_<SLURM_JOB_ID> when both env vars are set, else
        # <date>_<rand>.
        stage = os.environ.get('STAGE', '').strip()
        job_id = os.environ.get('SLURM_JOB_ID', '').strip()
        if stage and job_id:
            log_folder_name = f"{stage.upper()}_{job_id}"
        else:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            rand_num = random.randint(0, 99999)
            log_folder_name = f"{today}_{rand_num}"

        log_filename = f"{log_folder_name}.txt"
        log_folder_path = os.path.join(result_path, log_folder_name)
        log_file_path = os.path.join(log_folder_path, log_filename)

        if os.path.exists(log_folder_path):
            shutil.rmtree(log_folder_path)

        os.makedirs(log_folder_path, exist_ok=True)
        return log_folder_path,log_file_path


def timer_with_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        device = None
        if args and hasattr(args[0], "device"):
            device = args[0].device

        if torch.cuda.is_available():
            if device is not None:
                if isinstance(device, torch.device):
                    torch.cuda.set_device(device)
                elif isinstance(device, int):
                    torch.cuda.set_device(device)
                elif isinstance(device, str) and device.startswith("cuda"):
                    torch.cuda.set_device(torch.device(device))
            else:
                torch.cuda.set_device(torch.cuda.current_device())

        if args and hasattr(args[0], "__class__"):
            cls_name = args[0].__class__.__name__
            func_name = f"{cls_name}.{func.__name__}"
        else:
            func_name = func.__name__

        stop_flag = False
        cpu_samples = []
        gpu_samples = []

        process = psutil.Process(os.getpid())

        def monitor_memory():
            while not stop_flag:

                cpu_rss = process.memory_info().rss / (1024 ** 2)
                cpu_samples.append(cpu_rss)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                else:
                    gpu_mem = 0
                gpu_samples.append(gpu_mem)

                time.sleep(0.5)

        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        start_time = time.time()

        # All cuda calls guarded so CPU-only partitions (fitc/fitcq) don't fail
        # at reset_peak_memory_stats / synchronize / memory_allocated.
        _has_cuda = torch.cuda.is_available()
        if _has_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            gpu_start_alloc = torch.cuda.memory_allocated()
        else:
            gpu_start_alloc = 0

        result = func(*args, **kwargs)

        end_time = time.time()

        if _has_cuda:
            torch.cuda.synchronize()
            gpu_end_alloc = torch.cuda.memory_allocated()
            gpu_peak_alloc = torch.cuda.max_memory_allocated()
        else:
            gpu_end_alloc = 0
            gpu_peak_alloc = 0

        stop_flag = True
        monitor_thread.join()

        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        cpu_peak = max(cpu_samples) if cpu_samples else 0

        gpu_avg = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
        gpu_peak = max(gpu_samples) if gpu_samples else 0

        print(f"  | {func_name} Runtime | {end_time - start_time:.2f} s")
        print(f"  | {func_name} Max CPU Usage | {cpu_peak:.2f} MB")
        print(f"  | {func_name} Avg CPU Usage | {cpu_avg:.2f} MB")
        print(f"  | {func_name} Peak GPU Allocated | {gpu_peak_alloc / 1024**2:.2f} MB")

        return result

    return wrapper

