import torch
import torch.nn as nn
from Utils.Net.DDPM_net import UNet
from Utils.Net.DDPM_net_mv import UNet1D

class Model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T):
        super().__init__()
        self.device = device
        # Registered buffer so DataParallel moves it to each replica device automatically
        self.register_buffer(
            'alpha_bars',
            torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0),
        )
        self.backbone = UNet(T)

        self.to(device=self.device)

    def loss_fn(self, x, idx=None, T_prime=None):
        output, epsilon, alpha_bar = self.forward(x, idx=idx, get_target=True, T_prime=T_prime)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, idx=None, get_target=False, T_prime=None):
        if idx is None:
            if T_prime is None:
                T_prime = len(self.alpha_bars)
            idx = torch.randint(0, T_prime, (x.size(0),), device=x.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None, None]

            epsilon = torch.randn_like(x)

            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon

        else:
            # idx is a plain int when called from DiffusionProcess; use x.device for portability
            idx = torch.tensor([idx] * x.size(0), dtype=torch.long, device=x.device)
            x_tilde = x

        output = self.backbone(x_tilde, idx)
        return (output, epsilon, used_alpha_bars) if get_target else output



class ModelMV(nn.Module):
    """Multivariate DDPM on (B, F, L) tensors with the UNet1D backbone.

    If n_cond > 0, the last n_cond channels are clean conditioning (e.g.
    calendar features) and are NOT noised. The model reads all n_features
    channels and predicts noise only on the first n_features - n_cond.
    """
    def __init__(self, device, beta_1, beta_T, T, n_features, seq_len=None,
                 ch=64, ch_mult=(1, 2, 4), num_res_blocks=1, attn=(), n_cond=0):
        super().__init__()
        self.device = device
        self.n_features = n_features
        self.n_cond = n_cond
        self.n_stoch = n_features - n_cond
        assert self.n_stoch > 0, f'n_cond ({n_cond}) must be < n_features ({n_features})'
        self.seq_len = seq_len
        self.register_buffer(
            'alpha_bars',
            torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0),
        )
        # Defaults give ~1M params, fast on CPU; the 2D net's ch=128 had 15.85M.
        self.backbone = UNet1D(T, in_ch=n_features, ch=ch, ch_mult=list(ch_mult),
                               attn=list(attn), num_res_blocks=num_res_blocks,
                               out_ch=self.n_stoch)
        self.to(device=self.device)

    def loss_fn(self, x, idx=None, T_prime=None):
        output, epsilon, _ = self.forward(x, idx=idx, get_target=True, T_prime=T_prime)
        return (output - epsilon).square().mean()

    def forward(self, x, idx=None, get_target=False, T_prime=None):
        # x: (B, F, L), F = n_stoch + n_cond. Stoch channels lead so the noise
        # prediction (out_ch=n_stoch) aligns with x[:, :n_stoch].
        if idx is None:
            if T_prime is None:
                T_prime = len(self.alpha_bars)
            idx = torch.randint(0, T_prime, (x.size(0),), device=x.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None]   # (B, 1, 1)
            if self.n_cond == 0:
                epsilon = torch.randn_like(x)
                x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            else:
                x_stoch = x[:, :self.n_stoch]
                x_cond  = x[:, self.n_stoch:]
                epsilon = torch.randn_like(x_stoch)
                x_stoch_tilde = torch.sqrt(used_alpha_bars) * x_stoch + torch.sqrt(1 - used_alpha_bars) * epsilon
                x_tilde = torch.cat([x_stoch_tilde, x_cond], dim=1)
        else:
            idx = torch.tensor([idx] * x.size(0), dtype=torch.long, device=x.device)
            x_tilde = x
            used_alpha_bars = None
            epsilon = None

        output = self.backbone(x_tilde, idx)   # (B, n_stoch, L)
        return (output, epsilon, used_alpha_bars) if get_target else output


class DiffusionProcess():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape, n_cond=0):

        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape

        self.diffusion_fn = diffusion_fn
        self.device = device
        # Conditional mode: last n_cond channels of the (B, F, L) tensor are
        # clean conditioning that pass through every reverse step unchanged.
        self.n_cond = n_cond


    def _one_diffusion_step(self, x, guidance_fn=None):
        # x: (B, F, L). The trailing n_cond channels are re-attached unchanged
        # each step so the backbone always reads the conditioning.
        #
        # guidance_fn(idx, x_stoch, x0_hat) returns an additive delta on the
        # stochastic channels after each reverse step, or None. The loop runs
        # under no_grad. guidance_fn=None gives plain ancestral sampling.
        n_stoch = x.shape[1] - self.n_cond
        _sqrt_ab     = torch.sqrt(self.alpha_bars)
        _sqrt_1m_ab  = torch.sqrt(1 - self.alpha_bars)

        for idx in reversed(range(len(self.alpha_bars))):
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_fn(x, idx)   # (B, n_stoch, L)
            x_stoch = x[:, :n_stoch]
            x_cond  = x[:, n_stoch:]
            noise = torch.zeros_like(x_stoch) if idx == 0 else torch.randn_like(x_stoch)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                x_stoch - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon
            )
            x_stoch = mu_theta_xt + sqrt_tilde_beta * noise
            if guidance_fn is not None:
                # Tweedie estimate of the clean signal from the pre-step x.
                x0_hat = (x[:, :n_stoch] - _sqrt_1m_ab[idx] * predict_epsilon) / _sqrt_ab[idx]
                delta = guidance_fn(idx, x_stoch, x0_hat)
                if delta is not None:
                    x_stoch = x_stoch + delta
            x = torch.cat([x_stoch, x_cond], dim=1) if self.n_cond > 0 else x_stoch
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, save_trajectory=True, cond=None, guidance_fn=None):
        # self.shape is (n_stoch, L); cond is appended on the channel axis.
        sample = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        if cond is not None:
            if cond.shape[0] != sampling_number:
                # broadcast a single conditioning window across all samples
                cond = cond.expand(sampling_number, *cond.shape[1:])
            sample = torch.cat([sample, cond.to(self.device)], dim=1)
        sampling_list = []

        final = None
        for sample in self._one_diffusion_step(sample, guidance_fn=guidance_fn):
            final = sample
            if save_trajectory:
                sampling_list.append(final)

        if save_trajectory:
            trajectory = torch.stack(sampling_list)
            return final, trajectory
        else:
            return final, None