"""1D U-Net over time for the multivariate DDPM.

Tensors are shaped (B, F, L): F = feature channels, L = time axis that the
Conv1d layers slide over. The 2D net (DDPM_net.py) reshaped a day into a 28x28
grid and lost time order (lag-1 autocorr -0.025 vs 0.81 real).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _num_groups(ch):
    # GroupNorm needs num_groups | channels. 32 matches the 2D net; fall back
    # to the largest divisor <=32 for small channel counts.
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / torch.Tensor([d_model]) * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class AttnBlock1d(nn.Module):
    """Self-attention over the time axis. Input/output (B, C, L)."""
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1)
        self.proj = nn.Conv1d(in_ch, in_ch, 1)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, L = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)            # (B, C, L)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 1)        # (B, L, C)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))   # (B, L, L)
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1)        # (B, L, C)
        h = torch.bmm(w, v)           # (B, L, C)
        h = h.permute(0, 2, 1)        # (B, C, L)
        h = self.proj(h)
        return x + h


class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(_num_groups(in_ch), in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
        self.attn = AttnBlock1d(out_ch) if attn else nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None]   # broadcast over time axis
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class DownSample1d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        return self.main(x)


class UpSample1d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv1d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)


class UNet1D(nn.Module):
    """1D denoising U-Net. in_ch = total channels in (stoch + cond if conditional).
    out_ch = channels in the noise prediction; defaults to in_ch. A conditional
    DDPM predicts noise only on the stochastic subset: set out_ch = n_stoch.
    """
    def __init__(self, T, in_ch=2, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=0.1, out_ch=None):
        super().__init__()
        # The up/down loops reuse `out_ch` as a loop variable; the tail conv
        # must use the original parameter.
        tail_out_ch = in_ch if out_ch is None else out_ch
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv1d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock1d(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample1d(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock1d(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock1d(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock1d(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample1d(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(_num_groups(now_ch), now_ch),
            nn.GELU(),
            nn.Conv1d(now_ch, tail_out_ch, 3, stride=1, padding=1),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock1d):
                h_skip = hs.pop()
                if h.shape[2:] != h_skip.shape[2:]:
                    h_skip = F.interpolate(h_skip, size=h.shape[2:], mode='linear', align_corners=False)
                h = torch.cat([h, h_skip], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        # Restore exact input length (down/up rounds can drift for non-2^k L).
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode='linear', align_corners=False)
        assert len(hs) == 0, f"Unused skip connections: {len(hs)} left in stack."
        return h
