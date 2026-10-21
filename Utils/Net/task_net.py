import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantMeanPredictor(nn.Module):
    """Sanity-check baseline: emit a fixed normalised-price vector for every input.

    Acts as the "no signal" predictor that the trained models must beat. If
    PTO/DFL/WDRO/3DLearning all produce regret comparable to this predictor, the
    LSTM is stuck at the underfit floor.
    """

    def __init__(self, predict_len: int, value: float):
        super().__init__()
        self.predict_len = predict_len
        self.register_buffer('val', torch.full((predict_len,), float(value)))

    def forward(self, x):
        return self.val.unsqueeze(0).expand(x.shape[0], -1)


class DeepLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[128, 64], output_size=1, dropout=0.0,
                 avg_delta_init=None, zero_init_fc2=True,
                 head_mode='persistence_residual', init_bias_for_norm_mean=None,
                 pred_lo=None, pred_hi=None):
        super(DeepLSTM, self).__init__()

        # Output range in NORMALISED space; default (0,1). A wider range lets
        # denorm reach negative real prices. Set it via env so every model
        # shares one head range; PRED_NORM_LO=-0.5 PRED_NORM_HI=1.5 gives real
        # [-$107, $359] under the P1/P99 denorm bounds.
        if pred_lo is None:
            pred_lo = float(os.environ.get('PRED_NORM_LO', '0.0'))
        if pred_hi is None:
            pred_hi = float(os.environ.get('PRED_NORM_HI', '1.0'))
        self.pred_lo, self.pred_hi = float(pred_lo), float(pred_hi)

        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout)

        # fc1 takes concat of [last_hidden, mean_pooled_hidden] → 2 * hidden_sizes[1].
        self.fc1 = nn.Linear(2 * hidden_sizes[1], 32)
        self.fc2 = nn.Linear(32, output_size)

        # head_mode maps fc2's output to ŷ_norm:
        # - 'persistence_residual': ŷ = (last_price + 1.0·tanh(fc2)).clamp(lo,hi).
        #   The LSTM predicts only the deviation from the last price. A scale
        #   of 0.3 was too tight under P1/P99 bounds (±$70 real).
        # - 'sigmoid': ŷ = sigmoid(fc2). Under SPO+ the persistence anchor plus
        #   the one-sided clamp masks gradients and drifts fc2 upward.
        self.head_mode = head_mode

        if head_mode == 'persistence_residual':
            # zero_init_fc2=True: output at init is pure persistence (MSE moves
            # it). Under SPO+ keep the default init: zero-init makes the SPO+
            # subgradient zero and training freezes.
            if zero_init_fc2:
                nn.init.zeros_(self.fc2.weight)
                if avg_delta_init is not None:
                    assert avg_delta_init.shape == (output_size,), \
                        f'avg_delta_init shape {tuple(avg_delta_init.shape)} != ({output_size},)'
                    # Bias init solves 1.0·tanh(b) = avg_delta_norm.
                    arg = avg_delta_init.float().clamp(-0.99, 0.99)
                    self.fc2.bias.data.copy_(torch.atanh(arg))
                else:
                    nn.init.zeros_(self.fc2.bias)
        elif head_mode == 'sigmoid':
            # Default init on fc2.weight gives the output spread SPO+ needs.
            # The bias starts the output near the target mean, not at 0.5.
            if init_bias_for_norm_mean is not None:
                # sigmoid(b) is the fraction of the target within
                # [pred_lo, pred_hi].
                target = float(init_bias_for_norm_mean)
                p = (target - self.pred_lo) / (self.pred_hi - self.pred_lo)
                p = max(min(p, 1.0 - 1e-4), 1e-4)
                nn.init.constant_(self.fc2.bias, math.log(p / (1.0 - p)))
        else:
            raise ValueError(f'unknown head_mode: {head_mode!r} '
                             f"(expected 'persistence_residual' or 'sigmoid')")

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.norm1(out)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.norm2(out)
        out = self.dropout2(out)

        # fc1 reads the last-step and the mean-pooled hidden states.
        last_h = out[:, -1, :]                          # (B, 64)
        mean_h = out.mean(dim=1)                        # (B, 64)
        readout = torch.cat([last_h, mean_h], dim=-1)   # (B, 128)
        out = F.gelu(self.fc1(readout))

        # getattr defaults: old pickled models lack pred_lo/pred_hi and were
        # trained on (0,1). Warm-start callers may override after load.
        pred_lo = getattr(self, 'pred_lo', 0.0)
        pred_hi = getattr(self, 'pred_hi', 1.0)
        if self.head_mode == 'persistence_residual':
            # delta ∈ [-1, +1] in normalised space.
            last_price_norm = x[:, -1, 0:1]             # (B, 1)
            delta = 1.0 * torch.tanh(self.fc2(out))     # (B, output_size)
            return (last_price_norm + delta).clamp(pred_lo, pred_hi)
        else:  # 'sigmoid'
            # Sigmoid rescaled to [pred_lo, pred_hi].
            s = torch.sigmoid(self.fc2(out))
            if pred_lo == 0.0 and pred_hi == 1.0:
                return s
            return pred_lo + (pred_hi - pred_lo) * s
