import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bld(x):
    if x.dim() == 4:
        x = x.squeeze(1).transpose(1, 2)
    elif x.dim() == 3 and x.shape[1] < x.shape[2]:
        x = x.transpose(1, 2)
    return x


def _match_features(x, expected_dim):
    if x.shape[-1] == expected_dim:
        return x
    if x.shape[-1] > expected_dim:
        return x[:, :, :expected_dim]
    pad = expected_dim - x.shape[-1]
    return F.pad(x, (0, pad))


class TaylorKANProjection(nn.Module):
    def __init__(self, d_model, order=3):
        super().__init__()
        self.order = order
        self.proj = nn.Linear(d_model * order, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x):
        terms = [x]
        if self.order >= 2:
            terms.append(x * x)
        if self.order >= 3:
            terms.append(x * x * x)
        y = self.proj(torch.cat(terms, dim=-1))
        return y * self.gate(x)


class KarmaBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.time_mixer = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)
        self.freq_filter = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.kan = TaylorKANProjection(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        time_out, _ = self.time_mixer(self.norm1(x))

        freq = torch.fft.rfft(self.norm2(x), dim=1)
        amp = torch.abs(freq)
        phase = torch.angle(freq)
        smooth_amp = self.freq_filter(amp)
        freq_out = torch.fft.irfft(smooth_amp * torch.exp(1j * phase), n=x.shape[1], dim=1)

        mixed = time_out + freq_out
        return x + self.dropout(self.kan(mixed))


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        trend = F.avg_pool1d(
            F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """
    Project-native KARMA adapter.

    The upstream KARMA model uses mamba-ssm and pytorch-wavelets. This adapter
    follows the decomposition + frequency/time mixing + KAN projection idea with
    pure PyTorch modules, matching the local train.py model interface.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", 3)
        self.d_model = configs.d_model

        self.decomp = SeriesDecomposition(getattr(configs, "moving_avg", 25))
        self.input_proj = nn.Linear(self.enc_in, self.d_model)
        self.seasonal_proj = nn.Linear(self.enc_in, self.d_model)
        self.trend_proj = nn.Linear(self.enc_in, self.d_model)
        self.blocks = nn.ModuleList(
            [
                KarmaBlock(
                    self.d_model,
                    getattr(configs, "d_ff", self.d_model * 2),
                    configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )
        self.temporal_head = nn.Linear(self.seq_len, self.pred_len)
        self.output_head = nn.Linear(self.d_model, self.c_out)

    def forecast(self, x_enc):
        x_enc = _to_bld(x_enc)
        x_enc = _match_features(x_enc, self.enc_in)
        target_stats = x_enc[:, :, : self.c_out]
        means = target_stats.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(target_stats, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        x = x_enc.clone()
        x[:, :, : self.c_out] = (x[:, :, : self.c_out] - means) / stdev
        seasonal, trend = self.decomp(x)

        x_main = self.input_proj(x)
        x_seasonal = self.seasonal_proj(seasonal)
        x_trend = self.trend_proj(trend)
        x = self.fusion(torch.cat([x_main, x_seasonal, x_trend], dim=-1))

        for block in self.blocks:
            x = block(x)

        x = self.temporal_head(x.transpose(1, 2)).transpose(1, 2)
        out = self.output_head(x)
        out = out * stdev + means
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        return self.forecast(x_enc)
