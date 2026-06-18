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


class TemporalPrototypeBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, prototypes):
        context, _ = self.attn(x, prototypes, prototypes, need_weights=False)
        x = self.norm1(x + context)
        x = self.norm2(x + self.ffn(x))
        return x


class Model(nn.Module):
    """
    Project-native TimePro adapter.

    The original TimePro repository depends on custom selective-scan/DCNv4 CUDA
    extensions. This implementation keeps the same project interface while using
    a pure PyTorch temporal-prototype encoder so it can run directly in this
    Windows workspace.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", 3)
        self.d_model = configs.d_model

        self.value_embedding = nn.Linear(self.enc_in, self.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)
        self.prototypes = nn.Parameter(
            torch.randn(1, getattr(configs, "timepro_num_prototypes", 16), self.d_model) * 0.02
        )

        kernels = getattr(configs, "timepro_kernel_sizes", (3, 7, 15))
        self.multi_scale = nn.ModuleList(
            [
                nn.Conv1d(
                    self.d_model,
                    self.d_model,
                    kernel_size=k,
                    padding=k // 2,
                    groups=self.d_model,
                )
                for k in kernels
            ]
        )
        self.scale_fusion = nn.Linear(self.d_model * len(kernels), self.d_model)
        self.blocks = nn.ModuleList(
            [
                TemporalPrototypeBlock(
                    self.d_model,
                    getattr(configs, "n_heads", 4),
                    getattr(configs, "d_ff", self.d_model * 2),
                    configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.dropout = nn.Dropout(configs.dropout)
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
        x = self.value_embedding(x) + self.position_embedding[:, : x.shape[1], :]

        conv_in = x.transpose(1, 2)
        scales = [conv(conv_in).transpose(1, 2) for conv in self.multi_scale]
        x = self.scale_fusion(torch.cat(scales, dim=-1)) + x

        prototypes = self.prototypes.expand(x.shape[0], -1, -1)
        for block in self.blocks:
            x = block(self.dropout(x), prototypes)

        x = self.temporal_head(x.transpose(1, 2)).transpose(1, 2)
        out = self.output_head(x)
        out = out * stdev + means
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        return self.forecast(x_enc)
