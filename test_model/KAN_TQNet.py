import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from test_model.TimeMixer import PastDecomposableMixing


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=8, grid_min=-2.0, grid_max=2.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.base = nn.Linear(in_features, out_features, bias=bias)

        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)

        self.log_scale = nn.Parameter(torch.zeros(in_features))
        self.spline = nn.Linear(in_features * grid_size, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base.weight)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.xavier_uniform_(self.spline.weight)

    def forward(self, x):
        base_out = self.base(x)

        x_expand = x.unsqueeze(-1)
        grid = self.grid.view(*([1] * x.dim()), self.grid_size)
        scale = torch.exp(self.log_scale).view(*([1] * (x.dim() - 1)), self.in_features, 1) + 1e-6

        basis = torch.exp(-((x_expand - grid) / scale) ** 2)
        basis = basis.reshape(*x.shape[:-1], self.in_features * self.grid_size)

        spline_out = self.spline(basis)
        return base_out + spline_out


class KANBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=None, grid_size=8, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.kan1 = KANLinear(d_model, hidden_dim, grid_size=grid_size)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.kan2 = KANLinear(hidden_dim, d_model, grid_size=grid_size)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.kan1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.kan2(x)
        x = self.act2(x)
        x = self.drop2(x)

        return x + residual


class TimeMixerRefiner(nn.Module):
    def __init__(
        self,
        seq_len,
        enc_in,
        dropout,
        d_ff,
        e_layers,
        down_sampling_layers,
        down_sampling_window,
        moving_avg,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = max(1, down_sampling_window)

        mixer_cfg = SimpleNamespace(
            seq_len=seq_len,
            pred_in_len=seq_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            channel_independence=False,
            dropout=dropout,
            decomp_method='moving_avg',
            moving_avg=moving_avg,
            top_k=5,
            d_model=enc_in,
            d_ff=d_ff,
        )
        self.blocks = nn.ModuleList(
            [PastDecomposableMixing(mixer_cfg) for _ in range(max(1, e_layers))]
        )
        self.level_logits = nn.Parameter(torch.zeros(down_sampling_layers + 1))
        self.mix_gate = nn.Parameter(torch.tensor(0.0))

    def _build_multiscale_inputs(self, x):
        x_list = [x.permute(0, 2, 1)]
        current = x
        for _ in range(self.down_sampling_layers):
            next_len = max(1, current.size(-1) // self.down_sampling_window)
            if next_len == current.size(-1) and current.size(-1) > 1:
                next_len = current.size(-1) - 1
            current = F.adaptive_avg_pool1d(current, next_len)
            x_list.append(current.permute(0, 2, 1))
        return x_list

    def forward(self, x):
        x_list = self._build_multiscale_inputs(x)
        for block in self.blocks:
            x_list = block(x_list)

        upsampled = []
        for level_out in x_list:
            level_out = level_out.permute(0, 2, 1)
            if level_out.size(-1) != self.seq_len:
                level_out = F.interpolate(level_out, size=self.seq_len, mode='linear', align_corners=False)
            upsampled.append(level_out)

        level_weights = torch.softmax(self.level_logits[:len(upsampled)], dim=0)
        fused = sum(weight * feat for weight, feat in zip(level_weights, upsampled))

        mix_ratio = torch.sigmoid(self.mix_gate)
        return (1 - mix_ratio) * x + mix_ratio * fused


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = getattr(configs, 'cycle', self.seq_len)
        self.model_type = getattr(configs, 'model_type', 'KAN_TQNet')
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = getattr(configs, 'use_revin', 1)
        self.target_channels = getattr(configs, 'target_channels', 3)
        self.use_channel_heads = getattr(configs, 'use_channel_heads', 1)
        self.use_channel_adapter = getattr(configs, 'use_channel_adapter', 1)
        self.channel_adapter_hidden_dim = getattr(configs, 'channel_adapter_hidden_dim', max(64, self.d_model // 2))

        self.use_tq = getattr(configs, 'use_tq', 1)
        self.channel_aggre = True
        self.use_hda = getattr(configs, 'use_hda', 1)
        self.hda_consistency_prefix_len = getattr(configs, 'hda_consistency_prefix_len', 48)
        self.use_patch_multiscale = getattr(configs, 'use_patch_multiscale', 0)
        self.patch_fine_len = getattr(configs, 'patch_fine_len', 8)
        self.patch_fine_stride = getattr(configs, 'patch_fine_stride', 4)
        self.patch_coarse_len = getattr(configs, 'patch_coarse_len', 16)
        self.patch_coarse_stride = getattr(configs, 'patch_coarse_stride', 8)
        self.use_uncertainty = getattr(configs, 'use_uncertainty', 0)

        self.use_kan = getattr(configs, 'use_kan', 1)
        self.kan_grid_size = getattr(configs, 'kan_grid_size', 4)
        self.kan_hidden_dim = getattr(configs, 'kan_hidden_dim', self.d_model)
        self.use_trend_residual = getattr(configs, 'use_trend_residual', 0)
        self.trend_kernel_size = getattr(configs, 'trend_kernel_size', 25)
        self.trend_hidden_dim = getattr(configs, 'trend_hidden_dim', self.d_model)
        self.use_freq_branch = getattr(configs, 'use_freq_branch', 1)
        self.freq_topk = getattr(configs, 'freq_topk', 16)
        self.freq_hidden_dim = getattr(configs, 'freq_hidden_dim', self.d_model)
        self.use_complex_freq = getattr(configs, 'use_complex_freq', 0)
        self.use_freq_weight = getattr(configs, 'use_freq_weight', 1)
        self.wavelet_levels = max(1, getattr(configs, 'wavelet_levels', 3))
        self.wavelet_pool_len = max(1, getattr(configs, 'wavelet_pool_len', 4))
        self.use_fft_aux = getattr(configs, 'use_fft_aux', 1)
        self.fft_aux_scale = getattr(configs, 'fft_aux_scale', 0.35)
        self.use_multi_scale = getattr(configs, 'use_multi_scale', 0)
        self.multi_scale_factor = max(1, getattr(configs, 'multi_scale_factor', 2))
        self.low_branch_hidden_dim = getattr(configs, 'low_branch_hidden_dim', self.kan_hidden_dim)
        self.use_local_conv = getattr(configs, 'use_local_conv', 0)
        self.use_stat_gate = getattr(configs, 'use_stat_gate', 1)
        self.use_electricity_refine = getattr(configs, 'use_electricity_refine', 0)
        self.electricity_refine_hidden_dim = getattr(configs, 'electricity_refine_hidden_dim', self.d_model)
        self.electricity_refine_scale = getattr(configs, 'electricity_refine_scale', 0.1)
        self.use_cooling_refine = getattr(configs, 'use_cooling_refine', 0)
        self.cooling_refine_hidden_dim = getattr(configs, 'cooling_refine_hidden_dim', self.d_model)
        self.cooling_refine_scale = getattr(configs, 'cooling_refine_scale', 0.2)
        self.use_timemixer = getattr(configs, 'use_timemixer', 1)
        self.timemixer_e_layers = getattr(configs, 'timemixer_e_layers', 1)
        self.timemixer_down_sampling_layers = getattr(configs, 'timemixer_down_sampling_layers', 2)
        self.timemixer_down_sampling_window = getattr(configs, 'timemixer_down_sampling_window', 2)
        self.timemixer_moving_avg = getattr(configs, 'timemixer_moving_avg', 25)
        self.timemixer_d_ff = getattr(configs, 'timemixer_d_ff', max(64, self.enc_in * 2))

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(
                torch.zeros(self.cycle_len, self.enc_in), requires_grad=True
            )

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len,
                num_heads=4,
                batch_first=True,
                dropout=0.5
            )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = self._build_backbone(self.kan_hidden_dim)

        if self.use_patch_multiscale:
            self.patch_horizon_levels = [24, 48, 96, 192, 336, 720]
            self.patch_fine_proj = nn.Linear(self.patch_fine_len, self.d_model)
            self.patch_coarse_proj = nn.Linear(self.patch_coarse_len, self.d_model)
            self.patch_cross_to_fine = nn.Linear(self.d_model, self.d_model)
            self.patch_cross_to_coarse = nn.Linear(self.d_model, self.d_model)
            self.patch_fine_reconstruct = nn.Linear(self.d_model, self.patch_fine_len)
            self.patch_coarse_reconstruct = nn.Linear(self.d_model, self.patch_coarse_len)
            self.patch_horizon_embed = nn.Embedding(len(self.patch_horizon_levels), self.d_model)
            self.patch_horizon_gate = nn.Linear(self.d_model, 2)
        else:
            self.patch_horizon_levels = None
            self.patch_fine_proj = None
            self.patch_coarse_proj = None
            self.patch_cross_to_fine = None
            self.patch_cross_to_coarse = None
            self.patch_fine_reconstruct = None
            self.patch_coarse_reconstruct = None
            self.patch_horizon_embed = None
            self.patch_horizon_gate = None

        if self.use_local_conv:
            self.local_conv = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=1,
                groups=self.enc_in
            )
        else:
            self.local_conv = None

        if self.use_freq_branch:
            wavelet_input_dim = 2 * self.wavelet_pool_len * (self.wavelet_levels + 1)
            freq_input_dim = self.freq_topk * (2 if self.use_complex_freq else 1)
            if self.use_complex_freq:
                self.freq_weight = nn.Parameter(torch.ones(1, 1, self.freq_topk * 2))
            else:
                self.freq_weight = nn.Parameter(torch.ones(1, 1, self.freq_topk))
            self.wavelet_proj = nn.Linear(wavelet_input_dim, self.d_model)
            self.wavelet_encoder = nn.Sequential(
                nn.Linear(self.d_model, self.freq_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.freq_hidden_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            self.freq_proj = nn.Linear(freq_input_dim, self.d_model)
            self.freq_encoder = nn.Sequential(
                nn.Linear(self.d_model, self.freq_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.freq_hidden_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            self.fft_aux_gate = nn.Linear(self.d_model * 2, self.d_model)
            if self.use_stat_gate:
                self.gate_stat_proj = nn.Linear(4, self.d_model)
                gate_input_dim = self.d_model * 3
            else:
                self.gate_stat_proj = None
                gate_input_dim = self.d_model * 2
            self.fusion_gate = nn.Linear(gate_input_dim, self.d_model)
        else:
            self.wavelet_proj = nn.Identity()
            self.wavelet_encoder = nn.Identity()
            self.freq_weight = None
            self.freq_proj = nn.Identity()
            self.freq_encoder = nn.Identity()
            self.fft_aux_gate = None
            self.gate_stat_proj = None
            self.fusion_gate = None

        self.output_proj = self._build_feature_heads(self.pred_len)
        self.output_adapter = self._build_channel_feature_adapters(self.d_model)
        if self.use_uncertainty:
            self.uncertainty_head = self._build_feature_heads(self.pred_len)
            self.uncertainty_adapter = self._build_channel_feature_adapters(self.d_model)
        else:
            self.uncertainty_head = None
            self.uncertainty_adapter = None

        if self.use_hda:
            self.short_local_conv = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=1,
                groups=self.enc_in
            )
            self.short_local_proj = nn.Linear(self.seq_len, self.d_model)
            self.short_feature_adapter = self._build_channel_feature_adapters(self.d_model)
            self.short_output_proj = self._build_feature_heads(self.pred_len)
            self.long_trend_head = self._build_sequence_heads(self.seq_len, self.trend_hidden_dim, self.pred_len)
            self.horizon_levels = [24, 48, 96, 192, 336, 720]
            self.horizon_embed = nn.Embedding(len(self.horizon_levels), self.d_model)
            self.dataset_prompt = nn.Sequential(
                nn.Linear(6, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model)
            )
            self.hda_router = nn.Linear(self.d_model * 2, 3)
        else:
            self.short_local_conv = None
            self.short_local_proj = None
            self.short_feature_adapter = None
            self.short_output_proj = None
            self.long_trend_head = None
            self.horizon_levels = None
            self.horizon_embed = None
            self.dataset_prompt = None
            self.hda_router = None

        if self.use_trend_residual:
            self.trend_head = self._build_sequence_heads(self.seq_len, self.trend_hidden_dim, self.pred_len)
        else:
            self.trend_head = None

        if self.use_electricity_refine:
            self.electricity_refine_head = nn.Sequential(
                nn.Linear(self.d_model, self.electricity_refine_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.electricity_refine_hidden_dim, self.pred_len)
            )
        else:
            self.electricity_refine_head = None

        if self.use_cooling_refine:
            self.cooling_refine_head = nn.Sequential(
                nn.Linear(self.d_model, self.cooling_refine_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.cooling_refine_hidden_dim, self.pred_len)
            )
        else:
            self.cooling_refine_head = None

        if self.use_timemixer:
            self.timemixer_refiner = TimeMixerRefiner(
                seq_len=self.seq_len,
                enc_in=self.enc_in,
                dropout=self.dropout,
                d_ff=self.timemixer_d_ff,
                e_layers=self.timemixer_e_layers,
                down_sampling_layers=self.timemixer_down_sampling_layers,
                down_sampling_window=self.timemixer_down_sampling_window,
                moving_avg=self.timemixer_moving_avg,
            )
        else:
            self.timemixer_refiner = None

        self.has_low_branch = self.use_multi_scale or self.use_hda
        if self.has_low_branch:
            self.low_seq_len = max(1, self.seq_len // self.multi_scale_factor)
            self.low_pred_len = max(1, self.pred_len // self.multi_scale_factor)
            self.low_input_proj = nn.Linear(self.low_seq_len, self.d_model)
            self.low_model = self._build_backbone(self.low_branch_hidden_dim)
            self.low_feature_adapter = self._build_channel_feature_adapters(self.d_model)
            self.low_output_proj = self._build_feature_heads(self.low_pred_len)
            if self.channel_aggre:
                self.low_channel_aggregator = nn.MultiheadAttention(
                    embed_dim=self.low_seq_len,
                    num_heads=4,
                    batch_first=True,
                    dropout=0.5
                )
            else:
                self.low_channel_aggregator = None
            horizon_ramp = torch.linspace(-2.0, 2.0, self.pred_len)
            self.horizon_gate_logits = nn.Parameter(horizon_ramp)
        else:
            self.low_input_proj = None
            self.low_model = None
            self.low_feature_adapter = None
            self.low_output_proj = None
            self.low_channel_aggregator = None
            self.horizon_gate_logits = None

    def _moving_average(self, x_input):
        kernel_size = min(self.trend_kernel_size, self.seq_len)
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
        if kernel_size <= 1:
            return x_input

        pad = kernel_size // 2
        padded = F.pad(x_input, (pad, pad), mode='replicate')
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)

    def _make_feature_head(self, out_len):
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, out_len)
        )

    def _build_feature_heads(self, out_len):
        if self.use_channel_heads:
            return nn.ModuleList([self._make_feature_head(out_len) for _ in range(self.target_channels)])
        return self._make_feature_head(out_len)

    def _make_channel_feature_adapter(self, feature_dim):
        return nn.Sequential(
            nn.Linear(feature_dim, self.channel_adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channel_adapter_hidden_dim, feature_dim)
        )

    def _build_channel_feature_adapters(self, feature_dim):
        if not self.use_channel_adapter:
            return None
        return nn.ModuleList([
            self._make_channel_feature_adapter(feature_dim) for _ in range(self.target_channels)
        ])

    def _make_sequence_head(self, in_len, hidden_dim, out_len):
        return nn.Sequential(
            nn.Linear(in_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, out_len)
        )

    def _build_sequence_heads(self, in_len, hidden_dim, out_len):
        if self.use_channel_heads:
            return nn.ModuleList([
                self._make_sequence_head(in_len, hidden_dim, out_len) for _ in range(self.target_channels)
            ])
        return self._make_sequence_head(in_len, hidden_dim, out_len)

    def _apply_feature_heads(self, features, heads):
        if isinstance(heads, nn.ModuleList):
            outputs = [head(features[:, channel_idx, :]) for channel_idx, head in enumerate(heads)]
            return torch.stack(outputs, dim=-1)
        return heads(features).permute(0, 2, 1)[..., :self.target_channels]

    def _apply_channel_feature_adapters(self, features, adapters):
        if adapters is None:
            return features
        adapted = features.clone()
        for channel_idx, adapter in enumerate(adapters):
            channel_feature = features[:, channel_idx, :]
            adapted[:, channel_idx, :] = channel_feature + adapter(channel_feature)
        return adapted

    def _apply_sequence_heads(self, seq_features, heads):
        if isinstance(heads, nn.ModuleList):
            outputs = [head(seq_features[:, channel_idx, :]) for channel_idx, head in enumerate(heads)]
            return torch.stack(outputs, dim=-1)
        return heads(seq_features).permute(0, 2, 1)[..., :self.target_channels]

    def _build_backbone(self, hidden_dim):
        if self.use_kan:
            return KANBlock(
                d_model=self.d_model,
                hidden_dim=hidden_dim,
                grid_size=self.kan_grid_size,
                dropout=self.dropout
            )
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

    def _haar_wavelet_decompose(self, x_input):
        current = x_input
        details = []
        scale = 1.0 / math.sqrt(2.0)

        for _ in range(self.wavelet_levels):
            if current.size(-1) < 2:
                break
            if current.size(-1) % 2 != 0:
                current = F.pad(current, (0, 1), mode='replicate')
            even = current[..., 0::2]
            odd = current[..., 1::2]
            approx = (even + odd) * scale
            detail = (even - odd) * scale
            details.append(detail)
            current = approx

        return current, details

    def _summarize_wavelet_band(self, coeff):
        signed_summary = F.adaptive_avg_pool1d(coeff, self.wavelet_pool_len)
        magnitude_summary = F.adaptive_avg_pool1d(coeff.abs(), self.wavelet_pool_len)
        return torch.cat([signed_summary, magnitude_summary], dim=-1)

    def _build_wavelet_feature(self, x_input):
        approx, details = self._haar_wavelet_decompose(x_input)
        subband_summaries = [self._summarize_wavelet_band(approx)]
        subband_summaries.extend(self._summarize_wavelet_band(detail) for detail in reversed(details))

        while len(subband_summaries) < self.wavelet_levels + 1:
            subband_summaries.append(
                x_input.new_zeros(x_input.size(0), x_input.size(1), self.wavelet_pool_len * 2)
            )

        wavelet_feature = torch.cat(subband_summaries[:self.wavelet_levels + 1], dim=-1)
        wavelet_feature = self.wavelet_proj(wavelet_feature)
        wavelet_hidden = self.wavelet_encoder(wavelet_feature)
        return wavelet_hidden + wavelet_feature

    def _build_frequency_feature(self, x_input):
        freq_spectrum = torch.fft.rfft(x_input, dim=-1)
        valid_topk = min(self.freq_topk, freq_spectrum.size(-1))
        freq_spectrum = freq_spectrum[..., :valid_topk]

        if self.use_complex_freq:
            freq_feature = torch.cat([freq_spectrum.real, freq_spectrum.imag], dim=-1)
            target_dim = self.freq_topk * 2
        else:
            freq_feature = torch.abs(freq_spectrum)
            target_dim = self.freq_topk

        if freq_feature.size(-1) < target_dim:
            freq_feature = F.pad(freq_feature, (0, target_dim - freq_feature.size(-1)))

        if self.use_freq_weight:
            freq_feature = freq_feature * self.freq_weight

        freq_feature = self.freq_proj(freq_feature)
        freq_hidden = self.freq_encoder(freq_feature)
        return freq_hidden + freq_feature

    def _build_stat_feature(self, x_input):
        stats_mean = x_input.mean(dim=-1, keepdim=True)
        stats_std = x_input.std(dim=-1, keepdim=True, unbiased=False)

        if x_input.size(-1) > 1:
            diff = x_input[..., 1:] - x_input[..., :-1]
            stats_diff_mean = diff.mean(dim=-1, keepdim=True)
            stats_diff_std = diff.std(dim=-1, keepdim=True, unbiased=False)
        else:
            stats_diff_mean = torch.zeros_like(stats_mean)
            stats_diff_std = torch.zeros_like(stats_std)

        stats = torch.cat([stats_mean, stats_std, stats_diff_mean, stats_diff_std], dim=-1)
        return self.gate_stat_proj(stats)

    def _extract_patches(self, x_input, patch_len, stride):
        if self.seq_len < patch_len:
            pad_len = patch_len - self.seq_len
            x_input = F.pad(x_input, (0, pad_len), mode='replicate')
        return x_input.unfold(dimension=-1, size=patch_len, step=stride)

    def _restore_patches(self, patch_values, patch_len, stride, target_len):
        batch_size, channels, num_patches, _ = patch_values.shape
        output = patch_values.new_zeros(batch_size, channels, target_len)
        counts = patch_values.new_zeros(batch_size, channels, target_len)

        for patch_idx in range(num_patches):
            start = patch_idx * stride
            end = min(start + patch_len, target_len)
            valid_len = end - start
            output[:, :, start:end] += patch_values[:, :, patch_idx, :valid_len]
            counts[:, :, start:end] += 1

        counts = counts.clamp_min(1.0)
        return output / counts

    def _get_patch_horizon_weights(self, batch_size, device):
        pred_len_tensor = torch.tensor(self.pred_len, device=device)
        levels_tensor = torch.tensor(self.patch_horizon_levels, device=device)
        horizon_idx = torch.argmin(torch.abs(levels_tensor - pred_len_tensor)).long()
        horizon_ids = horizon_idx.expand(batch_size)
        horizon_embed = self.patch_horizon_embed(horizon_ids)
        return torch.softmax(self.patch_horizon_gate(horizon_embed), dim=-1)

    def _apply_patch_multiscale(self, x_input):
        fine_patches = self._extract_patches(x_input, self.patch_fine_len, self.patch_fine_stride)
        coarse_patches = self._extract_patches(x_input, self.patch_coarse_len, self.patch_coarse_stride)

        fine_hidden = self.patch_fine_proj(fine_patches)
        coarse_hidden = self.patch_coarse_proj(coarse_patches)

        coarse_context = self.patch_cross_to_fine(coarse_hidden.mean(dim=2, keepdim=True))
        fine_hidden = fine_hidden + coarse_context
        fine_context = self.patch_cross_to_coarse(fine_hidden.mean(dim=2, keepdim=True))
        coarse_hidden = coarse_hidden + fine_context

        fine_reconstructed = self.patch_fine_reconstruct(fine_hidden)
        coarse_reconstructed = self.patch_coarse_reconstruct(coarse_hidden)

        fine_sequence = self._restore_patches(
            fine_reconstructed, self.patch_fine_len, self.patch_fine_stride, self.seq_len
        )
        coarse_sequence = self._restore_patches(
            coarse_reconstructed, self.patch_coarse_len, self.patch_coarse_stride, self.seq_len
        )

        patch_weights = self._get_patch_horizon_weights(x_input.size(0), x_input.device)
        fused_patch = (
            patch_weights[:, 0].view(-1, 1, 1) * fine_sequence
            + patch_weights[:, 1].view(-1, 1, 1) * coarse_sequence
        )
        return x_input + fused_patch

    def _compute_dataset_stats(self, x_input):
        mean = x_input.mean(dim=(1, 2))
        std = x_input.std(dim=(1, 2), unbiased=False)

        if x_input.size(-1) > 1:
            diff = x_input[..., 1:] - x_input[..., :-1]
            diff_std = diff.std(dim=(1, 2), unbiased=False)
        else:
            diff_std = torch.zeros_like(mean)

        centered = x_input - x_input.mean(dim=-1, keepdim=True)
        var = centered.pow(2).mean(dim=-1) + 1e-6
        max_lag = min(24, self.seq_len - 1)
        if max_lag > 0:
            acf_scores = []
            for lag in range(1, max_lag + 1):
                corr = (centered[..., :-lag] * centered[..., lag:]).mean(dim=-1) / var
                acf_scores.append(corr.mean(dim=-1))
            acf_peak = torch.stack(acf_scores, dim=-1).amax(dim=-1)
        else:
            acf_peak = torch.zeros_like(mean)

        freq_mag = torch.abs(torch.fft.rfft(x_input, dim=-1)).mean(dim=1)
        if freq_mag.size(-1) > 1:
            mag_wo_dc = freq_mag[:, 1:]
            prob = mag_wo_dc / (mag_wo_dc.sum(dim=-1, keepdim=True) + 1e-6)
            spectral_entropy = -(prob * torch.log(prob + 1e-6)).sum(dim=-1) / torch.log(
                torch.tensor(prob.size(-1), device=x_input.device, dtype=x_input.dtype) + 1e-6
            )
            dominant_idx = mag_wo_dc.argmax(dim=-1).to(x_input.dtype) + 1.0
            dominant_period = self.seq_len / dominant_idx
        else:
            spectral_entropy = torch.zeros_like(mean)
            dominant_period = torch.full_like(mean, float(self.seq_len))

        dominant_period = dominant_period / max(self.seq_len, 1)
        return torch.stack([mean, std, diff_std, acf_peak, spectral_entropy, dominant_period], dim=-1)

    def _get_horizon_embedding(self, batch_size, device):
        pred_len_tensor = torch.tensor(self.pred_len, device=device)
        horizon_tensor = torch.tensor(self.horizon_levels, device=device)
        horizon_idx = torch.argmin(torch.abs(horizon_tensor - pred_len_tensor)).long()
        horizon_ids = horizon_idx.expand(batch_size)
        return self.horizon_embed(horizon_ids)

    def _build_query(self, cycle_index, seq_len, cycle_len):
        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(seq_len, device=cycle_index.device).view(1, -1)
        ) % cycle_len
        return self.temporalQuery[gather_index].permute(0, 2, 1)

    def _run_high_branch(self, x_input, cycle_index):
        if self.use_tq:
            query_input = self._build_query(cycle_index, self.seq_len, self.cycle_len)
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=query_input, key=x_input, value=x_input
                )[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(
                    query=x_input, key=x_input, value=x_input
                )[0]
            else:
                channel_information = 0

        model_input = self.input_proj(x_input + channel_information)
        hidden = self.model(model_input)
        time_feature = hidden + model_input

        if self.use_freq_branch:
            wavelet_feature = self._build_wavelet_feature(x_input)
            spectral_feature = wavelet_feature
            if self.use_fft_aux:
                fft_feature = self._build_frequency_feature(x_input)
                fft_gate = torch.sigmoid(self.fft_aux_gate(torch.cat([wavelet_feature, fft_feature], dim=-1)))
                spectral_feature = wavelet_feature + self.fft_aux_scale * fft_gate * fft_feature
            if self.use_stat_gate:
                stats_feature = self._build_stat_feature(x_input)
                gate_input = torch.cat([time_feature, spectral_feature, stats_feature], dim=-1)
            else:
                gate_input = torch.cat([time_feature, spectral_feature], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            fused_feature = gate * time_feature + (1 - gate) * spectral_feature
        else:
            fused_feature = time_feature

        fused_feature = self._apply_channel_feature_adapters(fused_feature, self.output_adapter)
        return self._apply_feature_heads(fused_feature, self.output_proj), fused_feature

    def _run_low_branch(self, x_input, cycle_index):
        low_source = self.timemixer_refiner(x_input) if self.timemixer_refiner is not None else x_input
        x_low = F.avg_pool1d(low_source, kernel_size=self.multi_scale_factor, stride=self.multi_scale_factor)

        if self.use_tq:
            query_high = self._build_query(cycle_index, self.seq_len, self.cycle_len)
            query_low = F.avg_pool1d(query_high, kernel_size=self.multi_scale_factor, stride=self.multi_scale_factor)
            if self.channel_aggre:
                channel_information = self.low_channel_aggregator(query=query_low, key=x_low, value=x_low)[0]
            else:
                channel_information = query_low
        else:
            if self.channel_aggre:
                channel_information = self.low_channel_aggregator(query=x_low, key=x_low, value=x_low)[0]
            else:
                channel_information = 0

        low_input = self.low_input_proj(x_low + channel_information)
        low_hidden = self.low_model(low_input)
        low_feature = low_hidden + low_input
        low_feature = self._apply_channel_feature_adapters(low_feature, self.low_feature_adapter)
        low_output = self._apply_feature_heads(low_feature, self.low_output_proj).permute(0, 2, 1)
        low_output = F.interpolate(low_output, size=self.pred_len, mode='linear', align_corners=False).permute(0, 2, 1)
        return low_output

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, cycle_index=None):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
            if x.size(1) == self.enc_in:
                x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.size(1) == self.enc_in:
            x = x.permute(0, 2, 1)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input shape for KAN_TQNet: {tuple(x.shape)}")

        if cycle_index is None:
            cycle_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)
        if self.use_patch_multiscale:
            x_input = self._apply_patch_multiscale(x_input)
        if self.use_local_conv:
            x_input = x_input + self.local_conv(x_input)

        if self.use_trend_residual:
            trend_input = self._moving_average(x_input)
            residual_input = x_input - trend_input
            trend_output = self._apply_sequence_heads(trend_input, self.trend_head)
        else:
            trend_output = 0
            residual_input = x_input

        high_output, _ = self._run_high_branch(residual_input, cycle_index)
        medium_output = high_output
        output = medium_output + trend_output
        uncertainty_source = None

        if self.use_hda:
            short_input = x_input + self.short_local_conv(x_input)
            short_feature = self.short_local_proj(short_input) + self.short_local_proj(x_input)
            short_feature = self._apply_channel_feature_adapters(short_feature, self.short_feature_adapter)
            short_output = self._apply_feature_heads(short_feature, self.short_output_proj)

            long_trend = self._apply_sequence_heads(self._moving_average(x_input), self.long_trend_head)
            long_low = self._run_low_branch(x_input, cycle_index)
            long_output = long_trend + long_low

            dataset_stats = self._compute_dataset_stats(x_input)
            dataset_embed = self.dataset_prompt(dataset_stats)
            horizon_embed = self._get_horizon_embedding(x_input.size(0), x_input.device)
            router_input = torch.cat([horizon_embed, dataset_embed], dim=-1)
            routing_weights = torch.softmax(self.hda_router(router_input), dim=-1)

            output = (
                routing_weights[:, 0].view(-1, 1, 1) * short_output
                + routing_weights[:, 1].view(-1, 1, 1) * medium_output
                + routing_weights[:, 2].view(-1, 1, 1) * long_output
            )
            uncertainty_source = (
                routing_weights[:, 0].view(-1, 1, 1) * short_feature
                + routing_weights[:, 1].view(-1, 1, 1) * self.input_proj(residual_input)
                + routing_weights[:, 2].view(-1, 1, 1) * self.low_input_proj(
                    F.avg_pool1d(x_input, kernel_size=self.multi_scale_factor, stride=self.multi_scale_factor)
                ).mean(dim=1, keepdim=True).expand(-1, self.enc_in, -1)
            )

            prefix_len = min(self.hda_consistency_prefix_len, self.pred_len)
            consistency_loss = F.l1_loss(output[:, :prefix_len, :], short_output[:, :prefix_len, :])
        else:
            routing_weights = None
            consistency_loss = None
            uncertainty_source = self.input_proj(residual_input)

        if self.use_multi_scale:
            low_output = self._run_low_branch(residual_input, cycle_index)
            horizon_gate = torch.sigmoid(self.horizon_gate_logits).view(1, self.pred_len, 1)
            output = (1 - horizon_gate) * output + horizon_gate * low_output

        if self.use_revin:
            target_seq_var = seq_var[:, :, :self.target_channels]
            target_seq_mean = seq_mean[:, :, :self.target_channels]
            output = output * torch.sqrt(target_seq_var) + target_seq_mean

        if self.use_electricity_refine and self.electricity_refine_head is not None and uncertainty_source is not None:
            electricity_channel_idx = 0
            electricity_feature = uncertainty_source[:, electricity_channel_idx, :]
            electricity_delta = self.electricity_refine_head(electricity_feature) * self.electricity_refine_scale
            output[:, :, electricity_channel_idx] = output[:, :, electricity_channel_idx] + electricity_delta

        if self.use_cooling_refine and self.cooling_refine_head is not None and uncertainty_source is not None:
            cooling_channel_idx = min(1, uncertainty_source.size(1) - 1)
            cooling_feature = uncertainty_source[:, cooling_channel_idx, :]
            cooling_delta = self.cooling_refine_head(cooling_feature) * self.cooling_refine_scale
            output[:, :, cooling_channel_idx] = output[:, :, cooling_channel_idx] + cooling_delta

        if self.use_uncertainty:
            uncertainty_source = self._apply_channel_feature_adapters(uncertainty_source, self.uncertainty_adapter)
            log_var = self._apply_feature_heads(uncertainty_source, self.uncertainty_head)
            log_var = torch.clamp(log_var, min=-6.0, max=4.0)
        else:
            log_var = None

        self.latest_aux = {
            'consistency_loss': consistency_loss,
            'routing_weights': routing_weights,
            'log_var': log_var
        }
        return output[:, :, :self.target_channels]
