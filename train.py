import argparse
import math
import time
import random
import torch
import torch.nn as nn
# from net import gtnet
import numpy as np
import importlib
from util import *
from trainer import Optim
from metrics import *
# from ptflops import get_model_complexity_info
import os
from get_config import get_config

try:
    from Save_result import show_pred
    from Save_result_multipredict import show_pred_final
except ModuleNotFoundError as exc:
    _plotting_import_error = exc

    def show_pred(*args, **kwargs):
        print(f'skip plotting because dependency is missing: {_plotting_import_error}')

    def show_pred_final(*args, **kwargs):
        print(f'skip final plotting because dependency is missing: {_plotting_import_error}')

bootstrap_parser = argparse.ArgumentParser(add_help=False)
bootstrap_parser.add_argument('--model_name', type=str, default='KAN_TQNet')
bootstrap_parser.add_argument('--pred_len_override', type=int, default=None)
bootstrap_parser.add_argument('--exp_tag', type=str, default='')
bootstrap_parser.add_argument('--seed', type=int, default=2020)
bootstrap_parser.add_argument('--channel_loss_weights', type=float, nargs=3, default=None)
bootstrap_parser.add_argument('--use_tq', type=int, default=None)
bootstrap_parser.add_argument('--use_kan', type=int, default=None)
bootstrap_parser.add_argument('--use_channel_heads', type=int, default=None)
bootstrap_parser.add_argument('--use_channel_adapter', type=int, default=None)
bootstrap_parser.add_argument('--use_hda', type=int, default=None)
bootstrap_parser.add_argument('--use_multi_scale', type=int, default=None)
bootstrap_parser.add_argument('--use_freq_branch', type=int, default=None)
bootstrap_parser.add_argument('--use_trend_residual', type=int, default=None)
bootstrap_parser.add_argument('--use_patch_multiscale', type=int, default=None)
bootstrap_parser.add_argument('--use_local_conv', type=int, default=None)
bootstrap_parser.add_argument('--use_electricity_refine', type=int, default=None)
bootstrap_parser.add_argument('--electricity_refine_scale', type=float, default=None)
bootstrap_parser.add_argument('--use_cooling_refine', type=int, default=None)
bootstrap_parser.add_argument('--cooling_refine_scale', type=float, default=None)
bootstrap_parser.add_argument('--electricity_peak_weight', type=float, default=None)
bootstrap_parser.add_argument('--electricity_peak_quantile', type=float, default=None)
bootstrap_parser.add_argument('--cooling_peak_weight', type=float, default=None)
bootstrap_parser.add_argument('--cooling_peak_quantile', type=float, default=None)
bootstrap_parser.add_argument('--use_time_features', type=int, default=None)
bootstrap_parser.add_argument('--use_timemixer', type=int, default=None)
bootstrap_args, _ = bootstrap_parser.parse_known_args()

test_model_name = bootstrap_args.model_name

config = get_config(test_model_name)
if bootstrap_args.pred_len_override is not None:
    config.pred_len = bootstrap_args.pred_len_override
if bootstrap_args.channel_loss_weights is not None:
    config.channel_loss_weights = tuple(bootstrap_args.channel_loss_weights)
if bootstrap_args.use_tq is not None:
    config.use_tq = bootstrap_args.use_tq
if bootstrap_args.use_kan is not None:
    config.use_kan = bootstrap_args.use_kan
if bootstrap_args.use_channel_heads is not None:
    config.use_channel_heads = bootstrap_args.use_channel_heads
if bootstrap_args.use_channel_adapter is not None:
    config.use_channel_adapter = bootstrap_args.use_channel_adapter
if bootstrap_args.use_hda is not None:
    config.use_hda = bootstrap_args.use_hda
if bootstrap_args.use_multi_scale is not None:
    config.use_multi_scale = bootstrap_args.use_multi_scale
if bootstrap_args.use_freq_branch is not None:
    config.use_freq_branch = bootstrap_args.use_freq_branch
if bootstrap_args.use_trend_residual is not None:
    config.use_trend_residual = bootstrap_args.use_trend_residual
if bootstrap_args.use_patch_multiscale is not None:
    config.use_patch_multiscale = bootstrap_args.use_patch_multiscale
if bootstrap_args.use_local_conv is not None:
    config.use_local_conv = bootstrap_args.use_local_conv
if bootstrap_args.use_electricity_refine is not None:
    config.use_electricity_refine = bootstrap_args.use_electricity_refine
if bootstrap_args.electricity_refine_scale is not None:
    config.electricity_refine_scale = bootstrap_args.electricity_refine_scale
if bootstrap_args.use_cooling_refine is not None:
    config.use_cooling_refine = bootstrap_args.use_cooling_refine
if bootstrap_args.cooling_refine_scale is not None:
    config.cooling_refine_scale = bootstrap_args.cooling_refine_scale
if bootstrap_args.electricity_peak_weight is not None:
    config.electricity_peak_weight = bootstrap_args.electricity_peak_weight
if bootstrap_args.electricity_peak_quantile is not None:
    config.electricity_peak_quantile = bootstrap_args.electricity_peak_quantile
if bootstrap_args.cooling_peak_weight is not None:
    config.cooling_peak_weight = bootstrap_args.cooling_peak_weight
if bootstrap_args.cooling_peak_quantile is not None:
    config.cooling_peak_quantile = bootstrap_args.cooling_peak_quantile
if bootstrap_args.use_time_features is None:
    config.use_time_features = 1 if test_model_name == 'KAN_TQNet' else 0
else:
    config.use_time_features = bootstrap_args.use_time_features
if bootstrap_args.use_timemixer is not None:
    config.use_timemixer = bootstrap_args.use_timemixer
Model = importlib.import_module(f'test_model.{test_model_name}').Model

pred_length = config.pred_len
seq_len = config.seq_len

save_name = f'{test_model_name}_{pred_length}_{seq_len}'
if bootstrap_args.exp_tag:
    save_name = f'{save_name}_{bootstrap_args.exp_tag}'

path1 = f'./output/{save_name}/result/'
os.makedirs(path1, exist_ok=True)

path2 = f'./output/{save_name}/model/'
os.makedirs(path2, exist_ok=True)

path3 = f'./output/{save_name}/assets/'
os.makedirs(path3, exist_ok=True)

CHANNEL_LOSS_WEIGHTS = tuple(getattr(config, 'channel_loss_weights', (1.0, 1.0, 1.0)))
MIXED_MAE_RATIO_WEIGHT = getattr(config, 'mixed_mae_ratio_weight', 0.2)
ELECTRICITY_PEAK_WEIGHT = getattr(config, 'electricity_peak_weight', 0.0)
ELECTRICITY_PEAK_QUANTILE = getattr(config, 'electricity_peak_quantile', 0.85)
COOLING_PEAK_WEIGHT = getattr(config, 'cooling_peak_weight', 0.0)
COOLING_PEAK_QUANTILE = getattr(config, 'cooling_peak_quantile', 0.75)
CHANNEL_NAMES = ('electricity', 'cooling', 'heating')
TARGET_CHANNEL_COUNT = min(getattr(config, 'target_channels', len(CHANNEL_NAMES)), len(CHANNEL_NAMES))
USE_BALANCED_CHECKPOINT = bool(getattr(config, 'use_balanced_checkpoint', 1))
BALANCED_GUARD_RELAX = float(getattr(config, 'balanced_guard_relax', 0.03))
BALANCED_GUARD_ABS = float(getattr(config, 'balanced_guard_abs', 0.2))
BALANCED_CHANNEL_PRIORITY = tuple(
    getattr(config, 'balanced_channel_priority', (1.0,) * TARGET_CHANNEL_COUNT)
)


def weighted_channel_mixed_loss(target, pred, channel_weights=CHANNEL_LOSS_WEIGHTS,
                                mae_ratio_weight=MIXED_MAE_RATIO_WEIGHT,
                                electricity_peak_weight=ELECTRICITY_PEAK_WEIGHT,
                                electricity_peak_quantile=ELECTRICITY_PEAK_QUANTILE,
                                cooling_peak_weight=COOLING_PEAK_WEIGHT,
                                cooling_peak_quantile=COOLING_PEAK_QUANTILE):
    weights = pred.new_tensor(channel_weights)
    losses = []

    for channel_idx, weight in enumerate(weights):
        channel_target = target[:, :, channel_idx]
        channel_pred = pred[:, :, channel_idx]
        abs_diff = torch.abs(channel_pred - channel_target)
        point_weight = torch.ones_like(abs_diff)

        if channel_idx == 0 and electricity_peak_weight > 0:
            peak_quantile = min(max(electricity_peak_quantile, 0.0), 0.99)
            threshold = torch.quantile(torch.abs(channel_target).reshape(-1), peak_quantile)
            point_weight = point_weight + electricity_peak_weight * (torch.abs(channel_target) >= threshold).float()

        if channel_idx == 1 and cooling_peak_weight > 0:
            peak_quantile = min(max(cooling_peak_quantile, 0.0), 0.99)
            threshold = torch.quantile(torch.abs(channel_target).reshape(-1), peak_quantile)
            point_weight = point_weight + cooling_peak_weight * (torch.abs(channel_target) >= threshold).float()

        channel_mape = (
            (abs_diff / (torch.abs(channel_target) + 1e-2)) * point_weight
        ).sum() / point_weight.sum() * 100
        channel_mae_ratio = (
            (abs_diff * point_weight).sum() /
            ((torch.abs(channel_target) * point_weight).sum() + 1e-2)
        ) * 100
        losses.append(weight * (channel_mape + mae_ratio_weight * channel_mae_ratio))

    return torch.stack(losses).sum() / weights.sum()


def compute_correlation_score(pred, target):
    sigma_p = pred.std(axis=0)
    sigma_g = target.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = target.mean(axis=0)
    denominator = sigma_p * sigma_g
    valid = denominator != 0
    if not np.any(valid):
        return 0.0
    correlation = ((pred - mean_p) * (target - mean_g)).mean(axis=0) / (denominator + 1e-8)
    return float(correlation[valid].mean())


def build_metric_summary(predict_scaled, target_scaled):
    summary = {
        'overall': {
            'mae': float(MAE(target_scaled, predict_scaled)),
            'mape': float(MAPE(target_scaled, predict_scaled)),
            'rmse': float(RMSE(target_scaled, predict_scaled)),
            'corr': compute_correlation_score(predict_scaled, target_scaled),
        },
        'per_channel': []
    }

    for channel_idx in range(min(target_scaled.shape[-1], TARGET_CHANNEL_COUNT)):
        channel_target = target_scaled[:, :, channel_idx]
        channel_predict = predict_scaled[:, :, channel_idx]
        summary['per_channel'].append({
            'name': CHANNEL_NAMES[channel_idx],
            'mae': float(MAE(channel_target, channel_predict)),
            'mape': float(MAPE(channel_target, channel_predict)),
            'rmse': float(RMSE(channel_target, channel_predict)),
            'corr': compute_correlation_score(channel_predict, channel_target),
        })

    return summary


def format_channel_summary(prefix, metric_summary):
    lines = []
    for channel_summary in metric_summary['per_channel']:
        lines.append(
            f"{prefix}_{channel_summary['name']} mae {channel_summary['mae']:.4f} | "
            f"mape {channel_summary['mape']:.4f} | corr {channel_summary['corr']:.4f} | "
            f"rmse {channel_summary['rmse']:.4f}"
        )
    return '\n'.join(lines)


def format_overall_summary(prefix, metric_summary):
    overall = metric_summary['overall']
    return (
        f"{prefix} mae {overall['mae']:.4f} | mape {overall['mape']:.4f} | "
        f"corr {overall['corr']:.4f} | rmse {overall['rmse']:.4f}"
    )


def save_prediction_artifacts(result_dir, file_prefix, target, predict):
    prefix = f'{file_prefix}_' if file_prefix else ''
    target_pt_path = os.path.join(result_dir, f'{prefix}all_y_true.pt')
    predict_pt_path = os.path.join(result_dir, f'{prefix}all_predict_value.pt')
    target_npy_path = os.path.join(result_dir, f'{prefix}all_y_true.npy')
    predict_npy_path = os.path.join(result_dir, f'{prefix}all_predict_value.npy')

    torch.save(target, target_pt_path)
    torch.save(predict, predict_pt_path)

    target_np = target.cpu().numpy()
    predict_np = predict.cpu().numpy()
    np.save(target_npy_path, target_np)
    np.save(predict_npy_path, predict_np)
    return target_np, predict_np


def check_balanced_candidate(metric_summary, best_channel_val, relax=BALANCED_GUARD_RELAX, abs_tol=BALANCED_GUARD_ABS):
    breaches = []
    for channel_summary in metric_summary['per_channel']:
        channel_name = channel_summary['name']
        frontier = best_channel_val.get(channel_name, float('inf'))
        if not math.isfinite(frontier):
            continue
        allowed_gap = max(abs_tol, frontier * relax)
        upper_bound = frontier + allowed_gap
        if channel_summary['mape'] > upper_bound:
            breaches.append({
                'name': channel_name,
                'current': channel_summary['mape'],
                'frontier': frontier,
                'upper_bound': upper_bound,
            })
    return len(breaches) == 0, breaches


def format_balanced_breaches(breaches):
    if not breaches:
        return ''
    return ' | '.join(
        f"{item['name']} {item['current']:.4f}>{item['upper_bound']:.4f} (best {item['frontier']:.4f})"
        for item in breaches
    )


def build_balanced_signature(metric_summary, channel_frontier):
    relative_gaps = []
    absolute_gaps = []
    priority_weights = []
    for channel_idx, channel_summary in enumerate(metric_summary['per_channel']):
        frontier = channel_frontier.get(channel_summary['name'], float('inf'))
        current = channel_summary['mape']
        if math.isfinite(frontier):
            absolute_gap = max(0.0, current - frontier)
            relative_gap = absolute_gap / max(frontier, 1e-8)
        else:
            absolute_gap = 0.0
            relative_gap = 0.0
        absolute_gaps.append(absolute_gap)
        relative_gaps.append(relative_gap)
        if channel_idx < len(BALANCED_CHANNEL_PRIORITY):
            priority_weights.append(BALANCED_CHANNEL_PRIORITY[channel_idx])
        else:
            priority_weights.append(1.0)

    if not relative_gaps:
        relative_gaps = [0.0]
        absolute_gaps = [0.0]
        priority_weights = [1.0]

    priority_weights = np.asarray(priority_weights, dtype=np.float32)
    priority_weights = priority_weights / np.clip(priority_weights.sum(), a_min=1e-8, a_max=None)

    return (
        max(relative_gaps),
        float(np.sum(np.asarray(relative_gaps) * priority_weights)),
        max(absolute_gaps),
        float(np.sum(np.asarray(absolute_gaps) * priority_weights)),
        metric_summary['overall']['mape'],
    )


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        # [64,12,3]
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

    scale = data.scale.expand(predict.size(0), predict.size(1), TARGET_CHANNEL_COUNT).cpu().numpy()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    predict_scaled = predict * scale
    target_scaled = Ytest * scale
    return build_metric_summary(predict_scaled, target_scaled)


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    total_mae_loss = 0

    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # print(X.shape, Y.shape)
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        # print(X.shape)
        tx = X
        ty = Y
        # print(tx.shape,ty.shape)
        output = model(tx)
        # print(output.shape)
        output = torch.squeeze(output)
        # print(output.shape)
        scale = data.scale.expand(output.size(0), output.size(1), TARGET_CHANNEL_COUNT)
        # print("ty", ty, "output", output, "scale", scale, "output*scale", output * scale, "ty*scale", ty * scale)
        # print(ty.shape,output.shape,scale.shape)
        scaled_target = ty * scale
        scaled_output = output * scale
        loss = weighted_channel_mixed_loss(scaled_target, scaled_output)
        loss_mae = MAE(scaled_target.cpu().detach().numpy(), scaled_output.cpu().detach().numpy())

        loss_mse = MSE(scaled_target, scaled_output)
        loss.backward()
        total_loss += loss.item()
        total_mae_loss += loss_mae.item()
        grad_norm = optim.step()
        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / TARGET_CHANNEL_COUNT))
        iter += 1

    return total_loss / iter, total_mae_loss / iter


def load_model_from_checkpoint(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Model(config).to(device)
    model.load_state_dict(state_dict)
    return model


def build_channel_best_bundle(data, model_dir, device, batch_size):
    channel_paths = [
        os.path.join(model_dir, f'model_best_{channel_name}.pt')
        for channel_name in CHANNEL_NAMES[:TARGET_CHANNEL_COUNT]
    ]
    if not all(os.path.exists(path) for path in channel_paths):
        return None

    hybrid_target = None
    hybrid_predict = None

    for channel_idx, checkpoint_path in enumerate(channel_paths):
        channel_model = load_model_from_checkpoint(checkpoint_path, device)
        channel_target, channel_predict = plow(
            data, data.test[0], data.test[1][:, :, :TARGET_CHANNEL_COUNT], channel_model, batch_size
        )
        if hybrid_target is None:
            hybrid_target = channel_target.clone()
            hybrid_predict = torch.zeros_like(channel_predict)
        hybrid_predict[:, :, channel_idx] = channel_predict[:, :, channel_idx]

    hybrid_summary = build_metric_summary(hybrid_predict.cpu().numpy(), hybrid_target.cpu().numpy())
    return {
        'target': hybrid_target,
        'predict': hybrid_predict,
        'summary': hybrid_summary,
    }


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:

        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            # print(param[0])
            # print(param[1])
            total_params = param[1].numel()
            # print(f'{total_params:,} total parameters.')
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
            # print('----------------')
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


# 原始参数表：
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/dataset_input_jiuzheng.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=f'./output/{save_name}/model/model_lnn.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=15, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=12, help='inputs dimension')

parser.add_argument('--seq_in_len', type=int, default=config.seq_len, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=config.pred_len, help='output sequence length')
parser.add_argument('--horizon', type=int, default=config.pred_len)

parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=config.batchsize, help='batch size')
parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=config.epochs, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--pred_len_override', type=int, default=bootstrap_args.pred_len_override,
                    help='override prediction length for multi-horizon training')
parser.add_argument('--seed', type=int, default=bootstrap_args.seed,
                    help='random seed for reproducible runs')
parser.add_argument('--model_name', type=str, default=bootstrap_args.model_name,
                    help='model name in test_model/ and get_config.py')
parser.add_argument('--exp_tag', type=str, default=bootstrap_args.exp_tag,
                    help='suffix for experiment output directory')
parser.add_argument('--channel_loss_weights', type=float, nargs=3, default=bootstrap_args.channel_loss_weights,
                    help='channel loss weights for electricity/cooling/heating')
parser.add_argument('--use_tq', type=int, default=bootstrap_args.use_tq,
                    help='1 to enable Temporal Query, 0 to disable it')
parser.add_argument('--use_kan', type=int, default=bootstrap_args.use_kan,
                    help='1 to use KAN blocks, 0 to replace them with MLP blocks')
parser.add_argument('--use_channel_heads', type=int, default=bootstrap_args.use_channel_heads,
                    help='1 to use per-channel output heads, 0 to share the output head')
parser.add_argument('--use_channel_adapter', type=int, default=bootstrap_args.use_channel_adapter,
                    help='1 to enable per-channel feature adapters, 0 to disable them')
parser.add_argument('--use_hda', type=int, default=bootstrap_args.use_hda,
                    help='1 to enable hierarchical decomposition aggregation, 0 to disable it')
parser.add_argument('--use_multi_scale', type=int, default=bootstrap_args.use_multi_scale,
                    help='1 to enable low-frequency multi-scale branch, 0 to disable it')
parser.add_argument('--use_freq_branch', type=int, default=bootstrap_args.use_freq_branch,
                    help='1 to enable frequency branch, 0 to disable it')
parser.add_argument('--use_trend_residual', type=int, default=bootstrap_args.use_trend_residual,
                    help='1 to enable trend residual decomposition, 0 to disable it')
parser.add_argument('--use_patch_multiscale', type=int, default=bootstrap_args.use_patch_multiscale,
                    help='1 to enable patch multi-scale enhancement, 0 to disable it')
parser.add_argument('--use_local_conv', type=int, default=bootstrap_args.use_local_conv,
                    help='1 to enable local depthwise convolution, 0 to disable it')
parser.add_argument('--use_electricity_refine', type=int, default=bootstrap_args.use_electricity_refine,
                    help='override electricity refine switch')
parser.add_argument('--electricity_refine_scale', type=float, default=bootstrap_args.electricity_refine_scale,
                    help='override electricity refine residual scale')
parser.add_argument('--use_cooling_refine', type=int, default=bootstrap_args.use_cooling_refine,
                    help='override cooling refine switch')
parser.add_argument('--cooling_refine_scale', type=float, default=bootstrap_args.cooling_refine_scale,
                    help='override cooling refine residual scale')
parser.add_argument('--electricity_peak_weight', type=float, default=bootstrap_args.electricity_peak_weight,
                    help='extra weight for high-electricity target timesteps')
parser.add_argument('--electricity_peak_quantile', type=float, default=bootstrap_args.electricity_peak_quantile,
                    help='quantile threshold for high-electricity weighting')
parser.add_argument('--cooling_peak_weight', type=float, default=bootstrap_args.cooling_peak_weight,
                    help='extra weight for high-cooling target timesteps')
parser.add_argument('--cooling_peak_quantile', type=float, default=bootstrap_args.cooling_peak_quantile,
                    help='quantile threshold for high-cooling weighting')
parser.add_argument('--use_time_features', type=int, default=bootstrap_args.use_time_features,
                    help='1 to append engineered time features, 0 to use raw input columns only')
parser.add_argument('--use_timemixer', type=int, default=bootstrap_args.use_timemixer,
                    help='1 to enable TimeMixer refinement in the low-frequency branch of KAN_TQNet')
parser.add_argument('--resume_from', type=str, default='',
                    help='optional checkpoint path to resume model weights from')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch index already completed when resuming')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

# 时序库的参数设置：
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


def main():
    seed = args.seed
    fix_seed(seed)

    fin = open(args.data)
    rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
    print(f'raw data shape: {rawdat.shape}')

    use_time_features = getattr(config, 'use_time_features', 1)
    Data = DataLoaderS(
        args.data, 0.8, 0.1, device, args.horizon, args.seq_in_len, args.normalize,
        add_time_features=bool(use_time_features)
    )
    config.enc_in = Data.m
    if hasattr(config, 'dec_in'):
        config.dec_in = Data.m
    args.in_dim = Data.m
    args.num_nodes = Data.m
    print(f'model input features after time augmentation: {Data.m}')

    model = Model(config)

    # flops, params = get_model_complexity_info(model, (1, 12, config.seq_len), as_strings=True, print_per_layer_stat=False)
    # print('flops: ', flops, 'params: ', params)
    # print('------------------------------------------------------')
    #
    # total_param, total_megabytes, _dict = count_parameters(model)
    # model = model.to(device)
    # # for k, v in _dict.items():
    # #     print("Module:", k, "param:", v, "%3.3fM" % (v / (1024 * 1024)))
    # print("Total megabytes:", total_megabytes, "M")
    # print("Total parameters:", total_param)
    # print(args)
    # # print('The recpetive field size is', model.receptive_field)
    # nParams = sum([p.nelement() for p in model.parameters()])
    # with open(f'./output/{save_name}/result/data.txt', 'a') as f:
    #     print('Parameters is', params, flush=True, file=f)
    #     print('FLOPs is', flops, flush=True, file=f)
    #
    # print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    result_log_path = os.path.join(path1, 'data.txt')
    model = Model(config).to(device)
    if args.resume_from:
        if os.path.exists(args.resume_from):
            state_dict = torch.load(args.resume_from, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f'resumed model weights from {args.resume_from}')
        else:
            raise FileNotFoundError(f'resume checkpoint not found: {args.resume_from}')

    best_val = float('inf')
    best_channel_val = {channel_name: float('inf') for channel_name in CHANNEL_NAMES[:TARGET_CHANNEL_COUNT]}
    best_channel_paths = {
        channel_name: os.path.join(path2, f'model_best_{channel_name}.pt')
        for channel_name in CHANNEL_NAMES[:TARGET_CHANNEL_COUNT]
    }
    balanced_best_val = float('inf')
    balanced_best_signature = None
    balanced_best_path = os.path.join(path2, 'model_balanced_best.pt')
    optim = Optim(
        model.parameters(), args.optim, config.lr, args.clip, 'min', config.weightdecay, config.decaypatience,
        lr_decay=args.weight_decay
    )

    try:
        print('begin training')
        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_mae_loss = train(
                Data, Data.train[0], Data.train[1][:, :, :TARGET_CHANNEL_COUNT], model, criterion, optim,
                args.batch_size
            )
            val_metrics = evaluate(
                Data, Data.valid[0], Data.valid[1][:, :, :TARGET_CHANNEL_COUNT], model,
                evaluateL2,
                evaluateL1,
                args.batch_size
            )
            val_mae = val_metrics['overall']['mae']
            val_mape = val_metrics['overall']['mape']
            val_corr = val_metrics['overall']['corr']
            val_rmse = val_metrics['overall']['rmse']
            optim.lronplateau(val_mape)

            epoch_line = (
                '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | '
                'valid mae {:5.4f} | valid mape {:5.4f} | valid corr {:5.4f} | valid rmse {:5.4f}'
            ).format(
                epoch, (time.time() - epoch_start_time), train_loss, train_mae_loss, val_mae, val_mape, val_corr,
                val_rmse
            )
            with open(result_log_path, 'a', encoding='utf-8') as f:
                print(epoch_line, flush=True, file=f)
                channel_log = format_channel_summary('valid', val_metrics)
                if channel_log:
                    print(channel_log, flush=True, file=f)
            print(epoch_line, flush=True)
            channel_console_log = format_channel_summary('valid', val_metrics)
            if channel_console_log:
                print(channel_console_log, flush=True)

            channel_frontier_before_update = dict(best_channel_val)

            for channel_summary in val_metrics['per_channel']:
                channel_name = channel_summary['name']
                channel_val = channel_summary['mape']
                if channel_val < best_channel_val[channel_name]:
                    torch.save(model.state_dict(), best_channel_paths[channel_name])
                    best_channel_val[channel_name] = channel_val
                    print(f'{channel_name} checkpoint updated: valid_mape={channel_val:.4f}', flush=True)

            if USE_BALANCED_CHECKPOINT:
                is_balanced_candidate, balanced_breaches = check_balanced_candidate(
                    val_metrics, channel_frontier_before_update
                )
                balanced_signature = build_balanced_signature(val_metrics, channel_frontier_before_update)
                if is_balanced_candidate and (
                    balanced_best_signature is None or balanced_signature < balanced_best_signature
                ):
                    torch.save(model.state_dict(), balanced_best_path)
                    balanced_best_val = val_mape
                    balanced_best_signature = balanced_signature
                    print(
                        f'balanced checkpoint updated: valid_mape={val_mape:.4f} | '
                        f'max_rel_gap={balanced_signature[0]:.4f} | mean_rel_gap={balanced_signature[1]:.4f} | '
                        f'guard_relax={BALANCED_GUARD_RELAX:.4f} | guard_abs={BALANCED_GUARD_ABS:.4f}',
                        flush=True
                    )
                elif (not is_balanced_candidate) and epoch % 10 == 0:
                    print(
                        f'balanced checkpoint skipped: {format_balanced_breaches(balanced_breaches)}',
                        flush=True
                    )

            if val_mape < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val = val_mape
                print(f'overall checkpoint updated: valid_mape={val_mape:.4f}', flush=True)

            test_metrics = evaluate(
                Data, Data.test[0], Data.test[1][:, :, :TARGET_CHANNEL_COUNT], model,
                evaluateL2,
                evaluateL1,
                args.batch_size
            )
            test_line = format_overall_summary('test', test_metrics)
            with open(result_log_path, 'a', encoding='utf-8') as f:
                print(test_line, flush=True, file=f)
                test_channel_log = format_channel_summary('test', test_metrics)
                if test_channel_log:
                    print(test_channel_log, flush=True, file=f)
            print(test_line, flush=True)
            test_channel_console_log = format_channel_summary('test', test_metrics)
            if test_channel_console_log:
                print(test_channel_console_log, flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if os.path.exists(args.save):
        best_model = load_model_from_checkpoint(args.save, device)
    else:
        best_model = model
        print(f'overall best checkpoint not found, fallback to in-memory model: {args.save}', flush=True)

    overall_metrics = evaluate(
        Data, Data.test[0], Data.test[1][:, :, :TARGET_CHANNEL_COUNT], best_model, evaluateL2, evaluateL1,
        args.batch_size
    )
    overall_line = format_overall_summary('final_overall_best', overall_metrics)
    with open(result_log_path, 'a', encoding='utf-8') as f:
        print(overall_line, file=f)
        overall_channel_log = format_channel_summary('final_overall_best', overall_metrics)
        if overall_channel_log:
            print(overall_channel_log, file=f)
    print(overall_line)
    overall_channel_console_log = format_channel_summary('final_overall_best', overall_metrics)
    if overall_channel_console_log:
        print(overall_channel_console_log)

    overall_y_true, overall_predict_value = plow(
        Data, Data.test[0], Data.test[1][:, :, :TARGET_CHANNEL_COUNT], best_model, args.batch_size
    )
    save_prediction_artifacts(path1, 'overall_best', overall_y_true, overall_predict_value)

    balanced_bundle = None
    if USE_BALANCED_CHECKPOINT and os.path.exists(balanced_best_path):
        balanced_model = load_model_from_checkpoint(balanced_best_path, device)
        balanced_metrics = evaluate(
            Data, Data.test[0], Data.test[1][:, :, :TARGET_CHANNEL_COUNT], balanced_model, evaluateL2, evaluateL1,
            args.batch_size
        )
        balanced_line = format_overall_summary('final_balanced_best', balanced_metrics)
        with open(result_log_path, 'a', encoding='utf-8') as f:
            print(balanced_line, file=f)
            balanced_channel_log = format_channel_summary('final_balanced_best', balanced_metrics)
            if balanced_channel_log:
                print(balanced_channel_log, file=f)
        print(balanced_line)
        balanced_channel_console_log = format_channel_summary('final_balanced_best', balanced_metrics)
        if balanced_channel_console_log:
            print(balanced_channel_console_log)
        balanced_y_true, balanced_predict_value = plow(
            Data, Data.test[0], Data.test[1][:, :, :TARGET_CHANNEL_COUNT], balanced_model, args.batch_size
        )
        save_prediction_artifacts(path1, 'balanced_best', balanced_y_true, balanced_predict_value)
        balanced_bundle = {
            'target': balanced_y_true,
            'predict': balanced_predict_value,
            'summary': balanced_metrics,
        }
    elif USE_BALANCED_CHECKPOINT:
        print('balanced best checkpoint not available, skip final_balanced_best evaluation', flush=True)

    hybrid_bundle = build_channel_best_bundle(Data, path2, device, args.batch_size)
    if hybrid_bundle is not None:
        hybrid_line = format_overall_summary('final_hybrid_best', hybrid_bundle['summary'])
        with open(result_log_path, 'a', encoding='utf-8') as f:
            print(hybrid_line, file=f)
            hybrid_channel_log = format_channel_summary('final_hybrid_best', hybrid_bundle['summary'])
            if hybrid_channel_log:
                print(hybrid_channel_log, file=f)
        print(hybrid_line)
        hybrid_channel_console_log = format_channel_summary('final_hybrid_best', hybrid_bundle['summary'])
        if hybrid_channel_console_log:
            print(hybrid_channel_console_log)
        save_prediction_artifacts(path1, 'hybrid_best', hybrid_bundle['target'], hybrid_bundle['predict'])
        final_target = hybrid_bundle['target']
        final_predict = hybrid_bundle['predict']
        final_metrics = hybrid_bundle['summary']
        final_export_name = 'hybrid_best'
    elif balanced_bundle is not None:
        final_target = balanced_bundle['target']
        final_predict = balanced_bundle['predict']
        final_metrics = balanced_bundle['summary']
        final_export_name = 'balanced_best'
    else:
        final_target = overall_y_true
        final_predict = overall_predict_value
        final_metrics = overall_metrics
        final_export_name = 'overall_best'
        print('hybrid best bundle skipped because not all per-channel checkpoints are available', flush=True)

    final_target_np, final_predict_np = save_prediction_artifacts(path1, '', final_target, final_predict)
    with open(result_log_path, 'a', encoding='utf-8') as f:
        print(f'final export source: {final_export_name}', file=f)
    print(f'final export source: {final_export_name}')

    show_pred(final_target_np, final_predict_np, config.pred_len, save_name)
    print('--------------------------------final result--------------------------------')
    show_pred_final(final_target_np, final_predict_np, config.pred_len, save_name)

    return final_metrics['overall']['mae'], final_metrics['overall']['mape'], final_metrics['overall']['corr']

def plow(data, X, Y, model, batch_size):
    model.eval()
    model.eval()

    all_predict_value = 0
    all_y_true = 0
    num = 0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), output.size(1), TARGET_CHANNEL_COUNT)  # zuijin xiugai
        y_true = Y * scale
        predict_value = output * scale

        if num == 0:
            all_predict_value = predict_value
            all_y_true = y_true
        else:
            all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
            all_y_true = torch.cat([all_y_true, y_true], dim=0)
        num = num + 1

    return all_y_true, all_predict_value


if __name__ == "__main__":
    main()

