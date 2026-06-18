"""
KAN_TQNet 噪声鲁棒性实验
======================
实验目的：评估 KAN_TQNet 在不同类型、不同强度噪声下的预测鲁棒性。

噪声类型：
  1. Gaussian  - 高斯加性噪声（模拟传感器随机误差）
  2. Impulse   - 脉冲噪声（模拟数据突变/异常值）
  3. Scaling   - 幅度缩放噪声（模拟传感器增益漂移）
  4. Missing   - 随机缺失值（用均值填充，模拟数据丢失）
  5. Periodic  - 周期性偏置噪声（模拟系统性漂移）

噪声强度（SNR 梯度）：
  noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
  （相对于归一化数据的标准差比例）

输出：
  - output/noise_experiment/results.csv   逐行记录每个实验条件的指标
  - output/noise_experiment/summary.txt   人类可读的汇总表
  - output/noise_experiment/plots/        各噪声类型的性能曲线图（可选）
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
from get_config import get_config
from util import DataLoaderS
from metrics import MAE, MAPE, RMSE

# ── 全局常量 ──────────────────────────────────────────────────────────────────
CHANNEL_NAMES = ('electricity', 'cooling', 'heating')
TARGET_CHANNEL_COUNT = 3

# ── 噪声实验配置 ──────────────────────────────────────────────────────────────
NOISE_TYPES = ['clean', 'gaussian', 'impulse', 'scaling', 'missing', 'periodic']
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_correlation_score(pred: np.ndarray, target: np.ndarray) -> float:
    sigma_p = pred.std(axis=0)
    sigma_g = target.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = target.mean(axis=0)
    denominator = sigma_p * sigma_g
    valid = denominator != 0
    if not np.any(valid):
        return 0.0
    corr = ((pred - mean_p) * (target - mean_g)).mean(axis=0) / (denominator + 1e-8)
    return float(corr[valid].mean())


# ─────────────────────────────────────────────────────────────────────────────
# 噪声注入函数（作用于归一化后的输入张量 X，形状 [B, seq_len, enc_in]）
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(X: torch.Tensor, level: float) -> torch.Tensor:
    """高斯加性噪声：x + N(0, level)"""
    if level == 0.0:
        return X
    noise = torch.randn_like(X) * level
    return X + noise


def add_impulse_noise(X: torch.Tensor, level: float) -> torch.Tensor:
    """脉冲噪声：以 level 概率将某时间步替换为随机极值"""
    if level == 0.0:
        return X
    X = X.clone()
    mask = torch.rand_like(X) < level
    # 极值取 [-3, 3] 均匀分布（归一化数据范围约 [-1, 1]，3 倍为明显异常）
    impulse = (torch.rand_like(X) * 6.0 - 3.0)
    X[mask] = impulse[mask]
    return X


def add_scaling_noise(X: torch.Tensor, level: float) -> torch.Tensor:
    """幅度缩放噪声：x * (1 + N(0, level))，逐通道独立"""
    if level == 0.0:
        return X
    # 每个样本、每个通道一个缩放因子
    scale = 1.0 + torch.randn(X.shape[0], 1, X.shape[2], device=X.device) * level
    return X * scale


def add_missing_noise(X: torch.Tensor, level: float) -> torch.Tensor:
    """随机缺失：以 level 概率将时间步置为该通道均值（归一化后约为 0）"""
    if level == 0.0:
        return X
    X = X.clone()
    mask = torch.rand(X.shape[0], X.shape[1], 1, device=X.device) < level
    mask = mask.expand_as(X)
    # 用通道均值填充（归一化数据均值接近 0）
    channel_mean = X.mean(dim=1, keepdim=True)
    X[mask] = channel_mean.expand_as(X)[mask]
    return X


def add_periodic_noise(X: torch.Tensor, level: float, period: int = 24) -> torch.Tensor:
    """周期性偏置噪声：叠加一个幅度为 level 的正弦偏置（模拟系统漂移）"""
    if level == 0.0:
        return X
    seq_len = X.shape[1]
    t = torch.arange(seq_len, dtype=torch.float32, device=X.device)
    bias = level * torch.sin(2 * np.pi * t / period)  # [seq_len]
    bias = bias.unsqueeze(0).unsqueeze(-1)             # [1, seq_len, 1]
    return X + bias


NOISE_FN = {
    'clean':    lambda x, l: x,
    'gaussian': add_gaussian_noise,
    'impulse':  add_impulse_noise,
    'scaling':  add_scaling_noise,
    'missing':  add_missing_noise,
    'periodic': add_periodic_noise,
}


# ─────────────────────────────────────────────────────────────────────────────
# 评估函数（带噪声注入）
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_noise(data, X_all, Y_all, model, batch_size,
                        noise_type: str, noise_level: float, device):
    model.eval()
    predict_list = []
    target_list = []

    noise_fn = NOISE_FN[noise_type]

    for X, Y in data.get_batches(X_all, Y_all, batch_size, shuffle=False):
        # X: [B, seq_len, enc_in]
        X_noisy = noise_fn(X, noise_level)

        # 与 train.py 保持一致的维度变换
        X_noisy = torch.unsqueeze(X_noisy, dim=1)   # [B, 1, seq_len, enc_in]
        X_noisy = X_noisy.transpose(2, 3)            # [B, 1, enc_in, seq_len]

        with torch.no_grad():
            output = model(X_noisy)
        output = torch.squeeze(output)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        predict_list.append(output)
        target_list.append(Y)

    predict = torch.cat(predict_list, dim=0)
    test    = torch.cat(target_list,  dim=0)

    scale = data.scale.expand(predict.size(0), predict.size(1), TARGET_CHANNEL_COUNT).cpu().numpy()
    predict_np = predict.cpu().numpy() * scale
    target_np  = test.cpu().numpy()    * scale

    metrics = {
        'overall_mae':  float(MAE(target_np, predict_np)),
        'overall_mape': float(MAPE(target_np, predict_np)),
        'overall_rmse': float(RMSE(target_np, predict_np)),
        'overall_corr': compute_correlation_score(predict_np, target_np),
    }
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch_pred   = predict_np[:, :, ch_idx]
        ch_target = target_np[:, :, ch_idx]
        metrics[f'{ch_name}_mae']  = float(MAE(ch_target, ch_pred))
        metrics[f'{ch_name}_mape'] = float(MAPE(ch_target, ch_pred))
        metrics[f'{ch_name}_rmse'] = float(RMSE(ch_target, ch_pred))
        metrics[f'{ch_name}_corr'] = compute_correlation_score(ch_pred, ch_target)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 结果保存
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list, out_path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row[k]) for k in keys) + '\n')
    print(f'[saved] {out_path}')


def save_summary(rows: list, out_path: str):
    """打印一个人类可读的汇总表（按噪声类型分组）"""
    lines = []
    lines.append('=' * 100)
    lines.append('KAN_TQNet 噪声鲁棒性实验汇总')
    lines.append('=' * 100)
    lines.append(f'{"噪声类型":<12} {"强度":>8} {"MAPE(%)":>10} {"MAE":>10} {"RMSE":>10} {"Corr":>8} '
                 f'{"Elec_MAPE":>12} {"Cool_MAPE":>12} {"Heat_MAPE":>12}')
    lines.append('-' * 100)

    current_type = None
    for row in rows:
        if row['noise_type'] != current_type:
            if current_type is not None:
                lines.append('')
            current_type = row['noise_type']

        lines.append(
            f"{row['noise_type']:<12} {row['noise_level']:>8.3f} "
            f"{row['overall_mape']:>10.4f} {row['overall_mae']:>10.4f} "
            f"{row['overall_rmse']:>10.4f} {row['overall_corr']:>8.4f} "
            f"{row['electricity_mape']:>12.4f} {row['cooling_mape']:>12.4f} "
            f"{row['heating_mape']:>12.4f}"
        )

    lines.append('=' * 100)
    text = '\n'.join(lines)
    print(text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'[saved] {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 可选：绘图
# ─────────────────────────────────────────────────────────────────────────────

def try_plot(rows: list, plot_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[skip] matplotlib not available, skipping plots')
        return

    os.makedirs(plot_dir, exist_ok=True)
    noise_types_to_plot = [t for t in NOISE_TYPES if t != 'clean']

    # 按噪声类型绘制 MAPE vs noise_level
    fig, axes = plt.subplots(1, len(noise_types_to_plot), figsize=(5 * len(noise_types_to_plot), 4), sharey=False)
    if len(noise_types_to_plot) == 1:
        axes = [axes]

    for ax, ntype in zip(axes, noise_types_to_plot):
        subset = [r for r in rows if r['noise_type'] in ('clean', ntype)]
        levels = [r['noise_level'] for r in subset]
        mapes  = [r['overall_mape'] for r in subset]
        ax.plot(levels, mapes, marker='o', linewidth=2)
        ax.set_title(ntype)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Overall MAPE (%)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('KAN_TQNet Noise Robustness', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(plot_dir, 'mape_vs_noise_level.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'[saved] {fig_path}')

    # 各通道 MAPE 对比（堆叠折线，选 gaussian 为代表）
    gaussian_rows = [r for r in rows if r['noise_type'] in ('clean', 'gaussian')]
    if gaussian_rows:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        levels = [r['noise_level'] for r in gaussian_rows]
        for ch in CHANNEL_NAMES:
            ch_mapes = [r[f'{ch}_mape'] for r in gaussian_rows]
            ax2.plot(levels, ch_mapes, marker='s', label=ch)
        ax2.set_title('Per-channel MAPE under Gaussian Noise')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('MAPE (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2_path = os.path.join(plot_dir, 'channel_mape_gaussian.png')
        plt.savefig(fig2_path, dpi=150)
        plt.close()
        print(f'[saved] {fig2_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='KAN_TQNet 噪声鲁棒性实验')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的模型权重路径（.pt 文件）')
    parser.add_argument('--data', type=str,
                        default='./data/dataset_input_jiuzheng.csv',
                        help='数据集路径')
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--noise_types', type=str, nargs='+',
                        default=NOISE_TYPES,
                        help='要测试的噪声类型，默认全部')
    parser.add_argument('--noise_levels', type=float, nargs='+',
                        default=NOISE_LEVELS,
                        help='噪声强度列表')
    parser.add_argument('--pred_len', type=int, default=None,
                        help='预测步长，覆盖config默认值（24/48/72/96）')
    parser.add_argument('--out_dir', type=str,
                        default='./output/noise_experiment',
                        help='结果输出目录')
    parser.add_argument('--plot', action='store_true',
                        help='是否生成可视化图表')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print(f'[data] loading {args.data}')
    config = get_config('KAN_TQNet')
    if args.pred_len is not None:
        config.pred_len = args.pred_len
    Data = DataLoaderS(
        args.data, 0.8, 0.1, device,
        config.pred_len, config.seq_len, normalize=2,
        add_time_features=True
    )
    config.enc_in = Data.m
    X_test = Data.test[0]
    Y_test = Data.test[1][:, :, :TARGET_CHANNEL_COUNT]
    print(f'[data] test samples: {X_test.shape[0]}, enc_in: {Data.m}')

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    from test_model.KAN_TQNet import Model
    print(f'[model] loading checkpoint: {args.checkpoint}')
    model = Model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print('[model] loaded successfully')

    # ── 实验循环 ──────────────────────────────────────────────────────────────
    all_rows = []
    total = sum(
        len(args.noise_levels) if nt != 'clean' else 1
        for nt in args.noise_types
    )
    done = 0

    for noise_type in args.noise_types:
        # clean 只跑一次（level=0）
        levels_to_run = [0.0] if noise_type == 'clean' else args.noise_levels

        for level in levels_to_run:
            t0 = time.time()
            metrics = evaluate_with_noise(
                Data, X_test, Y_test, model,
                args.batch_size, noise_type, level, device
            )
            elapsed = time.time() - t0
            done += 1

            row = {
                'noise_type':        noise_type,
                'noise_level':       level,
                'overall_mape':      round(metrics['overall_mape'], 6),
                'overall_mae':       round(metrics['overall_mae'], 6),
                'overall_rmse':      round(metrics['overall_rmse'], 6),
                'overall_corr':      round(metrics['overall_corr'], 6),
                'electricity_mape':  round(metrics['electricity_mape'], 6),
                'electricity_mae':   round(metrics['electricity_mae'], 6),
                'electricity_rmse':  round(metrics['electricity_rmse'], 6),
                'electricity_corr':  round(metrics['electricity_corr'], 6),
                'cooling_mape':      round(metrics['cooling_mape'], 6),
                'cooling_mae':       round(metrics['cooling_mae'], 6),
                'cooling_rmse':      round(metrics['cooling_rmse'], 6),
                'cooling_corr':      round(metrics['cooling_corr'], 6),
                'heating_mape':      round(metrics['heating_mape'], 6),
                'heating_mae':       round(metrics['heating_mae'], 6),
                'heating_rmse':      round(metrics['heating_rmse'], 6),
                'heating_corr':      round(metrics['heating_corr'], 6),
            }
            all_rows.append(row)

            print(
                f'[{done:>3}/{total}] {noise_type:<10} level={level:.3f} | '
                f'MAPE={row["overall_mape"]:.4f}% MAE={row["overall_mae"]:.4f} '
                f'RMSE={row["overall_rmse"]:.4f} Corr={row["overall_corr"]:.4f} '
                f'({elapsed:.1f}s)'
            )

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    csv_path     = os.path.join(args.out_dir, 'results.csv')
    summary_path = os.path.join(args.out_dir, 'summary.txt')
    save_csv(all_rows, csv_path)
    save_summary(all_rows, summary_path)

    if args.plot:
        plot_dir = os.path.join(args.out_dir, 'plots')
        try_plot(all_rows, plot_dir)

    print('\n[done] 噪声实验完成')


if __name__ == '__main__':
    main()
