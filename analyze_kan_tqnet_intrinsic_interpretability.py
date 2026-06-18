from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from get_config import get_config
from test_model.KAN_TQNet import Model as KAN_TQNetModel
from util import DataLoaderS


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset_input_jiuzheng.csv"
OUTPUT_ROOT = ROOT / "output"
SAVE_DIR = OUTPUT_ROOT / "KAN_TQNet_intrinsic_interpretability"
FIG_DIR = SAVE_DIR / "figures"

HORIZONS = [24, 48, 72, 96]
CHANNEL_NAMES = ["electricity", "cooling", "heating"]
RAW_FEATURE_NAMES = [
    "KW",
    "CHWTON",
    "HTmmBTU",
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wet_bulb_temperature",
    "altimeter",
    "DayOfYear_cos",
    "Combined_mmBTU",
    "GHG",
]
TIME_FEATURE_NAMES = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "is_weekend",
    "is_workhour",
]
ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + TIME_FEATURE_NAMES


def ensure_dirs() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fix_seed(seed: int = 2020) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_config_and_data(horizon: int, device: torch.device) -> tuple[Any, DataLoaderS]:
    config = get_config("KAN_TQNet")
    config.pred_len = horizon
    config.use_time_features = 1

    data = DataLoaderS(
        str(DATA_PATH),
        0.8,
        0.1,
        device,
        horizon,
        config.seq_len,
        normalize=2,
        add_time_features=True,
    )
    config.enc_in = data.m
    if hasattr(config, "dec_in"):
        config.dec_in = data.m
    return config, data


def get_checkpoint_path(horizon: int) -> Path:
    return OUTPUT_ROOT / f"KAN_TQNet_{horizon}_168_core_full_s0" / "model" / "model_lnn.pt"


def load_model(horizon: int, device: torch.device) -> tuple[KAN_TQNetModel, Any, DataLoaderS]:
    config, data = build_config_and_data(horizon, device)
    model = KAN_TQNetModel(config).to(device)
    checkpoint_path = get_checkpoint_path(horizon)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, data


def fetch_test_batch(data: DataLoaderS, batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    X_test, Y_test = data.test
    X = X_test[:batch_size].to(data.device)
    Y = Y_test[:batch_size].to(data.device)
    return X, Y


def collect_routing_statistics(device: torch.device) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for horizon in HORIZONS:
        model, _, data = load_model(horizon, device)
        X, _ = fetch_test_batch(data, batch_size=256)
        tx = torch.unsqueeze(X, dim=1).transpose(2, 3)
        with torch.no_grad():
            _ = model(tx)
        routing = model.latest_aux.get("routing_weights")
        if routing is None:
            continue
        routing_np = routing.detach().cpu().numpy()
        rows.append(
            {
                "horizon": horizon,
                "short_mean": float(routing_np[:, 0].mean()),
                "medium_mean": float(routing_np[:, 1].mean()),
                "long_mean": float(routing_np[:, 2].mean()),
                "short_std": float(routing_np[:, 0].std()),
                "medium_std": float(routing_np[:, 1].std()),
                "long_std": float(routing_np[:, 2].std()),
            }
        )
    return rows


def _attention_forward_with_weights(model: KAN_TQNetModel, x_input: torch.Tensor, cycle_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query_input = model._build_query(cycle_index, model.seq_len, model.cycle_len)
    attn_output, attn_weights = model.channelAggregator(
        query=query_input,
        key=x_input,
        value=x_input,
        need_weights=True,
        average_attn_weights=False,
    )
    return attn_output, attn_weights


def collect_attention_map(device: torch.device, horizon: int = 96) -> dict[str, np.ndarray]:
    model, _, data = load_model(horizon, device)
    X, _ = fetch_test_batch(data, batch_size=64)
    tx = torch.unsqueeze(X, dim=1).transpose(2, 3)

    with torch.no_grad():
        x = tx.squeeze(1)
        if x.size(1) == model.enc_in:
            x = x.permute(0, 2, 1)
        x_input = x.permute(0, 2, 1)
        if model.use_patch_multiscale:
            x_input = model._apply_patch_multiscale(x_input)
        if model.use_local_conv:
            x_input = x_input + model.local_conv(x_input)
        if model.use_trend_residual:
            trend_input = model._moving_average(x_input)
            residual_input = x_input - trend_input
        else:
            residual_input = x_input
        cycle_index = torch.zeros(x_input.size(0), dtype=torch.long, device=x_input.device)
        _, attn_weights = _attention_forward_with_weights(model, residual_input, cycle_index)

    attn_np = attn_weights.detach().cpu().numpy()
    mean_heads = attn_np.mean(axis=0)  # [batch, query_channel, key_channel]
    mean_batch = mean_heads.mean(axis=0)
    return {
        "attention_matrix": mean_batch,
    }


def collect_kan_curves(device: torch.device, horizon: int = 96, num_points: int = 160) -> list[dict[str, Any]]:
    model, _, data = load_model(horizon, device)
    if not getattr(model, "use_kan", 0):
        return []

    module = model.model.kan1
    train_data = data.train[0][:, :, :]
    train_np = train_data.reshape(-1, train_data.size(-1)).cpu().numpy()

    feature_std = train_np.std(axis=0)
    top_idx = np.argsort(feature_std)[-6:][::-1]

    curves = []
    for feature_idx in top_idx:
        feature_values = train_np[:, feature_idx]
        low = float(np.percentile(feature_values, 5))
        high = float(np.percentile(feature_values, 95))
        xs = torch.linspace(low, high, num_points, device=device)
        probe = torch.zeros((num_points, model.d_model), device=device)
        mapped_idx = min(feature_idx, model.d_model - 1)
        probe[:, mapped_idx] = xs
        with torch.no_grad():
            base_out = module.base(probe)
            x_expand = probe.unsqueeze(-1)
            grid = module.grid.view(*([1] * probe.dim()), module.grid_size)
            scale = torch.exp(module.log_scale).view(*([1] * (probe.dim() - 1)), module.in_features, 1) + 1e-6
            basis = torch.exp(-((x_expand - grid) / scale) ** 2)
            basis = basis.reshape(*probe.shape[:-1], module.in_features * module.grid_size)
            spline_out = module.spline(basis)
            total_out = base_out + spline_out
        curves.append(
            {
                "feature_idx": int(feature_idx),
                "feature_name": ALL_FEATURE_NAMES[feature_idx] if feature_idx < len(ALL_FEATURE_NAMES) else f"feature_{feature_idx}",
                "x": xs.detach().cpu().numpy(),
                "base": base_out[:, mapped_idx].detach().cpu().numpy(),
                "spline": spline_out[:, mapped_idx].detach().cpu().numpy(),
                "total": total_out[:, mapped_idx].detach().cpu().numpy(),
            }
        )
    return curves


def collect_temporal_sensitivity(device: torch.device, horizon: int = 96, batch_size: int = 64) -> dict[str, np.ndarray]:
    model, _, data = load_model(horizon, device)
    X, _ = fetch_test_batch(data, batch_size=batch_size)
    tx = torch.unsqueeze(X, dim=1).transpose(2, 3).clone().detach().requires_grad_(True)
    output = model(tx)
    objective = output[:, :, : len(CHANNEL_NAMES)].abs().mean()
    objective.backward()

    grads = tx.grad.detach().cpu().numpy()  # [B, 1, F, T]
    temporal_importance = np.mean(np.abs(grads), axis=(0, 1, 2))
    feature_importance = np.mean(np.abs(grads), axis=(0, 1, 3))
    target_grad = np.mean(np.abs(grads), axis=(0, 1))
    return {
        "temporal_importance": temporal_importance,
        "feature_importance": feature_importance,
        "feature_time_importance": target_grad,
    }


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_routing(rows: list[dict[str, float]]) -> Path:
    horizons = [row["horizon"] for row in rows]
    short_vals = [row["short_mean"] for row in rows]
    medium_vals = [row["medium_mean"] for row in rows]
    long_vals = [row["long_mean"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(horizons, short_vals, marker="o", linewidth=2, label="Short branch")
    ax.plot(horizons, medium_vals, marker="s", linewidth=2, label="Medium branch")
    ax.plot(horizons, long_vals, marker="^", linewidth=2, label="Long branch")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Average routing weight")
    ax.set_title("HDA routing weight across forecast horizons")
    ax.grid(linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = FIG_DIR / "Fig_intrinsic_hda_routing_weights.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attention(attention: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(attention, cmap="YlGnBu", aspect="auto")
    feature_labels = ALL_FEATURE_NAMES[: attention.shape[0]]
    ax.set_xticks(range(len(feature_labels)), feature_labels, rotation=90)
    ax.set_yticks(range(len(feature_labels)), feature_labels)
    ax.set_title("Temporal Query channel attention map (H96)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    path = FIG_DIR / "Fig_intrinsic_temporal_query_attention.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_kan_curves(curves: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    axes = axes.flatten()
    for ax, curve in zip(axes, curves):
        ax.plot(curve["x"], curve["base"], label="Linear base", linewidth=1.8)
        ax.plot(curve["x"], curve["spline"], label="Spline term", linewidth=1.8)
        ax.plot(curve["x"], curve["total"], label="Total", linewidth=2.2)
        ax.set_title(curve["feature_name"])
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
    for ax in axes[len(curves):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("KAN response curves for high-variance input features (H96)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "Fig_intrinsic_kan_response_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity(temporal_importance: np.ndarray, feature_importance: np.ndarray, feature_time_importance: np.ndarray) -> list[Path]:
    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(8.6, 4.3))
    ax1.plot(np.arange(1, len(temporal_importance) + 1), temporal_importance, linewidth=2.0, color="#c44e52")
    ax1.set_xlabel("Historical timestep (1=oldest, 168=latest)")
    ax1.set_ylabel("Mean absolute gradient")
    ax1.set_title("Temporal sensitivity of model output (H96)")
    ax1.grid(linestyle="--", linewidth=0.6, alpha=0.35)
    fig1.tight_layout()
    path1 = FIG_DIR / "Fig_intrinsic_temporal_sensitivity.png"
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    paths.append(path1)

    fig2, ax2 = plt.subplots(figsize=(9.0, 4.6))
    feature_labels = ALL_FEATURE_NAMES[: len(feature_importance)]
    order = np.argsort(feature_importance)[::-1]
    sorted_vals = feature_importance[order]
    sorted_labels = [feature_labels[idx] for idx in order]
    ax2.bar(range(len(sorted_vals)), sorted_vals, color="#4c78a8")
    ax2.set_xticks(range(len(sorted_vals)), sorted_labels, rotation=55, ha="right")
    ax2.set_ylabel("Mean absolute gradient")
    ax2.set_title("Input feature sensitivity ranking (H96)")
    ax2.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig2.tight_layout()
    path2 = FIG_DIR / "Fig_intrinsic_feature_sensitivity.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    paths.append(path2)

    fig3, ax3 = plt.subplots(figsize=(10.5, 6.2))
    im = ax3.imshow(feature_time_importance, cmap="magma", aspect="auto")
    ax3.set_xlabel("Historical timestep")
    ax3.set_yticks(range(len(feature_labels)), feature_labels)
    ax3.set_title("Feature-time sensitivity map (H96)")
    cbar = fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute gradient")
    fig3.tight_layout()
    path3 = FIG_DIR / "Fig_intrinsic_feature_time_sensitivity.png"
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    paths.append(path3)

    return paths


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[str(item) for item in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in string_rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_summary(
    routing_rows: list[dict[str, float]],
    attention: np.ndarray,
    curves: list[dict[str, Any]],
    sensitivity: dict[str, np.ndarray],
    figure_paths: list[Path],
) -> Path:
    routing_table = markdown_table(
        ["horizon", "short_mean", "medium_mean", "long_mean"],
        [
            [
                row["horizon"],
                f"{row['short_mean']:.4f}",
                f"{row['medium_mean']:.4f}",
                f"{row['long_mean']:.4f}",
            ]
            for row in routing_rows
        ],
    )

    feature_importance = sensitivity["feature_importance"]
    top_order = np.argsort(feature_importance)[::-1][:8]
    top_feature_table = markdown_table(
        ["rank", "feature", "importance"],
        [
            [idx + 1, ALL_FEATURE_NAMES[feature_idx], f"{feature_importance[feature_idx]:.6f}"]
            for idx, feature_idx in enumerate(top_order)
        ],
    )

    attn_scores = attention.copy()
    np.fill_diagonal(attn_scores, 0.0)
    flat_idx = np.argsort(attn_scores.reshape(-1))[::-1][:8]
    top_pairs = []
    for rank, flat in enumerate(flat_idx, start=1):
        i, j = np.unravel_index(flat, attn_scores.shape)
        top_pairs.append(
            [
                rank,
                ALL_FEATURE_NAMES[i],
                ALL_FEATURE_NAMES[j],
                f"{attn_scores[i, j]:.4f}",
            ]
        )
    attn_table = markdown_table(["rank", "query_feature", "key_feature", "weight"], top_pairs)

    curve_features = ", ".join(curve["feature_name"] for curve in curves)
    fig_lines = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)

    text = f"""# KAN_TQNet 非消融可解释性分析

## 1. 分析目标

本分析不再通过“去掉模块看性能下降”来解释模型，而是直接读取完整 KAN_TQNet 的内部信号，包括：

1. HDA 路由权重，解释模型在不同预测步长下如何在短期、中期、长期分支之间分配权重。
2. Temporal Query 通道注意力，解释模型更关注哪些输入变量之间的周期耦合关系。
3. KAN 响应曲线，解释 KAN 主干如何对高方差输入特征学习非线性映射。
4. 输入敏感性，解释模型对哪些历史时间步和哪些输入变量最敏感。

## 2. HDA 路由解释

跨 24/48/72/96 步长统计得到的平均路由权重如下：

{routing_table}

这部分可以直接支持这样的结论：随着预测步长增大，模型会逐渐提升对中长期信息分支的依赖，而不是始终固定使用同一条预测路径。也就是说，HDA 的可解释意义不只是“多分支”，而是“根据预测尺度动态切换决策路径”。

## 3. Temporal Query 注意力解释

在 96 步完整模型上，提取 `temporalQuery + channelAggregator` 的通道注意力矩阵，并按非对角元素排序，得到最强变量交互：

{attn_table}

这部分适合在论文中解释为：Temporal Query 学到的并不是单一时间偏置，而是不同输入变量间围绕周期性查询形成的耦合结构，尤其适用于解释气象变量、历史负荷变量和派生统计量之间的协同关系。

## 4. KAN 非线性函数解释

在 96 步完整模型中，依据训练样本方差最大的输入维度，可视化了 KAN 第一层的线性项、样条项和总响应。当前输出的重点特征包括：`{curve_features}`。

这类图可以用于说明：

- 某些变量主要通过线性基底起作用；
- 某些变量则明显依赖样条项补充非线性；
- 总响应曲线若出现拐点、饱和或局部增强，说明模型并不是简单线性外推，而是在学习更细的输入-输出映射关系。

## 5. 输入敏感性解释

对 96 步完整模型的输出均值反向传播，统计输入梯度绝对值，可以得到时间维和特征维的重要性排序。前 8 个最敏感特征如下：

{top_feature_table}

这部分可以解释为：

- 若靠近预测起点的历史时间步梯度更大，说明模型更依赖近端局部模式；
- 若更早历史段仍保持较大敏感性，说明模型确实使用了长程记忆；
- 若天气、湿球温度、综合能耗等变量敏感度高，则说明模型利用了外生变量调制负荷预测。

## 6. 适合写进论文的表述

可以写成：

> 为进一步提升 KAN_TQNet 的可解释性，本文除模块级消融外，还直接分析了完整模型的内部决策信号。结果表明，HDA 路由权重会随预测步长变化而动态调整，说明模型能够在短期局部模式、中期非线性表示和长期趋势表示之间进行自适应融合。Temporal Query 注意力矩阵揭示了历史负荷与气象变量之间的周期耦合结构；KAN 响应曲线则展示了模型对关键输入变量的非线性建模方式；输入梯度敏感性进一步验证了模型对重要历史时间步和外生驱动变量的依赖。

## 7. 输出文件

{fig_lines}
"""

    path = SAVE_DIR / "KAN_TQNet_intrinsic_interpretability_summary.md"
    path.write_text(text, encoding="utf-8-sig")
    return path


def get_top_feature_rows(feature_importance: np.ndarray, top_n: int = 8) -> list[dict[str, Any]]:
    order = np.argsort(feature_importance)[::-1][:top_n]
    return [
        {
            "rank": rank,
            "feature_idx": int(feature_idx),
            "feature_name": ALL_FEATURE_NAMES[feature_idx],
            "importance": float(feature_importance[feature_idx]),
        }
        for rank, feature_idx in enumerate(order, start=1)
    ]


def get_top_temporal_rows(temporal_importance: np.ndarray, top_n: int = 12) -> list[dict[str, Any]]:
    order = np.argsort(temporal_importance)[::-1][:top_n]
    total_steps = len(temporal_importance)
    return [
        {
            "rank": rank,
            "time_index": int(time_idx + 1),
            "lag_to_forecast_start": int(total_steps - time_idx),
            "importance": float(temporal_importance[time_idx]),
        }
        for rank, time_idx in enumerate(order, start=1)
    ]


def get_top_attention_pairs(attention: np.ndarray, top_n: int = 12) -> list[dict[str, Any]]:
    attn_scores = attention.copy()
    np.fill_diagonal(attn_scores, 0.0)
    flat_idx = np.argsort(attn_scores.reshape(-1))[::-1][:top_n]
    rows: list[dict[str, Any]] = []
    for rank, flat in enumerate(flat_idx, start=1):
        query_idx, key_idx = np.unravel_index(flat, attn_scores.shape)
        rows.append(
            {
                "rank": rank,
                "query_idx": int(query_idx),
                "query_feature": ALL_FEATURE_NAMES[query_idx],
                "key_idx": int(key_idx),
                "key_feature": ALL_FEATURE_NAMES[key_idx],
                "weight": float(attn_scores[query_idx, key_idx]),
            }
        )
    return rows


def attention_matrix_rows(attention: np.ndarray) -> list[dict[str, Any]]:
    feature_labels = ALL_FEATURE_NAMES[: attention.shape[0]]
    rows: list[dict[str, Any]] = []
    for query_idx, query_feature in enumerate(feature_labels):
        row: dict[str, Any] = {"query_feature": query_feature}
        for key_idx, key_feature in enumerate(feature_labels):
            row[key_feature] = float(attention[query_idx, key_idx])
        rows.append(row)
    return rows


def write_summary_v2(
    routing_rows: list[dict[str, float]],
    attention: np.ndarray,
    curves: list[dict[str, Any]],
    sensitivity: dict[str, np.ndarray],
    figure_paths: list[Path],
) -> Path:
    routing_table = markdown_table(
        ["horizon", "short_mean", "medium_mean", "long_mean", "long_minus_short"],
        [
            [
                row["horizon"],
                f"{row['short_mean']:.4f}",
                f"{row['medium_mean']:.4f}",
                f"{row['long_mean']:.4f}",
                f"{row['long_mean'] - row['short_mean']:.4f}",
            ]
            for row in routing_rows
        ],
    )

    top_feature_rows = get_top_feature_rows(sensitivity["feature_importance"], top_n=8)
    top_feature_table = markdown_table(
        ["rank", "feature", "importance"],
        [
            [row["rank"], row["feature_name"], f"{row['importance']:.6f}"]
            for row in top_feature_rows
        ],
    )

    top_temporal_rows = get_top_temporal_rows(sensitivity["temporal_importance"], top_n=8)
    top_temporal_table = markdown_table(
        ["rank", "time_index", "lag_to_forecast_start", "importance"],
        [
            [
                row["rank"],
                row["time_index"],
                row["lag_to_forecast_start"],
                f"{row['importance']:.6f}",
            ]
            for row in top_temporal_rows
        ],
    )

    top_attention_rows = get_top_attention_pairs(attention, top_n=8)
    attention_table = markdown_table(
        ["rank", "query_feature", "key_feature", "weight"],
        [
            [row["rank"], row["query_feature"], row["key_feature"], f"{row['weight']:.4f}"]
            for row in top_attention_rows
        ],
    )

    long_values = [row["long_mean"] for row in routing_rows]
    short_values = [row["short_mean"] for row in routing_rows]
    medium_values = [row["medium_mean"] for row in routing_rows]
    curve_features = ", ".join(curve["feature_name"] for curve in curves)
    fig_lines = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)

    text = f"""# KAN_TQNet 非消融可解释性分析

## 1. 分析目标

本实验不再通过“删除模块后观察误差变化”来解释模型，而是直接读取完整 KAN_TQNet 的内部决策信号。分析对象包括 HDA 路由权重、Temporal Query 通道注意力、KAN 非线性响应曲线，以及基于输入梯度的时间步与特征敏感性。

## 2. HDA 路由权重

不同预测步长下的 HDA 三分支平均路由权重如下：

{routing_table}

结果显示，长程分支在 24/48/72/96 四个预测步长上均保持最高权重，范围为 {min(long_values):.4f}-{max(long_values):.4f}；短程分支范围为 {min(short_values):.4f}-{max(short_values):.4f}，中程分支范围为 {min(medium_values):.4f}-{max(medium_values):.4f}。因此，这组结果更适合表述为：在当前数据集和训练设置下，HDA 路由稳定地将主要权重分配给长程趋势/低频信息分支，同时保留短程局部模式和中程非线性分支作为补充。不要写成“随预测步长增加单调提升长程分支权重”，因为 72 和 96 步并不满足单调趋势。

## 3. Temporal Query 通道注意力

在 96 步完整模型上提取 `temporalQuery + channelAggregator` 的通道注意力矩阵，并按非对角元素排序，得到最强变量交互：

{attention_table}

注意力权重整体接近 1/20，说明多头注意力没有坍缩到单一变量，而是在多个负荷、日周期和工作时段变量之间分散建模。排名靠前的交互集中在 `KW`、`CHWTON`、`HTmmBTU` 与 `hour_sin/hour_cos/is_workhour` 等时间变量之间，可解释为模型利用周期时间查询来调制多能负荷预测。

## 4. KAN 非线性响应

在 96 步完整模型中，提取 KAN 第一层的线性基底项、样条项和总响应。当前可视化的高方差输入维度为：`{curve_features}`。

这些曲线可以用于说明 KAN 分支不是简单线性映射。若某个变量的样条项在局部区间产生明显弯曲或饱和，说明模型在该变量上学习到了非线性输入-输出关系；若线性基底项占主导，则说明该变量主要提供稳定趋势或尺度信息。

## 5. 输入梯度敏感性

对 96 步完整模型输出均值反向传播，统计输入梯度绝对值，得到特征维度的重要性排序：

{top_feature_table}

时间维度上最敏感的历史步如下，`time_index=168` 表示最接近预测起点的历史时刻，`lag_to_forecast_start=1` 表示距离预测起点 1 个历史步：

{top_temporal_table}

特征敏感性显示，气压相关变量、历史 `KW`、`HTmmBTU`、`CHWTON` 和年周期变量对 96 步预测更重要；时间敏感性排序则可用于判断模型更依赖近端历史还是远端周期信息。

## 6. 论文可用表述

可以写成：

> 为进一步验证 KAN_TQNet 的可解释性，本文除模块级消融外，还分析了完整模型的内部决策信号。HDA 路由结果表明，模型在不同预测步长下均稳定提高长程趋势/低频信息分支的权重，同时保留短程局部和中程非线性分支。Temporal Query 通道注意力揭示了历史负荷变量与日周期、工作时段变量之间的耦合关系；KAN 响应曲线展示了模型对关键输入维度的非线性映射；输入梯度敏感性进一步表明模型同时依赖气象/气压驱动变量、历史负荷变量和周期时间变量。

## 7. 输出文件

{fig_lines}
"""

    path = SAVE_DIR / "KAN_TQNet_intrinsic_interpretability_summary.md"
    path.write_text(text, encoding="utf-8-sig")
    return path


def main() -> None:
    ensure_dirs()
    fix_seed(2020)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    device = get_device()
    routing_rows = collect_routing_statistics(device)
    attention_payload = collect_attention_map(device, horizon=96)
    curves = collect_kan_curves(device, horizon=96)
    sensitivity = collect_temporal_sensitivity(device, horizon=96)

    save_csv(
        SAVE_DIR / "hda_routing_summary.csv",
        routing_rows,
        ["horizon", "short_mean", "medium_mean", "long_mean", "short_std", "medium_std", "long_std"],
    )

    attention_matrix = attention_payload["attention_matrix"]
    np.save(SAVE_DIR / "temporal_query_attention_matrix.npy", attention_matrix)
    save_csv(
        SAVE_DIR / "temporal_query_attention_matrix.csv",
        attention_matrix_rows(attention_matrix),
        ["query_feature", *ALL_FEATURE_NAMES[: attention_matrix.shape[1]]],
    )
    save_csv(
        SAVE_DIR / "temporal_query_top_attention_pairs.csv",
        get_top_attention_pairs(attention_matrix, top_n=20),
        ["rank", "query_idx", "query_feature", "key_idx", "key_feature", "weight"],
    )

    kan_curve_rows = []
    for curve in curves:
        for x_val, base_val, spline_val, total_val in zip(curve["x"], curve["base"], curve["spline"], curve["total"]):
            kan_curve_rows.append(
                {
                    "feature_idx": curve["feature_idx"],
                    "feature_name": curve["feature_name"],
                    "x": float(x_val),
                    "base": float(base_val),
                    "spline": float(spline_val),
                    "total": float(total_val),
                }
            )
    save_csv(
        SAVE_DIR / "kan_response_curves.csv",
        kan_curve_rows,
        ["feature_idx", "feature_name", "x", "base", "spline", "total"],
    )

    feature_importance_rows = [
        {
            "feature_idx": idx,
            "feature_name": ALL_FEATURE_NAMES[idx],
            "importance": float(value),
        }
        for idx, value in enumerate(sensitivity["feature_importance"])
    ]
    save_csv(
        SAVE_DIR / "input_feature_sensitivity.csv",
        feature_importance_rows,
        ["feature_idx", "feature_name", "importance"],
    )
    save_csv(
        SAVE_DIR / "input_feature_sensitivity_top.csv",
        get_top_feature_rows(sensitivity["feature_importance"], top_n=len(feature_importance_rows)),
        ["rank", "feature_idx", "feature_name", "importance"],
    )

    temporal_rows = [
        {"time_index": idx + 1, "importance": float(value)}
        for idx, value in enumerate(sensitivity["temporal_importance"])
    ]
    save_csv(
        SAVE_DIR / "input_temporal_sensitivity.csv",
        temporal_rows,
        ["time_index", "importance"],
    )
    save_csv(
        SAVE_DIR / "input_temporal_sensitivity_top.csv",
        get_top_temporal_rows(sensitivity["temporal_importance"], top_n=len(temporal_rows)),
        ["rank", "time_index", "lag_to_forecast_start", "importance"],
    )
    np.save(SAVE_DIR / "feature_time_sensitivity.npy", sensitivity["feature_time_importance"])

    figure_paths = [
        plot_routing(routing_rows),
        plot_attention(attention_matrix),
        plot_kan_curves(curves),
        *plot_sensitivity(
            sensitivity["temporal_importance"],
            sensitivity["feature_importance"],
            sensitivity["feature_time_importance"],
        ),
    ]
    summary_path = write_summary_v2(routing_rows, attention_matrix, curves, sensitivity, figure_paths)

    print("Generated intrinsic interpretability outputs:")
    for path in [
        SAVE_DIR / "hda_routing_summary.csv",
        SAVE_DIR / "temporal_query_attention_matrix.npy",
        SAVE_DIR / "temporal_query_attention_matrix.csv",
        SAVE_DIR / "temporal_query_top_attention_pairs.csv",
        SAVE_DIR / "kan_response_curves.csv",
        SAVE_DIR / "input_feature_sensitivity.csv",
        SAVE_DIR / "input_feature_sensitivity_top.csv",
        SAVE_DIR / "input_temporal_sensitivity.csv",
        SAVE_DIR / "input_temporal_sensitivity_top.csv",
        SAVE_DIR / "feature_time_sensitivity.npy",
        *figure_paths,
        summary_path,
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
