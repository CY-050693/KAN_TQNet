from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
SAVE_DIR = OUTPUT_ROOT / "KAN_TQNet_interpretability_efficiency"
FIG_DIR = SAVE_DIR / "figures"

CORE_OVERALL = ROOT / "core_ablation_new_all_overall.csv"
CORE_CHANNELS = ROOT / "core_ablation_new_all_channels.csv"
COMPLEXITY = OUTPUT_ROOT / "KAN_TQNet_complexity_summary.csv"

HORIZONS = [24, 48, 72, 96]
CHANNELS = ["electricity", "cooling", "heating"]

MODULE_LABELS = {
    "core_no_kan": "KAN backbone",
    "core_no_tq": "Temporal Query",
    "core_no_freq": "Frequency branch",
    "core_no_trend": "Trend residual",
    "core_no_hda": "HDA router",
    "core_no_timemixer": "TimeMixer refiner",
}

MODULE_LABELS_CN = {
    "core_no_kan": "KAN 主干",
    "core_no_tq": "Temporal Query",
    "core_no_freq": "频域分支",
    "core_no_trend": "趋势残差分解",
    "core_no_hda": "HDA 路由",
    "core_no_timemixer": "TimeMixer 精修",
}

CHANNEL_LABELS_CN = {
    "electricity": "电负荷",
    "cooling": "冷负荷",
    "heating": "热负荷",
}


def ensure_dirs() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in [CORE_OVERALL, CORE_CHANNELS, COMPLEXITY] if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")

    overall = pd.read_csv(CORE_OVERALL, encoding="utf-8-sig")
    channels = pd.read_csv(CORE_CHANNELS, encoding="utf-8-sig")
    complexity = pd.read_csv(COMPLEXITY, encoding="utf-8-sig")
    return overall, channels, complexity


def signed_fmt(value: float, decimals: int = 4) -> str:
    return f"{value:+.{decimals}f}"


def pct_fmt(value: float, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}%"


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = SAVE_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_module_contribution(overall: pd.DataFrame) -> pd.DataFrame:
    subset = overall[overall["experiment_tag"].isin(MODULE_LABELS)].copy()
    full = overall[overall["experiment_tag"] == "core_full"][
        ["horizon", "selected_overall_mae", "selected_overall_rmse", "selected_overall_mape"]
    ].rename(
        columns={
            "selected_overall_mae": "full_mae",
            "selected_overall_rmse": "full_rmse",
            "selected_overall_mape": "full_mape",
        }
    )
    subset = subset.merge(full, on="horizon", how="left")

    subset["module"] = subset["experiment_tag"].map(MODULE_LABELS)
    subset["module_cn"] = subset["experiment_tag"].map(MODULE_LABELS_CN)
    subset["delta_mae_pct"] = subset["delta_mae_vs_full"] / subset["full_mae"] * 100.0
    subset["delta_rmse_pct"] = subset["delta_rmse_vs_full"] / subset["full_rmse"] * 100.0
    subset["impact_type"] = subset["delta_mae_vs_full"].apply(
        lambda value: "harmful_when_removed" if value > 0 else "neutral_or_negative_gain"
    )
    subset["rank_by_horizon"] = subset.groupby("horizon")["delta_mae_vs_full"].rank(
        ascending=False, method="dense"
    ).astype(int)

    columns = [
        "horizon",
        "experiment_tag",
        "module",
        "module_cn",
        "full_mae",
        "selected_overall_mae",
        "delta_mae_vs_full",
        "delta_mae_pct",
        "selected_overall_rmse",
        "delta_rmse_vs_full",
        "delta_rmse_pct",
        "selected_overall_mape",
        "delta_mape_vs_full",
        "selected_overall_corr",
        "impact_type",
        "rank_by_horizon",
    ]
    return subset[columns].sort_values(["horizon", "rank_by_horizon", "experiment_tag"])


def build_channel_contribution(channels: pd.DataFrame) -> pd.DataFrame:
    subset = channels[channels["experiment_tag"].isin(MODULE_LABELS)].copy()
    subset["module"] = subset["experiment_tag"].map(MODULE_LABELS)
    subset["module_cn"] = subset["experiment_tag"].map(MODULE_LABELS_CN)
    subset["channel_cn"] = subset["channel"].map(CHANNEL_LABELS_CN)
    subset["rank_by_horizon_channel"] = subset.groupby(["horizon", "channel"])["delta_mae_vs_full"].rank(
        ascending=False, method="dense"
    ).astype(int)
    columns = [
        "horizon",
        "experiment_tag",
        "module",
        "module_cn",
        "channel",
        "channel_cn",
        "mae",
        "delta_mae_vs_full",
        "rmse",
        "mape",
        "corr",
        "rank_by_horizon_channel",
    ]
    return subset[columns].sort_values(
        ["horizon", "channel", "rank_by_horizon_channel", "experiment_tag"]
    )


def build_efficiency_summary(overall: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    full = overall[overall["experiment_tag"] == "core_full"][
        [
            "horizon",
            "selected_overall_mae",
            "selected_overall_rmse",
            "selected_overall_mape",
            "selected_overall_corr",
            "electricity_mae",
            "cooling_mae",
            "heating_mae",
        ]
    ].copy()
    merged = complexity.merge(full, on="horizon", how="left")
    merged["parameters_m"] = merged["parameters_raw"] / 1_000_000.0
    merged["flops_m"] = merged["flops_raw_forward_batch1"] / 1_000_000.0
    merged["mae_per_million_params"] = merged["selected_overall_mae"] / merged["parameters_m"]
    merged["mae_per_mflop"] = merged["selected_overall_mae"] / merged["flops_m"]
    merged["pred_steps_per_mflop"] = merged["horizon"] / merged["flops_m"]
    merged["flops_growth_pct_vs_h24"] = (
        (merged["flops_raw_forward_batch1"] / merged["flops_raw_forward_batch1"].iloc[0]) - 1.0
    ) * 100.0
    merged["params_growth_pct_vs_h24"] = (
        (merged["parameters_raw"] / merged["parameters_raw"].iloc[0]) - 1.0
    ) * 100.0
    columns = [
        "horizon",
        "parameters_raw",
        "parameters_m",
        "flops_raw_forward_batch1",
        "flops_m",
        "epoch_time_seconds",
        "selected_overall_mae",
        "selected_overall_rmse",
        "selected_overall_mape",
        "selected_overall_corr",
        "electricity_mae",
        "cooling_mae",
        "heating_mae",
        "mae_per_million_params",
        "mae_per_mflop",
        "pred_steps_per_mflop",
        "flops_growth_pct_vs_h24",
        "params_growth_pct_vs_h24",
        "notes",
    ]
    return merged[columns]


def plot_module_heatmap(module_df: pd.DataFrame) -> Path:
    if plt is None:
        return Path()

    matrix = module_df.pivot(index="module", columns="horizon", values="delta_mae_vs_full")
    matrix = matrix.loc[list(MODULE_LABELS.values()), HORIZONS]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    max_abs = max(abs(matrix.min().min()), abs(matrix.max().max()))
    image = ax.imshow(matrix.values, cmap="RdBu_r", aspect="auto", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(HORIZONS)), HORIZONS)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Module ablation impact on MAE (removed module - full model)")

    for i, module in enumerate(matrix.index):
        for j, horizon in enumerate(HORIZONS):
            value = matrix.loc[module, horizon]
            color = "white" if abs(value) > max_abs * 0.55 else "black"
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta MAE")
    fig.tight_layout()
    path = FIG_DIR / "Fig_interpretability_module_delta_mae_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_channel_bars(channel_df: pd.DataFrame) -> Path:
    if plt is None:
        return Path()

    avg = (
        channel_df.groupby(["module", "channel"], as_index=False)["delta_mae_vs_full"]
        .mean()
        .pivot(index="module", columns="channel", values="delta_mae_vs_full")
    )
    avg = avg.loc[list(MODULE_LABELS.values()), CHANNELS]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = range(len(avg.index))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    for idx, channel in enumerate(CHANNELS):
        positions = [pos + offsets[idx] for pos in x]
        ax.bar(positions, avg[channel].values, width=width, label=channel.title(), color=colors[idx])

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(list(x), avg.index, rotation=25, ha="right")
    ax.set_ylabel("Mean delta MAE across horizons")
    ax.set_title("Channel-wise module contribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    path = FIG_DIR / "Fig_interpretability_channel_delta_mae_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_efficiency_tradeoff(efficiency_df: pd.DataFrame) -> Path:
    if plt is None:
        return Path()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    axes[0].plot(efficiency_df["horizon"], efficiency_df["parameters_m"], marker="o", label="Params (M)")
    axes[0].plot(efficiency_df["horizon"], efficiency_df["flops_m"], marker="s", label="FLOPs (M)")
    axes[0].set_xlabel("Forecast horizon")
    axes[0].set_ylabel("Million")
    axes[0].set_title("Complexity growth")
    axes[0].legend(frameon=False)
    axes[0].grid(linestyle="--", linewidth=0.6, alpha=0.35)

    scatter = axes[1].scatter(
        efficiency_df["flops_m"],
        efficiency_df["selected_overall_mae"],
        s=efficiency_df["parameters_m"] * 420,
        c=efficiency_df["horizon"],
        cmap="viridis",
        edgecolor="#222222",
        linewidth=0.8,
    )
    for _, row in efficiency_df.iterrows():
        axes[1].annotate(f"H{int(row['horizon'])}", (row["flops_m"], row["selected_overall_mae"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel("Forward FLOPs (M), batch=1")
    axes[1].set_ylabel("Overall MAE")
    axes[1].set_title("Accuracy-complexity tradeoff")
    axes[1].grid(linestyle="--", linewidth=0.6, alpha=0.35)
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Horizon")

    fig.tight_layout()
    path = FIG_DIR / "Fig_efficiency_tradeoff.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, columns: Iterable[str], float_digits: int = 4) -> str:
    view = df.loc[:, list(columns)].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{value:.{float_digits}f}")
    view = view.astype(str)
    widths = {
        col: max(len(col), *(len(value) for value in view[col].tolist()))
        for col in view.columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in view.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_markdown(
    module_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    avg_module = (
        module_df.groupby(["experiment_tag", "module_cn"], as_index=False)
        .agg(
            mean_delta_mae=("delta_mae_vs_full", "mean"),
            max_delta_mae=("delta_mae_vs_full", "max"),
            mean_delta_pct=("delta_mae_pct", "mean"),
        )
        .sort_values("mean_delta_mae", ascending=False)
    )
    avg_channel = (
        channel_df.groupby(["experiment_tag", "module_cn", "channel", "channel_cn"], as_index=False)
        .agg(mean_delta_mae=("delta_mae_vs_full", "mean"))
        .sort_values(["channel", "mean_delta_mae"], ascending=[True, False])
    )

    h96 = module_df[module_df["horizon"] == 96].sort_values("delta_mae_vs_full", ascending=False)
    top_h96 = h96.iloc[0]
    top_avg = avg_module.iloc[0]
    eff_first = efficiency_df.iloc[0]
    eff_last = efficiency_df.iloc[-1]
    flops_growth = eff_last["flops_growth_pct_vs_h24"]
    params_growth = eff_last["params_growth_pct_vs_h24"]

    existing_figures = [path for path in figure_paths if path and path.exists()]
    if existing_figures:
        fig_lines = "\n".join(f"- `{path.relative_to(ROOT)}`" for path in existing_figures)
    else:
        fig_lines = "- 当前 Python 环境未安装 Matplotlib，本次仅生成 CSV 和 Markdown 分析；安装 Matplotlib 后重新运行脚本即可生成图片。"

    content = f"""# KAN_TQNet 可解释性分析与计算效率分析

## 1. 分析依据

本分析不重新训练模型，而是复用项目中已经生成的实验结果：`core_ablation_new_all_overall.csv`、`core_ablation_new_all_channels.csv` 和 `output/KAN_TQNet_complexity_summary.csv`。其中，模块可解释性主要来自“删除某模块后的性能变化”，计算效率主要来自参数量、单样本前向 FLOPs 和训练日志中的 epoch 耗时。

源码层面的对应关系如下：`KANLinear/KANBlock` 对应 KAN 主干；`temporalQuery` 与 `channelAggregator` 对应 Temporal Query；`_build_wavelet_feature`、`_build_frequency_feature` 和 `fusion_gate` 对应频域分支；`_moving_average` 与 `trend_head` 对应趋势残差；`hda_router` 对应短期、中期、长期三路输出的 HDA 自适应融合；`TimeMixerRefiner` 对应低频长程分支前的多尺度精修。

## 2. 可解释性分析设计

KAN_TQNet 的可解释性可以分成三层：

1. 结构可解释性：模型显式拆成时间查询、非线性 KAN 映射、频域/小波特征、趋势残差、HDA 路由和 TimeMixer 精修，因此每个结构模块都有明确的时间序列含义。
2. 消融可解释性：用 `Delta MAE = MAE(删除模块) - MAE(完整模型)` 度量模块贡献。Delta 为正说明删除该模块会损害性能，该模块对预测有正贡献；Delta 为负说明该模块在该设置下没有带来净收益，可能存在冗余或过拟合。
3. 通道可解释性：对电、冷、热三个预测目标分别统计 Delta MAE，判断模块更主要服务于哪类负荷。

## 3. 模块贡献结论

跨 24/48/72/96 步长取平均后，模块贡献排序如下：

{markdown_table(avg_module, ["module_cn", "mean_delta_mae", "max_delta_mae", "mean_delta_pct"])}

最稳定的正贡献来自 `{top_avg["module_cn"]}`，平均 Delta MAE 为 `{top_avg["mean_delta_mae"]:.4f}`。在 96 步预测中，影响最大的删除项是 `{top_h96["module_cn"]}`，删除后 MAE 变化 `{signed_fmt(top_h96["delta_mae_vs_full"])}`，相对完整模型变化 `{pct_fmt(top_h96["delta_mae_pct"])}`。这说明模型在长预测步长下更依赖自适应路由和多尺度长程建模，而不是只依赖单一主干。

按单步长的模块 Delta MAE 如下：

{markdown_table(module_df, ["horizon", "module_cn", "selected_overall_mae", "delta_mae_vs_full", "delta_mae_pct", "rank_by_horizon"])}

需要注意，`core_no_freq` 在 96 步上出现负 Delta MAE，这说明频域分支在该次实验设置下并非所有预测长度都稳定增益。论文表述中建议写成“频域分支提供可解释的频率/小波证据，但其收益依赖预测长度和数据分布”，不要夸大为全局必然增益。

## 4. 通道贡献结论

通道层面平均 Delta MAE 如下：

{markdown_table(avg_channel, ["module_cn", "channel_cn", "mean_delta_mae"])}

从通道结果看，HDA 与 TimeMixer 对冷负荷的提升最明显，说明冷负荷的长程波动和多尺度成分更强；KAN 和趋势残差对电负荷的贡献在长步长中更明显，说明非线性映射与趋势分解对电负荷外推有帮助；热负荷整体数值尺度较小，因此 Delta MAE 绝对值也更小，建议在论文中同时报告百分比或归一化误差，避免被绝对尺度掩盖。

## 5. 计算效率分析

复杂度汇总如下：

{markdown_table(efficiency_df, ["horizon", "parameters_m", "flops_m", "epoch_time_seconds", "selected_overall_mae", "mae_per_mflop", "pred_steps_per_mflop"])}

从 24 步增加到 96 步时，参数量从 `{eff_first["parameters_m"]:.4f}M` 增加到 `{eff_last["parameters_m"]:.4f}M`，增长 `{params_growth:.2f}%`；单样本前向 FLOPs 从 `{eff_first["flops_m"]:.4f}M` 增加到 `{eff_last["flops_m"]:.4f}M`，增长 `{flops_growth:.2f}%`。预测步长扩大 4 倍，但 FLOPs 基本保持在 12M 左右，说明 KAN_TQNet 的主要计算开销来自共享编码、频域特征和路由分支，输出长度增加只带来很小的边际开销。

效率结论可以写成：KAN_TQNet 不是通过大幅增加计算量来换取长步预测能力，而是通过共享特征编码、低频/高频分支和 HDA 路由复用特征。其优势是长预测步长下计算量增长缓慢；代价是模型包含多个功能分支，结构比普通 MLP 或单分支时序模型更复杂。

## 6. 论文可用表述

可解释性表述：

> 为验证 KAN_TQNet 各组成模块的作用，本文设计了模块级消融实验，并以删除模块后的 MAE 增量作为贡献度指标。实验结果表明，HDA 路由和 TimeMixer 精修在中长预测步长上贡献最显著，说明模型需要根据预测尺度自适应融合短期局部模式、中期非线性表示和长期趋势表示。KAN 主干、Temporal Query、趋势残差与频域分支分别提供非线性函数拟合、周期查询、趋势外推和频率结构建模能力，从结构和实验两方面提升模型解释性。

计算效率表述：

> 在输入长度为 168、batch size 为 1 的前向计算条件下，KAN_TQNet 在 24/48/72/96 步预测上的 FLOPs 约为 11.89M/11.94M/11.98M/12.03M，参数量约为 0.55M/0.57M/0.59M/0.62M。随着预测步长从 24 增至 96，FLOPs 仅增长约 {flops_growth:.2f}%，表明该模型通过共享编码与分支复用控制了长步预测的计算开销。

## 7. 输出文件

本脚本生成的图表如下：

{fig_lines}

CSV 汇总文件：

- `output/KAN_TQNet_interpretability_efficiency/interpretability_module_contributions.csv`
- `output/KAN_TQNet_interpretability_efficiency/interpretability_channel_contributions.csv`
- `output/KAN_TQNet_interpretability_efficiency/efficiency_summary.csv`
"""

    path = SAVE_DIR / "KAN_TQNet_interpretability_efficiency_analysis.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    if plt is not None:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False

    overall, channels, complexity = read_inputs()
    module_df = build_module_contribution(overall)
    channel_df = build_channel_contribution(channels)
    efficiency_df = build_efficiency_summary(overall, complexity)

    module_csv = write_csv(module_df, "interpretability_module_contributions.csv")
    channel_csv = write_csv(channel_df, "interpretability_channel_contributions.csv")
    efficiency_csv = write_csv(efficiency_df, "efficiency_summary.csv")

    figure_paths = [
        plot_module_heatmap(module_df),
        plot_channel_bars(channel_df),
        plot_efficiency_tradeoff(efficiency_df),
    ]
    markdown_path = write_markdown(module_df, channel_df, efficiency_df, figure_paths)

    print("Generated KAN_TQNet interpretability and efficiency analysis:")
    for path in [module_csv, channel_csv, efficiency_csv, *figure_paths, markdown_path]:
        if path:
            print(path.relative_to(ROOT))
    if plt is None:
        print("Matplotlib is not installed in this Python environment; figure generation was skipped.")


if __name__ == "__main__":
    main()
