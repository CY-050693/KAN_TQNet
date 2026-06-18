from __future__ import annotations

import csv
import math
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\30217\Desktop\负荷预测（原件）\负荷预测（原件）")
OUTPUT_DIR = ROOT / "paper_figures_300dpi"
DPI = 300

COLORS = {
    "primary": "#0b3c5d",
    "accent": "#d95f02",
    "accent2": "#1b9e77",
    "muted": "#6b7280",
    "grid": "#d1d5db",
    "danger": "#b91c1c",
    "safe": "#047857",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def read_prediction_csv(path: Path, n: int = 336) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.iloc[:n].copy()
    for col in ["test_y", "predicted_values"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["test_y", "predicted_values"]).reset_index(drop=True)
    return df


def parse_final_metric(data_txt: Path, metric_key: str = "mae") -> float:
    patterns = [
        re.compile(rf"final_hybrid_best {metric_key}\s+([0-9.]+)", re.IGNORECASE),
        re.compile(rf"final test {metric_key}\s+([0-9.]+)", re.IGNORECASE),
    ]
    for line in data_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Could not parse {metric_key} from {data_txt}")


def parse_progressive_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stage_order = [
        "P0_backbone_mlp",
        "P1_plus_kan",
        "P2_plus_channel",
        "P3_plus_tq",
        "P4_plus_freq",
        "P5_plus_trend",
        "P6_plus_hda",
        "P7_plus_ele_refine",
        "P8_plus_timemixer",
    ]
    stage_label = {
        "P0_backbone_mlp": "P0",
        "P1_plus_kan": "P1",
        "P2_plus_channel": "P2",
        "P3_plus_tq": "P3",
        "P4_plus_freq": "P4",
        "P5_plus_trend": "P5",
        "P6_plus_hda": "P6",
        "P7_plus_ele_refine": "P7",
        "P8_plus_timemixer": "P8",
    }

    for horizon in [24, 48, 72]:
        for stage in stage_order:
            run_dir = ROOT / "output" / f"KAN_TQNet_{horizon}_168_{stage}_s0"
            data_txt = run_dir / "result" / "data.txt"
            if not data_txt.exists():
                continue
            rows.append(
                {
                    "horizon": horizon,
                    "stage": stage,
                    "stage_label": stage_label[stage],
                    "mae": parse_final_metric(data_txt, "mae"),
                }
            )
    if not rows:
        raise ValueError("No progressive ablation metrics found.")
    return pd.DataFrame(rows)


def parse_baseline_metrics() -> pd.DataFrame:
    experiments = {
        "KAN_TQNet": {
            24: ROOT / "output" / "KAN_TQNet_24_168_cmp_KAN_TQNet_24" / "result" / "data.txt",
            48: ROOT / "output" / "KAN_TQNet_48_168_cmp_KAN_TQNet_48" / "result" / "data.txt",
            72: ROOT / "output" / "KAN_TQNet_72_168_cmp_KAN_TQNet_72" / "result" / "data.txt",
            96: ROOT / "output" / "KAN_TQNet_96_168_cmp_KAN_TQNet_96" / "result" / "data.txt",
        },
        "TimeMixer": {
            24: ROOT / "output" / "TimeMixer_24_168_cmp_TimeMixer_24" / "result" / "data.txt",
            48: ROOT / "output" / "TimeMixer_48_168_cmp_TimeMixer_48" / "result" / "data.txt",
            72: ROOT / "output" / "TimeMixer_72_168_cmp_TimeMixer_72" / "result" / "data.txt",
            96: ROOT / "output" / "TimeMixer_96_168_cmp_TimeMixer_96" / "result" / "data.txt",
        },
        "PatchTST": {
            24: ROOT / "output" / "PatchTST_24_168_cmp_PatchTST_24" / "result" / "data.txt",
            48: ROOT / "output" / "PatchTST_48_168_cmp_PatchTST_48" / "result" / "data.txt",
            72: ROOT / "output" / "PatchTST_72_168_cmp_PatchTST_72" / "result" / "data.txt",
            96: ROOT / "output" / "PatchTST_96_168_cmp_PatchTST_96" / "result" / "data.txt",
        },
        "TimesNet": {
            24: ROOT / "output" / "TimesNet_24_168_cmp_TimesNet_24" / "result" / "data.txt",
            48: ROOT / "output" / "TimesNet_48_168_cmp_TimesNet_48" / "result" / "data.txt",
            72: ROOT / "output" / "TimesNet_72_168_cmp_TimesNet_72" / "result" / "data.txt",
            96: ROOT / "output" / "TimesNet_96_168_cmp_TimesNet_96" / "result" / "data.txt",
        },
        "TSMixer": {
            24: ROOT / "output" / "TSMixer_24_168_cmp_TSMixer_24" / "result" / "data.txt",
            48: ROOT / "output" / "TSMixer_48_168_cmp_TSMixer_48" / "result" / "data.txt",
            72: ROOT / "output" / "TSMixer_72_168_cmp_TSMixer_72" / "result" / "data.txt",
            96: ROOT / "output" / "TSMixer_96_168_cmp_TSMixer_96" / "result" / "data.txt",
        },
    }

    rows: list[dict[str, object]] = []
    for model_name, horizons in experiments.items():
        for horizon, data_txt in horizons.items():
            if not data_txt.exists():
                continue
            try:
                mae = parse_final_metric(data_txt, "mae")
            except ValueError:
                continue
            rows.append({"model": model_name, "horizon": horizon, "mae": mae})
    if not rows:
        raise ValueError("No baseline comparison metrics found.")
    return pd.DataFrame(rows)


def parse_docx_table_metrics(docx_path: Path) -> pd.DataFrame:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    rows: list[list[str]] = []
    for tr in root.findall(".//w:tr", ns):
        cells: list[str] = []
        for tc in tr.findall("./w:tc", ns):
            texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
            cells.append("".join(texts).strip())
        if any(cells):
            rows.append(cells)

    records: list[dict[str, object]] = []
    current_model = ""
    for row in rows[1:]:
        if row[0]:
            current_model = row[0]
        if not current_model:
            continue
        model_name = "KAN_TQNet" if current_model == "Your Model" else current_model
        try:
            records.append(
                {
                    "model": model_name,
                    "horizon": int(float(row[1])),
                    "electrical_mae": float(row[2]),
                    "cooling_mae": float(row[6]),
                    "heating_mae": float(row[10]),
                }
            )
        except (ValueError, IndexError):
            continue
    if not records:
        raise ValueError(f"No usable table data parsed from {docx_path}")
    return pd.DataFrame(records)


def parse_noise_docx(docx_path: Path) -> pd.DataFrame:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    rows: list[list[str]] = []
    for tr in root.findall(".//w:tr", ns):
        cells: list[str] = []
        for tc in tr.findall("./w:tc", ns):
            texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
            cells.append("".join(texts).strip())
        if any(cells):
            rows.append(cells)

    records: list[dict[str, object]] = []
    current_step = ""
    for row in rows[1:]:
        if row[0]:
            current_step = row[0]
        if not current_step:
            continue
        try:
            noise_level = float(row[1])
            mae = float(row[2])
            increase_ratio = 0.0 if row[3] in {"每", "", "-"} else float(row[3])
            mape = float(row[4])
        except (ValueError, IndexError):
            continue
        records.append(
            {
                "horizon": int(float(current_step)),
                "noise_level": noise_level,
                "mae": mae,
                "increase_ratio": increase_ratio,
                "mape": mape,
            }
        )
    if not records:
        raise ValueError(f"No usable noise table data parsed from {docx_path}")
    return pd.DataFrame(records)


def load_figure4_metrics() -> pd.DataFrame:
    corrected_path = ROOT / "fig4_corrected_baselines.csv"
    corrected_df = pd.read_csv(corrected_path)

    # Keep the KAN_TQNet row from the original benchmark table unless a corrected row is provided later.
    docx_df = parse_docx_table_metrics(Path(r"C:\Users\30217\Desktop\24.docx"))
    kan_df = docx_df[docx_df["model"] == "KAN_TQNet"].copy()

    combined = pd.concat([corrected_df, kan_df], ignore_index=True)
    combined["model"] = combined["model"].astype(str)
    combined["horizon"] = combined["horizon"].astype(int)
    return combined


def figure1_motivation() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        ("A", ROOT / "time_domain_overview.png", "Temporal dependency"),
        ("B", ROOT / "frequency_domain_periodogram.png", "Periodic structure"),
        ("C", ROOT / "nonlinear_feature_ranking.png", "Nonlinear coupling"),
    ]
    for ax, (label, img_path, title) in zip(axes, panels):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_label(ax, label)
    fig.suptitle("Fig. 1. Motivation for joint temporal, spectral, and nonlinear modeling", y=1.02)
    save_figure(fig, "Fig1_motivation_triptych.png")


def figure3_prediction_curves() -> None:
    base = ROOT / "output" / "KAN_TQNet_96_168_core_full_s0" / "result"
    files = [
        ("A", "Electricity load", base / "ele.csv", COLORS["primary"]),
        ("B", "Cooling load", base / "cooling.csv", COLORS["accent"]),
        ("C", "Heating load", base / "heating.csv", COLORS["accent2"]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (label, title, csv_path, color) in zip(axes, files):
        df = read_prediction_csv(csv_path, n=336)
        ax.plot(df.index, df["test_y"], color="#111827", linewidth=1.35, label="Ground truth")
        ax.plot(df.index, df["predicted_values"], color=color, linewidth=1.25, alpha=0.95, label="Prediction")
        mae = np.mean(np.abs(df["test_y"] - df["predicted_values"]))
        ax.set_title(f"{title} (MAE={mae:.2f})", loc="left", pad=6)
        ax.set_ylabel("Load")
        add_panel_label(ax, label)
        ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Fig. 3. Representative 96-step forecasting curves of KAN_TQNet", y=0.995)
    save_figure(fig, "Fig3_prediction_curves_96step.png")


def figure4_multi_horizon() -> None:
    df = load_figure4_metrics()
    channel_specs = [
        ("electrical_mae", "A", "Electrical load"),
        ("cooling_mae", "B", "Cooling load"),
        ("heating_mae", "C", "Heating load"),
    ]

    top_models: set[str] = {"KAN_TQNet"}
    for metric_col, _, _ in channel_specs:
        ranked = (
            df[df["model"] != "KAN_TQNet"]
            .groupby("model", as_index=False)[metric_col]
            .mean()
            .sort_values(metric_col)
            .head(5)["model"]
            .tolist()
        )
        top_models.update(ranked)

    plot_df = df[df["model"].isin(top_models)].copy()
    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=["KAN_TQNet"] + sorted(m for m in top_models if m != "KAN_TQNet"),
        ordered=True,
    )
    plot_df = plot_df.sort_values(["model", "horizon"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    palette = ["#0b3c5d", "#d95f02", "#1b9e77", "#7c3aed", "#374151", "#8b5e34"]
    models = [m for m in plot_df["model"].cat.categories if pd.notna(m)]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    color_map["KAN_TQNet"] = COLORS["primary"]
    marker_pool = ["o", "s", "^", "D", "P", "X"]
    marker_map = {m: marker_pool[i % len(marker_pool)] for i, m in enumerate(models)}

    for ax, (metric_col, panel_label, title) in zip(axes, channel_specs):
        for model_name, sub_df in plot_df.groupby("model", observed=False):
            if sub_df.empty:
                continue
            ax.plot(
                sub_df["horizon"],
                sub_df[metric_col],
                marker=marker_map[str(model_name)],
                linewidth=2.3 if str(model_name) == "KAN_TQNet" else 1.5,
                markersize=5.5,
                color=color_map[str(model_name)],
                alpha=1.0 if str(model_name) == "KAN_TQNet" else 0.9,
                label=str(model_name),
            )
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("MAE")
        ax.set_xticks([24, 48, 72, 96])
        add_panel_label(ax, panel_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fig. 4. Multi-horizon MAE comparison across representative baselines", y=1.14)
    save_figure(fig, "Fig4_multi_horizon_comparison.png")


def figure5_core_ablation() -> None:
    df = pd.read_csv(ROOT / "core_ablation_new_mae_matrix.csv")
    label_map = {
        "完整模型": "Full model",
        "去掉 KAN 主干": "Without KAN",
        "去掉 Temporal Query": "Without TQ",
        "去掉频域分支": "Without frequency branch",
        "去掉趋势残差分解": "Without trend residual",
        "去掉 HDA 路由": "Without HDA",
        "去掉 TimeMixer 精修": "Without TimeMixer",
    }
    df["description_en"] = df["description"].map(label_map).fillna(df["description"])
    heatmap_df = df.set_index("description_en")[["delta_h24", "delta_h48", "delta_h72", "delta_h96"]]
    data = heatmap_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    vmax = float(np.max(np.abs(data)))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4), labels=["24", "48", "72", "96"])
    ax.set_yticks(range(len(heatmap_df.index)), labels=list(heatmap_df.index))
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Ablation setting")
    ax.set_title("Fig. 5. Core ablation impact on MAE (ΔMAE vs. full model)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("ΔMAE")
    save_figure(fig, "Fig5_core_ablation_heatmap.png")


def figure6_channel_ablation() -> None:
    df = pd.read_csv(ROOT / "core_ablation_new_all_channels.csv")
    df = df[df["horizon"] == 96].copy()
    order = [
        "完整模型",
        "去掉 KAN 主干",
        "去掉 Temporal Query",
        "去掉频域分支",
        "去掉趋势残差分解",
        "去掉 HDA 路由",
        "去掉 TimeMixer 精修",
    ]
    df["description"] = pd.Categorical(df["description"], categories=order, ordered=True)
    df = df.sort_values(["description", "channel"])

    channel_titles = {
        "electricity": ("A", "Electricity"),
        "cooling": ("B", "Cooling"),
        "heating": ("C", "Heating"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, (channel, (label, title)) in zip(axes, channel_titles.items()):
        sub_df = df[df["channel"] == channel].sort_values("description")
        colors = [
            COLORS["primary"] if desc == "完整模型" else COLORS["accent"]
            for desc in sub_df["description"].astype(str).tolist()
        ]
        x = np.arange(len(sub_df))
        ax.bar(x, sub_df["mae"], color=colors, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x, labels=["Full", "-KAN", "-TQ", "-Freq", "-Trend", "-HDA", "-TM"], rotation=30)
        ax.set_title(title)
        ax.set_ylabel("MAE")
        add_panel_label(ax, label)
    fig.suptitle("Fig. 6. Channel-wise MAE under core ablation settings at horizon 96", y=1.02)
    save_figure(fig, "Fig6_channelwise_ablation_h96.png")


def figure7_progressive_gain() -> None:
    df = parse_progressive_metrics()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    line_colors = {24: COLORS["accent2"], 48: COLORS["accent"], 72: COLORS["primary"]}
    for horizon, sub_df in df.groupby("horizon"):
        sub_df = sub_df.sort_values("stage_label")
        ax.plot(
            sub_df["stage_label"],
            sub_df["mae"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=line_colors.get(int(horizon), COLORS["muted"]),
            label=f"Horizon {horizon}",
        )
    ax.set_xlabel("Progressive construction stage")
    ax.set_ylabel("MAE")
    ax.set_title("Fig. 7. Progressive performance gain from P0 to P8")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    save_figure(fig, "Fig7_progressive_gain.png")


def figure8_noise_robustness() -> None:
    df = parse_noise_docx(Path(r"C:\Users\30217\Desktop\噪声.docx"))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    horizons = sorted(df["horizon"].unique().tolist())
    line_colors = {
        24: COLORS["accent2"],
        48: COLORS["accent"],
        72: "#7c3aed",
        96: COLORS["primary"],
    }
    markers = {24: "o", 48: "s", 72: "^", 96: "D"}

    for horizon in horizons:
        sub_df = df[df["horizon"] == horizon].sort_values("noise_level")
        axes[0].plot(
            sub_df["noise_level"],
            sub_df["mae"],
            marker=markers.get(int(horizon), "o"),
            linewidth=1.9,
            markersize=5.2,
            color=line_colors.get(int(horizon), COLORS["muted"]),
            label=f"Horizon {horizon}",
        )
        axes[1].plot(
            sub_df["noise_level"],
            sub_df["increase_ratio"],
            marker=markers.get(int(horizon), "o"),
            linewidth=1.9,
            markersize=5.2,
            color=line_colors.get(int(horizon), COLORS["muted"]),
            label=f"Horizon {horizon}",
        )

    axes[0].set_title("MAE under random noise perturbation")
    axes[0].set_xlabel("Noise level")
    axes[0].set_ylabel("MAE")
    add_panel_label(axes[0], "A")

    axes[1].set_title("Relative MAE increase under noise")
    axes[1].set_xlabel("Noise level")
    axes[1].set_ylabel("Increase ratio (%)")
    add_panel_label(axes[1], "B")

    for ax in axes:
        ax.set_xticks([0.0, 0.01, 0.02, 0.05, 0.1, 0.2])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Robustness of KAN_TQNet under random noise masking", y=1.10)
    save_figure(fig, "Fig8_noise_robustness.png")


def main() -> None:
    configure_style()
    figure1_motivation()
    figure3_prediction_curves()
    figure4_multi_horizon()
    figure5_core_ablation()
    figure6_channel_ablation()
    figure7_progressive_gain()
    figure8_noise_robustness()
    print(f"Saved paper figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
