from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_figures_300dpi" / "ablation_boxplots_300dpi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENT_ORDER = [
    "core_full",
    "core_no_kan",
    "core_no_tq",
    "core_no_freq",
    "core_no_hda",
    "core_no_timemixer",
]

EXPERIMENT_LABELS = {
    "core_full": "Full",
    "core_no_kan": "w/o KAN",
    "core_no_tq": "w/o TQ",
    "core_no_freq": "w/o Freq",
    "core_no_hda": "w/o HDA",
    "core_no_timemixer": "w/o MSR",
}

BOX_COLORS = [
    "#2a9d8f",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#577590",
    "#b56576",
]


def _styled_boxplot(ax, data, labels, colors):
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        widths=0.62,
        medianprops={"color": "#111111", "linewidth": 1.3},
        whiskerprops={"color": "#444444", "linewidth": 1.1},
        capprops={"color": "#444444", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "#444444",
            "markeredgecolor": "#444444",
            "alpha": 0.65,
        },
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="x", rotation=28)
    return bp


def plot_overall_boxplots():
    df = pd.read_csv(ROOT / "core_ablation_new_all_overall.csv")
    df = df[df["experiment_tag"].isin(EXPERIMENT_ORDER)].copy()

    metrics = [
        ("selected_overall_mae", "MAE"),
        ("selected_overall_rmse", "RMSE"),
        ("selected_overall_mape", "MAPE (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, (col, ylabel) in zip(axes, metrics):
        data = [
            df.loc[df["experiment_tag"] == tag, col].dropna().tolist()
            for tag in EXPERIMENT_ORDER
        ]
        labels = [EXPERIMENT_LABELS[tag] for tag in EXPERIMENT_ORDER]
        _styled_boxplot(ax, data, labels, BOX_COLORS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Overall {ylabel}")

    fig.suptitle("Ablation Boxplots Across Forecast Horizons", y=1.03, fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "ablation_overall_boxplots_300dpi.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_channel_mae_boxplots():
    df = pd.read_csv(ROOT / "core_ablation_new_all_channels.csv")
    df = df[df["experiment_tag"].isin(EXPERIMENT_ORDER)].copy()

    channel_order = ["electricity", "cooling", "heating"]
    channel_titles = {
        "electricity": "Electricity MAE",
        "cooling": "Cooling MAE",
        "heating": "Heating MAE",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, channel in zip(axes, channel_order):
        sub = df[df["channel"] == channel]
        data = [
            sub.loc[sub["experiment_tag"] == tag, "mae"].dropna().tolist()
            for tag in EXPERIMENT_ORDER
        ]
        labels = [EXPERIMENT_LABELS[tag] for tag in EXPERIMENT_ORDER]
        _styled_boxplot(ax, data, labels, BOX_COLORS)
        ax.set_ylabel("MAE")
        ax.set_title(channel_titles[channel])

    fig.suptitle("Channel-wise Ablation Boxplots Across Forecast Horizons", y=1.03, fontsize=14)
    fig.tight_layout()
    out_path = OUT_DIR / "ablation_channel_mae_boxplots_300dpi.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    overall_path = plot_overall_boxplots()
    channel_path = plot_channel_mae_boxplots()

    print(overall_path)
    print(channel_path)


if __name__ == "__main__":
    main()
