from __future__ import annotations

import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
SAVE_DIR = ROOT / "paper_figures_300dpi" / "KAN_TQNet_paper_layout"
DPI = 300
HORIZONS = [24, 48, 72, 96]
CHANNELS = [
    ("Electricity", 0, "#0b3c5d"),
    ("Cooling", 1, "#d95f02"),
    ("Heating", 2, "#1b9e77"),
]
CURVE_POINTS = 240


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d1d5db",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / filename, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def load_first_step_predictions(horizon: int) -> tuple[np.ndarray, np.ndarray]:
    run_dir = OUTPUT_ROOT / f"KAN_TQNet_{horizon}_168" / "result"
    y_true = np.load(run_dir / "all_y_true.npy")[:, 0, :]
    y_pred = np.load(run_dir / "all_predict_value.npy")[:, 0, :]
    return y_true, y_pred


def smooth_density(values: np.ndarray, bins: int = 45) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel /= kernel.sum()
    smoothed = np.convolve(density, kernel, mode="same")
    return centers, smoothed


def plot_prediction_curves_paper(results: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(len(HORIZONS), len(CHANNELS), figsize=(13.5, 10.8), sharex=True)
    panel_iter = iter(string.ascii_uppercase)

    for row_idx, horizon in enumerate(HORIZONS):
        y_true, y_pred = results[horizon]
        n = min(CURVE_POINTS, len(y_true))
        for col_idx, (channel_name, channel_idx, color) in enumerate(CHANNELS):
            ax = axes[row_idx, col_idx]
            actual = y_true[:n, channel_idx]
            pred = y_pred[:n, channel_idx]

            ax.plot(actual, color="#111827", linewidth=1.15, label="Actual")
            ax.plot(pred, color=color, linewidth=1.05, alpha=0.95, label="Prediction")
            ax.set_title(f"{channel_name}", pad=5)
            add_panel_label(ax, next(panel_iter))

            if col_idx == 0:
                ax.set_ylabel(f"{horizon}-step\nLoad")
            if row_idx == len(HORIZONS) - 1:
                ax.set_xlabel("Time index")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.58, 1.20))

    fig.suptitle("Multi-horizon prediction curves of KAN_TQNet", y=0.995, fontsize=12)
    fig.subplots_adjust(hspace=0.32, wspace=0.18, top=0.93)
    save_figure(fig, "Fig9_KAN_TQNet_prediction_curves_paper.png")


def plot_error_distribution_paper(results: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(len(HORIZONS), len(CHANNELS), figsize=(13.5, 10.8), sharex=False)
    panel_iter = iter(string.ascii_uppercase)

    for row_idx, horizon in enumerate(HORIZONS):
        y_true, y_pred = results[horizon]
        for col_idx, (channel_name, channel_idx, color) in enumerate(CHANNELS):
            ax = axes[row_idx, col_idx]
            error = y_pred[:, channel_idx] - y_true[:, channel_idx]
            centers, density = smooth_density(error)

            ax.hist(error, bins=42, density=True, color=color, alpha=0.28, edgecolor="white")
            ax.plot(centers, density, color=color, linewidth=1.5)
            ax.axvline(0.0, color="#111827", linewidth=1.0, linestyle="--")
            ax.axvline(float(np.mean(error)), color="#b91c1c", linewidth=1.0)
            ax.set_title(f"{channel_name}", pad=5)
            add_panel_label(ax, next(panel_iter))

            if col_idx == 0:
                ax.set_ylabel(f"{horizon}-step\nDensity")
            if row_idx == len(HORIZONS) - 1:
                ax.set_xlabel("Prediction error")

    fig.suptitle("Error distributions of first-step forecasts across horizons", y=0.995, fontsize=12)
    fig.subplots_adjust(hspace=0.32, wspace=0.18, top=0.93)
    save_figure(fig, "Fig10_KAN_TQNet_error_distribution_paper.png")


def write_caption_notes() -> None:
    text = "\n".join(
        [
            "Recommended paper captions",
            "",
            "Multi-horizon prediction curves of KAN_TQNet.",
            "Each row corresponds to a forecasting horizon (24, 48, 72, and 96), and each column corresponds to electricity, cooling, and heating load, respectively. The figure compares the first-step forecast sequence against the ground truth.",
            "",
            "Error distributions of first-step forecasts across horizons.",
            "Each row corresponds to a forecasting horizon (24, 48, 72, and 96), and each column corresponds to electricity, cooling, and heating load, respectively. Histograms and smoothed density curves illustrate the distribution of prediction errors, with vertical lines marking zero error and mean error.",
        ]
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "caption_notes.txt").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    results = {horizon: load_first_step_predictions(horizon) for horizon in HORIZONS}
    plot_prediction_curves_paper(results)
    plot_error_distribution_paper(results)
    write_caption_notes()
    print(f"Saved paper-layout figures to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
